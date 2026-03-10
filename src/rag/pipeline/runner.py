from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from rag.pipeline.chunker import chunk_document
from rag.pipeline.classifier import classify
from rag.pipeline.normalizer import normalize
from rag.pipeline.parser.base import get_parser
from rag.results import ParseSuccess
from rag.types import (
    NAMESPACE_RAG,
    ChunkRow,
    DocumentRow,
    ProcessingLogEntry,
    QdrantPayloadModel,
    RecordType,
    SectionRow,
    SyncStateRow,
    VectorPoint,
)

if TYPE_CHECKING:
    from rag.config import AppConfig
    from rag.db.models import SqliteMetadataDB
    from rag.pipeline.dedup import DedupChecker
    from rag.protocols import Embedder, Parser, VectorStore
    from rag.types import FileEvent

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


class PipelineRunner:
    """Orchestrate the full indexing pipeline.

    Stages: classify -> parse -> normalize -> dedup -> chunk -> embed -> index.
    """

    def __init__(
        self,
        db: SqliteMetadataDB,
        vector_store: VectorStore,
        embedder: Embedder,
        parsers: list[Parser],
        dedup: DedupChecker,
        config: AppConfig,
    ) -> None:
        self._db = db
        self._vector_store = vector_store
        self._embedder = embedder
        self._parsers = parsers
        self._dedup = dedup
        self._config = config

    def process_file(self, event: FileEvent) -> bool:
        """Process a single file through the full pipeline. Returns True on success."""
        file_path = event.file_path
        start = time.monotonic()

        try:
            if event.event_type == "deleted":
                self._handle_deletion(event)
                return True

            path = Path(file_path)
            folder_path = str(path.parent)
            folder_ancestors = self._compute_ancestors(folder_path)

            existing_sync = self._db.get_sync_state(file_path)
            sync_id = existing_sync.id if existing_sync else str(uuid.uuid4())
            retry_count = existing_sync.retry_count if existing_sync else 0
            self._db.upsert_sync_state(SyncStateRow(
                id=sync_id,
                file_path=file_path,
                file_name=path.name,
                folder_path=folder_path,
                folder_ancestors=folder_ancestors,
                file_type=event.file_type.value,
                size_bytes=path.stat().st_size if path.exists() else None,
                modified_at=event.modified_at,
                content_hash=event.content_hash,
                process_status="processing",
                retry_count=retry_count,
            ))

            # 1. Classify
            classification = classify(file_path, folder_path)

            # 2. Parse
            parser = get_parser(classification.file_type, self._parsers)
            if parser is None:
                msg = f"No parser for file type: {classification.file_type}"
                raise ValueError(msg)

            result = parser.parse(file_path, classification.ocr_enabled)
            if not isinstance(result, ParseSuccess):
                msg = f"Parse failed: {result.error}"
                raise ValueError(msg)

            parsed_doc = result.document

            # 3. Normalize
            normalized = normalize(parsed_doc)

            # 4. Dedup check
            canonical = self._dedup.check_duplicate(
                normalized.raw_content_hash, normalized.normalized_content_hash
            )
            if canonical is not None:
                doc_id = str(uuid.uuid4())
                self._db.upsert_document(DocumentRow(
                    doc_id=doc_id,
                    file_path=file_path,
                    folder_path=folder_path,
                    folder_ancestors=folder_ancestors,
                    title=parsed_doc.title,
                    file_type=event.file_type.value,
                    modified_at=event.modified_at,
                    raw_content_hash=normalized.raw_content_hash,
                    normalized_content_hash=normalized.normalized_content_hash,
                    duplicate_of_doc_id=canonical,
                ))
                self._update_sync_status(file_path, "done")
                details = f"duplicate of {canonical}"
                self._log(doc_id, file_path, "dedup", "duplicate", start, details)
                return True

            doc_id = parsed_doc.doc_id

            # 5. Register hash for future dedup
            self._dedup.register_hash(
                file_path, normalized.raw_content_hash,
                normalized.normalized_content_hash, doc_id,
            )

            # 6. Save document metadata
            self._db.upsert_document(DocumentRow(
                doc_id=doc_id,
                file_path=file_path,
                folder_path=folder_path,
                folder_ancestors=folder_ancestors,
                title=parsed_doc.title,
                file_type=event.file_type.value,
                modified_at=event.modified_at,
                raw_content_hash=normalized.raw_content_hash,
                normalized_content_hash=normalized.normalized_content_hash,
                ocr_required=1 if parsed_doc.ocr_required else 0,
                ocr_confidence=parsed_doc.ocr_confidence,
                embedding_model_version=self._embedder.model_version,
            ))

            # 7. Save sections
            section_rows = [
                SectionRow(
                    section_id=str(uuid.uuid5(NAMESPACE_RAG, f"{doc_id}:section:{s.order}")),
                    doc_id=doc_id,
                    section_heading=s.heading,
                    section_order=s.order,
                    page_start=s.page_start,
                    page_end=s.page_end,
                )
                for s in normalized.sections
            ]
            self._db.insert_sections(section_rows)

            # 8. Chunk
            chunks = chunk_document(normalized)

            # 9. Save chunks to DB
            chunk_rows = [
                ChunkRow(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    section_id=c.section_id,
                    chunk_order=c.chunk_order,
                    chunk_text=c.text,
                    chunk_text_normalized=c.text_normalized,
                    page_start=c.page_start,
                    page_end=c.page_end,
                    section_heading=c.section_heading,
                    citation_label=c.citation_label,
                    token_count=c.token_count,
                    embedding_model_version=self._embedder.model_version,
                )
                for c in chunks
            ]
            self._db.insert_chunks(chunk_rows)

            # 10. Embed
            texts = [c.text for c in chunks]
            vectors = self._embedder.embed_batch(texts) if texts else []

            # 11. Build VectorPoints and upsert
            points: list[VectorPoint] = []
            for chunk, vector in zip(chunks, vectors, strict=True):
                points.append(VectorPoint(
                    point_id=chunk.chunk_id,
                    vector=vector,
                    payload=QdrantPayloadModel(
                        record_type=RecordType.CHUNK,
                        doc_id=doc_id,
                        section_id=chunk.section_id,
                        chunk_id=chunk.chunk_id,
                        title=parsed_doc.title or path.stem,
                        file_path=file_path,
                        folder_path=folder_path,
                        folder_ancestors=folder_ancestors,
                        file_type=event.file_type,
                        modified_at=event.modified_at,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        section_heading=chunk.section_heading,
                        chunk_order=chunk.chunk_order,
                        token_count=chunk.token_count,
                        citation_label=chunk.citation_label,
                        text=chunk.text,
                    ),
                ))

            if points:
                self._vector_store.upsert_points(doc_id, points)
                keep_ids = {p.point_id for p in points}
                self._vector_store.delete_stale_points(doc_id, keep_ids)

            self._update_sync_status(file_path, "done")
            self._log(doc_id, file_path, "pipeline", "success", start, f"{len(chunks)} chunks")
            return True

        except Exception:
            logger.exception("Error processing %s", file_path)
            self._update_sync_status(file_path, "error", str(file_path))
            self._log(None, file_path, "pipeline", "error", start, file_path)
            return False

    def process_batch(
        self,
        events: list[FileEvent],
        progress: ProgressCallback | None = None,
    ) -> tuple[int, int]:
        """Process a batch of file events. Returns (success_count, error_count)."""
        success = 0
        errors = 0
        for i, event in enumerate(events):
            if progress:
                progress(i + 1, len(events), Path(event.file_path).name)
            if self.process_file(event):
                success += 1
            else:
                errors += 1
        return success, errors

    def _handle_deletion(self, event: FileEvent) -> None:
        """Mark file as deleted in sync_state and remove vectors from Qdrant."""
        existing = self._db.get_sync_state(event.file_path)
        if existing:
            self._db.upsert_sync_state(SyncStateRow(
                id=existing.id,
                file_path=existing.file_path,
                file_name=existing.file_name,
                folder_path=existing.folder_path,
                folder_ancestors=existing.folder_ancestors,
                file_type=existing.file_type,
                modified_at=existing.modified_at,
                content_hash=existing.content_hash,
                process_status="done",
                is_deleted=1,
            ))
            doc = self._db.get_document_by_hash(existing.content_hash)
            if doc:
                self._vector_store.delete_stale_points(doc.doc_id, set())

    def _update_sync_status(
        self, file_path: str, status: str, error: str | None = None
    ) -> None:
        existing = self._db.get_sync_state(file_path)
        if existing:
            retry = existing.retry_count + (1 if status == "error" else 0)
            final_status = "poison" if status == "error" and retry >= 3 else status
            self._db.upsert_sync_state(SyncStateRow(
                id=existing.id,
                file_path=existing.file_path,
                file_name=existing.file_name,
                folder_path=existing.folder_path,
                folder_ancestors=existing.folder_ancestors,
                file_type=existing.file_type,
                modified_at=existing.modified_at,
                content_hash=existing.content_hash,
                process_status=final_status,
                error_message=error,
                retry_count=retry,
                is_deleted=existing.is_deleted,
            ))

    def _compute_ancestors(self, folder_path: str) -> list[str]:
        """Compute folder ancestor paths."""
        parts = Path(folder_path).parts
        ancestors: list[str] = []
        for i in range(len(parts)):
            ancestors.append(str(Path(*parts[: i + 1])))
        return ancestors

    def _log(
        self,
        doc_id: str | None,
        file_path: str,
        stage: str,
        status: str,
        start: float,
        details: str,
    ) -> None:
        duration = int((time.monotonic() - start) * 1000)
        self._db.log_processing(ProcessingLogEntry(
            doc_id=doc_id,
            file_path=file_path,
            stage=stage,
            status=status,
            duration_ms=duration,
            details=details,
        ))
