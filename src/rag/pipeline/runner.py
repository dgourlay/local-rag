from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rag.pipeline.chunker import chunk_document
from rag.pipeline.classifier import classify
from rag.pipeline.normalizer import normalize
from rag.pipeline.parser.base import get_parser
from rag.results import ParseSuccess, SectionSummarySuccess, SummarySuccess
from rag.types import (
    NAMESPACE_RAG,
    Chunk,
    ChunkRow,
    DocumentRow,
    FileType,
    NormalizedDocument,
    ParsedDocument,
    ProcessingLogEntry,
    ProcessingOutcome,
    QdrantPayloadModel,
    RecordType,
    SectionRow,
    SyncStateRow,
    VectorPoint,
)

if TYPE_CHECKING:
    from rag.config import AppConfig
    from rag.db.async_upsert import BackgroundUpsertWorker
    from rag.db.models import SqliteMetadataDB
    from rag.db.qdrant import AsyncQdrantVectorStore
    from rag.pipeline.dedup import DedupChecker
    from rag.protocols import Embedder, Parser, Summarizer, VectorStore
    from rag.types import FileEvent

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str, ProcessingOutcome, str], None]


@dataclass
class _ParsedFileResult:
    """Intermediate result from the parse-only stage of the pipeline.

    Holds everything needed by the main thread to run dedup, embed, and index.
    """

    event: FileEvent
    start: float
    parsed_doc: ParsedDocument
    normalized: NormalizedDocument
    chunks: list[Chunk]
    folder_path: str
    folder_ancestors: list[str]


@dataclass
class _ParseErrorResult:
    """Result when parsing failed for a file in the parser thread."""

    event: FileEvent
    error_msg: str


# Items that flow through the producer-consumer queue.
# None is the sentinel signaling the parser thread is done.
_QueueItem = _ParsedFileResult | _ParseErrorResult | None


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
        summarizer: Summarizer | None = None,
        async_vector_store: AsyncQdrantVectorStore | None = None,
    ) -> None:
        self._db = db
        self._vector_store = vector_store
        self._embedder = embedder
        self._parsers = parsers
        self._dedup = dedup
        self._config = config
        self._summarizer = summarizer
        self._async_vector_store = async_vector_store
        self._bg_worker: BackgroundUpsertWorker | None = None

    def _start_background_worker(self) -> None:
        """Start the async upsert background worker if an async store is available."""
        if self._async_vector_store is None:
            return
        try:
            from rag.db.async_upsert import BackgroundUpsertWorker

            self._bg_worker = BackgroundUpsertWorker(self._async_vector_store)
            self._bg_worker.start()
            logger.debug("Background upsert worker started")
        except Exception:
            logger.debug("Async Qdrant upserts not available, using sync fallback")
            self._bg_worker = None

    def _stop_background_worker(self) -> None:
        """Stop the async upsert background worker."""
        if self._bg_worker is not None:
            try:
                self._bg_worker.wait_all()
            except Exception:
                logger.exception("Error waiting for background upserts")
            finally:
                self._bg_worker.stop()
                self._bg_worker = None

    def process_file(self, event: FileEvent) -> tuple[ProcessingOutcome, str]:
        """Process a single file through the full pipeline.

        Returns (outcome, detail) where detail is a human-readable description.
        """
        file_path = event.file_path
        start = time.monotonic()

        try:
            if event.event_type == "deleted":
                self._handle_deletion(event)
                return ProcessingOutcome.DELETED, "removed from index"

            # Fast skip: if file content hasn't changed since last successful index, skip entirely
            existing_sync = self._db.get_sync_state(file_path)
            if (
                existing_sync is not None
                and existing_sync.content_hash == event.content_hash
                and existing_sync.process_status == "done"
                and not existing_sync.is_deleted
            ):
                return ProcessingOutcome.UNCHANGED, "content unchanged"

            path = Path(file_path)
            folder_path = str(path.parent)
            folder_ancestors = self._compute_ancestors(folder_path)

            sync_id = existing_sync.id if existing_sync else str(uuid.uuid4())
            retry_count = existing_sync.retry_count if existing_sync else 0
            self._db.upsert_sync_state(
                SyncStateRow(
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
                )
            )

            # 1. Classify
            classification = classify(file_path, folder_path)

            # 2. Parse
            parser = get_parser(classification.file_type, self._parsers)
            if parser is None:
                msg = f"No parser for file type: {classification.file_type}"
                raise ValueError(msg)

            result = parser.parse(
                file_path, classification.ocr_enabled, content_hash=event.content_hash
            )
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
                # Check if this is the same file unchanged (skip entirely)
                existing_doc = self._db.get_document_by_path(file_path)
                if existing_doc is not None and existing_doc.doc_id == canonical:
                    self._update_sync_status(file_path, "done")
                    self._log(canonical, file_path, "dedup", "unchanged", start, "skipped")
                    return ProcessingOutcome.UNCHANGED, "content unchanged"

                # Different file with same content — record as duplicate
                doc_id = str(uuid.uuid4())
                canonical_doc = self._db.get_document(canonical)
                canonical_name = Path(canonical_doc.file_path).name if canonical_doc else canonical
                self._db.upsert_document(
                    DocumentRow(
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
                    )
                )
                self._update_sync_status(file_path, "done")
                details = f"duplicate of {canonical}"
                self._log(doc_id, file_path, "dedup", "duplicate", start, details)
                return ProcessingOutcome.DUPLICATE, f"duplicate of {canonical_name}"

            doc_id = parsed_doc.doc_id

            # 5. Register hash for future dedup
            self._dedup.register_hash(
                file_path,
                normalized.raw_content_hash,
                normalized.normalized_content_hash,
                doc_id,
            )

            # 6. Save document metadata
            self._db.upsert_document(
                DocumentRow(
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
                )
            )

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
                points.append(
                    VectorPoint(
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
                    )
                )

            if points:
                self._upsert_points(doc_id, points)
                keep_ids = {p.point_id for p in points}
                self._delete_stale_points(doc_id, keep_ids)

            # 12. Summarize (if enabled)
            summary_points = self._summarize_document(
                doc_id=doc_id,
                title=parsed_doc.title or path.stem,
                file_path=file_path,
                folder_path=folder_path,
                folder_ancestors=folder_ancestors,
                file_type=event.file_type,
                modified_at=event.modified_at,
                normalized=normalized,
                section_rows=section_rows,
            )
            if summary_points:
                self._upsert_points(doc_id, summary_points)

            # Ensure all async upserts complete before marking file as done
            self._flush_background_upserts()

            self._update_sync_status(file_path, "done")
            self._log(doc_id, file_path, "pipeline", "success", start, f"{len(chunks)} chunks")
            return ProcessingOutcome.INDEXED, f"{len(chunks)} chunks"

        except Exception:
            logger.exception("Error processing %s", file_path)
            self._update_sync_status(file_path, "error", str(file_path))
            self._log(None, file_path, "pipeline", "error", start, file_path)
            return ProcessingOutcome.ERROR, "processing failed"

    def process_batch(
        self,
        events: list[FileEvent],
        progress: ProgressCallback | None = None,
    ) -> dict[ProcessingOutcome, int]:
        """Process a batch of file events with pipeline parallelism.

        Uses a producer-consumer pattern: a parser thread runs
        classify -> parse -> normalize -> chunk for each file, while the
        main thread consumes results to run dedup -> embed -> upsert ->
        summarize.

        Chunks from multiple documents are accumulated and embedded in
        cross-document batches (sized by ``config.embedding.batch_size``)
        for better GPU utilization.

        Returns counts per outcome.
        """
        if not events:
            return dict.fromkeys(ProcessingOutcome, 0)

        # Try to set up async Qdrant upserts for better I/O throughput
        self._start_background_worker()

        try:
            return self._process_batch_parallel(events, progress)
        finally:
            self._stop_background_worker()

    # ------------------------------------------------------------------
    # Pipeline parallelism internals
    # ------------------------------------------------------------------

    def _parse_stage(self, event: FileEvent) -> _ParsedFileResult:
        """Run the CPU/IO-bound parse stage for a single file.

        Runs in the parser thread.  Does NOT touch SQLite or Qdrant --
        only: classify, parse (subprocess), normalize, chunk.
        """
        start = time.monotonic()
        file_path = event.file_path
        path = Path(file_path)
        folder_path = str(path.parent)
        folder_ancestors = self._compute_ancestors(folder_path)

        # 1. Classify
        classification = classify(file_path, folder_path)

        # 2. Parse
        parser = get_parser(classification.file_type, self._parsers)
        if parser is None:
            msg = f"No parser for file type: {classification.file_type}"
            raise ValueError(msg)

        result = parser.parse(
            file_path, classification.ocr_enabled, content_hash=event.content_hash
        )
        if not isinstance(result, ParseSuccess):
            msg = f"Parse failed: {result.error}"
            raise ValueError(msg)

        parsed_doc = result.document

        # 3. Normalize
        normalized = normalize(parsed_doc)

        # 4. Chunk
        chunks = chunk_document(normalized)

        return _ParsedFileResult(
            event=event,
            start=start,
            parsed_doc=parsed_doc,
            normalized=normalized,
            chunks=chunks,
            folder_path=folder_path,
            folder_ancestors=folder_ancestors,
        )

    def _process_batch_parallel(
        self,
        events: list[FileEvent],
        progress: ProgressCallback | None,
    ) -> dict[ProcessingOutcome, int]:
        """Core implementation of the parallel batch pipeline."""
        counts: dict[ProcessingOutcome, int] = dict.fromkeys(ProcessingOutcome, 0)
        total = len(events)
        processed_count = 0
        batch_size = self._config.embedding.batch_size

        # Queue with maxsize=2 to limit memory (at most 2 parsed docs buffered)
        q: queue.Queue[_QueueItem] = queue.Queue(maxsize=2)

        def _parser_worker() -> None:
            """Parser thread: classify -> parse -> normalize -> chunk."""
            for event in events:
                try:
                    # Handle deletions immediately as skip results
                    if event.event_type == "deleted":
                        q.put(_ParseErrorResult(
                            event=event, error_msg="__deleted__",
                        ))
                        continue

                    item = self._parse_stage(event)
                    q.put(item)
                except Exception:
                    logger.exception("Parser thread error for %s", event.file_path)
                    q.put(_ParseErrorResult(
                        event=event, error_msg="processing failed",
                    ))
            # Sentinel: signal the consumer that we are done
            q.put(None)

        # Start the parser thread
        parser_thread = threading.Thread(target=_parser_worker, daemon=True)
        parser_thread.start()

        # -- Consumer state for cross-document chunk batching --
        pending: list[_ParsedFileResult] = []
        pending_chunk_count = 0

        def _report_progress(
            outcome: ProcessingOutcome, detail: str, file_path: str,
        ) -> None:
            nonlocal processed_count
            processed_count += 1
            counts[outcome] += 1
            if progress:
                progress(
                    processed_count, total,
                    Path(file_path).name, outcome, detail,
                )

        def _flush_pending() -> None:
            """Embed accumulated chunks across documents, then index each."""
            nonlocal pending, pending_chunk_count
            if not pending:
                return

            # Collect all chunk texts across pending documents
            all_texts: list[str] = []
            boundaries: list[tuple[int, int]] = []
            for pr in pending:
                start_idx = len(all_texts)
                all_texts.extend(c.text for c in pr.chunks)
                boundaries.append((start_idx, len(all_texts)))

            # Single cross-document embed_batch call
            all_vectors = self._embedder.embed_batch(all_texts) if all_texts else []

            # Index each document with its slice of vectors
            for pr, (si, ei) in zip(pending, boundaries, strict=True):
                doc_vectors = all_vectors[si:ei]
                try:
                    self._index_parsed_file(pr, doc_vectors)
                    _report_progress(
                        ProcessingOutcome.INDEXED,
                        f"{len(pr.chunks)} chunks",
                        pr.event.file_path,
                    )
                except Exception:
                    logger.exception("Error indexing %s", pr.event.file_path)
                    self._update_sync_status(
                        pr.event.file_path, "error", str(pr.event.file_path),
                    )
                    self._log(
                        None, pr.event.file_path, "pipeline", "error",
                        pr.start, pr.event.file_path,
                    )
                    _report_progress(
                        ProcessingOutcome.ERROR,
                        "processing failed",
                        pr.event.file_path,
                    )

            pending = []
            pending_chunk_count = 0

        # -- Main consumer loop --
        while True:
            item = q.get()

            # Sentinel: parser thread is done
            if item is None:
                _flush_pending()
                break

            # Parse error or deletion -- handle on main thread
            if isinstance(item, _ParseErrorResult):
                if item.error_msg == "__deleted__":
                    self._handle_deletion(item.event)
                    _report_progress(
                        ProcessingOutcome.DELETED,
                        "removed from index",
                        item.event.file_path,
                    )
                else:
                    start_t = time.monotonic()
                    self._ensure_sync_state(item.event)
                    self._update_sync_status(
                        item.event.file_path, "error", str(item.event.file_path),
                    )
                    self._log(
                        None, item.event.file_path, "pipeline", "error",
                        start_t, item.event.file_path,
                    )
                    _report_progress(
                        ProcessingOutcome.ERROR,
                        item.error_msg,
                        item.event.file_path,
                    )
                continue

            # _ParsedFileResult: run dedup on main thread (requires SQLite)
            pr = item
            dedup_result = self._run_dedup_check(pr)
            if dedup_result is not None:
                outcome, detail = dedup_result
                _report_progress(outcome, detail, pr.event.file_path)
                # Flush dedup hashes periodically
                if processed_count % 10 == 0:
                    self._dedup.flush()
                continue

            # File needs indexing -- accumulate for cross-document batching
            pending.append(pr)
            pending_chunk_count += len(pr.chunks)

            # Flush when accumulated chunks reach batch_size
            if pending_chunk_count >= batch_size:
                _flush_pending()

            # Flush dedup hashes periodically
            if processed_count % 10 == 0:
                self._dedup.flush()

        # Wait for the parser thread to finish
        parser_thread.join()

        # Final dedup flush
        self._dedup.flush()
        return counts

    def _ensure_sync_state(self, event: FileEvent) -> None:
        """Create sync_state for a file if it doesn't exist yet.

        In the parallel pipeline, sync_state isn't created during parsing
        (parser thread can't touch SQLite), so we create it on the main thread
        before any status updates.
        """
        existing = self._db.get_sync_state(event.file_path)
        if existing is not None:
            return
        path = Path(event.file_path)
        folder_path = str(path.parent)
        self._db.upsert_sync_state(
            SyncStateRow(
                id=str(uuid.uuid4()),
                file_path=event.file_path,
                file_name=path.name,
                folder_path=folder_path,
                folder_ancestors=self._compute_ancestors(folder_path),
                file_type=event.file_type.value,
                size_bytes=path.stat().st_size if path.exists() else None,
                modified_at=event.modified_at,
                content_hash=event.content_hash,
                process_status="pending",
            )
        )

    def _run_dedup_check(
        self, pr: _ParsedFileResult,
    ) -> tuple[ProcessingOutcome, str] | None:
        """Run dedup + fast-skip checks on the main thread.

        Returns (outcome, detail) if the file should be skipped, or None if
        it needs full indexing.
        """
        event = pr.event
        file_path = event.file_path
        normalized = pr.normalized

        # Fast skip: unchanged content
        existing_sync = self._db.get_sync_state(file_path)
        if (
            existing_sync is not None
            and existing_sync.content_hash == event.content_hash
            and existing_sync.process_status == "done"
            and not existing_sync.is_deleted
        ):
            return ProcessingOutcome.UNCHANGED, "content unchanged"

        # Dedup check
        canonical = self._dedup.check_duplicate(
            normalized.raw_content_hash, normalized.normalized_content_hash,
        )
        if canonical is not None:
            existing_doc = self._db.get_document_by_path(file_path)
            if existing_doc is not None and existing_doc.doc_id == canonical:
                self._ensure_sync_state(event)
                self._update_sync_status(file_path, "done")
                self._log(
                    canonical, file_path, "dedup", "unchanged", pr.start, "skipped",
                )
                return ProcessingOutcome.UNCHANGED, "content unchanged"

            # Different file with same content
            self._ensure_sync_state(event)
            doc_id = str(uuid.uuid4())
            canonical_doc = self._db.get_document(canonical)
            canonical_name = (
                Path(canonical_doc.file_path).name if canonical_doc else canonical
            )
            self._db.upsert_document(
                DocumentRow(
                    doc_id=doc_id,
                    file_path=file_path,
                    folder_path=pr.folder_path,
                    folder_ancestors=pr.folder_ancestors,
                    title=pr.parsed_doc.title,
                    file_type=event.file_type.value,
                    modified_at=event.modified_at,
                    raw_content_hash=normalized.raw_content_hash,
                    normalized_content_hash=normalized.normalized_content_hash,
                    duplicate_of_doc_id=canonical,
                )
            )
            self._update_sync_status(file_path, "done")
            details = f"duplicate of {canonical}"
            self._log(doc_id, file_path, "dedup", "duplicate", pr.start, details)
            return ProcessingOutcome.DUPLICATE, f"duplicate of {canonical_name}"

        # Register hash for future dedup
        self._dedup.register_hash(
            file_path,
            normalized.raw_content_hash,
            normalized.normalized_content_hash,
            pr.parsed_doc.doc_id,
        )

        return None

    def _index_parsed_file(
        self,
        pr: _ParsedFileResult,
        vectors: list[list[float]],
    ) -> None:
        """Index a parsed file: write metadata to SQLite, upsert vectors, summarize.

        Runs on the main thread.  ``vectors`` are pre-computed embeddings for
        ``pr.chunks`` (from the cross-document batch).
        """
        event = pr.event
        file_path = event.file_path
        path = Path(file_path)
        doc_id = pr.parsed_doc.doc_id
        parsed_doc = pr.parsed_doc

        # Upsert sync state as processing
        existing_sync = self._db.get_sync_state(file_path)
        sync_id = existing_sync.id if existing_sync else str(uuid.uuid4())
        retry_count = existing_sync.retry_count if existing_sync else 0
        self._db.upsert_sync_state(
            SyncStateRow(
                id=sync_id,
                file_path=file_path,
                file_name=path.name,
                folder_path=pr.folder_path,
                folder_ancestors=pr.folder_ancestors,
                file_type=event.file_type.value,
                size_bytes=path.stat().st_size if path.exists() else None,
                modified_at=event.modified_at,
                content_hash=event.content_hash,
                process_status="processing",
                retry_count=retry_count,
            )
        )

        # Save document metadata
        self._db.upsert_document(
            DocumentRow(
                doc_id=doc_id,
                file_path=file_path,
                folder_path=pr.folder_path,
                folder_ancestors=pr.folder_ancestors,
                title=parsed_doc.title,
                file_type=event.file_type.value,
                modified_at=event.modified_at,
                raw_content_hash=pr.normalized.raw_content_hash,
                normalized_content_hash=pr.normalized.normalized_content_hash,
                ocr_required=1 if parsed_doc.ocr_required else 0,
                ocr_confidence=parsed_doc.ocr_confidence,
                embedding_model_version=self._embedder.model_version,
            )
        )

        # Save sections
        section_rows = [
            SectionRow(
                section_id=str(uuid.uuid5(NAMESPACE_RAG, f"{doc_id}:section:{s.order}")),
                doc_id=doc_id,
                section_heading=s.heading,
                section_order=s.order,
                page_start=s.page_start,
                page_end=s.page_end,
            )
            for s in pr.normalized.sections
        ]
        self._db.insert_sections(section_rows)

        # Save chunks to DB
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
            for c in pr.chunks
        ]
        self._db.insert_chunks(chunk_rows)

        # Build VectorPoints from pre-computed vectors
        points: list[VectorPoint] = []
        for chunk, vector in zip(pr.chunks, vectors, strict=True):
            points.append(
                VectorPoint(
                    point_id=chunk.chunk_id,
                    vector=vector,
                    payload=QdrantPayloadModel(
                        record_type=RecordType.CHUNK,
                        doc_id=doc_id,
                        section_id=chunk.section_id,
                        chunk_id=chunk.chunk_id,
                        title=parsed_doc.title or path.stem,
                        file_path=file_path,
                        folder_path=pr.folder_path,
                        folder_ancestors=pr.folder_ancestors,
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
                )
            )

        if points:
            self._upsert_points(doc_id, points)
            keep_ids = {p.point_id for p in points}
            self._delete_stale_points(doc_id, keep_ids)

        # Summarize (if enabled)
        summary_points = self._summarize_document(
            doc_id=doc_id,
            title=parsed_doc.title or path.stem,
            file_path=file_path,
            folder_path=pr.folder_path,
            folder_ancestors=pr.folder_ancestors,
            file_type=event.file_type,
            modified_at=event.modified_at,
            normalized=pr.normalized,
            section_rows=section_rows,
        )
        if summary_points:
            self._upsert_points(doc_id, summary_points)

        # Ensure all async upserts complete before marking file as done
        self._flush_background_upserts()

        self._update_sync_status(file_path, "done")
        self._log(
            doc_id, file_path, "pipeline", "success",
            pr.start, f"{len(pr.chunks)} chunks",
        )

    # ------------------------------------------------------------------
    # Async Qdrant helpers
    # ------------------------------------------------------------------

    def _upsert_points(self, doc_id: str, points: list[VectorPoint]) -> None:
        """Upsert points using the background worker if available, else sync."""
        if self._bg_worker is not None:
            self._bg_worker.submit_upsert(doc_id, points)
        else:
            self._vector_store.upsert_points(doc_id, points)

    def _delete_stale_points(self, doc_id: str, keep_ids: set[str]) -> None:
        """Delete stale points using the background worker if available, else sync."""
        if self._bg_worker is not None:
            self._bg_worker.submit_delete_stale(doc_id, keep_ids)
        else:
            self._vector_store.delete_stale_points(doc_id, keep_ids)

    def _flush_background_upserts(self) -> None:
        """Wait for all pending background upserts to complete."""
        if self._bg_worker is not None:
            errors = self._bg_worker.wait_all()
            if errors:
                logger.warning(
                    "Background upserts had %d error(s)", len(errors)
                )

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def _summarize_document(
        self,
        doc_id: str,
        title: str,
        file_path: str,
        folder_path: str,
        folder_ancestors: list[str],
        file_type: FileType,
        modified_at: str,
        normalized: NormalizedDocument,
        section_rows: list[SectionRow],
    ) -> list[VectorPoint]:
        """Run summarization and create summary vector points. Returns summary points."""
        if self._summarizer is None or not self._summarizer.available:
            return []

        full_text = "\n\n".join(s.text for s in normalized.sections)

        # --- Phase 1: Document-level summary (sequential, provides context) ---
        doc_result = self._summarizer.summarize_document(
            full_text, title, normalized.file_type.value
        )

        doc_summary_text: str | None = None
        doc_type_guess: str | None = None
        if isinstance(doc_result, SummarySuccess):
            existing_doc = self._db.get_document(doc_id)
            if existing_doc is not None:
                updated = existing_doc.model_copy(
                    update={
                        "summary_l1": doc_result.summary_l1,
                        "summary_l2": doc_result.summary_l2,
                        "summary_l3": doc_result.summary_l3,
                        "key_topics": doc_result.key_topics,
                        "doc_type_guess": doc_result.doc_type_guess,
                        "summary_content_hash": normalized.normalized_content_hash,
                    }
                )
                self._db.upsert_document(updated)

            doc_summary_text = doc_result.summary_l3
            doc_type_guess = doc_result.doc_type_guess
            logger.info("Generated document summary for %s", file_path)
        else:
            logger.warning("Document summarization failed for %s: %s", file_path, doc_result.error)

        # --- Phase 2: Section-level summaries (parallel via ThreadPoolExecutor) ---
        doc_context = f"{title} ({normalized.file_type.value})"

        # Build list of (section, section_row) pairs to summarize
        section_pairs = [
            (section, section_row)
            for section, section_row in zip(normalized.sections, section_rows, strict=False)
            if section.text.strip()
        ]

        # Submit all section summaries in parallel
        section_results: list[tuple[SectionSummarySuccess, SectionRow, int]] = []
        max_workers = self._config.summarization.max_workers

        def _summarize_one_section(
            section_text: str, heading: str | None, context: str
        ) -> SectionSummarySuccess | None:
            result = self._summarizer.summarize_section(section_text, heading, context)  # type: ignore[union-attr]
            if isinstance(result, SectionSummarySuccess):
                return result
            return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    _summarize_one_section, section.text, section.heading, doc_context
                ): i
                for i, (section, _section_row) in enumerate(section_pairs)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                section, section_row = section_pairs[idx]
                try:
                    sec_result = future.result()
                except Exception:
                    logger.exception(
                        "Section summarization raised for %s section %d",
                        file_path,
                        section.order,
                    )
                    continue

                if sec_result is not None:
                    # Update section row with summary
                    updated_section = section_row.model_copy(
                        update={
                            "section_summary": sec_result.section_summary,
                            "section_summary_l2": sec_result.section_summary_l2,
                            "embedding_model_version": self._embedder.model_version,
                        }
                    )
                    self._db.insert_sections([updated_section])
                    section_results.append((sec_result, section_row, section.order))
                    logger.info(
                        "Generated section summary for %s section %d",
                        file_path,
                        section.order,
                    )
                else:
                    logger.warning(
                        "Section summarization failed for %s section %d",
                        file_path,
                        section.order,
                    )

        # --- Phase 3: Batch-embed ALL summary texts in one call ---
        embed_entries: list[dict[str, object]] = []

        if doc_summary_text is not None:
            embed_entries.append({
                "text": doc_summary_text,
                "type": "document",
                "doc_type_guess": doc_type_guess,
            })

        # Sort section results by order for deterministic embedding order
        section_results.sort(key=lambda t: t[2])
        for sec_result, section_row, order in section_results:
            embed_entries.append({
                "text": sec_result.section_summary,
                "type": "section",
                "section_row": section_row,
                "order": order,
                "heading": section_row.section_heading,
            })

        if not embed_entries:
            return []

        all_texts = [str(entry["text"]) for entry in embed_entries]
        all_vectors = self._embedder.embed_batch(all_texts)

        # --- Phase 4: Build VectorPoints from zipped results ---
        summary_points: list[VectorPoint] = []
        for entry, vector in zip(embed_entries, all_vectors, strict=True):
            if entry["type"] == "document":
                point_id = str(uuid.uuid5(NAMESPACE_RAG, f"{doc_id}:document_summary"))
                summary_points.append(
                    VectorPoint(
                        point_id=point_id,
                        vector=vector,
                        payload=QdrantPayloadModel(
                            record_type=RecordType.DOCUMENT_SUMMARY,
                            summary_level="l3",
                            doc_id=doc_id,
                            title=title,
                            file_path=file_path,
                            folder_path=folder_path,
                            folder_ancestors=folder_ancestors,
                            file_type=file_type,
                            modified_at=modified_at,
                            doc_type_guess=(
                                str(entry["doc_type_guess"]) if entry["doc_type_guess"] else None
                            ),
                            text=str(entry["text"]),
                        ),
                    )
                )
            else:
                sec_row = entry["section_row"]
                order = entry["order"]
                sec_point_id = str(
                    uuid.uuid5(NAMESPACE_RAG, f"{doc_id}:section_summary:{order}")
                )
                summary_points.append(
                    VectorPoint(
                        point_id=sec_point_id,
                        vector=vector,
                        payload=QdrantPayloadModel(
                            record_type=RecordType.SECTION_SUMMARY,
                            doc_id=doc_id,
                            section_id=sec_row.section_id,  # type: ignore[union-attr]
                            title=title,
                            file_path=file_path,
                            folder_path=folder_path,
                            folder_ancestors=folder_ancestors,
                            file_type=file_type,
                            modified_at=modified_at,
                            section_heading=str(entry["heading"]) if entry["heading"] else None,
                            text=str(entry["text"]),
                        ),
                    )
                )

        return summary_points

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _handle_deletion(self, event: FileEvent) -> None:
        """Mark file as deleted in sync_state and remove vectors from Qdrant."""
        existing = self._db.get_sync_state(event.file_path)
        if existing:
            self._db.upsert_sync_state(
                SyncStateRow(
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
                )
            )
            doc = self._db.get_document_by_hash(existing.content_hash)
            if doc:
                self._delete_stale_points(doc.doc_id, set())

    def _update_sync_status(self, file_path: str, status: str, error: str | None = None) -> None:
        existing = self._db.get_sync_state(file_path)
        if existing:
            retry = existing.retry_count + (1 if status == "error" else 0)
            final_status = "poison" if status == "error" and retry >= 3 else status
            self._db.upsert_sync_state(
                SyncStateRow(
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
                )
            )

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
        self._db.log_processing(
            ProcessingLogEntry(
                doc_id=doc_id,
                file_path=file_path,
                stage=stage,
                status=status,
                duration_ms=duration,
                details=details,
            )
        )
