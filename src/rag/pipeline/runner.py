from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, cast

from rag.pipeline.chunker import chunk_document, get_chunker_version
from rag.pipeline.classifier import classify
from rag.pipeline.normalizer import normalize
from rag.pipeline.parser.base import get_parser
from rag.pipeline.summarizer import build_augmented_text
from rag.results import (
    CombinedSummarySuccess,
    ParseSuccess,
    SectionSummarySuccess,
    SummarySuccess,
)
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
StartCallback = Callable[[int, int, str], None]
StatusCallback = Callable[[int, int, str, str], None]


@dataclass
class _ParsedFileResult:
    """Intermediate result from the parse-only stage of the pipeline.

    Holds everything needed by the main thread to run dedup, embed, and index.
    """

    event: FileEvent
    file_index: int
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
    file_index: int
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

            # Check poison quarantine before processing
            existing_sync = self._db.get_sync_state(file_path)
            if (
                existing_sync is not None
                and existing_sync.process_status == "poison"
            ):
                return (
                    ProcessingOutcome.ERROR,
                    "quarantined \u2014 file failed 3+ times",
                )

            # Fast skip: if file content hasn't changed since last successful index, skip entirely
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

            # 5. Save document metadata
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
                    chunker_version=get_chunker_version(self._config.chunking.strategy),
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
            chunks = chunk_document(normalized, self._config.chunking, self._embedder)

            # 8.5. Generate questions (if enabled)
            if (
                self._summarizer is not None
                and self._summarizer.available
                and self._config.questions.enabled
            ):
                from rag.pipeline.summarizer import CliSummarizer

                if isinstance(self._summarizer, CliSummarizer):
                    chunks = self._summarizer.generate_chunk_questions(
                        chunks, parsed_doc.title,
                    )

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
                    generated_questions=(
                        json.dumps(c.generated_questions)
                        if c.generated_questions else None
                    ),
                    embedding_model_version=self._embedder.model_version,
                )
                for c in chunks
            ]
            self._db.insert_chunks(chunk_rows)

            # 10. Embed (using augmented text when questions are available)
            texts = [
                build_augmented_text(c.text, c.generated_questions)
                if c.generated_questions else c.text
                for c in chunks
            ]
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
                            generated_questions=chunk.generated_questions,
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

            # Register dedup hash only after successful indexing
            self._dedup.register_hash(
                file_path,
                normalized.raw_content_hash,
                normalized.normalized_content_hash,
                doc_id,
            )

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
        on_start: StartCallback | None = None,
        on_status: StatusCallback | None = None,
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
            return self._process_batch_parallel(events, progress, on_start, on_status)
        finally:
            self._stop_background_worker()

    # ------------------------------------------------------------------
    # Pipeline parallelism internals
    # ------------------------------------------------------------------

    def _parse_stage(self, event: FileEvent, file_index: int = 0) -> _ParsedFileResult:
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
        chunks = chunk_document(normalized, self._config.chunking, self._embedder)

        return _ParsedFileResult(
            event=event,
            file_index=file_index,
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
        on_start: StartCallback | None = None,
        on_status: StatusCallback | None = None,
    ) -> dict[ProcessingOutcome, int]:
        """Core implementation of the parallel batch pipeline."""
        counts: dict[ProcessingOutcome, int] = dict.fromkeys(ProcessingOutcome, 0)
        total = len(events)

        # Pre-filter: skip poisoned and backed-off files before starting threads
        eligible: list[tuple[int, FileEvent]] = []  # (file_index, event)
        for file_idx, event in enumerate(events, 1):
            if event.event_type == "deleted":
                eligible.append((file_idx, event))
                continue
            existing = self._db.get_sync_state(event.file_path)
            skip = self._check_skip_retry(existing)
            if skip is not None:
                outcome, detail = skip
                counts[outcome] += 1
                if progress:
                    progress(
                        file_idx, total,
                        Path(event.file_path).name, outcome, detail,
                    )
                continue
            eligible.append((file_idx, event))

        processed_count = sum(counts.values())
        batch_size = self._config.embedding.batch_size

        # Queue with maxsize=2 to limit memory (at most 2 parsed docs buffered)
        q: queue.Queue[_QueueItem] = queue.Queue(maxsize=2)

        def _parser_worker() -> None:
            """Parser thread: classify -> parse -> normalize -> chunk."""
            for file_idx, event in eligible:
                try:
                    # Handle deletions immediately as skip results
                    if event.event_type == "deleted":
                        q.put(_ParseErrorResult(
                            event=event, file_index=file_idx,
                            error_msg="__deleted__",
                        ))
                        continue

                    item = self._parse_stage(event, file_index=file_idx)
                    q.put(item)
                except Exception as exc:
                    logger.warning("Parse error for %s: %s", event.file_path, exc)
                    q.put(_ParseErrorResult(
                        event=event, file_index=file_idx,
                        error_msg="processing failed",
                    ))
            # Sentinel: signal the consumer that we are done
            q.put(None)

        # Start the parser thread
        parser_thread = threading.Thread(target=_parser_worker, daemon=True)
        parser_thread.start()

        # -- Consumer state for cross-document chunk batching --
        pending: list[_ParsedFileResult] = []
        pending_chunk_count = 0

        # -- Cross-file question generation parallelism --
        # Files whose question generation is in-flight via the summarizer's pool.
        in_flight_questions: list[tuple[_ParsedFileResult, Future[list[Chunk]]]] = []
        _questions_enabled = (
            self._summarizer is not None
            and self._summarizer.available
            and self._config.questions.enabled
        )
        from rag.pipeline.summarizer import CliSummarizer

        _cli_summarizer: CliSummarizer | None = None
        if _questions_enabled and isinstance(self._summarizer, CliSummarizer):
            _cli_summarizer = self._summarizer

        def _report_progress(
            outcome: ProcessingOutcome, detail: str, file_path: str,
            file_index: int = 0,
        ) -> None:
            nonlocal processed_count
            processed_count += 1
            counts[outcome] += 1
            if progress:
                progress(
                    file_index, total,
                    Path(file_path).name, outcome, detail,
                )

        def _collect_completed_questions() -> None:
            """Move files with completed question generation into the pending buffer."""
            nonlocal pending, pending_chunk_count
            still_in_flight: list[tuple[_ParsedFileResult, Future[list[Chunk]]]] = []
            for pr, future in in_flight_questions:
                if future.done():
                    try:
                        pr.chunks = future.result()
                    except Exception:
                        logger.warning(
                            "Question generation failed for %s, proceeding without",
                            pr.event.file_path,
                        )
                    pending.append(pr)
                    pending_chunk_count += len(pr.chunks)
                else:
                    still_in_flight.append((pr, future))
            in_flight_questions[:] = still_in_flight

        def _drain_all_questions() -> None:
            """Wait for all in-flight question generation to complete."""
            nonlocal pending, pending_chunk_count
            for pr, future in in_flight_questions:
                try:
                    pr.chunks = future.result()
                except Exception:
                    logger.warning(
                        "Question generation failed for %s, proceeding without",
                        pr.event.file_path,
                    )
                pending.append(pr)
                pending_chunk_count += len(pr.chunks)
            in_flight_questions.clear()

        def _flush_pending() -> None:
            """Embed accumulated chunks across documents, then index each."""
            nonlocal pending, pending_chunk_count
            if not pending:
                return

            # Collect all chunk texts across pending documents (augmented when questions available)
            all_texts: list[str] = []
            boundaries: list[tuple[int, int]] = []
            for pr in pending:
                start_idx = len(all_texts)
                all_texts.extend(
                    build_augmented_text(c.text, c.generated_questions)
                    if c.generated_questions else c.text
                    for c in pr.chunks
                )
                boundaries.append((start_idx, len(all_texts)))

            # Single cross-document embed_batch call
            all_vectors = self._embedder.embed_batch(all_texts) if all_texts else []

            # Index each document with its slice of vectors
            for pr, (si, ei) in zip(pending, boundaries, strict=True):
                doc_vectors = all_vectors[si:ei]
                try:
                    self._index_parsed_file(pr, doc_vectors, on_status=on_status)
                    # Register dedup hash only after successful indexing
                    self._dedup.register_hash(
                        pr.event.file_path,
                        pr.normalized.raw_content_hash,
                        pr.normalized.normalized_content_hash,
                        pr.parsed_doc.doc_id,
                    )
                    _report_progress(
                        ProcessingOutcome.INDEXED,
                        f"{len(pr.chunks)} chunks",
                        pr.event.file_path,
                        pr.file_index,
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
                        pr.file_index,
                    )

            pending = []
            pending_chunk_count = 0

        # -- In-flight question generation cap --
        # Keep the LLM pool fed without unbounded queue buildup.
        max_in_flight = self._config.summarization.max_concurrent_llm + 2

        def _wait_for_in_flight_room() -> None:
            """Block until in_flight_questions drops below max_in_flight."""
            while len(in_flight_questions) >= max_in_flight:
                # Brief sleep to avoid busy-waiting, then sweep completed futures
                time.sleep(0.05)
                _collect_completed_questions()
                if pending_chunk_count >= batch_size:
                    _flush_pending()

        # -- Main consumer loop --
        while True:
            # Collect any completed question-generation futures
            _collect_completed_questions()

            # Flush when accumulated chunks reach batch_size
            if pending_chunk_count >= batch_size:
                _flush_pending()

            item = q.get()

            # Sentinel: parser thread is done
            if item is None:
                _drain_all_questions()
                _flush_pending()
                break

            # Parse error or deletion -- handle on main thread
            if isinstance(item, _ParseErrorResult):
                if on_start:
                    on_start(item.file_index, total, Path(item.event.file_path).name)
                if item.error_msg == "__deleted__":
                    self._handle_deletion(item.event)
                    _report_progress(
                        ProcessingOutcome.DELETED,
                        "removed from index",
                        item.event.file_path,
                        item.file_index,
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
                        item.file_index,
                    )
                continue

            # _ParsedFileResult: run dedup on main thread (requires SQLite)
            pr = item
            dedup_result = self._run_dedup_check(pr)
            if dedup_result is not None:
                # Report start + immediate completion for duplicates/skips
                if on_start:
                    on_start(pr.file_index, total, Path(pr.event.file_path).name)
                outcome, detail = dedup_result
                _report_progress(outcome, detail, pr.event.file_path, pr.file_index)
                # Flush dedup hashes periodically
                if processed_count % 10 == 0:
                    self._dedup.flush()
                continue

            # Back-pressure: wait if too many files are in-flight for question generation
            _wait_for_in_flight_room()

            # Report start only for files that will actually be processed
            if on_start:
                on_start(pr.file_index, total, Path(pr.event.file_path).name)

            # Submit question generation to the summarizer's shared pool (non-blocking)
            if _cli_summarizer is not None:
                future = _cli_summarizer._pool.submit(
                    _cli_summarizer.generate_chunk_questions,
                    pr.chunks, pr.parsed_doc.title,
                )
                in_flight_questions.append((pr, future))
            else:
                # No question generation — add directly to pending
                pending.append(pr)
                pending_chunk_count += len(pr.chunks)

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

        # Hash registration deferred to _flush_pending (after successful indexing)
        # to avoid marking files as canonical before they are fully indexed.

        return None

    def _index_parsed_file(
        self,
        pr: _ParsedFileResult,
        vectors: list[list[float]],
        on_status: StatusCallback | None = None,
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
                chunker_version=get_chunker_version(self._config.chunking.strategy),
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
                generated_questions=json.dumps(c.generated_questions) if c.generated_questions else None,
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
                        generated_questions=chunk.generated_questions,
                        text=chunk.text,
                    ),
                )
            )

        if points:
            self._upsert_points(doc_id, points)
            keep_ids = {p.point_id for p in points}
            self._delete_stale_points(doc_id, keep_ids)

        # Summarize (if enabled)
        if on_status and self._summarizer and self._summarizer.available:
            section_count = len([s for s in pr.normalized.sections if s.text.strip()])
            on_status(pr.file_index, 0, Path(file_path).name, f"summarizing (1 combined call, {section_count} sections)...")

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

        # Build list of (section, section_row) pairs to summarize
        section_pairs = [
            (section, section_row)
            for section, section_row in zip(normalized.sections, section_rows, strict=False)
            if section.text.strip()
        ]

        # --- Try combined summarization first ---
        sections_for_combined: list[tuple[str | None, str]] = [
            (section.heading, section.text) for section, _row in section_pairs
        ]

        combined_result = self._summarizer.summarize_combined(
            full_text, title, normalized.file_type.value, sections_for_combined
        )

        if isinstance(combined_result, CombinedSummarySuccess):
            return self._process_combined_result(
                combined_result, doc_id, title, file_path, folder_path,
                folder_ancestors, file_type, modified_at, normalized,
                section_pairs,
            )

        # --- Fallback: separate doc + parallel section calls ---
        logger.warning(
            "Combined summarization failed for %s: %s, falling back to separate calls",
            file_path,
            combined_result.error,
        )
        return self._summarize_document_fallback(
            doc_id, title, file_path, folder_path, folder_ancestors,
            file_type, modified_at, normalized, section_pairs,
        )

    def _process_combined_result(
        self,
        combined: CombinedSummarySuccess,
        doc_id: str,
        title: str,
        file_path: str,
        folder_path: str,
        folder_ancestors: list[str],
        file_type: FileType,
        modified_at: str,
        normalized: NormalizedDocument,
        section_pairs: list[tuple[object, SectionRow]],
    ) -> list[VectorPoint]:
        """Process a successful combined summarization result into vector points."""
        # Update document row with doc-level summaries
        existing_doc = self._db.get_document(doc_id)
        if existing_doc is not None:
            updated = existing_doc.model_copy(
                update={
                    "summary_8w": combined.summary_8w,
                    "summary_16w": combined.summary_16w,
                    "summary_32w": combined.summary_32w,
                    "summary_64w": combined.summary_64w,
                    "summary_128w": combined.summary_128w,
                    "key_topics": combined.key_topics,
                    "doc_type_guess": combined.doc_type_guess,
                    "summary_content_hash": normalized.normalized_content_hash,
                }
            )
            self._db.upsert_document(updated)

        if len(combined.sections) < len(section_pairs):
            logger.warning(
                "Combined summary for %s has %d/%d sections (output may have been truncated)",
                file_path, len(combined.sections), len(section_pairs),
            )
        else:
            logger.info("Generated combined summary for %s", file_path)

        # Update section rows with section-level summaries
        section_results: list[tuple[SectionSummarySuccess, SectionRow, int]] = []
        for i, (section, section_row) in enumerate(section_pairs):
            if i < len(combined.sections):
                sec = combined.sections[i]
                sec_success = SectionSummarySuccess(
                    section_summary_8w=sec.section_summary_8w,
                    section_summary_32w=sec.section_summary_32w,
                    section_summary_128w=sec.section_summary_128w,
                )
                updated_section = section_row.model_copy(
                    update={
                        "section_summary_8w": sec.section_summary_8w,
                        "section_summary_32w": sec.section_summary_32w,
                        "section_summary_128w": sec.section_summary_128w,
                        "embedding_model_version": self._embedder.model_version,
                    }
                )
                self._db.insert_sections([updated_section])
                section_results.append((sec_success, section_row, i))

        return self._build_summary_points(
            doc_id, title, file_path, folder_path, folder_ancestors,
            file_type, modified_at, combined.summary_128w,
            combined.doc_type_guess, combined.key_topics, section_results,
        )

    def _summarize_document_fallback(
        self,
        doc_id: str,
        title: str,
        file_path: str,
        folder_path: str,
        folder_ancestors: list[str],
        file_type: FileType,
        modified_at: str,
        normalized: NormalizedDocument,
        section_pairs: list[tuple[object, SectionRow]],
    ) -> list[VectorPoint]:
        """Fallback: separate doc summary + parallel section summaries."""
        full_text = "\n\n".join(s.text for s in normalized.sections)

        doc_result = self._summarizer.summarize_document(  # type: ignore[union-attr]
            full_text, title, normalized.file_type.value
        )

        doc_summary_text: str | None = None
        doc_type_guess: str | None = None
        doc_key_topics: list[str] | None = None
        if isinstance(doc_result, SummarySuccess):
            existing_doc = self._db.get_document(doc_id)
            if existing_doc is not None:
                updated = existing_doc.model_copy(
                    update={
                        "summary_8w": doc_result.summary_8w,
                        "summary_16w": doc_result.summary_16w,
                        "summary_32w": doc_result.summary_32w,
                        "summary_64w": doc_result.summary_64w,
                        "summary_128w": doc_result.summary_128w,
                        "key_topics": doc_result.key_topics,
                        "doc_type_guess": doc_result.doc_type_guess,
                        "summary_content_hash": normalized.normalized_content_hash,
                    }
                )
                self._db.upsert_document(updated)

            doc_summary_text = doc_result.summary_128w
            doc_type_guess = doc_result.doc_type_guess
            doc_key_topics = doc_result.key_topics
            logger.info("Generated document summary for %s", file_path)
        else:
            logger.warning("Document summarization failed for %s: %s", file_path, doc_result.error)

        # Parallel section summaries (uses summarizer's shared LLM pool)
        doc_context = f"{title} ({normalized.file_type.value})"
        section_results: list[tuple[SectionSummarySuccess, SectionRow, int]] = []

        sections_for_batch: list[tuple[str | None, str]] = [
            (section.heading, section.text)  # type: ignore[union-attr]
            for section, _row in section_pairs
        ]
        batch_results = self._summarizer.summarize_sections_batch(  # type: ignore[union-attr]
            sections_for_batch, doc_context,
        )

        for i, sec_summary in enumerate(batch_results):
            if i >= len(section_pairs):
                break
            _section, section_row = section_pairs[i]
            sec_result = SectionSummarySuccess(
                section_summary_8w=sec_summary.section_summary_8w,
                section_summary_32w=sec_summary.section_summary_32w,
                section_summary_128w=sec_summary.section_summary_128w,
            )
            updated_section = section_row.model_copy(
                update={
                    "section_summary_8w": sec_result.section_summary_8w,
                    "section_summary_32w": sec_result.section_summary_32w,
                    "section_summary_128w": sec_result.section_summary_128w,
                    "embedding_model_version": self._embedder.model_version,
                }
            )
            self._db.insert_sections([updated_section])
            section_results.append((sec_result, section_row, i))

        return self._build_summary_points(
            doc_id, title, file_path, folder_path, folder_ancestors,
            file_type, modified_at, doc_summary_text, doc_type_guess,
            doc_key_topics, section_results,
        )

    def _build_summary_points(
        self,
        doc_id: str,
        title: str,
        file_path: str,
        folder_path: str,
        folder_ancestors: list[str],
        file_type: FileType,
        modified_at: str,
        doc_summary_text: str | None,
        doc_type_guess: str | None,
        doc_key_topics: list[str] | None,
        section_results: list[tuple[SectionSummarySuccess, SectionRow, int]],
    ) -> list[VectorPoint]:
        """Batch-embed all summaries and build VectorPoints."""
        embed_entries: list[dict[str, object]] = []

        if doc_summary_text is not None:
            embed_entries.append({
                "text": doc_summary_text,
                "type": "document",
                "doc_type_guess": doc_type_guess,
            })

        section_results.sort(key=lambda t: t[2])
        for sec_result, section_row, order in section_results:
            embed_entries.append({
                "text": sec_result.section_summary_128w,
                "type": "section",
                "section_row": section_row,
                "order": order,
                "heading": section_row.section_heading,
            })

        if not embed_entries:
            return []

        all_texts = [str(entry["text"]) for entry in embed_entries]
        all_vectors = self._embedder.embed_batch(all_texts)

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
                            summary_level="128w",
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
                            key_topics=doc_key_topics,
                            text=str(entry["text"]),
                        ),
                    )
                )
            else:
                sec_row = cast("SectionRow", entry["section_row"])
                order = int(str(entry["order"]))
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
                            section_id=sec_row.section_id,
                            title=title,
                            file_path=file_path,
                            folder_path=folder_path,
                            folder_ancestors=folder_ancestors,
                            file_type=file_type,
                            modified_at=modified_at,
                            section_heading=str(entry["heading"]) if entry["heading"] else None,
                            key_topics=doc_key_topics,
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

    @staticmethod
    def _check_skip_retry(
        sync_state: SyncStateRow | None,
    ) -> tuple[ProcessingOutcome, str] | None:
        """Check if a file should be skipped due to poison quarantine or backoff.

        Returns (outcome, detail) if the file should be skipped, or None to proceed.
        """
        if sync_state is None:
            return None
        if sync_state.process_status == "poison":
            return ProcessingOutcome.ERROR, "quarantined \u2014 file failed 3+ times"
        if (
            sync_state.process_status == "error"
            and sync_state.retry_count > 0
            and sync_state.synced_at
        ):
            try:
                last_attempt = datetime.fromisoformat(sync_state.synced_at)
                backoff = timedelta(seconds=(2 ** sync_state.retry_count) * 30)
                retry_after = last_attempt + backoff
                now = datetime.now(tz=last_attempt.tzinfo or UTC)
                if now < retry_after:
                    remaining = int((retry_after - now).total_seconds())
                    return (
                        ProcessingOutcome.ERROR,
                        f"backing off \u2014 retry in {remaining}s",
                    )
            except (ValueError, TypeError):
                pass  # If timestamp is unparseable, allow retry
        return None

    def _update_sync_status(self, file_path: str, status: str, error: str | None = None) -> None:
        existing = self._db.get_sync_state(file_path)
        if existing:
            retry = existing.retry_count + (1 if status == "error" else 0)
            final_status = "poison" if status == "error" and retry >= 3 else status
            now = datetime.now(tz=UTC).isoformat()
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
                    synced_at=now,
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
