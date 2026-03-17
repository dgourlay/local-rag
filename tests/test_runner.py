from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from rag.config import SummarizationConfig
from rag.pipeline.classifier import classify
from rag.pipeline.dedup import DedupChecker
from rag.pipeline.runner import PipelineRunner
from rag.results import (
    CombinedSectionSummary,
    CombinedSummaryError,
    CombinedSummarySuccess,
    ParseError,
    ParseSuccess,
    SectionSummaryError,
    SectionSummarySuccess,
    SummarySuccess,
)
from rag.types import (
    FileEvent,
    FileType,
    ParsedDocument,
    ParsedSection,
    ProcessingOutcome,
)

# Resolve forward references for Pydantic models using `from __future__ import annotations`
ParseSuccess.model_rebuild()
ParseError.model_rebuild()

# --- Helpers ---


def _create_db() -> sqlite3.Connection:
    """Create an in-memory SQLite DB with all tables."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    migrations_dir = Path(__file__).parent.parent / "migrations"
    conn.executescript((migrations_dir / "001_initial.sql").read_text())
    conn.executescript((migrations_dir / "002_pyramid_summaries.sql").read_text())
    conn.executescript((migrations_dir / "003_chunk_questions.sql").read_text())
    return conn


def _make_event(tmp_path: Path, content: str = "Hello world test content.") -> FileEvent:
    """Create a test file and matching FileEvent."""
    test_file = tmp_path / "test.txt"
    test_file.write_text(content)
    return FileEvent(
        file_path=str(test_file),
        content_hash="abc123hash",
        file_type=FileType.TXT,
        event_type="created",
        modified_at="2026-01-01T00:00:00+00:00",
    )


def _make_parsed_doc(file_path: str) -> ParsedDocument:
    """Create a minimal ParsedDocument."""
    return ParsedDocument(
        doc_id=str(uuid.uuid4()),
        title="Test Document",
        file_type=FileType.TXT,
        sections=[
            ParsedSection(
                heading="Introduction",
                order=0,
                text=(
                    "This is test content for the pipeline runner."
                    " It contains enough text to form at least one chunk."
                ),
            ),
        ],
        raw_content_hash="rawhash123",
    )


def _make_runner(
    tmp_path: Path,
    conn: sqlite3.Connection,
    *,
    parse_result: ParseSuccess | ParseError | None = None,
) -> tuple[PipelineRunner, dict[str, Any]]:
    """Build a PipelineRunner with mocked dependencies."""
    from rag.config import AppConfig, FoldersConfig
    from rag.db.models import SqliteMetadataDB

    db = SqliteMetadataDB(conn)
    dedup = DedupChecker(conn)

    mock_embedder = MagicMock()
    mock_embedder.embed_batch.return_value = [[0.1, 0.2, 0.3]]
    mock_embedder.model_version = "test-model-v1"

    mock_vector_store = MagicMock()

    mock_parser = MagicMock()
    mock_parser.supported_types = {FileType.TXT, FileType.MD}
    if parse_result is not None:
        mock_parser.parse.return_value = parse_result
    else:
        # Default: return success with a parsed doc
        doc = _make_parsed_doc(str(tmp_path / "test.txt"))
        mock_parser.parse.return_value = ParseSuccess(document=doc)

    config = AppConfig(folders=FoldersConfig(paths=[tmp_path]))

    runner = PipelineRunner(
        db=db,
        vector_store=mock_vector_store,
        embedder=mock_embedder,
        parsers=[mock_parser],
        dedup=dedup,
        config=config,
    )

    mocks = {
        "db": db,
        "dedup": dedup,
        "embedder": mock_embedder,
        "vector_store": mock_vector_store,
        "parser": mock_parser,
    }
    return runner, mocks


# --- Tests ---


class TestClassifier:
    def test_classify_txt(self, tmp_path: Path) -> None:
        f = tmp_path / "readme.txt"
        f.write_text("hello")
        result = classify(str(f), str(tmp_path))
        assert result.file_type == FileType.TXT
        assert result.complexity_estimate == "low"
        assert result.ocr_enabled is False

    def test_classify_pdf(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"fake pdf content")
        result = classify(str(f), str(tmp_path))
        assert result.file_type == FileType.PDF
        assert result.complexity_estimate == "medium"
        assert result.ocr_enabled is True

    def test_classify_unknown_extension_falls_back_to_txt(self, tmp_path: Path) -> None:
        f = tmp_path / "data.xyz"
        f.write_text("some data")
        result = classify(str(f), str(tmp_path))
        assert result.file_type == FileType.TXT


class TestProcessFile:
    def test_success(self, tmp_path: Path) -> None:
        conn = _create_db()
        event = _make_event(tmp_path)
        runner, mocks = _make_runner(tmp_path, conn)

        outcome, detail = runner.process_file(event)

        assert outcome == ProcessingOutcome.INDEXED
        assert "chunk" in detail
        mocks["parser"].parse.assert_called_once()

    def test_content_hash_forwarded_to_parser(self, tmp_path: Path) -> None:
        """Verify that the runner passes event.content_hash to parser.parse()."""
        conn = _create_db()
        event = _make_event(tmp_path)
        runner, mocks = _make_runner(tmp_path, conn)

        runner.process_file(event)

        mocks["parser"].parse.assert_called_once()
        call_kwargs = mocks["parser"].parse.call_args
        # content_hash should be passed as a keyword argument
        assert call_kwargs[1]["content_hash"] == "abc123hash"
        mocks["embedder"].embed_batch.assert_called_once()
        mocks["vector_store"].upsert_points.assert_called_once()

    def test_parse_error_returns_error(self, tmp_path: Path) -> None:
        conn = _create_db()
        event = _make_event(tmp_path)
        parse_err = ParseError(error="bad file", file_path=str(tmp_path / "test.txt"))
        runner, mocks = _make_runner(tmp_path, conn, parse_result=parse_err)

        outcome, _detail = runner.process_file(event)

        assert outcome == ProcessingOutcome.ERROR
        mocks["embedder"].embed_batch.assert_not_called()
        mocks["vector_store"].upsert_points.assert_not_called()

    def test_unchanged_file_skips_pipeline(self, tmp_path: Path) -> None:
        conn = _create_db()
        event = _make_event(tmp_path)
        runner, mocks = _make_runner(tmp_path, conn)

        # Process first time
        outcome1, _detail = runner.process_file(event)
        assert outcome1 == ProcessingOutcome.INDEXED

        # Reset mocks for second call
        mocks["parser"].parse.reset_mock()
        mocks["embedder"].embed_batch.reset_mock()
        mocks["vector_store"].upsert_points.reset_mock()

        # Process same event again (same content_hash)
        outcome2, detail = runner.process_file(event)
        assert outcome2 == ProcessingOutcome.UNCHANGED
        assert "unchanged" in detail

        # Nothing should have been called — skipped before parsing
        mocks["parser"].parse.assert_not_called()
        mocks["embedder"].embed_batch.assert_not_called()
        mocks["vector_store"].upsert_points.assert_not_called()

    def test_dedup_skips_remaining_pipeline(self, tmp_path: Path) -> None:
        conn = _create_db()
        event = _make_event(tmp_path)
        runner, mocks = _make_runner(tmp_path, conn)

        # Process first file to register the hash
        outcome1, _detail = runner.process_file(event)
        assert outcome1 == ProcessingOutcome.INDEXED

        # Reset mocks for second call
        mocks["embedder"].embed_batch.reset_mock()
        mocks["vector_store"].upsert_points.reset_mock()

        # Create second file with same content (will produce same hash)
        event2 = _make_event(tmp_path, content="Hello world test content.")
        event2 = FileEvent(
            file_path=str(tmp_path / "test.txt"),
            content_hash="different_hash",
            file_type=FileType.TXT,
            event_type="created",
            modified_at="2026-01-02T00:00:00+00:00",
        )
        outcome2, _detail = runner.process_file(event2)
        assert outcome2 in (ProcessingOutcome.UNCHANGED, ProcessingOutcome.DUPLICATE)

        # The parser was called (it doesn't skip parse), but since the
        # normalized hash matches, embedding/indexing should be skipped
        # Actually the dedup checks raw_hash first, then normalized_hash
        # The raw_hash from the parsed doc is "rawhash123" which was registered
        # So the second call should find the duplicate
        mocks["embedder"].embed_batch.assert_not_called()
        mocks["vector_store"].upsert_points.assert_not_called()

    def test_deletion_marks_deleted(self, tmp_path: Path) -> None:
        conn = _create_db()
        runner, mocks = _make_runner(tmp_path, conn)

        # First, process a file to create sync state
        event = _make_event(tmp_path)
        runner.process_file(event)

        # Now delete it
        delete_event = FileEvent(
            file_path=str(tmp_path / "test.txt"),
            content_hash="abc123hash",
            file_type=FileType.TXT,
            event_type="deleted",
            modified_at="2026-01-01T00:00:00+00:00",
        )
        outcome, _detail = runner.process_file(delete_event)
        assert outcome == ProcessingOutcome.DELETED

        # Verify sync state is marked deleted
        state = mocks["db"].get_sync_state(str(tmp_path / "test.txt"))
        assert state is not None
        assert state.is_deleted == 1

    def test_error_increments_retry_count(self, tmp_path: Path) -> None:
        conn = _create_db()
        event = _make_event(tmp_path)

        # Make parser raise an exception
        parse_err = ParseError(error="corrupt file", file_path=str(tmp_path / "test.txt"))
        runner, mocks = _make_runner(tmp_path, conn, parse_result=parse_err)

        # First failure
        outcome, _detail = runner.process_file(event)
        assert outcome == ProcessingOutcome.ERROR

        state = mocks["db"].get_sync_state(str(tmp_path / "test.txt"))
        assert state is not None
        assert state.process_status == "error"
        assert state.retry_count == 1

    def test_poison_after_three_retries(self, tmp_path: Path) -> None:
        conn = _create_db()
        event = _make_event(tmp_path)
        parse_err = ParseError(error="always fails", file_path=str(tmp_path / "test.txt"))
        runner, mocks = _make_runner(tmp_path, conn, parse_result=parse_err)

        # Process 3 times to trigger poison
        runner.process_file(event)
        runner.process_file(event)
        runner.process_file(event)

        state = mocks["db"].get_sync_state(str(tmp_path / "test.txt"))
        assert state is not None
        assert state.process_status == "poison"
        assert state.retry_count == 3


class TestProcessBatch:
    def test_batch_counts(self, tmp_path: Path) -> None:
        conn = _create_db()

        # Create two files
        f1 = tmp_path / "a.txt"
        f1.write_text("file a content")
        f2 = tmp_path / "b.txt"
        f2.write_text("file b content")

        events = [
            FileEvent(
                file_path=str(f1),
                content_hash="hash_a",
                file_type=FileType.TXT,
                event_type="created",
                modified_at="2026-01-01T00:00:00+00:00",
            ),
            FileEvent(
                file_path=str(f2),
                content_hash="hash_b",
                file_type=FileType.TXT,
                event_type="created",
                modified_at="2026-01-01T00:00:00+00:00",
            ),
        ]

        runner, mocks = _make_runner(tmp_path, conn)
        # Make second parse unique doc_id so dedup doesn't kick in
        call_count = 0

        def side_effect_parse(
            fp: str, ocr: bool, content_hash: str | None = None,
        ) -> ParseSuccess:
            nonlocal call_count
            call_count += 1
            doc = _make_parsed_doc(fp)
            doc = doc.model_copy(update={"raw_content_hash": f"unique_hash_{call_count}"})
            return ParseSuccess(document=doc)

        mocks["parser"].parse.side_effect = side_effect_parse

        counts = runner.process_batch(events)
        # Both succeed (one indexed, one may dedup due to same normalized text)
        assert counts[ProcessingOutcome.INDEXED] + counts[ProcessingOutcome.DUPLICATE] == 2
        assert counts[ProcessingOutcome.ERROR] == 0

    def test_batch_with_errors(self, tmp_path: Path) -> None:
        conn = _create_db()

        f1 = tmp_path / "good.txt"
        f1.write_text("good content")
        f2 = tmp_path / "bad.txt"
        f2.write_text("bad content")

        events = [
            FileEvent(
                file_path=str(f1),
                content_hash="hash_good",
                file_type=FileType.TXT,
                event_type="created",
                modified_at="2026-01-01T00:00:00+00:00",
            ),
            FileEvent(
                file_path=str(f2),
                content_hash="hash_bad",
                file_type=FileType.TXT,
                event_type="created",
                modified_at="2026-01-01T00:00:00+00:00",
            ),
        ]

        runner, mocks = _make_runner(tmp_path, conn)

        call_count = 0

        def side_effect_parse(
            fp: str, ocr: bool, content_hash: str | None = None,
        ) -> ParseSuccess | ParseError:
            nonlocal call_count
            call_count += 1
            if "bad" in fp:
                return ParseError(error="bad file", file_path=fp)
            doc = _make_parsed_doc(fp)
            return ParseSuccess(document=doc)

        mocks["parser"].parse.side_effect = side_effect_parse

        counts = runner.process_batch(events)
        assert counts[ProcessingOutcome.INDEXED] == 1
        assert counts[ProcessingOutcome.ERROR] == 1

    def test_progress_callback(self, tmp_path: Path) -> None:
        conn = _create_db()
        event = _make_event(tmp_path)
        runner, _mocks = _make_runner(tmp_path, conn)

        calls: list[tuple[int, int, str, ProcessingOutcome, str]] = []

        def on_progress(
            current: int, total: int, filename: str,
            outcome: ProcessingOutcome, detail: str,
        ) -> None:
            calls.append((current, total, filename, outcome, detail))

        runner.process_batch([event], progress=on_progress)
        assert len(calls) == 1
        assert calls[0][0:3] == (1, 1, "test.txt")
        assert calls[0][3] == ProcessingOutcome.INDEXED


# --- Helpers for summarization tests ---


def _make_parsed_doc_with_sections(file_path: str, num_sections: int = 3) -> ParsedDocument:
    """Create a ParsedDocument with multiple sections for summarization tests."""
    sections = [
        ParsedSection(
            heading=f"Section {i}",
            order=i,
            text=f"Content for section {i} with enough text to be meaningful.",
        )
        for i in range(num_sections)
    ]
    return ParsedDocument(
        doc_id=str(uuid.uuid4()),
        title="Multi-Section Doc",
        file_type=FileType.TXT,
        sections=sections,
        raw_content_hash="rawhash_multi",
    )


def _make_runner_with_summarizer(
    tmp_path: Path,
    conn: sqlite3.Connection,
    *,
    summarizer: MagicMock | None = None,
    num_sections: int = 3,
) -> tuple[PipelineRunner, dict[str, Any]]:
    """Build a PipelineRunner with a mock summarizer."""
    from rag.config import AppConfig, FoldersConfig
    from rag.db.models import SqliteMetadataDB

    db = SqliteMetadataDB(conn)
    dedup = DedupChecker(conn)

    mock_embedder = MagicMock()
    # Return a unique vector for each text
    mock_embedder.embed_batch.side_effect = lambda texts: [
        [float(i) * 0.1] * 3 for i in range(len(texts))
    ]
    mock_embedder.model_version = "test-model-v1"
    mock_embedder.dimensions = 3

    mock_vector_store = MagicMock()

    doc = _make_parsed_doc_with_sections(str(tmp_path / "test.txt"), num_sections)
    mock_parser = MagicMock()
    mock_parser.supported_types = {FileType.TXT, FileType.MD}
    mock_parser.parse.return_value = ParseSuccess(document=doc)

    config = AppConfig(folders=FoldersConfig(paths=[tmp_path]))

    runner = PipelineRunner(
        db=db,
        vector_store=mock_vector_store,
        embedder=mock_embedder,
        parsers=[mock_parser],
        dedup=dedup,
        config=config,
        summarizer=summarizer,
    )

    mocks = {
        "db": db,
        "dedup": dedup,
        "embedder": mock_embedder,
        "vector_store": mock_vector_store,
        "parser": mock_parser,
        "doc": doc,
    }
    return runner, mocks


class TestSummarizationBatchEmbedding:
    """Tests for batched summary embeddings and parallel summarization."""

    def test_embed_batch_called_once_with_all_summaries(self, tmp_path: Path) -> None:
        """embed_batch should be called ONCE for all summary texts (not N times)."""
        conn = _create_db()
        num_sections = 3

        mock_summarizer = MagicMock()
        mock_summarizer.available = True
        mock_summarizer.summarize_combined.return_value = CombinedSummarySuccess(
            summary_8w="Short",
            summary_16w="Medium summary.",
            summary_32w="A moderate summary of the document.",
            summary_64w="An extended summary covering more details.",
            summary_128w="Long detailed summary of the document.",
            key_topics=["topic1"],
            doc_type_guess="report",
            sections=[
                CombinedSectionSummary(
                    heading=f"Section {i}",
                    section_summary_8w="Short section",
                    section_summary_32w="Section summary text.",
                    section_summary_128w="Detailed section summary text with more context.",
                )
                for i in range(num_sections)
            ],
        )

        runner, mocks = _make_runner_with_summarizer(
            tmp_path, conn, summarizer=mock_summarizer, num_sections=num_sections
        )
        event = _make_event(tmp_path)
        outcome, _detail = runner.process_file(event)
        assert outcome == ProcessingOutcome.INDEXED

        # embed_batch is called twice: once for chunks, once for ALL summaries
        embed_calls = mocks["embedder"].embed_batch.call_args_list

        # First call is for chunks (from process_file step 10)
        # Second call is for all summaries (1 doc + 3 sections = 4 texts)
        assert len(embed_calls) == 2, f"Expected 2 embed_batch calls, got {len(embed_calls)}"

        summary_call_texts = embed_calls[1][0][0]  # positional arg 0 of second call
        # 1 doc summary + 3 section summaries = 4 total
        assert len(summary_call_texts) == 1 + num_sections, (
            f"Expected {1 + num_sections} summary texts, got {len(summary_call_texts)}"
        )

    def test_no_summarizer_returns_empty(self, tmp_path: Path) -> None:
        """When summarizer is None, _summarize_document returns empty list."""
        conn = _create_db()
        runner, mocks = _make_runner_with_summarizer(
            tmp_path, conn, summarizer=None, num_sections=2
        )
        event = _make_event(tmp_path)
        outcome, _detail = runner.process_file(event)
        assert outcome == ProcessingOutcome.INDEXED

        # embed_batch called only once (for chunks), not for summaries
        assert mocks["embedder"].embed_batch.call_count == 1

    def test_unavailable_summarizer_returns_empty(self, tmp_path: Path) -> None:
        """When summarizer.available is False, returns empty list."""
        conn = _create_db()
        mock_summarizer = MagicMock()
        mock_summarizer.available = False

        runner, mocks = _make_runner_with_summarizer(
            tmp_path, conn, summarizer=mock_summarizer, num_sections=2
        )
        event = _make_event(tmp_path)
        outcome, _detail = runner.process_file(event)
        assert outcome == ProcessingOutcome.INDEXED

        # Summarizer should not be called
        mock_summarizer.summarize_document.assert_not_called()
        assert mocks["embedder"].embed_batch.call_count == 1

    def test_section_failure_does_not_block_others(self, tmp_path: Path) -> None:
        """If combined fails and one section is missing in fallback batch, others still produce points."""
        conn = _create_db()
        num_sections = 3

        mock_summarizer = MagicMock()
        mock_summarizer.available = True
        # Combined fails, triggering fallback
        mock_summarizer.summarize_combined.return_value = CombinedSummaryError(
            error="Combined call failed"
        )
        mock_summarizer.summarize_document.return_value = SummarySuccess(
            summary_8w="Short",
            summary_16w="Medium.",
            summary_32w="Moderate doc summary.",
            summary_64w="Extended doc summary.",
            summary_128w="Long doc summary.",
            key_topics=["t1"],
            doc_type_guess="notes",
        )

        # summarize_sections_batch returns only 2 of 3 sections (one failed in batch)
        mock_summarizer.summarize_sections_batch.return_value = [
            CombinedSectionSummary(
                section_summary_8w="Short",
                section_summary_32w="Summary for section 0",
                section_summary_128w="Detailed summary of Section 0",
            ),
            CombinedSectionSummary(
                section_summary_8w="Short",
                section_summary_32w="Summary for section 2",
                section_summary_128w="Detailed summary of Section 2",
            ),
        ]

        runner, mocks = _make_runner_with_summarizer(
            tmp_path, conn, summarizer=mock_summarizer, num_sections=num_sections
        )
        event = _make_event(tmp_path)
        outcome, _detail = runner.process_file(event)
        assert outcome == ProcessingOutcome.INDEXED

        # embed_batch for summaries: 1 doc + 2 successful sections = 3
        summary_call = mocks["embedder"].embed_batch.call_args_list[1]
        summary_texts = summary_call[0][0]
        assert len(summary_texts) == 3, f"Expected 3 summary texts, got {len(summary_texts)}"

    def test_section_exception_does_not_block_others(self, tmp_path: Path) -> None:
        """If combined fails and batch returns partial results, other sections still complete."""
        conn = _create_db()
        num_sections = 3

        mock_summarizer = MagicMock()
        mock_summarizer.available = True
        mock_summarizer.summarize_combined.return_value = CombinedSummaryError(
            error="Combined call failed"
        )
        mock_summarizer.summarize_document.return_value = SummarySuccess(
            summary_8w="Short",
            summary_16w="Medium.",
            summary_32w="Moderate doc summary.",
            summary_64w="Extended doc summary.",
            summary_128w="Long doc summary.",
            key_topics=["t1"],
            doc_type_guess="notes",
        )

        # summarize_sections_batch returns only 2 of 3 sections
        mock_summarizer.summarize_sections_batch.return_value = [
            CombinedSectionSummary(
                section_summary_8w="Short",
                section_summary_32w="Summary for section 0",
                section_summary_128w="Detailed summary of Section 0",
            ),
            CombinedSectionSummary(
                section_summary_8w="Short",
                section_summary_32w="Summary for section 2",
                section_summary_128w="Detailed summary of Section 2",
            ),
        ]

        runner, mocks = _make_runner_with_summarizer(
            tmp_path, conn, summarizer=mock_summarizer, num_sections=num_sections
        )
        event = _make_event(tmp_path)
        outcome, _detail = runner.process_file(event)
        assert outcome == ProcessingOutcome.INDEXED

        # embed_batch for summaries: 1 doc + 2 successful sections = 3
        summary_call = mocks["embedder"].embed_batch.call_args_list[1]
        summary_texts = summary_call[0][0]
        assert len(summary_texts) == 3

    def test_doc_summary_failure_still_embeds_sections(self, tmp_path: Path) -> None:
        """If document summary fails, section summaries still get embedded via fallback."""
        conn = _create_db()
        num_sections = 2
        from rag.results import SummaryError

        mock_summarizer = MagicMock()
        mock_summarizer.available = True
        # Combined fails, fallback doc summary also fails, but sections succeed
        mock_summarizer.summarize_combined.return_value = CombinedSummaryError(
            error="combined failed"
        )
        mock_summarizer.summarize_document.return_value = SummaryError(error="timeout")
        mock_summarizer.summarize_sections_batch.return_value = [
            CombinedSectionSummary(
                section_summary_8w="Short",
                section_summary_32w="Section summary.",
                section_summary_128w="Detailed section summary.",
            )
            for _ in range(num_sections)
        ]

        runner, mocks = _make_runner_with_summarizer(
            tmp_path, conn, summarizer=mock_summarizer, num_sections=num_sections
        )
        event = _make_event(tmp_path)
        outcome, _detail = runner.process_file(event)
        assert outcome == ProcessingOutcome.INDEXED

        # embed_batch for summaries: 0 doc + 2 sections = 2
        summary_call = mocks["embedder"].embed_batch.call_args_list[1]
        summary_texts = summary_call[0][0]
        assert len(summary_texts) == num_sections

    def test_parallel_produces_same_results(self, tmp_path: Path) -> None:
        """Combined summarization should produce correct vector points."""
        conn = _create_db()
        num_sections = 4

        mock_summarizer = MagicMock()
        mock_summarizer.available = True
        mock_summarizer.summarize_combined.return_value = CombinedSummarySuccess(
            summary_8w="Short",
            summary_16w="Medium.",
            summary_32w="Moderate doc summary.",
            summary_64w="Extended doc summary.",
            summary_128w="Long doc summary.",
            key_topics=["t1"],
            doc_type_guess="report",
            sections=[
                CombinedSectionSummary(
                    heading=f"Section {i}",
                    section_summary_8w=f"Short Section {i}",
                    section_summary_32w=f"Summary of Section {i}",
                    section_summary_128w=f"Detailed summary of Section {i}",
                )
                for i in range(num_sections)
            ],
        )

        runner, mocks = _make_runner_with_summarizer(
            tmp_path, conn, summarizer=mock_summarizer, num_sections=num_sections
        )
        event = _make_event(tmp_path)
        outcome, _detail = runner.process_file(event)
        assert outcome == ProcessingOutcome.INDEXED

        # Check the summary embed_batch call
        summary_call = mocks["embedder"].embed_batch.call_args_list[1]
        summary_texts = summary_call[0][0]
        # First text is doc summary, then sections in order
        assert summary_texts[0] == "Long doc summary."
        for i in range(num_sections):
            assert summary_texts[i + 1] == f"Detailed summary of Section {i}"


class TestMaxConcurrentLlmConfig:
    """Tests for max_concurrent_llm config validation."""

    def test_default_max_concurrent_llm(self) -> None:
        config = SummarizationConfig()
        assert config.max_concurrent_llm == 3

    def test_max_concurrent_llm_min_valid(self) -> None:
        config = SummarizationConfig(max_concurrent_llm=1)
        assert config.max_concurrent_llm == 1

    def test_max_concurrent_llm_max_valid(self) -> None:
        config = SummarizationConfig(max_concurrent_llm=4)
        assert config.max_concurrent_llm == 4

    def test_max_concurrent_llm_below_min_raises(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            SummarizationConfig(max_concurrent_llm=0)

    def test_max_concurrent_llm_above_max_raises(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            SummarizationConfig(max_concurrent_llm=5)


class TestPipelineParallelism:
    """Tests for producer-consumer pipeline parallelism in process_batch."""

    def test_parallel_batch_same_results_as_sequential(self, tmp_path: Path) -> None:
        """Pipeline parallelism should produce the same outcomes as sequential."""
        conn = _create_db()

        files = []
        events = []
        for i in range(4):
            f = tmp_path / f"file_{i}.txt"
            f.write_text(f"Content for file {i} with unique text.")
            files.append(f)
            events.append(
                FileEvent(
                    file_path=str(f),
                    content_hash=f"hash_{i}",
                    file_type=FileType.TXT,
                    event_type="created",
                    modified_at="2026-01-01T00:00:00+00:00",
                )
            )

        runner, mocks = _make_runner(tmp_path, conn)
        call_count = 0

        def side_effect_parse(
            fp: str, ocr: bool, content_hash: str | None = None,
        ) -> ParseSuccess:
            nonlocal call_count
            call_count += 1
            doc = _make_parsed_doc(fp)
            doc = doc.model_copy(update={"raw_content_hash": f"unique_{call_count}"})
            return ParseSuccess(document=doc)

        mocks["parser"].parse.side_effect = side_effect_parse

        counts = runner.process_batch(events)

        total_processed = sum(counts.values())
        assert total_processed == 4
        assert counts[ProcessingOutcome.ERROR] == 0
        assert counts[ProcessingOutcome.INDEXED] + counts[ProcessingOutcome.DUPLICATE] == 4

    def test_cross_document_batching_combines_chunks(self, tmp_path: Path) -> None:
        """embed_batch should be called with chunks from multiple documents combined."""
        from rag.config import AppConfig, EmbeddingConfig, FoldersConfig
        from rag.db.models import SqliteMetadataDB

        conn = _create_db()
        db = SqliteMetadataDB(conn)
        dedup = DedupChecker(conn)

        mock_embedder = MagicMock()
        # Return a vector for each text
        mock_embedder.embed_batch.side_effect = lambda texts: [
            [0.1] * 3 for _ in texts
        ]
        mock_embedder.model_version = "test-model-v1"

        mock_vector_store = MagicMock()

        mock_parser = MagicMock()
        mock_parser.supported_types = {FileType.TXT, FileType.MD}

        call_count = 0

        def _make_unique_doc(fp: str, idx: int) -> ParsedDocument:
            """Create a doc with unique content so dedup doesn't trigger."""
            return ParsedDocument(
                doc_id=str(uuid.uuid4()),
                title=f"Document {idx}",
                file_type=FileType.TXT,
                sections=[
                    ParsedSection(
                        heading=f"Section for doc {idx}",
                        order=0,
                        text=(
                            f"Unique content for document number {idx}."
                            " It contains enough text to form at least one chunk."
                        ),
                    ),
                ],
                raw_content_hash=f"raw_unique_{idx}",
            )

        def side_effect_parse(
            fp: str, ocr: bool, content_hash: str | None = None,
        ) -> ParseSuccess:
            nonlocal call_count
            call_count += 1
            doc = _make_unique_doc(fp, call_count)
            return ParseSuccess(document=doc)

        mock_parser.parse.side_effect = side_effect_parse

        # Set batch_size very large so all chunks accumulate in one batch
        config = AppConfig(
            folders=FoldersConfig(paths=[tmp_path]),
            embedding=EmbeddingConfig(batch_size=1000),
        )

        runner = PipelineRunner(
            db=db,
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            parsers=[mock_parser],
            dedup=dedup,
            config=config,
        )

        # Create 3 files
        events = []
        for i in range(3):
            f = tmp_path / f"doc_{i}.txt"
            f.write_text(f"Content for document {i} that is unique.")
            events.append(
                FileEvent(
                    file_path=str(f),
                    content_hash=f"hash_{i}",
                    file_type=FileType.TXT,
                    event_type="created",
                    modified_at="2026-01-01T00:00:00+00:00",
                )
            )

        counts = runner.process_batch(events)
        assert counts[ProcessingOutcome.INDEXED] == 3
        assert counts[ProcessingOutcome.ERROR] == 0

        # With batch_size=1000, all chunks should be embedded in a single call
        # (since 3 small docs will have few chunks total)
        embed_calls = mock_embedder.embed_batch.call_args_list
        assert len(embed_calls) == 1, (
            f"Expected 1 embed_batch call for cross-doc batching, got {len(embed_calls)}"
        )
        # The single call should contain chunks from all 3 documents
        total_texts = len(embed_calls[0][0][0])
        assert total_texts >= 3, f"Expected at least 3 texts batched, got {total_texts}"

    def test_error_in_one_file_does_not_block_others(self, tmp_path: Path) -> None:
        """If parsing fails for one file, other files should still be processed."""
        conn = _create_db()

        f_good1 = tmp_path / "good1.txt"
        f_good1.write_text("Good content one")
        f_bad = tmp_path / "bad.txt"
        f_bad.write_text("Bad content")
        f_good2 = tmp_path / "good2.txt"
        f_good2.write_text("Good content two")

        events = [
            FileEvent(
                file_path=str(f_good1),
                content_hash="hash_good1",
                file_type=FileType.TXT,
                event_type="created",
                modified_at="2026-01-01T00:00:00+00:00",
            ),
            FileEvent(
                file_path=str(f_bad),
                content_hash="hash_bad",
                file_type=FileType.TXT,
                event_type="created",
                modified_at="2026-01-01T00:00:00+00:00",
            ),
            FileEvent(
                file_path=str(f_good2),
                content_hash="hash_good2",
                file_type=FileType.TXT,
                event_type="created",
                modified_at="2026-01-01T00:00:00+00:00",
            ),
        ]

        runner, mocks = _make_runner(tmp_path, conn)
        call_count = 0

        def side_effect_parse(
            fp: str, ocr: bool, content_hash: str | None = None,
        ) -> ParseSuccess | ParseError:
            nonlocal call_count
            call_count += 1
            if "bad" in fp:
                return ParseError(error="corrupt file", file_path=fp)
            doc = _make_parsed_doc(fp)
            doc = doc.model_copy(update={"raw_content_hash": f"unique_{call_count}"})
            return ParseSuccess(document=doc)

        mocks["parser"].parse.side_effect = side_effect_parse

        counts = runner.process_batch(events)
        assert counts[ProcessingOutcome.ERROR] == 1
        assert counts[ProcessingOutcome.INDEXED] + counts[ProcessingOutcome.DUPLICATE] == 2

    def test_single_file_still_works_via_process_file(self, tmp_path: Path) -> None:
        """process_file should still work for single-file indexing."""
        conn = _create_db()
        event = _make_event(tmp_path)
        runner, mocks = _make_runner(tmp_path, conn)

        outcome, detail = runner.process_file(event)

        assert outcome == ProcessingOutcome.INDEXED
        assert "chunk" in detail
        mocks["parser"].parse.assert_called_once()
        mocks["embedder"].embed_batch.assert_called_once()
        mocks["vector_store"].upsert_points.assert_called_once()

    def test_progress_callback_in_parallel_batch(self, tmp_path: Path) -> None:
        """Progress callback should be called correctly with parallel batch."""
        conn = _create_db()

        events = []
        for i in range(3):
            f = tmp_path / f"prog_{i}.txt"
            f.write_text(f"Progress test content {i}")
            events.append(
                FileEvent(
                    file_path=str(f),
                    content_hash=f"prog_hash_{i}",
                    file_type=FileType.TXT,
                    event_type="created",
                    modified_at="2026-01-01T00:00:00+00:00",
                )
            )

        runner, mocks = _make_runner(tmp_path, conn)
        call_count = 0

        def side_effect_parse(
            fp: str, ocr: bool, content_hash: str | None = None,
        ) -> ParseSuccess:
            nonlocal call_count
            call_count += 1
            doc = _make_parsed_doc(fp)
            doc = doc.model_copy(update={"raw_content_hash": f"prog_unique_{call_count}"})
            return ParseSuccess(document=doc)

        mocks["parser"].parse.side_effect = side_effect_parse

        calls: list[tuple[int, int, str, ProcessingOutcome, str]] = []

        def on_progress(
            current: int, total: int, filename: str,
            outcome: ProcessingOutcome, detail: str,
        ) -> None:
            calls.append((current, total, filename, outcome, detail))

        runner.process_batch(events, progress=on_progress)

        assert len(calls) == 3
        # All calls should have total=3
        for c in calls:
            assert c[1] == 3
        # current values should be 1, 2, 3 (in some order since parallel)
        currents = sorted(c[0] for c in calls)
        assert currents == [1, 2, 3]

    def test_empty_batch_returns_zero_counts(self, tmp_path: Path) -> None:
        """Empty event list should return zero counts without errors."""
        conn = _create_db()
        runner, _mocks = _make_runner(tmp_path, conn)

        counts = runner.process_batch([])
        assert all(v == 0 for v in counts.values())

    def test_batch_with_small_batch_size_triggers_multiple_embeds(
        self, tmp_path: Path,
    ) -> None:
        """When batch_size is small, embed_batch should be called multiple times."""
        from rag.config import AppConfig, EmbeddingConfig, FoldersConfig
        from rag.db.models import SqliteMetadataDB

        conn = _create_db()
        db = SqliteMetadataDB(conn)
        dedup = DedupChecker(conn)

        mock_embedder = MagicMock()
        mock_embedder.embed_batch.side_effect = lambda texts: [
            [0.1] * 3 for _ in texts
        ]
        mock_embedder.model_version = "test-model-v1"

        mock_vector_store = MagicMock()
        mock_parser = MagicMock()
        mock_parser.supported_types = {FileType.TXT, FileType.MD}

        call_count = 0

        def side_effect_parse(
            fp: str, ocr: bool, content_hash: str | None = None,
        ) -> ParseSuccess:
            nonlocal call_count
            call_count += 1
            doc = ParsedDocument(
                doc_id=str(uuid.uuid4()),
                title=f"Small Batch Doc {call_count}",
                file_type=FileType.TXT,
                sections=[
                    ParsedSection(
                        heading=f"Section {call_count}",
                        order=0,
                        text=(
                            f"Unique small batch content number {call_count}."
                            " Extra text to form a chunk."
                        ),
                    ),
                ],
                raw_content_hash=f"small_batch_raw_{call_count}",
            )
            return ParseSuccess(document=doc)

        mock_parser.parse.side_effect = side_effect_parse

        # Set batch_size=1 so each document's chunks trigger a flush
        config = AppConfig(
            folders=FoldersConfig(paths=[tmp_path]),
            embedding=EmbeddingConfig(batch_size=1),
        )

        runner = PipelineRunner(
            db=db,
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            parsers=[mock_parser],
            dedup=dedup,
            config=config,
        )

        events = []
        for i in range(3):
            f = tmp_path / f"small_{i}.txt"
            f.write_text(f"Small batch content {i}")
            events.append(
                FileEvent(
                    file_path=str(f),
                    content_hash=f"small_hash_{i}",
                    file_type=FileType.TXT,
                    event_type="created",
                    modified_at="2026-01-01T00:00:00+00:00",
                )
            )

        counts = runner.process_batch(events)
        assert counts[ProcessingOutcome.ERROR] == 0
        assert counts[ProcessingOutcome.INDEXED] == 3

        # With batch_size=1, each document should trigger its own embed call
        embed_call_count = mock_embedder.embed_batch.call_count
        assert embed_call_count >= 2, (
            f"Expected multiple embed_batch calls with small batch_size, got {embed_call_count}"
        )
