from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from rag.pipeline.classifier import classify
from rag.pipeline.dedup import DedupChecker
from rag.pipeline.runner import PipelineRunner
from rag.results import ParseError, ParseSuccess
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
    schema = Path(__file__).parent.parent / "migrations" / "001_initial.sql"
    conn.executescript(schema.read_text())
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

        def side_effect_parse(fp: str, ocr: bool) -> ParseSuccess:
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

        def side_effect_parse(fp: str, ocr: bool) -> ParseSuccess | ParseError:
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
