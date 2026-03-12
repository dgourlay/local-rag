from __future__ import annotations

from typing import TYPE_CHECKING

from rag.types import ProcessingOutcome

if TYPE_CHECKING:
    from rag.db.models import SqliteMetadataDB
    from rag.db.qdrant import QdrantVectorStore
    from rag.pipeline.runner import PipelineRunner
    from rag.types import FileEvent


class TestIndexMarkdown:
    def test_index_quarterly_report(
        self,
        pipeline_runner: PipelineRunner,
        file_events: list[FileEvent],
        metadata_db: SqliteMetadataDB,
        vector_store: QdrantVectorStore,
    ) -> None:
        """Index quarterly-report.md, verify DB and vector state."""
        event = next(e for e in file_events if "quarterly-report" in e.file_path)
        outcome, _detail = pipeline_runner.process_file(event)
        assert outcome == ProcessingOutcome.INDEXED

        # Verify sync_state
        sync = metadata_db.get_sync_state(event.file_path)
        assert sync is not None
        assert sync.process_status == "done"

        # Verify document was inserted
        doc_row = metadata_db._conn.execute(
            "SELECT * FROM documents WHERE file_path = ?", (event.file_path,)
        ).fetchone()
        assert doc_row is not None
        assert doc_row["file_type"] == "md"
        assert doc_row["title"] == "quarterly-report"

        doc_id = doc_row["doc_id"]

        # Verify sections exist
        sections = metadata_db.get_sections(doc_id)
        assert len(sections) >= 3  # Revenue, Expenses, Outlook (+ heading)

        # Verify chunks exist
        chunks = metadata_db.get_chunks(doc_id)
        assert len(chunks) > 0

        # Verify Qdrant has points
        points, _ = vector_store._client.scroll(
            collection_name="test_documents",
            limit=100,
            with_payload=True,
        )
        doc_points = [p for p in points if p.payload and p.payload.get("doc_id") == doc_id]
        assert len(doc_points) > 0
        assert doc_points[0].payload["record_type"] == "chunk"


class TestIndexPlaintext:
    def test_index_readme_txt(
        self,
        pipeline_runner: PipelineRunner,
        file_events: list[FileEvent],
        metadata_db: SqliteMetadataDB,
    ) -> None:
        """Index readme.txt as a single section."""
        event = next(e for e in file_events if "readme.txt" in e.file_path)
        outcome, _detail = pipeline_runner.process_file(event)
        assert outcome == ProcessingOutcome.INDEXED

        sync = metadata_db.get_sync_state(event.file_path)
        assert sync is not None
        assert sync.process_status == "done"

        doc_row = metadata_db._conn.execute(
            "SELECT * FROM documents WHERE file_path = ?", (event.file_path,)
        ).fetchone()
        assert doc_row is not None

        sections = metadata_db.get_sections(doc_row["doc_id"])
        assert len(sections) == 1


class TestIndexEmptyFile:
    def test_empty_file_handled_gracefully(
        self,
        pipeline_runner: PipelineRunner,
        file_events: list[FileEvent],
        metadata_db: SqliteMetadataDB,
    ) -> None:
        """Empty file should be handled without crashing."""
        empty_events = [e for e in file_events if "empty.txt" in e.file_path]
        if empty_events:
            outcome, _detail = pipeline_runner.process_file(empty_events[0])
            assert outcome == ProcessingOutcome.ERROR
            sync = metadata_db.get_sync_state(empty_events[0].file_path)
            assert sync is not None
            assert sync.process_status in ("error", "poison")


class TestIndexCorruptedFile:
    def test_corrupted_pdf_returns_error(
        self,
        pipeline_runner: PipelineRunner,
        file_events: list[FileEvent],
        metadata_db: SqliteMetadataDB,
    ) -> None:
        """corrupted.pdf returns error, doesn't crash batch."""
        corrupted = [e for e in file_events if "corrupted.pdf" in e.file_path]
        if corrupted:
            outcome, _detail = pipeline_runner.process_file(corrupted[0])
            assert outcome == ProcessingOutcome.ERROR
            sync = metadata_db.get_sync_state(corrupted[0].file_path)
            assert sync is not None
            assert sync.process_status in ("error", "poison")


class TestIndexFullFolder:
    def test_index_all_fixtures(
        self,
        indexed_pipeline: tuple[PipelineRunner, int, int],
        metadata_db: SqliteMetadataDB,
        vector_store: QdrantVectorStore,
    ) -> None:
        """Index all fixtures, verify aggregate counts."""
        _runner, success, _errors = indexed_pipeline

        # We expect at least 4 successful: quarterly-report, project-plan,
        # meeting-notes, readme.txt. The duplicate is also "done".
        assert success >= 4

        doc_count = metadata_db.get_document_count()
        assert doc_count >= 4

        chunk_count = metadata_db.get_chunk_count()
        assert chunk_count > 0

        points, _ = vector_store._client.scroll(
            collection_name="test_documents",
            limit=200,
            with_payload=False,
        )
        assert len(points) > 0
