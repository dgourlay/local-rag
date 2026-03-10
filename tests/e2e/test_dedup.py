from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.db.models import SqliteMetadataDB
    from rag.db.qdrant import QdrantVectorStore
    from rag.pipeline.runner import PipelineRunner
    from rag.types import FileEvent


class TestExactDuplicateSuppressed:
    def test_duplicate_marked_correctly(
        self,
        pipeline_runner: PipelineRunner,
        file_events: list[FileEvent],
        metadata_db: SqliteMetadataDB,
        vector_store: QdrantVectorStore,
    ) -> None:
        """Index quarterly-report.md then duplicate-report.md: only one canonical."""
        original = next(e for e in file_events if "quarterly-report" in e.file_path)
        duplicate = next(e for e in file_events if "duplicate-report" in e.file_path)

        assert pipeline_runner.process_file(original) is True
        assert pipeline_runner.process_file(duplicate) is True

        orig_row = metadata_db._conn.execute(
            "SELECT * FROM documents WHERE file_path = ?", (original.file_path,)
        ).fetchone()
        dup_row = metadata_db._conn.execute(
            "SELECT * FROM documents WHERE file_path = ?", (duplicate.file_path,)
        ).fetchone()

        assert orig_row is not None
        assert dup_row is not None

        # The duplicate should reference the original's doc_id
        assert dup_row["duplicate_of_doc_id"] is not None
        assert dup_row["duplicate_of_doc_id"] == orig_row["doc_id"]

        # Only the original should have chunks in Qdrant
        points, _ = vector_store._client.scroll(
            collection_name="test_documents",
            limit=200,
            with_payload=True,
        )
        orig_points = [
            p for p in points if p.payload and p.payload.get("doc_id") == orig_row["doc_id"]
        ]
        dup_points = [
            p for p in points if p.payload and p.payload.get("doc_id") == dup_row["doc_id"]
        ]
        assert len(orig_points) > 0
        assert len(dup_points) == 0
