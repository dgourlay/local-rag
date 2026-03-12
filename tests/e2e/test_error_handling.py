from __future__ import annotations

from typing import TYPE_CHECKING

from rag.types import ProcessingOutcome

if TYPE_CHECKING:
    from rag.db.models import SqliteMetadataDB
    from rag.pipeline.runner import PipelineRunner
    from rag.types import FileEvent


class TestSingleBadFileDoesntCrashBatch:
    def test_batch_with_mixed_files(
        self,
        pipeline_runner: PipelineRunner,
        file_events: list[FileEvent],
        metadata_db: SqliteMetadataDB,
    ) -> None:
        """Batch with corrupted + valid files: valid files indexed, corrupted errored."""
        counts = pipeline_runner.process_batch(file_events)
        success = counts[ProcessingOutcome.INDEXED] + counts[ProcessingOutcome.DUPLICATE]

        # At least some files should succeed
        assert success >= 4

        # Check that valid files have process_status='done'
        for event in file_events:
            if "quarterly-report" in event.file_path or "readme.txt" in event.file_path:
                sync = metadata_db.get_sync_state(event.file_path)
                assert sync is not None
                assert sync.process_status == "done"
