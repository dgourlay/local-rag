from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from rag.db.async_upsert import BackgroundUpsertWorker


@pytest.fixture()
def mock_async_store() -> MagicMock:
    """Create a mock AsyncQdrantVectorStore."""
    store = MagicMock()
    store.upsert_points = AsyncMock(return_value=None)
    store.delete_stale_points = AsyncMock(return_value=None)
    return store


@pytest.fixture()
def worker(mock_async_store: MagicMock) -> BackgroundUpsertWorker:
    """Create and start a BackgroundUpsertWorker."""
    w = BackgroundUpsertWorker(mock_async_store)
    w.start()
    yield w
    w.stop()


class TestBackgroundUpsertWorker:
    def test_submit_upsert_delegates_to_async_store(
        self, worker: BackgroundUpsertWorker, mock_async_store: MagicMock
    ) -> None:
        points = [MagicMock()]
        worker.submit_upsert("doc-1", points)
        errors = worker.wait_all()

        assert errors == []
        mock_async_store.upsert_points.assert_awaited_once_with("doc-1", points)

    def test_submit_delete_stale_delegates_to_async_store(
        self, worker: BackgroundUpsertWorker, mock_async_store: MagicMock
    ) -> None:
        keep = {"point-1", "point-2"}
        worker.submit_delete_stale("doc-1", keep)
        errors = worker.wait_all()

        assert errors == []
        mock_async_store.delete_stale_points.assert_awaited_once_with("doc-1", keep)

    def test_multiple_upserts_all_complete(
        self, worker: BackgroundUpsertWorker, mock_async_store: MagicMock
    ) -> None:
        for i in range(5):
            worker.submit_upsert(f"doc-{i}", [MagicMock()])

        assert worker.pending_count == 5
        errors = worker.wait_all()

        assert errors == []
        assert mock_async_store.upsert_points.await_count == 5
        assert worker.pending_count == 0

    def test_error_in_upsert_returned_by_wait_all(
        self, mock_async_store: MagicMock
    ) -> None:
        mock_async_store.upsert_points = AsyncMock(
            side_effect=RuntimeError("connection refused")
        )
        w = BackgroundUpsertWorker(mock_async_store)
        w.start()
        try:
            w.submit_upsert("doc-1", [MagicMock()])
            errors = w.wait_all()

            assert len(errors) == 1
            assert "connection refused" in str(errors[0])
        finally:
            w.stop()

    def test_submit_after_stop_raises(
        self, mock_async_store: MagicMock
    ) -> None:
        w = BackgroundUpsertWorker(mock_async_store)
        w.start()
        w.stop()

        with pytest.raises(RuntimeError, match="not running"):
            w.submit_upsert("doc-1", [MagicMock()])

    def test_submit_before_start_raises(
        self, mock_async_store: MagicMock
    ) -> None:
        w = BackgroundUpsertWorker(mock_async_store)

        with pytest.raises(RuntimeError, match="not running"):
            w.submit_upsert("doc-1", [MagicMock()])

    def test_stop_is_idempotent(
        self, mock_async_store: MagicMock
    ) -> None:
        w = BackgroundUpsertWorker(mock_async_store)
        w.start()
        w.stop()
        # Should not raise
        w.stop()

    def test_wait_all_clears_pending(
        self, worker: BackgroundUpsertWorker, mock_async_store: MagicMock
    ) -> None:
        worker.submit_upsert("doc-1", [MagicMock()])
        worker.submit_upsert("doc-2", [MagicMock()])
        assert worker.pending_count == 2

        worker.wait_all()
        assert worker.pending_count == 0

        # Second wait_all should be a no-op
        errors = worker.wait_all()
        assert errors == []
