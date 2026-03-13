from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from concurrent.futures import Future

    from rag.db.qdrant import AsyncQdrantVectorStore
    from rag.types import VectorPoint

logger = logging.getLogger(__name__)


class BackgroundUpsertWorker:
    """Runs async Qdrant upserts on a background event loop thread.

    Usage::

        worker = BackgroundUpsertWorker(async_store)
        worker.start()
        future = worker.submit_upsert(doc_id, points)
        # ... do other work ...
        worker.wait_all()   # block until all pending upserts finish
        worker.stop()
    """

    def __init__(self, async_store: AsyncQdrantVectorStore) -> None:
        self._async_store = async_store
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._pending: list[Future[None]] = []

    def start(self) -> None:
        """Start the background event loop thread."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="qdrant-upsert-worker"
        )
        self._thread.start()

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit_upsert(self, doc_id: str, points: list[VectorPoint]) -> Future[None]:
        """Submit an async upsert to the background loop. Returns a Future."""
        if self._loop is None or self._thread is None or not self._thread.is_alive():
            msg = "BackgroundUpsertWorker is not running"
            raise RuntimeError(msg)

        future = asyncio.run_coroutine_threadsafe(
            self._async_store.upsert_points(doc_id, points),
            self._loop,
        )
        self._pending.append(future)
        return future

    def submit_delete_stale(
        self, doc_id: str, keep_ids: set[str]
    ) -> Future[None]:
        """Submit an async delete_stale_points to the background loop."""
        if self._loop is None or self._thread is None or not self._thread.is_alive():
            msg = "BackgroundUpsertWorker is not running"
            raise RuntimeError(msg)

        future = asyncio.run_coroutine_threadsafe(
            self._async_store.delete_stale_points(doc_id, keep_ids),
            self._loop,
        )
        self._pending.append(future)
        return future

    def wait_all(self, timeout: float | None = None) -> list[Exception]:
        """Wait for all pending upserts to complete. Returns list of exceptions (if any)."""
        errors: list[Exception] = []
        for future in self._pending:
            try:
                future.result(timeout=timeout)
            except Exception as exc:
                logger.exception("Background upsert failed")
                errors.append(exc)
        self._pending.clear()
        return errors

    def stop(self) -> None:
        """Stop the background event loop and join the thread."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._loop = None
        self._thread = None
        self._pending.clear()

    @property
    def pending_count(self) -> int:
        """Number of pending upserts."""
        return len(self._pending)
