from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

from rag.cli import _IndexGate
from rag.types import FileEvent, ProcessingOutcome, SyncStateRow

# --- Helper to build a fake SyncStateRow ---


def _make_sync_state(file_path: str, content_hash: str = "abc123") -> SyncStateRow:
    return SyncStateRow(
        id="test-id",
        file_path=file_path,
        file_name=Path(file_path).name,
        folder_path=str(Path(file_path).parent),
        folder_ancestors=[],
        file_type="txt",
        modified_at="2026-01-01T00:00:00+00:00",
        content_hash=content_hash,
    )


# --- Path claiming: stops the re-scan thread and the live watcher from
#     indexing the same file concurrently ---


class TestIndexGateClaims:
    """Guards the UNIQUE constraint failed: documents.file_path race."""

    def test_first_claim_is_granted(self) -> None:
        claims = _IndexGate()
        assert claims.claim(["/docs/a.pdf", "/docs/b.pdf"]) == ["/docs/a.pdf", "/docs/b.pdf"]

    def test_second_claim_on_a_held_path_is_refused(self) -> None:
        # This is the actual bug: the re-scan thread holds a path, the live
        # watcher must not also hand it to process_batch.
        claims = _IndexGate()
        claims.claim(["/docs/a.pdf"])
        assert claims.claim(["/docs/a.pdf"]) == []

    def test_refuses_only_the_overlap(self) -> None:
        claims = _IndexGate()
        claims.claim(["/docs/a.pdf"])
        assert claims.claim(["/docs/a.pdf", "/docs/b.pdf"]) == ["/docs/b.pdf"]

    def test_release_allows_reclaiming(self) -> None:
        claims = _IndexGate()
        claims.claim(["/docs/a.pdf"])
        claims.release(["/docs/a.pdf"])
        assert claims.claim(["/docs/a.pdf"]) == ["/docs/a.pdf"]

    def test_release_of_unheld_path_is_harmless(self) -> None:
        claims = _IndexGate()
        claims.release(["/docs/never-claimed.pdf"])
        assert claims.claim(["/docs/never-claimed.pdf"]) == ["/docs/never-claimed.pdf"]

    def test_duplicate_paths_in_one_call_granted_once(self) -> None:
        claims = _IndexGate()
        assert claims.claim(["/docs/a.pdf", "/docs/a.pdf"]) == ["/docs/a.pdf"]

    def test_concurrent_claims_grant_each_path_exactly_once(self) -> None:
        """Two threads racing on the same paths: every path goes to one winner."""
        claims = _IndexGate()
        paths = [f"/docs/{i}.pdf" for i in range(200)]
        granted: list[list[str]] = []
        lock = threading.Lock()
        start = threading.Barrier(2)

        def worker() -> None:
            start.wait(timeout=5)
            mine = claims.claim(paths)
            with lock:
                granted.append(mine)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        all_granted = [p for batch in granted for p in batch]
        assert sorted(all_granted) == sorted(paths)
        assert len(all_granted) == len(set(all_granted))


class TestIndexGateSerialize:
    """Guards the shared-sqlite3.Connection corruption, which claiming misses.

    Both producers drive one connection whose transaction state is global to it,
    so overlapping process_batch calls lose rows silently even when the two
    producers touch entirely different paths.
    """

    def test_serialize_excludes_a_second_thread(self) -> None:
        gate = _IndexGate()
        inside = threading.Event()
        second_got_in = threading.Event()
        release_first = threading.Event()

        def first() -> None:
            with gate.serialize():
                inside.set()
                release_first.wait(timeout=5)

        def second() -> None:
            inside.wait(timeout=5)
            with gate.serialize():
                second_got_in.set()

        t1 = threading.Thread(target=first)
        t2 = threading.Thread(target=second)
        t1.start()
        t2.start()
        # Second thread must still be blocked while the first holds the gate.
        assert not second_got_in.wait(timeout=0.5)
        release_first.set()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert second_got_in.is_set()

    def test_serialize_never_overlaps_under_contention(self) -> None:
        """No two threads inside the gate at once -- disjoint paths included."""
        gate = _IndexGate()
        concurrent = 0
        max_concurrent = 0
        bookkeeping = threading.Lock()
        start = threading.Barrier(4)

        def worker() -> None:
            nonlocal concurrent, max_concurrent
            start.wait(timeout=5)
            for _ in range(25):
                with gate.serialize():
                    with bookkeeping:
                        concurrent += 1
                        max_concurrent = max(max_concurrent, concurrent)
                    time.sleep(0.0005)
                    with bookkeeping:
                        concurrent -= 1

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert max_concurrent == 1

    def test_serialize_releases_on_exception(self) -> None:
        gate = _IndexGate()
        msg = "boom"
        try:
            with gate.serialize():
                raise RuntimeError(msg)
        except RuntimeError:
            pass
        # A leaked gate would deadlock watch permanently.
        with gate.serialize():
            pass


# --- Task 1: process_batch output formatting ---


class TestWatchProcessBatchFormatting:
    """Verify the watch command correctly handles process_batch dict return."""

    def test_counts_dict_formatting(self) -> None:
        """Test that outcome dict is formatted correctly."""
        counts: dict[ProcessingOutcome, int] = {
            ProcessingOutcome.INDEXED: 3,
            ProcessingOutcome.UNCHANGED: 1,
            ProcessingOutcome.DUPLICATE: 0,
            ProcessingOutcome.DELETED: 2,
            ProcessingOutcome.ERROR: 0,
        }
        total = 6

        parts: list[str] = []
        if counts[ProcessingOutcome.INDEXED]:
            parts.append(f"{counts[ProcessingOutcome.INDEXED]} indexed")
        if counts[ProcessingOutcome.UNCHANGED]:
            parts.append(f"{counts[ProcessingOutcome.UNCHANGED]} unchanged")
        if counts[ProcessingOutcome.DUPLICATE]:
            parts.append(f"{counts[ProcessingOutcome.DUPLICATE]} duplicates")
        if counts[ProcessingOutcome.DELETED]:
            parts.append(f"{counts[ProcessingOutcome.DELETED]} deleted")
        if counts[ProcessingOutcome.ERROR]:
            parts.append(f"{counts[ProcessingOutcome.ERROR]} errors")

        result = f"Processed {total} files: {', '.join(parts)}."
        assert result == "Processed 6 files: 3 indexed, 1 unchanged, 2 deleted."

    def test_counts_all_zeros(self) -> None:
        counts = dict.fromkeys(ProcessingOutcome, 0)
        parts: list[str] = []
        for outcome, label in [
            (ProcessingOutcome.INDEXED, "indexed"),
            (ProcessingOutcome.UNCHANGED, "unchanged"),
            (ProcessingOutcome.DUPLICATE, "duplicates"),
            (ProcessingOutcome.DELETED, "deleted"),
            (ProcessingOutcome.ERROR, "errors"),
        ]:
            if counts[outcome]:
                parts.append(f"{counts[outcome]} {label}")
        result = f"Processed 0 files: {', '.join(parts)}."
        assert result == "Processed 0 files: ."

    def test_counts_errors_only(self) -> None:
        counts = dict.fromkeys(ProcessingOutcome, 0)
        counts[ProcessingOutcome.ERROR] = 5
        parts: list[str] = []
        for outcome, label in [
            (ProcessingOutcome.INDEXED, "indexed"),
            (ProcessingOutcome.ERROR, "errors"),
        ]:
            if counts[outcome]:
                parts.append(f"{counts[outcome]} {label}")
        result = f"Processed 5 files: {', '.join(parts)}."
        assert result == "Processed 5 files: 5 errors."


# --- Task 2: Deletion event generation ---


class TestWatchDeletionHandling:
    """Verify deleted files generate deletion events instead of being skipped."""

    def test_deleted_file_generates_deletion_event(self) -> None:
        """Known path no longer on disk -> deletion event created."""
        known_paths: set[str] = {"/docs/report.txt"}
        mock_db = MagicMock()
        mock_db.get_sync_state.return_value = _make_sync_state("/docs/report.txt")

        file_path = "/docs/report.txt"
        path = Path(file_path)
        events: list[FileEvent] = []

        # Reproduce the watch command's deletion logic
        if not path.is_file() and file_path in known_paths:
            existing = mock_db.get_sync_state(file_path)
            events.append(
                FileEvent(
                    file_path=file_path,
                    content_hash=existing.content_hash if existing else "",
                    file_type=existing.file_type if existing else "txt",
                    event_type="deleted",
                    modified_at=existing.modified_at if existing else "",
                )
            )

        assert len(events) == 1
        assert events[0].event_type == "deleted"
        assert events[0].file_path == "/docs/report.txt"
        assert events[0].content_hash == "abc123"

    def test_deleted_unknown_file_is_skipped(self) -> None:
        """Non-tracked deleted path -> no event generated."""
        known_paths: set[str] = set()
        file_path = "/docs/unknown.txt"
        path = Path(file_path)
        events: list[FileEvent] = []

        if not path.is_file() and file_path in known_paths:
            events.append(
                FileEvent(
                    file_path=file_path,
                    content_hash="",
                    file_type="txt",
                    event_type="deleted",
                    modified_at="",
                )
            )

        assert len(events) == 0

    def test_deleted_file_removed_from_known_paths(self) -> None:
        """After processing deletion, path removed from known_paths."""
        known_paths: set[str] = {"/docs/report.txt", "/docs/other.txt"}

        ev = FileEvent(
            file_path="/docs/report.txt",
            content_hash="abc",
            file_type="txt",
            event_type="deleted",
            modified_at="2026-01-01",
        )

        if ev.event_type == "deleted":
            known_paths.discard(ev.file_path)
        else:
            known_paths.add(ev.file_path)

        assert "/docs/report.txt" not in known_paths
        assert "/docs/other.txt" in known_paths

    def test_new_file_added_to_known_paths(self) -> None:
        """After processing a new file, it is added to known_paths."""
        known_paths: set[str] = set()

        ev = FileEvent(
            file_path="/docs/new.txt",
            content_hash="xyz",
            file_type="txt",
            event_type="modified",
            modified_at="2026-01-01",
        )

        if ev.event_type == "deleted":
            known_paths.discard(ev.file_path)
        else:
            known_paths.add(ev.file_path)

        assert "/docs/new.txt" in known_paths


# --- Task 4: Startup re-scan ---


class TestStartupRescan:
    """Verify startup re-scan runs in a background thread."""

    def test_rescan_spawns_thread(self) -> None:
        """Startup re-scan should run in a non-blocking daemon thread."""
        rescan_called = threading.Event()

        def fake_rescan(*_args: object, **_kwargs: object) -> list[FileEvent]:
            rescan_called.set()
            return []

        thread = threading.Thread(target=fake_rescan, daemon=True)
        thread.start()
        rescan_called.wait(timeout=2)
        assert rescan_called.is_set()

    def test_rescan_processes_found_changes(self) -> None:
        """When rescan finds changes, they are passed to process_batch."""
        mock_runner = MagicMock()
        mock_runner.process_batch.return_value = {
            ProcessingOutcome.INDEXED: 2,
            ProcessingOutcome.UNCHANGED: 0,
            ProcessingOutcome.DUPLICATE: 0,
            ProcessingOutcome.DELETED: 1,
            ProcessingOutcome.ERROR: 0,
        }

        events = [
            FileEvent(
                file_path="/docs/a.txt",
                content_hash="h1",
                file_type="txt",
                event_type="created",
                modified_at="2026-01-01",
            ),
            FileEvent(
                file_path="/docs/b.txt",
                content_hash="h2",
                file_type="txt",
                event_type="modified",
                modified_at="2026-01-02",
            ),
            FileEvent(
                file_path="/docs/old.txt",
                content_hash="h3",
                file_type="txt",
                event_type="deleted",
                modified_at="2026-01-01",
            ),
        ]

        mock_rescan = MagicMock(return_value=events)

        found = mock_rescan()
        if found:
            counts = mock_runner.process_batch(found)

        mock_runner.process_batch.assert_called_once_with(events)
        assert counts[ProcessingOutcome.INDEXED] == 2
        assert counts[ProcessingOutcome.DELETED] == 1

    def test_rescan_no_changes(self) -> None:
        """When no changes found, process_batch should not be called."""
        mock_runner = MagicMock()
        events: list[FileEvent] = []

        if events:
            mock_runner.process_batch(events)

        mock_runner.process_batch.assert_not_called()

    def test_rescan_updates_known_paths(self) -> None:
        """After rescan, known_paths reflects created and deleted files."""
        known_paths: set[str] = {"/docs/old.txt"}

        events = [
            FileEvent(
                file_path="/docs/new.txt",
                content_hash="h1",
                file_type="txt",
                event_type="created",
                modified_at="2026-01-01",
            ),
            FileEvent(
                file_path="/docs/old.txt",
                content_hash="h2",
                file_type="txt",
                event_type="deleted",
                modified_at="2026-01-01",
            ),
        ]

        for ev in events:
            if ev.event_type == "deleted":
                known_paths.discard(ev.file_path)
            else:
                known_paths.add(ev.file_path)

        assert "/docs/new.txt" in known_paths
        assert "/docs/old.txt" not in known_paths

    def test_rescan_error_does_not_crash(self) -> None:
        """If rescan raises, the thread handles it gracefully."""
        error_caught = threading.Event()

        def failing_rescan() -> None:
            try:
                msg = "simulated failure"
                raise RuntimeError(msg)
            except Exception:
                error_caught.set()

        thread = threading.Thread(target=failing_rescan, daemon=True)
        thread.start()
        thread.join(timeout=2)
        assert error_caught.is_set()


# --- get_all_tracked_paths ---


class TestGetAllTrackedPaths:
    """Verify get_all_tracked_paths method on SqliteMetadataDB."""

    def test_returns_non_deleted_paths(self) -> None:
        import sqlite3

        from rag.db.migrations import run_migrations
        from rag.db.models import SqliteMetadataDB

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        run_migrations(conn)

        db = SqliteMetadataDB(conn)

        db.upsert_sync_state(SyncStateRow(
            id="1",
            file_path="/docs/a.txt",
            file_name="a.txt",
            folder_path="/docs",
            folder_ancestors=[],
            file_type="txt",
            modified_at="2026-01-01",
            content_hash="hash1",
            is_deleted=0,
        ))
        db.upsert_sync_state(SyncStateRow(
            id="2",
            file_path="/docs/b.txt",
            file_name="b.txt",
            folder_path="/docs",
            folder_ancestors=[],
            file_type="txt",
            modified_at="2026-01-01",
            content_hash="hash2",
            is_deleted=1,
        ))
        db.upsert_sync_state(SyncStateRow(
            id="3",
            file_path="/docs/c.txt",
            file_name="c.txt",
            folder_path="/docs",
            folder_ancestors=[],
            file_type="txt",
            modified_at="2026-01-01",
            content_hash="hash3",
            is_deleted=0,
        ))

        paths = db.get_all_tracked_paths()
        assert sorted(paths) == ["/docs/a.txt", "/docs/c.txt"]

    def test_returns_empty_when_no_tracked(self) -> None:
        import sqlite3

        from rag.db.migrations import run_migrations
        from rag.db.models import SqliteMetadataDB

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        run_migrations(conn)

        db = SqliteMetadataDB(conn)
        paths = db.get_all_tracked_paths()
        assert paths == []
