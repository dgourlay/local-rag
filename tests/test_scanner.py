from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

from rag.config import FoldersConfig
from rag.sync.scanner import (
    classify_file_type,
    compute_file_hash,
    rescan_for_changes,
    scan_folders,
    should_ignore,
)
from rag.types import FileType, SyncStateRow

# --- classify_file_type ---


class TestClassifyFileType:
    def test_pdf(self) -> None:
        assert classify_file_type(Path("doc.pdf")) == FileType.PDF

    def test_docx(self) -> None:
        assert classify_file_type(Path("doc.docx")) == FileType.DOCX

    def test_txt(self) -> None:
        assert classify_file_type(Path("notes.txt")) == FileType.TXT

    def test_md(self) -> None:
        assert classify_file_type(Path("README.md")) == FileType.MD

    def test_uppercase_extension(self) -> None:
        assert classify_file_type(Path("DOC.PDF")) == FileType.PDF

    def test_unsupported_extension(self) -> None:
        assert classify_file_type(Path("image.png")) is None

    def test_no_extension(self) -> None:
        assert classify_file_type(Path("Makefile")) is None


# --- should_ignore ---


class TestShouldIgnore:
    def test_matches_node_modules(self) -> None:
        assert should_ignore(
            Path("/project/node_modules/pkg/file.txt"),
            ["**/node_modules"],
        )

    def test_matches_git_dir(self) -> None:
        assert should_ignore(
            Path("/project/.git/config"),
            ["**/.git"],
        )

    def test_no_match(self) -> None:
        assert not should_ignore(
            Path("/project/src/main.py"),
            ["**/node_modules", "**/.git"],
        )

    def test_pycache_pattern(self) -> None:
        assert should_ignore(
            Path("/project/src/__pycache__/module.pyc"),
            ["**/__pycache__"],
        )


# --- compute_file_hash ---


class TestComputeFileHash:
    def test_deterministic(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h1 = compute_file_hash(f)
        h2 = compute_file_hash(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        assert compute_file_hash(f1) != compute_file_hash(f2)


# --- scan_folders ---


class TestScanFolders:
    def test_scan_finds_matching_files(self, tmp_path: Path) -> None:
        (tmp_path / "doc.pdf").write_bytes(b"pdf content")
        (tmp_path / "notes.txt").write_text("text content")
        (tmp_path / "image.png").write_bytes(b"png content")

        config = FoldersConfig(
            paths=[tmp_path],
            extensions=[FileType.PDF, FileType.TXT],
            ignore=[],
        )
        events = scan_folders(config)
        paths = {e.file_path for e in events}
        assert str(tmp_path / "doc.pdf") in paths
        assert str(tmp_path / "notes.txt") in paths
        assert str(tmp_path / "image.png") not in paths

    def test_scan_respects_ignore_patterns(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "module.txt").write_text("cached")
        (tmp_path / "real.txt").write_text("real")

        config = FoldersConfig(
            paths=[tmp_path],
            extensions=[FileType.TXT],
            ignore=["**/__pycache__"],
        )
        events = scan_folders(config)
        paths = {e.file_path for e in events}
        assert str(tmp_path / "real.txt") in paths
        assert str(cache_dir / "module.txt") not in paths

    def test_scan_walks_subdirectories(self, tmp_path: Path) -> None:
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "deep.md").write_text("# Deep")
        (tmp_path / "top.md").write_text("# Top")

        config = FoldersConfig(
            paths=[tmp_path],
            extensions=[FileType.MD],
            ignore=[],
        )
        events = scan_folders(config)
        paths = {e.file_path for e in events}
        assert str(sub / "deep.md") in paths
        assert str(tmp_path / "top.md") in paths

    def test_scan_nonexistent_directory_graceful(self, tmp_path: Path) -> None:
        config = FoldersConfig(
            paths=[tmp_path / "nonexistent"],
            ignore=[],
        )
        events = scan_folders(config)
        assert events == []

    def test_scan_all_events_are_created(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("hello")

        config = FoldersConfig(
            paths=[tmp_path],
            extensions=[FileType.TXT],
            ignore=[],
        )
        events = scan_folders(config)
        assert len(events) == 1
        assert events[0].event_type == "created"

    def test_scan_populates_fields(self, tmp_path: Path) -> None:
        (tmp_path / "test.pdf").write_bytes(b"pdf bytes")

        config = FoldersConfig(
            paths=[tmp_path],
            extensions=[FileType.PDF],
            ignore=[],
        )
        events = scan_folders(config)
        assert len(events) == 1
        ev = events[0]
        assert ev.file_type == FileType.PDF
        assert len(ev.content_hash) == 64
        assert ev.modified_at  # ISO 8601 string
        assert ev.file_path == str(tmp_path / "test.pdf")

    def test_scan_multiple_folders(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "one.txt").write_text("one")
        (dir_b / "two.txt").write_text("two")

        config = FoldersConfig(
            paths=[dir_a, dir_b],
            extensions=[FileType.TXT],
            ignore=[],
        )
        events = scan_folders(config)
        assert len(events) == 2


# --- rescan_for_changes ---


def _make_sync_state(file_path: str, content_hash: str, modified_at: str) -> SyncStateRow:
    """Helper to build a minimal SyncStateRow for testing."""
    return SyncStateRow(
        id="test-id",
        file_path=file_path,
        file_name=Path(file_path).name,
        folder_path=str(Path(file_path).parent),
        folder_ancestors=[],
        file_type=Path(file_path).suffix.lstrip("."),
        modified_at=modified_at,
        content_hash=content_hash,
    )


class TestRescanForChanges:
    def test_detect_new_file(self, tmp_path: Path) -> None:
        (tmp_path / "new.txt").write_text("brand new")

        config = FoldersConfig(
            paths=[tmp_path],
            extensions=[FileType.TXT],
            ignore=[],
        )
        events = rescan_for_changes(
            config,
            get_sync_state=lambda _path: None,
            get_all_tracked_paths=lambda: [],
        )
        assert len(events) == 1
        assert events[0].event_type == "created"
        assert events[0].file_path == str(tmp_path / "new.txt")

    def test_detect_modified_file(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("original")
        original_hash = compute_file_hash(f)

        # Record original state
        stat = f.stat()
        original_mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
        state = _make_sync_state(str(f), original_hash, original_mtime)

        # Modify the file (ensure mtime changes)
        time.sleep(0.05)
        f.write_text("modified content")

        config = FoldersConfig(
            paths=[tmp_path],
            extensions=[FileType.TXT],
            ignore=[],
        )
        events = rescan_for_changes(
            config,
            get_sync_state=lambda _path: state,
            get_all_tracked_paths=lambda: [str(f)],
        )
        assert len(events) == 1
        assert events[0].event_type == "modified"
        assert events[0].content_hash != original_hash

    def test_skip_unchanged_file(self, tmp_path: Path) -> None:
        f = tmp_path / "stable.txt"
        f.write_text("no changes")
        content_hash = compute_file_hash(f)

        stat = f.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
        state = _make_sync_state(str(f), content_hash, mtime)

        config = FoldersConfig(
            paths=[tmp_path],
            extensions=[FileType.TXT],
            ignore=[],
        )
        events = rescan_for_changes(
            config,
            get_sync_state=lambda _path: state,
            get_all_tracked_paths=lambda: [str(f)],
        )
        assert events == []

    def test_detect_deleted_file(self, tmp_path: Path) -> None:
        deleted_path = str(tmp_path / "gone.txt")
        state = _make_sync_state(deleted_path, "abc123", "2025-01-01T00:00:00+00:00")

        config = FoldersConfig(
            paths=[tmp_path],
            extensions=[FileType.TXT],
            ignore=[],
        )
        events = rescan_for_changes(
            config,
            get_sync_state=lambda _path: state,
            get_all_tracked_paths=lambda: [deleted_path],
        )
        assert len(events) == 1
        assert events[0].event_type == "deleted"
        assert events[0].file_path == deleted_path

    def test_mtime_prefilter_skips_hash(self, tmp_path: Path) -> None:
        """When mtime matches, we skip hash computation entirely."""
        f = tmp_path / "cached.txt"
        f.write_text("cached content")
        content_hash = compute_file_hash(f)
        stat = f.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()

        # State matches current mtime -> should be skipped
        state = _make_sync_state(str(f), content_hash, mtime)

        call_count = 0
        original_get = lambda _path: state  # noqa: E731

        def tracking_get(path: str) -> SyncStateRow | None:
            nonlocal call_count
            call_count += 1
            return original_get(path)

        config = FoldersConfig(
            paths=[tmp_path],
            extensions=[FileType.TXT],
            ignore=[],
        )
        events = rescan_for_changes(
            config,
            get_sync_state=tracking_get,
            get_all_tracked_paths=lambda: [str(f)],
        )
        assert events == []
        assert call_count == 1  # Called once for the file lookup

    def test_mtime_changed_but_same_hash(self, tmp_path: Path) -> None:
        """If mtime changed but content hash is the same, no event emitted."""
        f = tmp_path / "touched.txt"
        f.write_text("same content")
        content_hash = compute_file_hash(f)

        # Record with a different mtime
        state = _make_sync_state(str(f), content_hash, "2020-01-01T00:00:00+00:00")

        config = FoldersConfig(
            paths=[tmp_path],
            extensions=[FileType.TXT],
            ignore=[],
        )
        events = rescan_for_changes(
            config,
            get_sync_state=lambda _path: state,
            get_all_tracked_paths=lambda: [str(f)],
        )
        # Hash same -> no modification event
        assert events == []
