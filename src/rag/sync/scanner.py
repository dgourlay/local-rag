from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from rag.config import FoldersConfig
    from rag.types import SyncStateRow

from rag.types import FileEvent, FileType


def classify_file_type(path: Path) -> FileType | None:
    """Map file extension to FileType, or None if not supported."""
    ext = path.suffix.lower().lstrip(".")
    try:
        return FileType(ext)
    except ValueError:
        return None


def compute_file_hash(path: Path) -> str:
    """SHA-256 hash of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def should_ignore(path: Path, ignore_patterns: list[str]) -> bool:
    """Check if path matches any ignore glob pattern."""
    path_str = str(path)
    for pattern in ignore_patterns:
        if fnmatch(path_str, pattern):
            return True
        # Also check each path component against stripped pattern
        stripped = pattern.strip("*/")
        if stripped:
            for part in path.parts:
                if fnmatch(part, stripped):
                    return True
    return False


def scan_folders(config: FoldersConfig) -> list[FileEvent]:
    """Walk all configured paths and return FileEvent for each matching file."""
    events: list[FileEvent] = []
    valid_extensions = {ft.value for ft in config.extensions}

    for folder in config.paths:
        folder = Path(folder).expanduser().resolve()
        if not folder.is_dir():
            continue
        for root, _dirs, files in os.walk(folder):
            root_path = Path(root)
            for fname in files:
                file_path = root_path / fname
                if should_ignore(file_path, config.ignore):
                    continue
                ft = classify_file_type(file_path)
                if ft is None or ft.value not in valid_extensions:
                    continue
                try:
                    content_hash = compute_file_hash(file_path)
                    stat = file_path.stat()
                    modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
                    events.append(
                        FileEvent(
                            file_path=str(file_path),
                            content_hash=content_hash,
                            file_type=ft,
                            event_type="created",
                            modified_at=modified_at,
                        )
                    )
                except OSError:
                    continue
    return events


def rescan_for_changes(
    config: FoldersConfig,
    get_sync_state: Callable[[str], SyncStateRow | None],
    get_all_tracked_paths: Callable[[], list[str]],
) -> list[FileEvent]:
    """Compare filesystem against sync_state, using mtime as pre-filter.

    1. Walk all files matching config (like scan_folders).
    2. For each file, check sync_state:
       - No entry -> "created"
       - Entry exists, mtime changed -> compute hash -> if hash differs -> "modified"
       - Entry exists, mtime same -> skip
    3. For tracked paths not found on disk -> "deleted"
    """
    events: list[FileEvent] = []
    valid_extensions = {ft.value for ft in config.extensions}
    seen_paths: set[str] = set()

    for folder in config.paths:
        folder = Path(folder).expanduser().resolve()
        if not folder.is_dir():
            continue
        for root, _dirs, files in os.walk(folder):
            root_path = Path(root)
            for fname in files:
                file_path = root_path / fname
                if should_ignore(file_path, config.ignore):
                    continue
                ft = classify_file_type(file_path)
                if ft is None or ft.value not in valid_extensions:
                    continue

                path_str = str(file_path)
                seen_paths.add(path_str)

                try:
                    stat = file_path.stat()
                except OSError:
                    continue

                modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()

                existing = get_sync_state(path_str)
                if existing is None:
                    # New file
                    content_hash = compute_file_hash(file_path)
                    events.append(
                        FileEvent(
                            file_path=path_str,
                            content_hash=content_hash,
                            file_type=ft,
                            event_type="created",
                            modified_at=modified_at,
                        )
                    )
                elif existing.modified_at != modified_at:
                    # Mtime changed — check hash
                    content_hash = compute_file_hash(file_path)
                    if content_hash != existing.content_hash:
                        events.append(
                            FileEvent(
                                file_path=path_str,
                                content_hash=content_hash,
                                file_type=ft,
                                event_type="modified",
                                modified_at=modified_at,
                            )
                        )
                # else: mtime same -> skip

    # Detect deleted files
    for tracked_path in get_all_tracked_paths():
        if tracked_path not in seen_paths:
            existing = get_sync_state(tracked_path)
            if existing is not None:
                ft = classify_file_type(Path(tracked_path))
                events.append(
                    FileEvent(
                        file_path=tracked_path,
                        content_hash=existing.content_hash,
                        file_type=ft if ft is not None else FileType.TXT,
                        event_type="deleted",
                        modified_at=existing.modified_at,
                    )
                )

    return events
