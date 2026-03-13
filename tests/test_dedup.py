from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rag.pipeline.dedup import DedupChecker

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


@pytest.fixture()
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    schema = (MIGRATIONS_DIR / "001_initial.sql").read_text()
    conn.executescript(schema)
    return conn


@pytest.fixture()
def checker(db: sqlite3.Connection) -> DedupChecker:
    return DedupChecker(db)


class TestCheckDuplicate:
    def test_unique_document_returns_none(self, checker: DedupChecker) -> None:
        result = checker.check_duplicate("abc123")
        assert result is None

    def test_raw_hash_match_detected(self, checker: DedupChecker) -> None:
        checker.register_hash("/docs/a.pdf", "raw1", "norm1", "doc-001")
        result = checker.check_duplicate("raw1")
        assert result == "doc-001"

    def test_normalized_hash_match_detected(self, checker: DedupChecker) -> None:
        checker.register_hash("/docs/a.pdf", "raw1", "norm1", "doc-001")
        result = checker.check_duplicate("different_raw", "norm1")
        assert result == "doc-001"

    def test_different_raw_same_normalized_is_duplicate(self, checker: DedupChecker) -> None:
        checker.register_hash("/docs/a.pdf", "raw_a", "norm_shared", "doc-001")
        result = checker.check_duplicate("raw_b", "norm_shared")
        assert result == "doc-001"

    def test_no_normalized_hash_still_checks_raw(self, checker: DedupChecker) -> None:
        checker.register_hash("/docs/a.pdf", "raw1", None, "doc-001")
        result = checker.check_duplicate("raw1")
        assert result == "doc-001"

    def test_no_match_returns_none(self, checker: DedupChecker) -> None:
        checker.register_hash("/docs/a.pdf", "raw1", "norm1", "doc-001")
        result = checker.check_duplicate("raw_other", "norm_other")
        assert result is None


class TestRegisterHash:
    def test_register_and_retrieve(self, checker: DedupChecker) -> None:
        checker.register_hash("/docs/b.pdf", "hash_b", "norm_b", "doc-002")
        assert checker.check_duplicate("hash_b") == "doc-002"

    def test_overwrite_on_same_file_path(
        self,
        checker: DedupChecker,
        db: sqlite3.Connection,
    ) -> None:
        checker.register_hash("/docs/a.pdf", "old_hash", "old_norm", "doc-001")
        checker.register_hash("/docs/a.pdf", "new_hash", "new_norm", "doc-002")

        # Old hashes should no longer match
        assert checker.check_duplicate("old_hash") is None
        # New hashes should match
        assert checker.check_duplicate("new_hash") == "doc-002"

        # Only one row for this file_path
        row = db.execute(
            "SELECT COUNT(*) FROM document_hashes WHERE file_path = ?",
            ("/docs/a.pdf",),
        ).fetchone()
        assert row[0] == 1


class TestWriteBatching:
    """Tests for deferred commit (write batching) behavior."""

    def test_register_hash_does_not_auto_commit(self, db: sqlite3.Connection) -> None:
        """register_hash should NOT commit; data should be lost if connection rolls back."""
        checker = DedupChecker(db)
        checker.register_hash("/docs/a.pdf", "raw1", "norm1", "doc-001")

        # Data is visible within the same connection (uncommitted)
        assert checker.check_duplicate("raw1") == "doc-001"

        # Roll back — if register_hash had committed, rollback would have no effect
        db.rollback()

        # After rollback, the uncommitted insert should be gone
        assert checker.check_duplicate("raw1") is None

    def test_flush_commits_data(self, db: sqlite3.Connection) -> None:
        """flush() should persist data so it survives a rollback."""
        checker = DedupChecker(db)
        checker.register_hash("/docs/a.pdf", "raw1", "norm1", "doc-001")
        checker.flush()

        # After flush + rollback, data should still be there
        db.rollback()
        assert checker.check_duplicate("raw1") == "doc-001"

    def test_flush_multiple_registers(self, db: sqlite3.Connection) -> None:
        """Multiple register_hash calls followed by one flush commits all."""
        checker = DedupChecker(db)
        checker.register_hash("/docs/a.pdf", "raw_a", "norm_a", "doc-001")
        checker.register_hash("/docs/b.pdf", "raw_b", "norm_b", "doc-002")
        checker.register_hash("/docs/c.pdf", "raw_c", "norm_c", "doc-003")
        checker.flush()

        db.rollback()
        assert checker.check_duplicate("raw_a") == "doc-001"
        assert checker.check_duplicate("raw_b") == "doc-002"
        assert checker.check_duplicate("raw_c") == "doc-003"

    def test_clear_all_still_commits_immediately(self, db: sqlite3.Connection) -> None:
        """clear_all should commit immediately (not deferred)."""
        checker = DedupChecker(db)
        checker.register_hash("/docs/a.pdf", "raw1", "norm1", "doc-001")
        checker.flush()

        checker.clear_all()
        db.rollback()

        # clear_all committed, so data is gone even after rollback
        assert checker.check_duplicate("raw1") is None
