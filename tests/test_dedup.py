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
