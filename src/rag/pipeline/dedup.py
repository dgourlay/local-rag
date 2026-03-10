from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3


class DedupChecker:
    """Check for duplicate documents using raw and normalized content hashes."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def check_duplicate(self, raw_hash: str, normalized_hash: str | None = None) -> str | None:
        """Check if a document with this hash already exists.

        Returns canonical doc_id if duplicate found, None if unique.
        """
        cursor = self._conn.execute(
            "SELECT canonical_doc_id FROM document_hashes WHERE raw_hash = ?",
            (raw_hash,),
        )
        row = cursor.fetchone()
        if row:
            return str(row[0])

        if normalized_hash:
            cursor = self._conn.execute(
                "SELECT canonical_doc_id FROM document_hashes WHERE normalized_hash = ?",
                (normalized_hash,),
            )
            row = cursor.fetchone()
            if row:
                return str(row[0])

        return None

    def register_hash(
        self,
        file_path: str,
        raw_hash: str,
        normalized_hash: str | None,
        canonical_doc_id: str,
    ) -> None:
        """Register a document's hashes for future dedup checks."""
        self._conn.execute(
            """INSERT OR REPLACE INTO document_hashes
               (file_path, raw_hash, normalized_hash, canonical_doc_id)
               VALUES (?, ?, ?, ?)""",
            (file_path, raw_hash, normalized_hash, canonical_doc_id),
        )
        self._conn.commit()
