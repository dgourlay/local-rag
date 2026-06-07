from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import sqlite3

from rag.db.connection import get_connection
from rag.db.migrations import run_migrations
from rag.db.models import SqliteMetadataDB
from rag.types import (
    ChunkRow,
    DocumentRow,
    ProcessingLogEntry,
    SectionRow,
    SyncStateRow,
)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


@pytest.fixture()
def conn(tmp_path: Path) -> sqlite3.Connection:
    db_path = tmp_path / "test.db"
    c = get_connection(db_path)
    run_migrations(c, migrations_dir=MIGRATIONS_DIR)
    return c


@pytest.fixture()
def db(conn: sqlite3.Connection) -> SqliteMetadataDB:
    return SqliteMetadataDB(conn)


class TestConnection:
    def test_wal_mode(self, conn: sqlite3.Connection) -> None:
        row = conn.execute("PRAGMA journal_mode").fetchone()
        assert row is not None
        assert row[0] == "wal"

    def test_foreign_keys_enabled(self, conn: sqlite3.Connection) -> None:
        row = conn.execute("PRAGMA foreign_keys").fetchone()
        assert row is not None
        assert row[0] == 1

    def test_busy_timeout(self, conn: sqlite3.Connection) -> None:
        row = conn.execute("PRAGMA busy_timeout").fetchone()
        assert row is not None
        assert row[0] == 30000


class TestMigrations:
    def test_tables_created(self, conn: sqlite3.Connection) -> None:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        expected = {
            "sync_state",
            "documents",
            "sections",
            "chunks",
            "document_hashes",
            "processing_log",
        }
        assert expected.issubset(tables)

    def test_idempotent(self, conn: sqlite3.Connection) -> None:
        run_migrations(conn, migrations_dir=MIGRATIONS_DIR)
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "sync_state" in tables


class TestSyncState:
    def _make_sync_state(self, **overrides: object) -> SyncStateRow:
        defaults: dict[str, object] = {
            "id": "sync-1",
            "file_path": "/docs/test.pdf",
            "file_name": "test.pdf",
            "folder_path": "/docs",
            "folder_ancestors": ["/docs", "/"],
            "file_type": "pdf",
            "size_bytes": 1024,
            "modified_at": "2026-01-01T00:00:00",
            "content_hash": "abc123",
            "process_status": "pending",
        }
        defaults.update(overrides)
        return SyncStateRow.model_validate(defaults)

    def test_upsert_and_get(self, db: SqliteMetadataDB) -> None:
        state = self._make_sync_state()
        db.upsert_sync_state(state)
        result = db.get_sync_state("/docs/test.pdf")
        assert result is not None
        assert result.id == "sync-1"
        assert result.folder_ancestors == ["/docs", "/"]
        assert result.process_status == "pending"

    def test_get_nonexistent(self, db: SqliteMetadataDB) -> None:
        assert db.get_sync_state("/nonexistent") is None

    def test_upsert_updates(self, db: SqliteMetadataDB) -> None:
        state = self._make_sync_state()
        db.upsert_sync_state(state)
        updated = self._make_sync_state(process_status="done")
        db.upsert_sync_state(updated)
        result = db.get_sync_state("/docs/test.pdf")
        assert result is not None
        assert result.process_status == "done"

    def test_get_pending_files(self, db: SqliteMetadataDB) -> None:
        db.upsert_sync_state(self._make_sync_state(id="s1", file_path="/a.pdf"))
        db.upsert_sync_state(
            self._make_sync_state(id="s2", file_path="/b.pdf", process_status="done")
        )
        db.upsert_sync_state(self._make_sync_state(id="s3", file_path="/c.pdf"))
        pending = db.get_pending_files(limit=10)
        assert len(pending) == 2
        paths = {p.file_path for p in pending}
        assert paths == {"/a.pdf", "/c.pdf"}

    def test_get_pending_excludes_deleted(self, db: SqliteMetadataDB) -> None:
        db.upsert_sync_state(self._make_sync_state(id="s1", file_path="/a.pdf", is_deleted=1))
        assert db.get_pending_files(limit=10) == []


class TestDocuments:
    def _make_document(self, **overrides: object) -> DocumentRow:
        defaults: dict[str, object] = {
            "doc_id": "doc-1",
            "file_path": "/docs/test.pdf",
            "folder_path": "/docs",
            "folder_ancestors": ["/docs", "/"],
            "title": "Test Document",
            "file_type": "pdf",
            "modified_at": "2026-01-01T00:00:00",
            "raw_content_hash": "abc123",
            "normalized_content_hash": "def456",
            "key_topics": ["python", "testing"],
        }
        defaults.update(overrides)
        return DocumentRow.model_validate(defaults)

    def test_upsert_and_get(self, db: SqliteMetadataDB) -> None:
        doc = self._make_document()
        db.upsert_document(doc)
        result = db.get_document("doc-1")
        assert result is not None
        assert result.title == "Test Document"
        assert result.folder_ancestors == ["/docs", "/"]
        assert result.key_topics == ["python", "testing"]

    def test_get_nonexistent(self, db: SqliteMetadataDB) -> None:
        assert db.get_document("nonexistent") is None

    def test_get_documents_batch(self, db: SqliteMetadataDB) -> None:
        db.upsert_document(
            self._make_document(doc_id="d1", file_path="/docs/a.pdf", title="Doc A")
        )
        db.upsert_document(
            self._make_document(doc_id="d2", file_path="/docs/b.pdf", title="Doc B")
        )

        result = db.get_documents(["d2", "missing", "d1", "d1"])

        assert set(result) == {"d1", "d2"}
        assert result["d1"].title == "Doc A"
        assert result["d2"].title == "Doc B"

    def test_get_documents_batch_empty(self, db: SqliteMetadataDB) -> None:
        assert db.get_documents([]) == {}

    def test_get_by_hash(self, db: SqliteMetadataDB) -> None:
        db.upsert_document(self._make_document())
        result = db.get_document_by_hash("def456")
        assert result is not None
        assert result.doc_id == "doc-1"

    def test_get_by_hash_nonexistent(self, db: SqliteMetadataDB) -> None:
        assert db.get_document_by_hash("nonexistent") is None

    def test_null_key_topics(self, db: SqliteMetadataDB) -> None:
        db.upsert_document(self._make_document(key_topics=None))
        result = db.get_document("doc-1")
        assert result is not None
        assert result.key_topics is None

    def test_document_count(self, db: SqliteMetadataDB) -> None:
        assert db.get_document_count() == 0
        db.upsert_document(self._make_document())
        assert db.get_document_count() == 1

    def test_index_fingerprint_changes_with_index_state(self, db: SqliteMetadataDB) -> None:
        before = db.get_index_fingerprint()

        db.upsert_document(self._make_document(summary_content_hash="summary-v1"))
        db.insert_chunks(
            [
                ChunkRow(
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    chunk_order=0,
                    chunk_text="Chunk text",
                    chunk_text_normalized="chunk text",
                )
            ]
        )
        db.log_processing(
            ProcessingLogEntry(
                doc_id="doc-1",
                file_path="/docs/test.pdf",
                stage="index",
                status="success",
            )
        )

        after = db.get_index_fingerprint()

        assert before != after
        assert "docs=1" in after
        assert "chunks=1" in after
        assert "log=1" in after
        assert "modified=2026-01-01T00:00:00" in after
        assert "raw=abc123" in after
        assert "normalized=def456" in after
        assert "summary=summary-v1" in after

    def test_recent_documents(self, db: SqliteMetadataDB) -> None:
        db.upsert_document(self._make_document(doc_id="d1", file_path="/a.pdf"))
        db.upsert_document(self._make_document(doc_id="d2", file_path="/b.pdf"))
        recent = db.get_recent_documents(limit=10)
        assert len(recent) == 2

    def test_recent_documents_folder_filter(self, db: SqliteMetadataDB) -> None:
        db.upsert_document(
            self._make_document(doc_id="d1", file_path="/a.pdf", folder_path="/docs")
        )
        db.upsert_document(
            self._make_document(doc_id="d2", file_path="/b.pdf", folder_path="/other")
        )
        recent = db.get_recent_documents(limit=10, folder_filter="/docs")
        assert len(recent) == 1
        assert recent[0].doc_id == "d1"


class TestSections:
    def _setup_doc(
        self, db: SqliteMetadataDB, doc_id: str = "doc-1", file_path: str = "/docs/test.pdf"
    ) -> None:
        doc = DocumentRow(
            doc_id=doc_id,
            file_path=file_path,
            folder_path="/docs",
            folder_ancestors=["/docs"],
            file_type="pdf",
            modified_at="2026-01-01T00:00:00",
            raw_content_hash="abc",
        )
        db.upsert_document(doc)

    def test_insert_and_get(self, db: SqliteMetadataDB) -> None:
        self._setup_doc(db)
        sections = [
            SectionRow(
                section_id="sec-1",
                doc_id="doc-1",
                section_heading="Introduction",
                section_order=0,
            ),
            SectionRow(
                section_id="sec-2",
                doc_id="doc-1",
                section_heading="Methods",
                section_order=1,
            ),
        ]
        db.insert_sections(sections)
        result = db.get_sections("doc-1")
        assert len(result) == 2
        assert result[0].section_heading == "Introduction"
        assert result[1].section_heading == "Methods"

    def test_get_empty(self, db: SqliteMetadataDB) -> None:
        assert db.get_sections("nonexistent") == []


class TestChunks:
    def _setup_doc(
        self, db: SqliteMetadataDB, doc_id: str = "doc-1", file_path: str = "/docs/test.pdf"
    ) -> None:
        doc = DocumentRow(
            doc_id=doc_id,
            file_path=file_path,
            folder_path="/docs",
            folder_ancestors=["/docs"],
            file_type="pdf",
            modified_at="2026-01-01T00:00:00",
            raw_content_hash="abc",
        )
        db.upsert_document(doc)

    def _make_chunk(
        self, order: int, doc_id: str = "doc-1", section_id: str | None = None
    ) -> ChunkRow:
        chunk_id = f"chunk-{order}" if doc_id == "doc-1" else f"{doc_id}-chunk-{order}"
        return ChunkRow(
            chunk_id=chunk_id,
            doc_id=doc_id,
            section_id=section_id,
            chunk_order=order,
            chunk_text=f"Text for chunk {order}",
            chunk_text_normalized=f"text for chunk {order}",
            token_count=10,
        )

    def test_insert_and_get(self, db: SqliteMetadataDB) -> None:
        self._setup_doc(db)
        chunks = [self._make_chunk(0), self._make_chunk(1)]
        db.insert_chunks(chunks)
        result = db.get_chunks("doc-1")
        assert len(result) == 2
        assert result[0].chunk_order == 0

    def test_get_chunks_by_documents_batch(self, db: SqliteMetadataDB) -> None:
        self._setup_doc(db)
        self._setup_doc(db, doc_id="doc-2", file_path="/docs/other.pdf")
        db.insert_chunks(
            [
                self._make_chunk(2),
                self._make_chunk(0),
                self._make_chunk(1),
                self._make_chunk(1, doc_id="doc-2"),
                self._make_chunk(0, doc_id="doc-2"),
            ]
        )

        result = db.get_chunks_by_documents(["doc-2", "missing", "doc-1"])

        assert [chunk.chunk_order for chunk in result["doc-1"]] == [0, 1, 2]
        assert [chunk.chunk_order for chunk in result["doc-2"]] == [0, 1]
        assert result["missing"] == []

    def test_get_chunks_by_documents_batch_with_limit(self, db: SqliteMetadataDB) -> None:
        self._setup_doc(db)
        self._setup_doc(db, doc_id="doc-2", file_path="/docs/other.pdf")
        db.insert_chunks(
            [
                self._make_chunk(0),
                self._make_chunk(1),
                self._make_chunk(2),
                self._make_chunk(0, doc_id="doc-2"),
                self._make_chunk(1, doc_id="doc-2"),
                self._make_chunk(2, doc_id="doc-2"),
            ]
        )

        result = db.get_chunks_by_documents(["doc-1", "doc-2"], limit_per_doc=2)

        assert [chunk.chunk_order for chunk in result["doc-1"]] == [0, 1]
        assert [chunk.chunk_order for chunk in result["doc-2"]] == [0, 1]

    def test_get_chunk(self, db: SqliteMetadataDB) -> None:
        self._setup_doc(db)
        db.insert_chunks([self._make_chunk(0)])
        result = db.get_chunk("chunk-0")
        assert result is not None
        assert result.chunk_text == "Text for chunk 0"

    def test_get_chunk_nonexistent(self, db: SqliteMetadataDB) -> None:
        assert db.get_chunk("nonexistent") is None

    def test_chunk_count(self, db: SqliteMetadataDB) -> None:
        self._setup_doc(db)
        assert db.get_chunk_count() == 0
        db.insert_chunks([self._make_chunk(0), self._make_chunk(1)])
        assert db.get_chunk_count() == 2

    def test_get_adjacent_chunks(self, db: SqliteMetadataDB) -> None:
        self._setup_doc(db)
        chunks = [self._make_chunk(i) for i in range(5)]
        db.insert_chunks(chunks)
        adjacent = db.get_adjacent_chunks("doc-1", chunk_order=2, window=1)
        assert len(adjacent) == 3
        orders = [c.chunk_order for c in adjacent]
        assert orders == [1, 2, 3]

    def test_get_adjacent_chunks_at_boundary(self, db: SqliteMetadataDB) -> None:
        self._setup_doc(db)
        chunks = [self._make_chunk(i) for i in range(3)]
        db.insert_chunks(chunks)
        adjacent = db.get_adjacent_chunks("doc-1", chunk_order=0, window=1)
        assert len(adjacent) == 2
        orders = [c.chunk_order for c in adjacent]
        assert orders == [0, 1]

    def test_get_chunks_by_sections_batch(self, db: SqliteMetadataDB) -> None:
        self._setup_doc(db)
        db.insert_chunks(
            [
                self._make_chunk(1, section_id="sec-1"),
                self._make_chunk(0, section_id="sec-1"),
                self._make_chunk(2, section_id="sec-2"),
            ]
        )

        result = db.get_chunks_by_sections(["sec-2", "missing", "sec-1"])

        assert [chunk.chunk_order for chunk in result["sec-1"]] == [0, 1]
        assert [chunk.chunk_order for chunk in result["sec-2"]] == [2]
        assert result["missing"] == []

    def test_get_adjacent_chunks_batch(self, db: SqliteMetadataDB) -> None:
        self._setup_doc(db)
        self._setup_doc(db, doc_id="doc-2", file_path="/docs/other.pdf")
        db.insert_chunks(
            [self._make_chunk(i) for i in range(5)]
            + [self._make_chunk(i, doc_id="doc-2") for i in range(3)]
        )

        result = db.get_adjacent_chunks_batch(
            [("doc-1", 2), ("doc-2", 0), ("missing", 0)], window=1
        )

        assert [chunk.chunk_order for chunk in result[("doc-1", 2)]] == [1, 2, 3]
        assert [chunk.chunk_order for chunk in result[("doc-2", 0)]] == [0, 1]
        assert result[("missing", 0)] == []


class TestProcessingLog:
    def test_log_processing(self, db: SqliteMetadataDB) -> None:
        entry = ProcessingLogEntry(
            doc_id="doc-1",
            file_path="/docs/test.pdf",
            stage="parse",
            status="success",
            duration_ms=150,
            details="Parsed OK",
        )
        db.log_processing(entry)
        row = db._conn.execute(
            "SELECT * FROM processing_log WHERE doc_id = ?", ("doc-1",)
        ).fetchone()
        assert row is not None
        assert row["stage"] == "parse"
        assert row["duration_ms"] == 150


class TestErrorCount:
    def test_error_count(self, db: SqliteMetadataDB) -> None:
        assert db.get_error_count() == 0
        state = SyncStateRow(
            id="s1",
            file_path="/a.pdf",
            file_name="a.pdf",
            folder_path="/",
            folder_ancestors=["/"],
            file_type="pdf",
            modified_at="2026-01-01T00:00:00",
            content_hash="abc",
            process_status="error",
            error_message="fail",
        )
        db.upsert_sync_state(state)
        assert db.get_error_count() == 1
