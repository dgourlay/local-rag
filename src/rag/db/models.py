from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

    from rag.types import (
        ChunkRow,
        DocumentRow,
        ProcessingLogEntry,
        SectionRow,
        SyncStateRow,
    )


class SqliteMetadataDB:
    """SQLite implementation of the MetadataDB protocol."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert_sync_state(self, state: SyncStateRow) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO sync_state
            (id, file_path, file_name, folder_path, folder_ancestors,
             file_type, size_bytes, modified_at, content_hash,
             synced_at, process_status, error_message, retry_count,
             is_deleted, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                state.id,
                state.file_path,
                state.file_name,
                state.folder_path,
                json.dumps(state.folder_ancestors),
                state.file_type,
                state.size_bytes,
                state.modified_at,
                state.content_hash,
                state.synced_at,
                state.process_status,
                state.error_message,
                state.retry_count,
                state.is_deleted,
                state.created_at,
            ),
        )
        self._conn.commit()

    def get_sync_state(self, file_path: str) -> SyncStateRow | None:
        row = self._conn.execute(
            "SELECT * FROM sync_state WHERE file_path = ?", (file_path,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_sync_state(row)

    def get_pending_files(self, limit: int) -> list[SyncStateRow]:
        rows = self._conn.execute(
            """SELECT * FROM sync_state
            WHERE process_status = 'pending' AND NOT is_deleted
            ORDER BY created_at ASC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [_row_to_sync_state(r) for r in rows]

    def upsert_document(self, doc: DocumentRow) -> None:
        self._conn.execute(
            """INSERT INTO documents
            (doc_id, file_path, folder_path, folder_ancestors, title,
             file_type, modified_at, indexed_at, parser_version,
             raw_content_hash, normalized_content_hash, duplicate_of_doc_id,
             ocr_required, ocr_confidence, doc_type_guess, key_topics,
             summary_l1, summary_l2, summary_l3, summary_content_hash,
             embedding_model_version, chunker_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                file_path = excluded.file_path,
                folder_path = excluded.folder_path,
                folder_ancestors = excluded.folder_ancestors,
                title = excluded.title,
                file_type = excluded.file_type,
                modified_at = excluded.modified_at,
                indexed_at = excluded.indexed_at,
                parser_version = excluded.parser_version,
                raw_content_hash = excluded.raw_content_hash,
                normalized_content_hash = excluded.normalized_content_hash,
                duplicate_of_doc_id = excluded.duplicate_of_doc_id,
                ocr_required = excluded.ocr_required,
                ocr_confidence = excluded.ocr_confidence,
                doc_type_guess = excluded.doc_type_guess,
                key_topics = excluded.key_topics,
                summary_l1 = excluded.summary_l1,
                summary_l2 = excluded.summary_l2,
                summary_l3 = excluded.summary_l3,
                summary_content_hash = excluded.summary_content_hash,
                embedding_model_version = excluded.embedding_model_version,
                chunker_version = excluded.chunker_version""",
            (
                doc.doc_id,
                doc.file_path,
                doc.folder_path,
                json.dumps(doc.folder_ancestors),
                doc.title,
                doc.file_type,
                doc.modified_at,
                doc.indexed_at,
                doc.parser_version,
                doc.raw_content_hash,
                doc.normalized_content_hash,
                doc.duplicate_of_doc_id,
                doc.ocr_required,
                doc.ocr_confidence,
                doc.doc_type_guess,
                json.dumps(doc.key_topics) if doc.key_topics is not None else None,
                doc.summary_l1,
                doc.summary_l2,
                doc.summary_l3,
                doc.summary_content_hash,
                doc.embedding_model_version,
                doc.chunker_version,
            ),
        )
        self._conn.commit()

    def get_document(self, doc_id: str) -> DocumentRow | None:
        row = self._conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        if row is None:
            return None
        return _row_to_document(row)

    def get_document_by_hash(self, content_hash: str) -> DocumentRow | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE normalized_content_hash = ?",
            (content_hash,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_document(row)

    def insert_sections(self, sections: list[SectionRow]) -> None:
        self._conn.executemany(
            """INSERT OR REPLACE INTO sections
            (section_id, doc_id, section_heading, section_order,
             page_start, page_end, section_summary, section_summary_l2,
             embedding_model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    s.section_id,
                    s.doc_id,
                    s.section_heading,
                    s.section_order,
                    s.page_start,
                    s.page_end,
                    s.section_summary,
                    s.section_summary_l2,
                    s.embedding_model_version,
                )
                for s in sections
            ],
        )
        self._conn.commit()

    def get_sections(self, doc_id: str) -> list[SectionRow]:
        rows = self._conn.execute(
            "SELECT * FROM sections WHERE doc_id = ? ORDER BY section_order",
            (doc_id,),
        ).fetchall()
        return [_row_to_section(r) for r in rows]

    def insert_chunks(self, chunks: list[ChunkRow]) -> None:
        self._conn.executemany(
            """INSERT OR REPLACE INTO chunks
            (chunk_id, doc_id, section_id, chunk_order, chunk_text,
             chunk_text_normalized, page_start, page_end,
             section_heading, citation_label, token_count,
             embedding_model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    c.chunk_id,
                    c.doc_id,
                    c.section_id,
                    c.chunk_order,
                    c.chunk_text,
                    c.chunk_text_normalized,
                    c.page_start,
                    c.page_end,
                    c.section_heading,
                    c.citation_label,
                    c.token_count,
                    c.embedding_model_version,
                )
                for c in chunks
            ],
        )
        self._conn.commit()

    def get_chunks(self, doc_id: str) -> list[ChunkRow]:
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_order",
            (doc_id,),
        ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def get_chunk(self, chunk_id: str) -> ChunkRow | None:
        row = self._conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        if row is None:
            return None
        return _row_to_chunk(row)

    def get_adjacent_chunks(self, doc_id: str, chunk_order: int, window: int) -> list[ChunkRow]:
        rows = self._conn.execute(
            """SELECT * FROM chunks
            WHERE doc_id = ?
              AND chunk_order BETWEEN ? AND ?
            ORDER BY chunk_order""",
            (doc_id, chunk_order - window, chunk_order + window),
        ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def log_processing(self, entry: ProcessingLogEntry) -> None:
        self._conn.execute(
            """INSERT INTO processing_log
            (doc_id, file_path, stage, status, duration_ms, details)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                entry.doc_id,
                entry.file_path,
                entry.stage,
                entry.status,
                entry.duration_ms,
                entry.details,
            ),
        )
        self._conn.commit()

    def get_document_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return int(row[0])

    def get_chunk_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0])

    def get_error_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM sync_state WHERE process_status = 'error' AND NOT is_deleted"
        ).fetchone()
        return int(row[0])

    def get_recent_documents(
        self, limit: int, folder_filter: str | None = None
    ) -> list[DocumentRow]:
        if folder_filter is not None:
            rows = self._conn.execute(
                """SELECT * FROM documents
                WHERE folder_path = ?
                ORDER BY indexed_at DESC LIMIT ?""",
                (folder_filter, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM documents ORDER BY indexed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_document(r) for r in rows]


def _row_to_sync_state(row: sqlite3.Row) -> SyncStateRow:
    from rag.types import SyncStateRow as _SyncStateRow

    return _SyncStateRow(
        id=row["id"],
        file_path=row["file_path"],
        file_name=row["file_name"],
        folder_path=row["folder_path"],
        folder_ancestors=json.loads(row["folder_ancestors"]),
        file_type=row["file_type"],
        size_bytes=row["size_bytes"],
        modified_at=row["modified_at"],
        content_hash=row["content_hash"],
        synced_at=row["synced_at"],
        process_status=row["process_status"],
        error_message=row["error_message"],
        retry_count=row["retry_count"],
        is_deleted=row["is_deleted"],
        created_at=row["created_at"],
    )


def _row_to_document(row: sqlite3.Row) -> DocumentRow:
    from rag.types import DocumentRow as _DocumentRow

    key_topics_raw = row["key_topics"]
    key_topics: list[str] | None = (
        json.loads(key_topics_raw) if key_topics_raw is not None else None
    )

    return _DocumentRow(
        doc_id=row["doc_id"],
        file_path=row["file_path"],
        folder_path=row["folder_path"],
        folder_ancestors=json.loads(row["folder_ancestors"]),
        title=row["title"],
        file_type=row["file_type"],
        modified_at=row["modified_at"],
        indexed_at=row["indexed_at"],
        parser_version=row["parser_version"],
        raw_content_hash=row["raw_content_hash"],
        normalized_content_hash=row["normalized_content_hash"],
        duplicate_of_doc_id=row["duplicate_of_doc_id"],
        ocr_required=row["ocr_required"],
        ocr_confidence=row["ocr_confidence"],
        doc_type_guess=row["doc_type_guess"],
        key_topics=key_topics,
        summary_l1=row["summary_l1"],
        summary_l2=row["summary_l2"],
        summary_l3=row["summary_l3"],
        summary_content_hash=row["summary_content_hash"],
        embedding_model_version=row["embedding_model_version"],
        chunker_version=row["chunker_version"],
    )


def _row_to_section(row: sqlite3.Row) -> SectionRow:
    from rag.types import SectionRow as _SectionRow

    return _SectionRow(
        section_id=row["section_id"],
        doc_id=row["doc_id"],
        section_heading=row["section_heading"],
        section_order=row["section_order"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        section_summary=row["section_summary"],
        section_summary_l2=row["section_summary_l2"],
        embedding_model_version=row["embedding_model_version"],
    )


def _row_to_chunk(row: sqlite3.Row) -> ChunkRow:
    from rag.types import ChunkRow as _ChunkRow

    return _ChunkRow(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        section_id=row["section_id"],
        chunk_order=row["chunk_order"],
        chunk_text=row["chunk_text"],
        chunk_text_normalized=row["chunk_text_normalized"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        section_heading=row["section_heading"],
        citation_label=row["citation_label"],
        token_count=row["token_count"],
        embedding_model_version=row["embedding_model_version"],
    )
