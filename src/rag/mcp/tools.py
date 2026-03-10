from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

import mcp.types as types

if TYPE_CHECKING:
    from mcp.server import Server

    from rag.config import AppConfig
    from rag.db.models import SqliteMetadataDB
    from rag.retrieval.engine import RetrievalEngine

logger = logging.getLogger(__name__)


class _Components:
    """Lazy-initialized backend components for MCP tool handlers."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._db: SqliteMetadataDB | None = None
        self._engine: RetrievalEngine | None = None

    def _init(self) -> None:
        from rag.db.connection import get_connection
        from rag.db.migrations import run_migrations
        from rag.db.models import SqliteMetadataDB
        from rag.db.qdrant import QdrantVectorStore
        from rag.pipeline.embedder import SentenceTransformerEmbedder
        from rag.retrieval.citations import CitationAssembler
        from rag.retrieval.engine import RetrievalEngine
        from rag.retrieval.reranker import OnnxReranker

        conn = get_connection(self._config.database.path)
        run_migrations(conn)
        db = SqliteMetadataDB(conn)

        vector_store = QdrantVectorStore(self._config.qdrant)
        vector_store.ensure_collection()

        embedder = SentenceTransformerEmbedder(self._config.embedding)
        reranker = OnnxReranker(self._config.reranker)
        citations = CitationAssembler(db)

        engine = RetrievalEngine(
            vector_store=vector_store,
            embedder=embedder,
            reranker=reranker,
            citation_assembler=citations,
            top_k_candidates=self._config.reranker.top_k_candidates,
            top_k_final=self._config.reranker.top_k_final,
        )

        self._db = db
        self._engine = engine

    @property
    def db(self) -> SqliteMetadataDB:
        if self._db is None:
            self._init()
        assert self._db is not None
        return self._db

    @property
    def engine(self) -> RetrievalEngine:
        if self._engine is None:
            self._init()
        assert self._engine is not None
        return self._engine


_TOOLS: list[types.Tool] = [
    types.Tool(
        name="search_documents",
        description=(
            "Search indexed documents using hybrid dense+keyword retrieval "
            "with cross-encoder reranking. Returns cited evidence passages."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text"},
                "folder_filter": {
                    "type": "string",
                    "description": "Restrict to a specific folder path",
                },
                "date_filter": {
                    "type": "string",
                    "description": (
                        "ISO 8601 date; only return docs modified on or after"
                        " this date"
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 10)",
                    "default": 10,
                },
                "debug": {
                    "type": "boolean",
                    "description": "Include timing and debug info in response",
                    "default": False,
                },
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="get_document_context",
        description=(
            "Get context for a document or chunk. "
            "Provide doc_id for document overview (summary + sections), "
            "or chunk_id for a chunk with surrounding context window."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "Document ID to retrieve overview for",
                },
                "chunk_id": {
                    "type": "string",
                    "description": "Chunk ID to retrieve with context window",
                },
                "window": {
                    "type": "integer",
                    "description": "Number of adjacent chunks to include (default 1)",
                    "default": 1,
                },
            },
        },
    ),
    types.Tool(
        name="list_recent_documents",
        description="List recently indexed documents, optionally filtered by folder.",
        inputSchema={
            "type": "object",
            "properties": {
                "folder_filter": {
                    "type": "string",
                    "description": "Restrict to a specific folder path",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of documents to return (default 20)",
                    "default": 20,
                },
            },
        },
    ),
    types.Tool(
        name="get_sync_status",
        description=(
            "Get the current indexing sync status: total files, indexed, "
            "pending, errors, per-folder breakdown."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
]


def _error_content(message: str) -> list[types.TextContent]:
    return [types.TextContent(type="text", text=json.dumps({"error": message}))]


def register_tools(server: Server, config: AppConfig) -> None:
    """Register all MCP tools with the server."""
    components = _Components(config)

    @server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
    async def handle_list_tools() -> list[types.Tool]:
        return _TOOLS

    @server.call_tool()  # type: ignore[untyped-decorator]
    async def handle_call_tool(
        name: str, arguments: dict[str, Any] | None
    ) -> list[types.TextContent]:
        args = arguments or {}
        try:
            if name == "search_documents":
                return await _handle_search(components, args)
            if name == "get_document_context":
                return await _handle_get_context(components, args)
            if name == "list_recent_documents":
                return await _handle_list_recent(components, args)
            if name == "get_sync_status":
                return await _handle_sync_status(components)
            return _error_content(f"Unknown tool: {name}")
        except Exception:
            logger.exception("Tool %s failed", name)
            return _error_content(f"Internal error executing {name}")


async def _handle_search(
    components: _Components, args: dict[str, Any]
) -> list[types.TextContent]:
    from rag.types import SearchDocumentsInput, SearchDocumentsOutput, SearchFilters

    inp = SearchDocumentsInput.model_validate(args)
    filters = SearchFilters(
        folder_filter=inp.folder_filter,
        date_filter=inp.date_filter,
    )

    result = await components.engine.async_search(
        query=inp.query,
        filters=filters,
        top_k=inp.top_k,
        debug=inp.debug,
    )

    output = SearchDocumentsOutput(
        results=result.hits,
        query_classification=result.query_classification,
        debug_info=result.debug_info,
    )
    return [types.TextContent(type="text", text=output.model_dump_json())]


async def _handle_get_context(
    components: _Components, args: dict[str, Any]
) -> list[types.TextContent]:
    from rag.types import DocumentContextOutput, GetDocumentContextInput

    inp = GetDocumentContextInput.model_validate(args)

    if inp.doc_id is None and inp.chunk_id is None:
        return _error_content("Either doc_id or chunk_id must be provided")

    db = components.db

    if inp.doc_id is not None:
        doc = await asyncio.to_thread(db.get_document, inp.doc_id)
        if doc is None:
            return _error_content(f"Document not found: {inp.doc_id}")

        sections = await asyncio.to_thread(db.get_sections, inp.doc_id)

        output = DocumentContextOutput(
            doc_id=doc.doc_id,
            title=doc.title,
            summary=doc.summary_l1,
            sections=sections if sections else None,
        )
        return [types.TextContent(type="text", text=output.model_dump_json())]

    # chunk_id path
    assert inp.chunk_id is not None
    chunk = await asyncio.to_thread(db.get_chunk, inp.chunk_id)
    if chunk is None:
        return _error_content(f"Chunk not found: {inp.chunk_id}")

    adjacent = await asyncio.to_thread(
        db.get_adjacent_chunks, chunk.doc_id, chunk.chunk_order, inp.window
    )

    doc = await asyncio.to_thread(db.get_document, chunk.doc_id)
    output = DocumentContextOutput(
        doc_id=chunk.doc_id,
        title=doc.title if doc else None,
        summary=doc.summary_l1 if doc else None,
        chunks=adjacent if adjacent else None,
    )
    return [types.TextContent(type="text", text=output.model_dump_json())]


async def _handle_list_recent(
    components: _Components, args: dict[str, Any]
) -> list[types.TextContent]:
    from rag.types import (
        ListRecentDocumentsInput,
        ListRecentDocumentsOutput,
        RecentDocumentEntry,
    )

    inp = ListRecentDocumentsInput.model_validate(args)
    db = components.db

    docs = await asyncio.to_thread(
        db.get_recent_documents, inp.limit, inp.folder_filter
    )

    entries = [
        RecentDocumentEntry(
            doc_id=d.doc_id,
            title=d.title,
            file_path=d.file_path,
            file_type=d.file_type,
            modified_at=d.modified_at,
            indexed_at=d.indexed_at,
            folder_path=d.folder_path,
        )
        for d in docs
    ]

    output = ListRecentDocumentsOutput(documents=entries)
    return [types.TextContent(type="text", text=output.model_dump_json())]


async def _handle_sync_status(
    components: _Components,
) -> list[types.TextContent]:
    from rag.types import FolderStatusEntry, SyncStatusOutput

    db = components.db
    conn = db._conn

    total_files = await asyncio.to_thread(
        lambda: conn.execute(
            "SELECT COUNT(*) FROM sync_state WHERE NOT is_deleted"
        ).fetchone()[0]
    )
    indexed_count = await asyncio.to_thread(db.get_document_count)
    error_count = await asyncio.to_thread(db.get_error_count)
    pending_count = await asyncio.to_thread(
        lambda: conn.execute(
            "SELECT COUNT(*) FROM sync_state WHERE process_status = 'pending' AND NOT is_deleted"
        ).fetchone()[0]
    )
    last_sync_row = await asyncio.to_thread(
        lambda: conn.execute(
            "SELECT MAX(synced_at) FROM sync_state WHERE synced_at IS NOT NULL"
        ).fetchone()
    )
    last_sync_time: str | None = last_sync_row[0] if last_sync_row else None

    folder_rows = await asyncio.to_thread(
        lambda: conn.execute(
            """SELECT
                folder_path,
                COUNT(*) AS file_count,
                SUM(CASE WHEN process_status = 'done' THEN 1 ELSE 0 END) AS indexed_count,
                SUM(CASE WHEN process_status = 'error' THEN 1 ELSE 0 END) AS error_count
            FROM sync_state
            WHERE NOT is_deleted
            GROUP BY folder_path"""
        ).fetchall()
    )

    folders = [
        FolderStatusEntry(
            folder_path=row[0],
            file_count=row[1],
            indexed_count=row[2],
            error_count=row[3],
        )
        for row in folder_rows
    ]

    output = SyncStatusOutput(
        total_files=total_files,
        indexed_count=indexed_count,
        pending_count=pending_count,
        error_count=error_count,
        last_sync_time=last_sync_time,
        folders=folders,
    )
    return [types.TextContent(type="text", text=output.model_dump_json())]
