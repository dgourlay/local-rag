from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import mcp.types as types

if TYPE_CHECKING:
    from mcp.server import Server

    from rag.config import AppConfig
    from rag.db.models import SqliteMetadataDB
    from rag.retrieval.engine import RetrievalEngine
    from rag.types import CitedEvidence, DetailLevel, DocumentRow

logger = logging.getLogger(__name__)

_DETAIL_SCHEMA: dict[str, object] = {
    "type": "string",
    "enum": ["8w", "16w", "32w", "64w", "128w"],
    "description": (
        "Summary detail level: '8w' (short phrase), '16w' (one sentence), "
        "'32w' (1-2 sentences), '64w' (short paragraph), '128w' (detailed paragraph)"
    ),
}

_QUERY_SCHEMA: dict[str, object] = {
    "type": "string",
    "description": (
        "Natural-language search query. Use specific multi-word phrases "
        'rather than single keywords — "employee onboarding process '
        'timeline" works better than "onboarding". Include domain-specific '
        "terms that would appear in the target documents. The query is used "
        "for both semantic (meaning-based) and keyword matching, so exact "
        "terminology helps."
    ),
}

_FOLDER_FILTER_SCHEMA: dict[str, object] = {
    "type": "string",
    "description": (
        "Restrict results to documents within this folder path. Use an "
        'absolute path prefix — e.g., "/Users/you/Documents/Work". '
        "Documents in subfolders are included. Omit to search all indexed "
        "folders."
    ),
}


def _get_doc_summary(doc: DocumentRow, detail: DetailLevel) -> str | None:
    """Look up the summary field on DocumentRow for the given detail level."""
    return getattr(doc, f"summary_{detail}", None)


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
            retrieval_config=self._config.retrieval,
            summarization_config=self._config.summarization,
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
            "Search indexed documents using hybrid dense+keyword retrieval with "
            "cross-encoder reranking. Returns cited evidence passages grouped by "
            "source document, with section locations and page numbers. Use "
            "natural-language queries with specific terms from the domain — e.g., "
            '"quarterly revenue growth methodology" rather than single words like '
            '"revenue". Use folder_filter to narrow results when you know which '
            "folder contains the relevant documents. For a broad overview of which "
            "documents match a topic, use quick_search first, then search_documents "
            "to extract specific evidence."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": _QUERY_SCHEMA,
                "folder_filter": _FOLDER_FILTER_SCHEMA,
                "date_filter": {
                    "type": "string",
                    "description": (
                        'ISO 8601 date string (e.g., "2025-01-01"). Only return documents '
                        "modified on or after this date. Useful for scoping to recent "
                        'content when the user asks about "recent" or "latest" information.'
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 10)",
                    "default": 10,
                },
                "debug": {
                    "type": "boolean",
                    "description": (
                        "Include retrieval debug info: query classification "
                        "(broad/specific/navigational), per-lane hit counts, fusion "
                        "weights, timing breakdown, and reranker scores. Useful for "
                        "understanding why results are ranked the way they are. Does not "
                        "change the results themselves."
                    ),
                    "default": False,
                },
                "format": {
                    "type": "string",
                    "enum": ["text", "json"],
                    "description": (
                        "Output format: 'text' (default) returns results grouped by "
                        "document with summaries, topics, and ranked passages — best for "
                        "answering questions. 'json' returns raw structured data with "
                        "scores, doc_ids, and chunk_ids — use this when you need IDs for "
                        "follow-up calls to get_document_context."
                    ),
                    "default": "text",
                },
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="get_document_context",
        description=(
            "Retrieve detailed context for a specific document or chunk. Provide "
            "doc_id to get a document overview with its summary and all section "
            "summaries — useful after quick_search identifies a relevant document. "
            "Provide chunk_id to get a specific passage with surrounding chunks for "
            "context — useful after search_documents returns a passage you want to "
            "read more around. The window parameter controls how many adjacent chunks "
            "to include (default 1, meaning 1 before + 1 after). Always requires "
            "either doc_id or chunk_id — do not call without one."
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
                "detail": {**_DETAIL_SCHEMA, "default": "128w"},
            },
        },
    ),
    types.Tool(
        name="list_recent_documents",
        description=(
            "List recently indexed documents sorted by modification date, with "
            "titles and summaries. Use this to answer \"what's new?\" or \"what "
            "changed recently?\" questions, or to browse the document collection "
            "without a specific search query. Use folder_filter to scope to a "
            'specific folder. The detail parameter controls summary length — use '
            '"8w" for a quick list, "32w" or "64w" when the user wants to '
            "understand what each document covers. This tool does not perform any "
            "search — it simply lists documents by recency."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "folder_filter": _FOLDER_FILTER_SCHEMA,
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of documents to return (default 20)",
                    "default": 20,
                },
                "detail": {**_DETAIL_SCHEMA, "default": "8w"},
            },
        },
    ),
    types.Tool(
        name="get_sync_status",
        description=(
            "Get the current indexing status: total files tracked, how many are "
            "indexed, pending, or errored, with a per-folder breakdown. Use this to "
            "check whether the index is up to date before searching, or to diagnose "
            "why expected documents are not appearing in search results. Returns "
            "counts only, not document content — use list_recent_documents or "
            "search_documents to see actual documents."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    types.Tool(
        name="quick_search",
        description=(
            "Quick document-level scan — returns document titles, summaries, and "
            "topics matching a query, without individual passage extraction. Use "
            "this as a first step to discover which documents are relevant before "
            "drilling into specific ones with search_documents. Runs the same "
            "retrieval pipeline but returns document-level results instead of "
            'chunk-level evidence. Good for questions like "what documents do we '
            'have about X?" or "which reports cover Y?". Not suitable when you need '
            "specific quotes, data points, or cited passages — use search_documents "
            "for that."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": _QUERY_SCHEMA,
                "folder_filter": _FOLDER_FILTER_SCHEMA,
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 5)",
                    "default": 5,
                },
                "detail": {**_DETAIL_SCHEMA, "default": "32w"},
            },
            "required": ["query"],
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
            if name == "quick_search":
                return await _handle_quick_search(components, args)
            return _error_content(f"Unknown tool: {name}")
        except Exception as exc:
            logger.exception("Tool %s failed", name)
            return _error_content(f"Error executing {name}: {type(exc).__name__}: {exc}")


def _append_debug_info(lines: list[str], debug_info: dict[str, Any] | None) -> None:
    """Append debug info block to output lines if present."""
    if debug_info:
        lines.append("")
        lines.append("Debug info:")
        for key, value in debug_info.items():
            lines.append(f"  {key}: {value}")


def _format_results_as_text(
    hits: list[CitedEvidence],
    doc_lookup: dict[str, DocumentRow | None],
    query_classification: str | None,
    debug_info: dict[str, Any] | None,
) -> str:
    """Format search results as LLM-friendly grouped text."""
    if not hits:
        lines: list[str] = ["No results found."]
        if query_classification:
            lines.append(f"\n---\nQuery classified as: {query_classification}")
        _append_debug_info(lines, debug_info)
        return "\n".join(lines)

    # Group hits by doc_id, preserving rank order
    groups: dict[str, list[tuple[int, CitedEvidence]]] = defaultdict(list)
    for rank, hit in enumerate(hits, 1):
        groups[hit.doc_id].append((rank, hit))

    unique_docs = len(groups)
    total_results = len(hits)
    lines = [f"Found {total_results} results across {unique_docs} documents.\n"]

    for _doc_path, ranked_hits in groups.items():
        # Use the first hit's citation for the title
        first_hit = ranked_hits[0][1]
        title = first_hit.citation.title

        lines.append(f"## {title}")

        # Look up doc for summary and topics
        doc = doc_lookup.get(first_hit.doc_id)
        if doc is not None:
            if doc.summary_32w:
                lines.append(f"Summary: {doc.summary_32w}")
            if doc.key_topics:
                lines.append(f"Topics: {', '.join(doc.key_topics)}")

        lines.append("")

        for rank, hit in ranked_hits:
            # Build location string
            loc_parts: list[str] = []
            if hit.citation.section:
                loc_parts.append(f"§ {hit.citation.section}")
            if hit.citation.pages:
                loc_parts.append(hit.citation.pages)
            loc_str = ", ".join(loc_parts)

            lines.append(f"[{rank}] (score: {hit.score:.3f}) {loc_str}")

            # Truncate long text to ~800 chars
            text = hit.text
            if len(text) > 800:
                text = text[:797] + "..."
            lines.append(text)
            lines.append("")

    # Footer
    footer_parts: list[str] = []
    if query_classification:
        footer_parts.append(f"Query classified as: {query_classification}")
    footer_parts.append(f"{total_results} results from {unique_docs} documents")
    lines.append("---")
    lines.append(" | ".join(footer_parts))

    _append_debug_info(lines, debug_info)

    return "\n".join(lines)


async def _handle_search(components: _Components, args: dict[str, Any]) -> list[types.TextContent]:
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

    if inp.format == "json":
        output = SearchDocumentsOutput(
            results=result.hits,
            query_classification=result.query_classification,
            debug_info=result.debug_info,
        )
        return [types.TextContent(type="text", text=output.model_dump_json())]

    # Text format: fetch document info for summaries/topics
    doc_ids = {hit.doc_id for hit in result.hits}
    doc_lookup: dict[str, DocumentRow | None] = {}
    for doc_id in doc_ids:
        doc_lookup[doc_id] = await asyncio.to_thread(components.db.get_document, doc_id)

    text = _format_results_as_text(
        hits=result.hits,
        doc_lookup=doc_lookup,
        query_classification=result.query_classification,
        debug_info=result.debug_info,
    )
    return [types.TextContent(type="text", text=text)]


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
            summary=_get_doc_summary(doc, inp.detail),
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
        summary=_get_doc_summary(doc, inp.detail) if doc else None,
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

    docs = await asyncio.to_thread(db.get_recent_documents, inp.limit, inp.folder_filter)

    entries = [
        RecentDocumentEntry(
            doc_id=d.doc_id,
            title=d.title,
            summary=_get_doc_summary(d, inp.detail),
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
        lambda: conn.execute("SELECT COUNT(*) FROM sync_state WHERE NOT is_deleted").fetchone()[0]
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

    # Question generation stats
    chunks_with_questions = await asyncio.to_thread(
        lambda: conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE generated_questions IS NOT NULL"
        ).fetchone()[0]
    )

    output = SyncStatusOutput(
        total_files=total_files,
        indexed_count=indexed_count,
        pending_count=pending_count,
        error_count=error_count,
        last_sync_time=last_sync_time,
        chunking_strategy=components._config.chunking.strategy,
        questions_enabled=components._config.questions.enabled,
        chunks_with_questions=chunks_with_questions,
        folders=folders,
    )
    return [types.TextContent(type="text", text=output.model_dump_json())]


async def _handle_quick_search(
    components: _Components, args: dict[str, Any]
) -> list[types.TextContent]:
    from rag.types import QuickSearchInput, SearchFilters

    inp = QuickSearchInput.model_validate(args)
    filters = SearchFilters(folder_filter=inp.folder_filter)

    result = await components.engine.async_search(
        query=inp.query,
        filters=filters,
        top_k=inp.top_k,
        debug=False,
    )

    # Collect unique doc_ids from results, preserving order
    seen_doc_ids = list(dict.fromkeys(hit.doc_id for hit in result.hits))

    # Fetch document details from SQLite
    lines: list[str] = []
    doc_count = 0
    for doc_id in seen_doc_ids:
        doc = await asyncio.to_thread(components.db.get_document, doc_id)
        if doc is None:
            continue
        doc_count += 1
        title = doc.title or "Untitled"
        lines.append(f"## {title}")
        summary = _get_doc_summary(doc, inp.detail)
        if summary:
            lines.append(f"Summary: {summary}")
        if doc.key_topics:
            lines.append(f"Topics: {', '.join(doc.key_topics)}")
        lines.append(f"Path: {doc.file_path}")
        lines.append(f"Modified: {doc.modified_at}")
        lines.append(f"Doc ID: {doc.doc_id}")
        lines.append("")

    if not lines:
        return [types.TextContent(type="text", text="No matching documents found.")]

    header = f"Found {doc_count} matching documents.\n\n"
    return [types.TextContent(type="text", text=header + "\n".join(lines))]
