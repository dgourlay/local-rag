from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from rag.mcp.server import create_server
from rag.mcp.tools import (
    _TOOLS,
    _Components,
    _error_content,
    _handle_get_context,
    _handle_list_recent,
    _handle_search,
    _handle_sync_status,
)
from rag.types import (
    ChunkRow,
    Citation,
    CitedEvidence,
    DocumentRow,
    RetrievalResult,
    SectionRow,
)

# --- Fixtures ---


def _make_config() -> MagicMock:
    config = MagicMock()
    config.database.path = ":memory:"
    config.qdrant.url = "http://localhost:6333"
    config.qdrant.collection = "documents"
    config.embedding.model = "BAAI/bge-m3"
    config.embedding.dimensions = 1024
    config.embedding.batch_size = 32
    config.reranker.top_k_candidates = 30
    config.reranker.top_k_final = 10
    config.mcp.transport = "stdio"
    config.mcp.host = "127.0.0.1"
    config.mcp.port = 8080
    return config


def _make_components() -> tuple[_Components, MagicMock, MagicMock]:
    """Create components with mocked db and engine."""
    config = _make_config()
    components = _Components(config)

    mock_db = MagicMock()
    mock_engine = MagicMock()
    mock_engine.async_search = AsyncMock()
    components._db = mock_db
    components._engine = mock_engine

    return components, mock_db, mock_engine


def _make_cited_evidence(text: str = "sample text", score: float = 0.95) -> CitedEvidence:
    return CitedEvidence(
        text=text,
        citation=Citation(
            title="Test Doc",
            path="/docs/test.pdf",
            section="Intro",
            pages="p. 1",
            modified="2025-01-01T00:00:00",
            label="test.pdf, \u00a7 Intro, p. 1",
        ),
        score=score,
        record_type="chunk",
    )


def _make_document_row(doc_id: str = "doc-1") -> DocumentRow:
    return DocumentRow(
        doc_id=doc_id,
        file_path="/docs/test.pdf",
        folder_path="/docs",
        folder_ancestors=["/docs"],
        title="Test Document",
        file_type="pdf",
        modified_at="2025-01-01T00:00:00",
        indexed_at="2025-01-02T00:00:00",
        raw_content_hash="abc123",
        summary_l1="A test document about testing.",
    )


def _make_section_row(doc_id: str = "doc-1", order: int = 0) -> SectionRow:
    return SectionRow(
        section_id=f"sec-{order}",
        doc_id=doc_id,
        section_heading=f"Section {order}",
        section_order=order,
    )


def _make_chunk_row(
    doc_id: str = "doc-1",
    chunk_id: str = "chunk-1",
    order: int = 0,
) -> ChunkRow:
    return ChunkRow(
        chunk_id=chunk_id,
        doc_id=doc_id,
        chunk_order=order,
        chunk_text="chunk text content",
        chunk_text_normalized="chunk text content",
        token_count=5,
    )


# --- Tool Registration Tests ---


class TestToolRegistration:
    def test_tools_defined(self) -> None:
        """All 4 tools are defined in _TOOLS list."""
        names = {t.name for t in _TOOLS}
        assert names == {
            "search_documents",
            "get_document_context",
            "list_recent_documents",
            "get_sync_status",
        }

    def test_tool_schemas_have_required_fields(self) -> None:
        """Each tool has name, description, and inputSchema."""
        for tool in _TOOLS:
            assert tool.name
            assert tool.description
            assert tool.inputSchema is not None
            assert tool.inputSchema["type"] == "object"

    def test_search_documents_requires_query(self) -> None:
        """search_documents tool requires 'query' parameter."""
        tool = next(t for t in _TOOLS if t.name == "search_documents")
        assert "query" in tool.inputSchema["required"]

    def test_server_creation(self) -> None:
        """create_server returns a configured Server."""
        config = _make_config()
        server = create_server(config)
        assert server.name == "dropbox-rag"


# --- search_documents Tests ---


class TestSearchDocuments:
    def test_returns_results(self) -> None:
        """search_documents returns cited evidence."""
        components, _db, mock_engine = _make_components()
        cited = [_make_cited_evidence()]
        mock_engine.async_search.return_value = RetrievalResult(
            hits=cited,
            query_classification="broad",
        )

        result = asyncio.run(_handle_search(components, {"query": "test query"}))

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert len(data["results"]) == 1
        assert data["results"][0]["text"] == "sample text"
        assert data["query_classification"] == "broad"

    def test_with_filters(self) -> None:
        """search_documents passes folder and date filters."""
        components, _db, mock_engine = _make_components()
        mock_engine.async_search.return_value = RetrievalResult(
            hits=[], query_classification="broad"
        )

        asyncio.run(
            _handle_search(
                components,
                {
                    "query": "test",
                    "folder_filter": "/docs",
                    "date_filter": "2025-01-01",
                    "top_k": 5,
                    "debug": True,
                },
            )
        )

        call_args = mock_engine.async_search.call_args
        assert call_args.kwargs["query"] == "test"
        assert call_args.kwargs["filters"].folder_filter == "/docs"
        assert call_args.kwargs["filters"].date_filter == "2025-01-01"
        assert call_args.kwargs["top_k"] == 5
        assert call_args.kwargs["debug"] is True

    def test_debug_info_included(self) -> None:
        """search_documents includes debug_info when debug=True."""
        components, _db, mock_engine = _make_components()
        mock_engine.async_search.return_value = RetrievalResult(
            hits=[],
            query_classification="broad",
            debug_info={"total_ms": 42},
        )

        result = asyncio.run(_handle_search(components, {"query": "test", "debug": True}))

        data = json.loads(result[0].text)
        assert data["debug_info"]["total_ms"] == 42


# --- get_document_context Tests ---


class TestGetDocumentContext:
    def test_with_doc_id(self) -> None:
        """get_document_context with doc_id returns document overview."""
        components, mock_db, _engine = _make_components()
        mock_db.get_document.return_value = _make_document_row()
        mock_db.get_sections.return_value = [
            _make_section_row(order=0),
            _make_section_row(order=1),
        ]

        result = asyncio.run(_handle_get_context(components, {"doc_id": "doc-1"}))

        data = json.loads(result[0].text)
        assert data["doc_id"] == "doc-1"
        assert data["title"] == "Test Document"
        assert data["summary"] == "A test document about testing."
        assert len(data["sections"]) == 2

    def test_with_chunk_id(self) -> None:
        """get_document_context with chunk_id returns chunk + adjacent."""
        components, mock_db, _engine = _make_components()
        center_chunk = _make_chunk_row(chunk_id="chunk-2", order=1)
        mock_db.get_chunk.return_value = center_chunk
        mock_db.get_adjacent_chunks.return_value = [
            _make_chunk_row(chunk_id="chunk-1", order=0),
            _make_chunk_row(chunk_id="chunk-2", order=1),
            _make_chunk_row(chunk_id="chunk-3", order=2),
        ]
        mock_db.get_document.return_value = _make_document_row()

        result = asyncio.run(_handle_get_context(components, {"chunk_id": "chunk-2", "window": 1}))

        data = json.loads(result[0].text)
        assert data["doc_id"] == "doc-1"
        assert len(data["chunks"]) == 3

    def test_missing_both_ids_returns_error(self) -> None:
        """get_document_context without doc_id or chunk_id returns error."""
        components, _db, _engine = _make_components()

        result = asyncio.run(_handle_get_context(components, {}))

        data = json.loads(result[0].text)
        assert "error" in data

    def test_doc_not_found_returns_error(self) -> None:
        """get_document_context with unknown doc_id returns error."""
        components, mock_db, _engine = _make_components()
        mock_db.get_document.return_value = None

        result = asyncio.run(_handle_get_context(components, {"doc_id": "nonexistent"}))

        data = json.loads(result[0].text)
        assert "error" in data

    def test_chunk_not_found_returns_error(self) -> None:
        """get_document_context with unknown chunk_id returns error."""
        components, mock_db, _engine = _make_components()
        mock_db.get_chunk.return_value = None

        result = asyncio.run(_handle_get_context(components, {"chunk_id": "nonexistent"}))

        data = json.loads(result[0].text)
        assert "error" in data


# --- list_recent_documents Tests ---


class TestListRecentDocuments:
    def test_returns_documents(self) -> None:
        """list_recent_documents returns formatted document entries."""
        components, mock_db, _engine = _make_components()
        mock_db.get_recent_documents.return_value = [
            _make_document_row("doc-1"),
            _make_document_row("doc-2"),
        ]

        result = asyncio.run(_handle_list_recent(components, {}))

        data = json.loads(result[0].text)
        assert len(data["documents"]) == 2
        assert data["documents"][0]["doc_id"] == "doc-1"

    def test_with_limit(self) -> None:
        """list_recent_documents respects limit parameter."""
        components, mock_db, _engine = _make_components()
        mock_db.get_recent_documents.return_value = [_make_document_row()]

        asyncio.run(_handle_list_recent(components, {"limit": 5}))

        mock_db.get_recent_documents.assert_called_once_with(5, None)

    def test_with_folder_filter(self) -> None:
        """list_recent_documents passes folder_filter to DB."""
        components, mock_db, _engine = _make_components()
        mock_db.get_recent_documents.return_value = []

        asyncio.run(_handle_list_recent(components, {"folder_filter": "/docs", "limit": 10}))

        mock_db.get_recent_documents.assert_called_once_with(10, "/docs")


# --- get_sync_status Tests ---


class TestGetSyncStatus:
    def test_returns_counts(self) -> None:
        """get_sync_status returns correct counts and folder breakdown."""
        components, mock_db, _engine = _make_components()

        mock_conn = MagicMock()
        mock_db._conn = mock_conn
        mock_db.get_document_count.return_value = 10
        mock_db.get_error_count.return_value = 2

        total_row = MagicMock()
        total_row.__getitem__ = lambda self, i: 15
        pending_row = MagicMock()
        pending_row.__getitem__ = lambda self, i: 3
        last_sync_row = MagicMock()
        last_sync_row.__getitem__ = lambda self, i: "2025-06-01T12:00:00"

        folder_row = MagicMock()
        folder_row.__getitem__ = lambda self, i: {
            0: "/docs",
            1: 10,
            2: 8,
            3: 1,
        }[i]

        q_total = "SELECT COUNT(*) FROM sync_state WHERE NOT is_deleted"
        q_pending = (
            "SELECT COUNT(*) FROM sync_state WHERE process_status = 'pending' AND NOT is_deleted"
        )
        q_sync = "SELECT MAX(synced_at) FROM sync_state WHERE synced_at IS NOT NULL"

        execute_results: dict[str, MagicMock] = {
            q_total: MagicMock(fetchone=MagicMock(return_value=total_row)),
            q_pending: MagicMock(fetchone=MagicMock(return_value=pending_row)),
            q_sync: MagicMock(fetchone=MagicMock(return_value=last_sync_row)),
        }

        def side_effect(sql: str) -> MagicMock:
            for key, val in execute_results.items():
                if key in sql:
                    return val
            result = MagicMock()
            result.fetchall.return_value = [folder_row]
            return result

        mock_conn.execute.side_effect = side_effect

        result = asyncio.run(_handle_sync_status(components))

        data = json.loads(result[0].text)
        assert data["total_files"] == 15
        assert data["indexed_count"] == 10
        assert data["pending_count"] == 3
        assert data["error_count"] == 2
        assert data["last_sync_time"] == "2025-06-01T12:00:00"
        assert len(data["folders"]) == 1
        assert data["folders"][0]["folder_path"] == "/docs"


# --- Error Handling Tests ---


class TestErrorHandling:
    def test_unknown_tool_returns_error(self) -> None:
        """_error_content produces structured error JSON."""
        result = _error_content("Unknown tool: foo")
        data = json.loads(result[0].text)
        assert data["error"] == "Unknown tool: foo"

    def test_invalid_search_input(self) -> None:
        """search_documents with missing query raises ValidationError."""
        components, _db, _engine = _make_components()

        with pytest.raises(ValidationError):
            asyncio.run(_handle_search(components, {}))
