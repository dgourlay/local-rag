from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import pytest

from rag.config import (
    AppConfig,
    DatabaseConfig,
    EmbeddingConfig,
    FoldersConfig,
    MCPConfig,
    QdrantConfig,
    RerankerConfig,
    SummarizationConfig,
    WatcherConfig,
)
from rag.db.connection import get_connection
from rag.db.migrations import run_migrations
from rag.db.models import SqliteMetadataDB
from rag.db.qdrant import QdrantVectorStore
from rag.pipeline.dedup import DedupChecker
from rag.pipeline.parser.text_parser import TextParser
from rag.pipeline.runner import PipelineRunner
from rag.results import SectionSummarySuccess, SummarySuccess
from rag.retrieval.citations import CitationAssembler
from rag.retrieval.engine import RetrievalEngine
from rag.sync.scanner import scan_folders
from rag.types import RecordType

if TYPE_CHECKING:
    import sqlite3
    from pathlib import Path

    from rag.types import FileEvent, SearchHit

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


# --- Fake implementations ---


class FakeEmbedder:
    """Deterministic mock embedder that hashes text into 1024-dim vectors."""

    @property
    def dimensions(self) -> int:
        return 1024

    @property
    def model_version(self) -> str:
        return "fake-embedder-v1"

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._hash_to_vector(query)

    def _hash_to_vector(self, text: str) -> list[float]:
        import hashlib

        words = text.lower().split()
        vector = [0.0] * 1024
        for word in words:
            h = hashlib.sha256(word.encode()).digest()
            for i in range(0, min(len(h), 32), 2):
                idx = int.from_bytes(h[i : i + 2], "big") % 1024
                vector[idx] += 1.0
        magnitude = sum(v * v for v in vector) ** 0.5
        if magnitude > 0:
            vector = [v / magnitude for v in vector]
        return vector


class FakeReranker:
    """Pass-through reranker that just truncates to top_k."""

    def rerank(self, query: str, candidates: list[SearchHit], top_k: int) -> list[SearchHit]:
        return candidates[:top_k]


class FakeSummarizer:
    """Fake summarizer that returns deterministic summaries without calling CLI."""

    @property
    def available(self) -> bool:
        return True

    def summarize_document(self, text: str, title: str | None, file_type: str) -> SummarySuccess:
        return SummarySuccess(
            summary_l1=f"Summary of {title or 'doc'}",
            summary_l2=f"Document '{title}' covers key topics in {file_type} format.",
            summary_l3=(
                f"This is a detailed summary of '{title}'. "
                f"The document is in {file_type} format. "
                "It contains several sections with important information. "
                f"Key content: {text[:100]}..."
            ),
            key_topics=["topic1", "topic2", "topic3"],
            doc_type_guess="document",
        )

    def summarize_section(
        self, text: str, heading: str | None, doc_context: str
    ) -> SectionSummarySuccess:
        return SectionSummarySuccess(
            section_summary=f"Section '{heading or 'untitled'}' discusses: {text[:80]}...",
            section_summary_l2=f"About {heading or 'content'}",
        )


# --- Fixtures ---


@pytest.fixture()
def tmp_config(tmp_path: Path) -> AppConfig:
    import shutil

    fixtures_dest = tmp_path / "fixtures"
    shutil.copytree(FIXTURES_DIR, fixtures_dest)

    return AppConfig(
        folders=FoldersConfig(paths=[fixtures_dest]),
        database=DatabaseConfig(path=tmp_path / "test.db"),
        qdrant=QdrantConfig(url="http://localhost:6333", collection="test_improvements"),
        embedding=EmbeddingConfig(),
        reranker=RerankerConfig(),
        summarization=SummarizationConfig(enabled=True),
        mcp=MCPConfig(),
        watcher=WatcherConfig(),
    )


@pytest.fixture()
def db_conn(tmp_config: AppConfig) -> sqlite3.Connection:
    conn = get_connection(tmp_config.database.path)
    run_migrations(conn)
    return conn


@pytest.fixture()
def metadata_db(db_conn: sqlite3.Connection) -> SqliteMetadataDB:
    return SqliteMetadataDB(db_conn)


@pytest.fixture()
def qdrant_store() -> QdrantVectorStore:
    from qdrant_client import QdrantClient

    client = QdrantClient(location=":memory:")
    store = QdrantVectorStore.from_client(client, "test_improvements")
    store.ensure_collection()
    return store


@pytest.fixture()
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture()
def reranker() -> FakeReranker:
    return FakeReranker()


@pytest.fixture()
def summarizer() -> FakeSummarizer:
    return FakeSummarizer()


@pytest.fixture()
def pipeline_with_summaries(
    metadata_db: SqliteMetadataDB,
    qdrant_store: QdrantVectorStore,
    embedder: FakeEmbedder,
    summarizer: FakeSummarizer,
    tmp_config: AppConfig,
    db_conn: sqlite3.Connection,
) -> PipelineRunner:
    dedup = DedupChecker(db_conn)
    text_parser = TextParser()
    return PipelineRunner(
        db=metadata_db,
        vector_store=qdrant_store,
        embedder=embedder,
        parsers=[text_parser],
        dedup=dedup,
        config=tmp_config,
        summarizer=summarizer,
    )


@pytest.fixture()
def retrieval_engine(
    qdrant_store: QdrantVectorStore,
    embedder: FakeEmbedder,
    reranker: FakeReranker,
    metadata_db: SqliteMetadataDB,
) -> RetrievalEngine:
    citations = CitationAssembler(metadata_db)
    return RetrievalEngine(
        vector_store=qdrant_store,
        embedder=embedder,
        reranker=reranker,
        citation_assembler=citations,
        top_k_candidates=30,
        top_k_final=10,
    )


@pytest.fixture()
def file_events(tmp_config: AppConfig) -> list[FileEvent]:
    return scan_folders(tmp_config.folders)


@pytest.fixture()
def indexed_with_summaries(
    pipeline_with_summaries: PipelineRunner,
    file_events: list[FileEvent],
) -> tuple[PipelineRunner, int, int]:
    """Index all fixtures with summarization enabled."""
    from rag.types import ProcessingOutcome

    counts = pipeline_with_summaries.process_batch(file_events)
    success = counts[ProcessingOutcome.INDEXED] + counts[ProcessingOutcome.DUPLICATE]
    errors = counts[ProcessingOutcome.ERROR]
    return pipeline_with_summaries, success, errors


# --- Test 1: Summary vectors in Qdrant ---


class TestSummaryVectorsInQdrant:
    def test_summary_points_exist(
        self,
        indexed_with_summaries: tuple[PipelineRunner, int, int],
        qdrant_store: QdrantVectorStore,
        metadata_db: SqliteMetadataDB,
    ) -> None:
        """Qdrant has document_summary and section_summary points."""
        _runner, success, _errors = indexed_with_summaries
        assert success >= 4

        # Scroll all points
        points, _ = qdrant_store._client.scroll(
            collection_name="test_improvements",
            limit=500,
            with_payload=True,
        )

        record_types = {p.payload["record_type"] for p in points if p.payload}

        assert "chunk" in record_types, "Expected chunk points in Qdrant"
        assert "document_summary" in record_types, "Expected document_summary points in Qdrant"
        assert "section_summary" in record_types, "Expected section_summary points in Qdrant"

    def test_summary_metadata_in_sqlite(
        self,
        indexed_with_summaries: tuple[PipelineRunner, int, int],
        metadata_db: SqliteMetadataDB,
        db_conn: sqlite3.Connection,
    ) -> None:
        """Documents should have summary_l1, key_topics populated."""
        _runner, _success, _errors = indexed_with_summaries

        docs = db_conn.execute(
            "SELECT doc_id, summary_l1, key_topics FROM documents WHERE summary_l1 IS NOT NULL"
        ).fetchall()
        assert len(docs) > 0, "Expected at least one document with summaries"

        for doc in docs:
            assert doc["summary_l1"] is not None
            assert doc["summary_l1"].startswith("Summary of")


# --- Test 2: Prefetch lanes debug info ---


class TestPrefetchLanesDebug:
    def test_debug_info_has_lane_counts(
        self,
        indexed_with_summaries: tuple[PipelineRunner, int, int],
        retrieval_engine: RetrievalEngine,
    ) -> None:
        """Search with debug=True returns per-lane hit counts."""
        result = retrieval_engine.search("revenue expenses outlook", debug=True)
        assert result.debug_info is not None

        assert "dense_doc_summary_count" in result.debug_info
        assert "dense_section_summary_count" in result.debug_info
        assert "dense_chunk_count" in result.debug_info

        # Total dense count should equal sum of lanes
        total = (
            result.debug_info["dense_doc_summary_count"]
            + result.debug_info["dense_section_summary_count"]
            + result.debug_info["dense_chunk_count"]
        )
        assert result.debug_info["dense_count"] == total


# --- Test 3: Layer weighting broad vs specific ---


class TestLayerWeightingBroadVsSpecific:
    def test_broad_vs_specific_weights_differ(
        self,
        indexed_with_summaries: tuple[PipelineRunner, int, int],
        retrieval_engine: RetrievalEngine,
    ) -> None:
        """Broad and specific queries produce different layer_weights in debug_info."""
        broad_result = retrieval_engine.search("what documents do I have", debug=True)
        specific_result = retrieval_engine.search(
            "truck gate check-in fraud rejection procedure step by step", debug=True
        )

        assert broad_result.debug_info is not None
        assert specific_result.debug_info is not None

        broad_weights = broad_result.debug_info["layer_weights"]
        specific_weights = specific_result.debug_info["layer_weights"]

        assert broad_weights != specific_weights

        # Broad should boost document_summary more
        assert (
            broad_weights[RecordType.DOCUMENT_SUMMARY]
            > specific_weights[RecordType.DOCUMENT_SUMMARY]
        )

        # Query classifications should differ
        assert broad_result.debug_info["query_classification"] == "broad"
        assert specific_result.debug_info["query_classification"] == "specific"


# --- Test 4: Recency boost applied ---


class TestRecencyBoostApplied:
    def test_recency_applied_in_debug(
        self,
        indexed_with_summaries: tuple[PipelineRunner, int, int],
        retrieval_engine: RetrievalEngine,
    ) -> None:
        """Debug info should include recency_applied=True."""
        result = retrieval_engine.search("project plan timeline", debug=True)
        assert result.debug_info is not None
        assert result.debug_info.get("recency_applied") is True


# --- Test 5: MCP text format (tested via _format_results_as_text) ---


class TestMCPTextFormat:
    def test_search_returns_text_format(
        self,
        indexed_with_summaries: tuple[PipelineRunner, int, int],
        retrieval_engine: RetrievalEngine,
        metadata_db: SqliteMetadataDB,
    ) -> None:
        """search_documents text format returns grouped LLM-friendly output."""
        from rag.mcp.tools import _format_results_as_text

        result = retrieval_engine.search("revenue expenses", debug=True)
        assert len(result.hits) > 0

        # Build doc_lookup like the MCP handler does
        doc_ids = {hit.citation.path for hit in result.hits}
        doc_lookup = {}
        for doc_id in doc_ids:
            doc_lookup[doc_id] = metadata_db.get_document(doc_id)

        text = _format_results_as_text(
            hits=result.hits,
            doc_lookup=doc_lookup,
            query_classification=result.query_classification,
            debug_info=result.debug_info,
        )

        # Should be plain text, not JSON
        assert not text.startswith("{"), "Expected text format, got JSON"
        assert not text.startswith("["), "Expected text format, got JSON array"

        # Should contain document grouping headers
        assert "##" in text
        assert "Found" in text

        # Should contain summary info if available
        assert "score:" in text

        # Footer should have query classification
        assert "Query classified as:" in text

        # Debug info should be present
        assert "Debug info:" in text


class TestMCPJsonFormat:
    def test_search_returns_json_format(
        self,
        indexed_with_summaries: tuple[PipelineRunner, int, int],
        retrieval_engine: RetrievalEngine,
    ) -> None:
        """search_documents json format returns valid JSON with results list."""
        from rag.types import SearchDocumentsOutput

        result = retrieval_engine.search("revenue", debug=False)

        output = SearchDocumentsOutput(
            results=result.hits,
            query_classification=result.query_classification,
            debug_info=result.debug_info,
        )
        text = output.model_dump_json()

        data = json.loads(text)
        assert "results" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) > 0

        # Each result should have citation
        for r in data["results"]:
            assert "citation" in r
            assert "text" in r
            assert "score" in r


# --- Test 7: MCP quick_search and tool list ---


class TestMCPQuickSearch:
    def test_five_tools_defined(self) -> None:
        """MCP tool list contains exactly 5 tools including quick_search."""
        from rag.mcp.tools import _TOOLS

        tool_names = [t.name for t in _TOOLS]
        assert len(_TOOLS) == 5, f"Expected 5 MCP tools, got {len(_TOOLS)}: {tool_names}"
        assert "quick_search" in tool_names
        assert "search_documents" in tool_names
        assert "get_document_context" in tool_names
        assert "list_recent_documents" in tool_names
        assert "get_sync_status" in tool_names

    def test_quick_search_returns_doc_level(
        self,
        indexed_with_summaries: tuple[PipelineRunner, int, int],
        retrieval_engine: RetrievalEngine,
        metadata_db: SqliteMetadataDB,
        db_conn: sqlite3.Connection,
    ) -> None:
        """quick_search returns document-level results (titles, summaries, paths)."""
        result = retrieval_engine.search("project plan", top_k=5)
        assert len(result.hits) > 0

        # Collect unique doc_ids from search results
        seen_doc_ids: list[str] = []
        for hit in result.hits:
            # The actual doc_id comes from the payload, not citation.path
            doc_id = hit.citation.path  # This is file_path in CitedEvidence
            # Look up by file_path since that's what quick_search uses
            row = db_conn.execute(
                "SELECT * FROM documents WHERE file_path = ?", (doc_id,)
            ).fetchone()
            if row and row["doc_id"] not in seen_doc_ids:
                seen_doc_ids.append(row["doc_id"])

        assert len(seen_doc_ids) > 0, "Expected docs in search results"

        # Verify we can get full doc details for quick_search output
        for doc_id in seen_doc_ids:
            doc = metadata_db.get_document(doc_id)
            assert doc is not None
            assert doc.title is not None or doc.file_path is not None

    def test_quick_search_has_summaries(
        self,
        indexed_with_summaries: tuple[PipelineRunner, int, int],
        metadata_db: SqliteMetadataDB,
    ) -> None:
        """Documents indexed with summarizer have summary_l1 and key_topics in DB."""
        docs = metadata_db._conn.execute(
            "SELECT doc_id, title, summary_l1, key_topics FROM documents "
            "WHERE summary_l1 IS NOT NULL"
        ).fetchall()

        assert len(docs) > 0
        for doc in docs:
            assert doc["summary_l1"] is not None
            # Verify topics are stored (JSON list in SQLite)
            assert doc["key_topics"] is not None


# --- Test 8: Known query results ---


class TestKnownQueryResults:
    def test_search_revenue(
        self,
        indexed_with_summaries: tuple[PipelineRunner, int, int],
        retrieval_engine: RetrievalEngine,
    ) -> None:
        """Search 'Q3 revenue year-over-year' finds quarterly-report content."""
        result = retrieval_engine.search("Q3 revenue year-over-year")
        assert len(result.hits) > 0
        texts = " ".join(h.text for h in result.hits)
        assert "12%" in texts or "revenue" in texts.lower()

    def test_search_postgresql_migration(
        self,
        indexed_with_summaries: tuple[PipelineRunner, int, int],
        retrieval_engine: RetrievalEngine,
    ) -> None:
        """Search 'PostgreSQL migration' finds meeting-notes content."""
        result = retrieval_engine.search("PostgreSQL migration")
        assert len(result.hits) > 0
        texts = " ".join(h.text for h in result.hits)
        assert "postgresql" in texts.lower() or "migration" in texts.lower()

    def test_search_project_plan_timeline(
        self,
        indexed_with_summaries: tuple[PipelineRunner, int, int],
        retrieval_engine: RetrievalEngine,
    ) -> None:
        """Search 'project plan timeline milestones' finds project-plan content."""
        result = retrieval_engine.search("project plan timeline milestones")
        assert len(result.hits) > 0
        # Verify at least one hit references project plan
        paths = [h.citation.path for h in result.hits]
        assert any("project-plan" in p for p in paths)

    def test_every_result_has_citation(
        self,
        indexed_with_summaries: tuple[PipelineRunner, int, int],
        retrieval_engine: RetrievalEngine,
    ) -> None:
        """Every result has a citation with title, path, and label."""
        result = retrieval_engine.search("revenue expenses outlook")
        for hit in result.hits:
            assert hit.citation is not None
            assert hit.citation.title
            assert hit.citation.path
            assert hit.citation.label
