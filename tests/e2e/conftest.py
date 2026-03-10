from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from qdrant_client import QdrantClient

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
from rag.retrieval.citations import CitationAssembler
from rag.retrieval.engine import RetrievalEngine
from rag.sync.scanner import scan_folders

if TYPE_CHECKING:
    import sqlite3

    from rag.types import FileEvent, SearchHit

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


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
        """Hash text into a deterministic 1024-dim unit vector.

        Uses word-level hashing so texts sharing words have higher cosine similarity.
        """
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


@pytest.fixture()
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture()
def temp_config(tmp_path: Path) -> AppConfig:
    """Config pointing at a copy of fixtures with in-memory Qdrant and temp SQLite."""
    fixtures_dest = tmp_path / "fixtures"
    shutil.copytree(FIXTURES_DIR, fixtures_dest)

    db_path = tmp_path / "test.db"

    return AppConfig(
        folders=FoldersConfig(paths=[fixtures_dest]),
        database=DatabaseConfig(path=db_path),
        qdrant=QdrantConfig(url="http://localhost:6333", collection="test_documents"),
        embedding=EmbeddingConfig(),
        reranker=RerankerConfig(),
        summarization=SummarizationConfig(enabled=False),
        mcp=MCPConfig(),
        watcher=WatcherConfig(),
    )


@pytest.fixture()
def db_conn(temp_config: AppConfig) -> sqlite3.Connection:
    """SQLite connection with migrations run."""
    conn = get_connection(temp_config.database.path)
    run_migrations(conn)
    return conn


@pytest.fixture()
def metadata_db(db_conn: sqlite3.Connection) -> SqliteMetadataDB:
    return SqliteMetadataDB(db_conn)


@pytest.fixture()
def qdrant_client() -> QdrantClient:
    """In-memory Qdrant client."""
    return QdrantClient(location=":memory:")


@pytest.fixture()
def vector_store(qdrant_client: QdrantClient) -> QdrantVectorStore:
    store = QdrantVectorStore.from_client(qdrant_client, "test_documents")
    store.ensure_collection()
    return store


@pytest.fixture()
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture()
def reranker() -> FakeReranker:
    return FakeReranker()


@pytest.fixture()
def dedup(db_conn: sqlite3.Connection) -> DedupChecker:
    return DedupChecker(db_conn)


@pytest.fixture()
def text_parser() -> TextParser:
    return TextParser()


@pytest.fixture()
def pipeline_runner(
    metadata_db: SqliteMetadataDB,
    vector_store: QdrantVectorStore,
    embedder: FakeEmbedder,
    text_parser: TextParser,
    dedup: DedupChecker,
    temp_config: AppConfig,
) -> PipelineRunner:
    return PipelineRunner(
        db=metadata_db,
        vector_store=vector_store,
        embedder=embedder,
        parsers=[text_parser],
        dedup=dedup,
        config=temp_config,
    )


@pytest.fixture()
def retrieval_engine(
    vector_store: QdrantVectorStore,
    embedder: FakeEmbedder,
    reranker: FakeReranker,
    metadata_db: SqliteMetadataDB,
) -> RetrievalEngine:
    citations = CitationAssembler(metadata_db)
    return RetrievalEngine(
        vector_store=vector_store,
        embedder=embedder,
        reranker=reranker,
        citation_assembler=citations,
        top_k_candidates=30,
        top_k_final=10,
    )


@pytest.fixture()
def file_events(temp_config: AppConfig) -> list[FileEvent]:
    """Scan the fixtures directory for file events."""
    return scan_folders(temp_config.folders)


@pytest.fixture()
def indexed_pipeline(
    pipeline_runner: PipelineRunner,
    file_events: list[FileEvent],
) -> tuple[PipelineRunner, int, int]:
    """Index all valid fixtures and return (runner, success_count, error_count)."""
    success, errors = pipeline_runner.process_batch(file_events)
    return pipeline_runner, success, errors
