from __future__ import annotations

import uuid

from qdrant_client import QdrantClient

from rag.db.qdrant import QdrantVectorStore
from rag.types import (
    FileType,
    QdrantPayloadModel,
    RecordType,
    SearchFilters,
    VectorPoint,
)

COLLECTION = "test_documents"


def _make_store() -> QdrantVectorStore:
    client = QdrantClient(location=":memory:")
    store = QdrantVectorStore.from_client(client, COLLECTION)
    store.ensure_collection()
    return store


def _make_point(
    point_id: str,
    doc_id: str = "doc-1",
    text: str = "hello world",
    folder_path: str = "/docs",
    file_type: FileType = FileType.PDF,
    record_type: RecordType = RecordType.CHUNK,
) -> VectorPoint:
    vector = [0.0] * 1024
    vector[0] = 1.0
    return VectorPoint(
        point_id=point_id,
        vector=vector,
        payload=QdrantPayloadModel(
            record_type=record_type,
            doc_id=doc_id,
            title="Test Doc",
            file_path=f"{folder_path}/test.pdf",
            folder_path=folder_path,
            folder_ancestors=[folder_path],
            file_type=file_type,
            modified_at="2026-01-01T00:00:00Z",
            text=text,
        ),
    )


class TestEnsureCollection:
    def test_creates_collection(self) -> None:
        client = QdrantClient(location=":memory:")
        store = QdrantVectorStore.from_client(client, COLLECTION)
        store.ensure_collection()
        assert client.collection_exists(COLLECTION)

    def test_idempotent(self) -> None:
        store = _make_store()
        store.ensure_collection()  # second call should not raise

    def test_collection_has_correct_vector_config(self) -> None:
        client = QdrantClient(location=":memory:")
        store = QdrantVectorStore.from_client(client, COLLECTION)
        store.ensure_collection()
        info = client.get_collection(COLLECTION)
        assert info.config.params.vectors.size == 1024  # type: ignore[union-attr]
        assert info.config.params.vectors.distance.name == "COSINE"  # type: ignore[union-attr]


class TestUpsertAndQueryDense:
    def test_upsert_and_find(self) -> None:
        store = _make_store()
        point = _make_point(str(uuid.uuid4()), text="machine learning algorithms")
        store.upsert_points("doc-1", [point])

        results = store.query_dense(
            vector=point.vector,
            filters=SearchFilters(),
            limit=5,
        )
        assert len(results) == 1
        assert results[0].doc_id == "doc-1"
        assert results[0].text == "machine learning algorithms"

    def test_upsert_overwrites(self) -> None:
        store = _make_store()
        pid = str(uuid.uuid4())
        point1 = _make_point(pid, text="original text")
        store.upsert_points("doc-1", [point1])

        point2 = _make_point(pid, text="updated text")
        store.upsert_points("doc-1", [point2])

        results = store.query_dense(
            vector=point2.vector,
            filters=SearchFilters(),
            limit=5,
        )
        assert len(results) == 1
        assert results[0].text == "updated text"

    def test_empty_points_noop(self) -> None:
        store = _make_store()
        store.upsert_points("doc-1", [])  # should not raise


class TestDeleteStalePoints:
    def test_deletes_stale(self) -> None:
        store = _make_store()
        id1 = str(uuid.uuid4())
        id2 = str(uuid.uuid4())
        store.upsert_points(
            "doc-1",
            [
                _make_point(id1, doc_id="doc-1", text="keep this"),
                _make_point(id2, doc_id="doc-1", text="delete this"),
            ],
        )

        store.delete_stale_points("doc-1", keep_ids={id1})

        results = store.query_dense(
            vector=_make_point(id1).vector,
            filters=SearchFilters(),
            limit=10,
        )
        assert len(results) == 1
        assert results[0].point_id == id1

    def test_no_stale_no_error(self) -> None:
        store = _make_store()
        id1 = str(uuid.uuid4())
        store.upsert_points("doc-1", [_make_point(id1)])
        store.delete_stale_points("doc-1", keep_ids={id1})  # nothing to delete


class TestQueryKeyword:
    def test_keyword_search(self) -> None:
        store = _make_store()
        store.upsert_points(
            "doc-1",
            [
                _make_point(str(uuid.uuid4()), text="python programming language"),
                _make_point(str(uuid.uuid4()), text="java programming language"),
                _make_point(str(uuid.uuid4()), text="cooking recipes for dinner"),
            ],
        )

        results = store.query_keyword(
            query="programming",
            filters=SearchFilters(),
            limit=10,
        )
        assert len(results) == 2
        texts = {r.text for r in results}
        assert "python programming language" in texts
        assert "java programming language" in texts

    def test_keyword_no_results(self) -> None:
        store = _make_store()
        store.upsert_points(
            "doc-1",
            [
                _make_point(str(uuid.uuid4()), text="hello world test"),
            ],
        )

        results = store.query_keyword(
            query="nonexistent",
            filters=SearchFilters(),
            limit=10,
        )
        assert len(results) == 0


class TestFilterQueries:
    def test_folder_filter(self) -> None:
        store = _make_store()
        v1 = [0.0] * 1024
        v1[0] = 1.0
        v2 = [0.0] * 1024
        v2[0] = 0.9
        v2[1] = 0.1

        store.upsert_points(
            "doc-1",
            [
                _make_point(str(uuid.uuid4()), folder_path="/docs/work"),
            ],
        )
        store.upsert_points(
            "doc-2",
            [
                _make_point(str(uuid.uuid4()), doc_id="doc-2", folder_path="/docs/personal"),
            ],
        )

        results = store.query_dense(
            vector=v1,
            filters=SearchFilters(folder_filter="/docs/work"),
            limit=10,
        )
        assert len(results) == 1
        assert results[0].payload["folder_path"] == "/docs/work"

    def test_file_type_filter(self) -> None:
        store = _make_store()
        store.upsert_points(
            "doc-1",
            [
                _make_point(str(uuid.uuid4()), file_type=FileType.PDF),
            ],
        )
        store.upsert_points(
            "doc-2",
            [
                _make_point(str(uuid.uuid4()), doc_id="doc-2", file_type=FileType.MD),
            ],
        )

        results = store.query_dense(
            vector=_make_point("x").vector,
            filters=SearchFilters(file_type=FileType.PDF),
            limit=10,
        )
        assert len(results) == 1
        assert results[0].payload["file_type"] == "pdf"

    def test_keyword_with_folder_filter(self) -> None:
        store = _make_store()
        store.upsert_points(
            "doc-1",
            [
                _make_point(
                    str(uuid.uuid4()),
                    text="deep learning neural networks",
                    folder_path="/research",
                ),
            ],
        )
        store.upsert_points(
            "doc-2",
            [
                _make_point(
                    str(uuid.uuid4()),
                    doc_id="doc-2",
                    text="deep learning transformers",
                    folder_path="/notes",
                ),
            ],
        )

        results = store.query_keyword(
            query="learning",
            filters=SearchFilters(folder_filter="/research"),
            limit=10,
        )
        assert len(results) == 1
        assert results[0].payload["folder_path"] == "/research"


class TestEmptyResults:
    def test_dense_empty_collection(self) -> None:
        store = _make_store()
        results = store.query_dense(
            vector=[0.0] * 1024,
            filters=SearchFilters(),
            limit=5,
        )
        assert results == []

    def test_keyword_empty_collection(self) -> None:
        store = _make_store()
        results = store.query_keyword(
            query="anything",
            filters=SearchFilters(),
            limit=5,
        )
        assert results == []
