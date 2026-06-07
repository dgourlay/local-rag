from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient

from rag.config import QdrantConfig
from rag.db.qdrant import QdrantVectorStore, _resolve_prefer_grpc
from rag.types import (
    FileType,
    QdrantPayloadModel,
    RecordType,
    SearchFilters,
    VectorPoint,
)

COLLECTION = "test_documents"


class _FakeSocket:
    def __enter__(self) -> _FakeSocket:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        return None


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


class TestGrpcTransportFallback:
    def test_prefer_grpc_false_skips_socket_probe(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fail_create_connection(address: tuple[str, int], timeout: float) -> _FakeSocket:
            raise AssertionError("socket probe should not run")

        monkeypatch.setattr(
            "rag.db.qdrant.socket.create_connection",
            fail_create_connection,
        )

        config = QdrantConfig(prefer_grpc=False)

        assert _resolve_prefer_grpc(config) is False

    def test_prefer_grpc_uses_reachable_port(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: dict[str, object] = {}

        def fake_create_connection(address: tuple[str, int], timeout: float) -> _FakeSocket:
            seen["address"] = address
            seen["timeout"] = timeout
            return _FakeSocket()

        monkeypatch.setattr(
            "rag.db.qdrant.socket.create_connection",
            fake_create_connection,
        )

        config = QdrantConfig(
            url="http://qdrant.local:6333",
            grpc_port=6334,
            prefer_grpc=True,
        )

        assert _resolve_prefer_grpc(config) is True
        assert seen["address"] == ("qdrant.local", 6334)
        assert seen["timeout"] == 0.25

    def test_prefer_grpc_falls_back_when_port_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_create_connection(address: tuple[str, int], timeout: float) -> _FakeSocket:
            raise OSError("connection refused")

        monkeypatch.setattr(
            "rag.db.qdrant.socket.create_connection",
            fake_create_connection,
        )

        config = QdrantConfig(prefer_grpc=True)

        assert _resolve_prefer_grpc(config) is False

    def test_constructor_passes_rest_fallback_to_client(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_create_connection(address: tuple[str, int], timeout: float) -> _FakeSocket:
            raise OSError("connection refused")

        class FakeQdrantClient:
            def __init__(
                self,
                *,
                url: str,
                grpc_port: int,
                prefer_grpc: bool,
                check_compatibility: bool,
            ) -> None:
                captured["url"] = url
                captured["grpc_port"] = grpc_port
                captured["prefer_grpc"] = prefer_grpc
                captured["check_compatibility"] = check_compatibility

        monkeypatch.setattr(
            "rag.db.qdrant.socket.create_connection",
            fake_create_connection,
        )
        monkeypatch.setattr("rag.db.qdrant.QdrantClient", FakeQdrantClient)

        config = QdrantConfig(
            url="http://localhost:6333",
            grpc_port=6334,
            prefer_grpc=True,
        )

        QdrantVectorStore(config)

        assert captured["url"] == "http://localhost:6333"
        assert captured["grpc_port"] == 6334
        assert captured["prefer_grpc"] is False
        assert captured["check_compatibility"] is False


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

    def test_query_dense_can_limit_payload_fields(self) -> None:
        store = _make_store()
        point = _make_point(str(uuid.uuid4()), text="machine learning algorithms")
        store.upsert_points("doc-1", [point])

        results = store.query_dense(
            vector=point.vector,
            filters=SearchFilters(),
            limit=5,
            payload_fields=["record_type", "doc_id", "text"],
        )

        assert len(results) == 1
        assert results[0].text == "machine learning algorithms"
        assert set(results[0].payload) == {"record_type", "doc_id", "text"}

    def test_get_hits_by_ids_fetches_full_payload(self) -> None:
        store = _make_store()
        point_id = str(uuid.uuid4())
        point = _make_point(point_id, text="machine learning algorithms")
        store.upsert_points("doc-1", [point])

        hits = store.get_hits_by_ids([point_id])

        assert list(hits) == [point_id]
        assert hits[point_id].text == "machine learning algorithms"
        assert hits[point_id].payload["file_path"] == "/docs/test.pdf"


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

    def test_query_keyword_can_limit_payload_fields(self) -> None:
        store = _make_store()
        store.upsert_points(
            "doc-1",
            [
                _make_point(str(uuid.uuid4()), text="python programming language"),
            ],
        )

        results = store.query_keyword(
            query="programming",
            filters=SearchFilters(),
            limit=10,
            payload_fields=["record_type", "doc_id", "text"],
        )

        assert len(results) == 1
        assert results[0].text == "python programming language"
        assert set(results[0].payload) == {"record_type", "doc_id", "text"}


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


class TestDateFilter:
    """Test date_filter on dense and keyword queries."""

    def _make_store_with_dates(self) -> QdrantVectorStore:
        store = _make_store()
        v_recent = [0.0] * 1024
        v_recent[0] = 1.0
        v_old = [0.0] * 1024
        v_old[0] = 0.9
        v_old[1] = 0.1
        store.upsert_points(
            "doc-recent",
            [
                VectorPoint(
                    point_id=str(uuid.uuid4()),
                    vector=v_recent,
                    payload=QdrantPayloadModel(
                        record_type=RecordType.CHUNK,
                        doc_id="doc-recent",
                        title="Recent Doc",
                        file_path="/docs/recent.pdf",
                        folder_path="/docs",
                        folder_ancestors=["/docs"],
                        file_type=FileType.PDF,
                        modified_at="2026-03-15T10:00:00+00:00",
                        text="recent quarterly business review",
                    ),
                ),
            ],
        )
        store.upsert_points(
            "doc-old",
            [
                VectorPoint(
                    point_id=str(uuid.uuid4()),
                    vector=v_old,
                    payload=QdrantPayloadModel(
                        record_type=RecordType.CHUNK,
                        doc_id="doc-old",
                        title="Old Doc",
                        file_path="/docs/old.pdf",
                        folder_path="/docs",
                        folder_ancestors=["/docs"],
                        file_type=FileType.PDF,
                        modified_at="2025-06-01T10:00:00+00:00",
                        text="old quarterly business review",
                    ),
                ),
            ],
        )
        return store

    def test_date_filter_full_iso(self) -> None:
        store = self._make_store_with_dates()
        results = store.query_dense(
            vector=[1.0] + [0.0] * 1023,
            filters=SearchFilters(date_filter="2026-01-01T00:00:00+00:00"),
            limit=10,
        )
        assert len(results) == 1
        assert results[0].doc_id == "doc-recent"

    def test_date_filter_bare_date(self) -> None:
        store = self._make_store_with_dates()
        results = store.query_dense(
            vector=[1.0] + [0.0] * 1023,
            filters=SearchFilters(date_filter="2026-01-01"),
            limit=10,
        )
        assert len(results) == 1
        assert results[0].doc_id == "doc-recent"

    def test_date_filter_includes_all_when_old_enough(self) -> None:
        store = self._make_store_with_dates()
        results = store.query_dense(
            vector=[1.0] + [0.0] * 1023,
            filters=SearchFilters(date_filter="2025-01-01"),
            limit=10,
        )
        assert len(results) == 2

    def test_date_filter_excludes_all_when_future(self) -> None:
        store = self._make_store_with_dates()
        results = store.query_dense(
            vector=[1.0] + [0.0] * 1023,
            filters=SearchFilters(date_filter="2027-01-01"),
            limit=10,
        )
        assert len(results) == 0

    def test_date_filter_keyword_search(self) -> None:
        store = self._make_store_with_dates()
        results = store.query_keyword(
            query="quarterly",
            filters=SearchFilters(date_filter="2026-01-01"),
            limit=10,
        )
        assert len(results) == 1
        assert results[0].doc_id == "doc-recent"

    def test_date_filter_rejects_numeric_range(self) -> None:
        """Regression: models.Range rejects string dates, DatetimeRange is required."""
        from qdrant_client import models as qdrant_models

        with pytest.raises(Exception, match="valid number"):
            qdrant_models.Range(gte="2026-01-01T00:00:00+00:00")


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
