from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qdrant_client import AsyncQdrantClient, QdrantClient, models

if TYPE_CHECKING:
    from rag.config import QdrantConfig
    from rag.types import SearchFilters, SearchHit, VectorPoint

# Payload fields that get keyword indices
_KEYWORD_INDEX_FIELDS: list[str] = [
    "record_type",
    "summary_level",
    "doc_id",
    "folder_path",
    "file_type",
    "file_path",
    "doc_type_guess",
    "section_id",
    "chunk_id",
    "key_topics",
]

_VECTOR_DIM = 1024
_COLLECTION_DISTANCE = models.Distance.COSINE


def _build_filter(
    filters: SearchFilters, *, record_type: str | None = None
) -> models.Filter | None:
    """Build a Qdrant filter from SearchFilters and optional record_type."""
    conditions: list[models.FieldCondition] = []

    if filters.folder_filter is not None:
        conditions.append(
            models.FieldCondition(
                key="folder_path",
                match=models.MatchValue(value=filters.folder_filter),
            )
        )

    if filters.file_type is not None:
        conditions.append(
            models.FieldCondition(
                key="file_type",
                match=models.MatchValue(value=str(filters.file_type)),
            )
        )

    if filters.date_filter is not None:
        # Normalize bare date ("2026-02-18") to full ISO 8601 datetime
        # so Qdrant's datetime range filter matches correctly.
        date_val = filters.date_filter
        if len(date_val) == 10 and "T" not in date_val:
            date_val = f"{date_val}T00:00:00+00:00"
        conditions.append(
            models.FieldCondition(
                key="modified_at",
                range=models.DatetimeRange(gte=date_val),
            )
        )

    if record_type is not None:
        conditions.append(
            models.FieldCondition(
                key="record_type",
                match=models.MatchValue(value=record_type),
            )
        )

    if not conditions:
        return None

    return models.Filter(must=conditions)


def _scored_point_to_search_hit(point: models.ScoredPoint) -> SearchHit:
    """Convert a Qdrant ScoredPoint to a SearchHit."""
    from rag.types import RecordType
    from rag.types import SearchHit as SearchHitModel

    payload: dict[str, Any] = point.payload or {}
    return SearchHitModel(
        point_id=str(point.id),
        score=point.score,
        record_type=RecordType(payload.get("record_type", "chunk")),
        doc_id=payload.get("doc_id", ""),
        text=payload.get("text", ""),
        payload=payload,
    )


class QdrantVectorStore:
    """Sync Qdrant vector store implementing the VectorStore protocol."""

    def __init__(self, config: QdrantConfig) -> None:
        self._client = QdrantClient(
            url=config.url,
            grpc_port=config.grpc_port,
            prefer_grpc=config.prefer_grpc,
            check_compatibility=False,
        )
        self._collection = config.collection

    @classmethod
    def from_client(cls, client: QdrantClient, collection: str) -> QdrantVectorStore:
        """Create from an existing client (useful for tests with :memory:)."""
        instance = object.__new__(cls)
        instance._client = client
        instance._collection = collection
        return instance

    def recreate_collection(self) -> None:
        """Drop and recreate the collection, removing all vectors."""
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        self.ensure_collection()

    def ensure_collection(self) -> None:
        """Create collection with 1024-dim cosine vectors and payload indices.

        Idempotent: creates the collection if missing, then ensures all payload
        indexes exist (safe to call on existing collections to pick up new indexes).
        """
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=_VECTOR_DIM,
                    distance=_COLLECTION_DISTANCE,
                ),
            )

        # Qdrant's create_payload_index is idempotent — no-op if index exists.
        for field_name in _KEYWORD_INDEX_FIELDS:
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="text",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                min_token_len=3,
                max_token_len=20,
                lowercase=True,
            ),
        )

        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="generated_questions",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                min_token_len=2,
                max_token_len=20,
                lowercase=True,
            ),
        )

        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="modified_at",
            field_schema=models.PayloadSchemaType.DATETIME,
        )

    def upsert_points(self, doc_id: str, points: list[VectorPoint]) -> None:
        """Upsert points with deterministic IDs (overwrite semantics)."""
        if not points:
            return

        qdrant_points = [
            models.PointStruct(
                id=p.point_id,
                vector=p.vector,
                payload=p.payload.model_dump(mode="json"),
            )
            for p in points
        ]

        self._client.upsert(
            collection_name=self._collection,
            points=qdrant_points,
        )

    def delete_stale_points(self, doc_id: str, keep_ids: set[str]) -> None:
        """Delete points for doc_id that are NOT in keep_ids."""
        all_ids: list[str] = []
        next_page_offset: Any = None

        while True:
            points, next_page_offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id),
                        )
                    ]
                ),
                limit=100,
                offset=next_page_offset,
                with_payload=False,
                with_vectors=False,
            )
            all_ids.extend(str(p.id) for p in points)
            if next_page_offset is None:
                break

        stale_ids = [pid for pid in all_ids if pid not in keep_ids]
        if stale_ids:
            self._client.delete(
                collection_name=self._collection,
                points_selector=models.PointIdsList(points=stale_ids),
            )

    def query_dense(
        self,
        vector: list[float],
        filters: SearchFilters,
        limit: int,
        record_type: str | None = None,
    ) -> list[SearchHit]:
        """Dense vector search using query_points() API."""
        query_filter = _build_filter(filters, record_type=record_type)

        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        return [_scored_point_to_search_hit(p) for p in response.points]

    def query_keyword(self, query: str, filters: SearchFilters, limit: int) -> list[SearchHit]:
        """Keyword search using text index via query_points()."""
        text_condition = models.FieldCondition(
            key="text",
            match=models.MatchText(text=query),
        )
        questions_condition = models.FieldCondition(
            key="generated_questions",
            match=models.MatchText(text=query),
        )

        # Text matching uses should (OR) -- match either text or generated_questions
        text_should: list[
            models.FieldCondition
            | models.IsEmptyCondition
            | models.IsNullCondition
            | models.HasIdCondition
            | models.HasVectorCondition
            | models.NestedCondition
            | models.Filter
        ] = [text_condition, questions_condition]

        base_filter = _build_filter(filters)
        if base_filter is not None and base_filter.must is not None:
            combined_filter = models.Filter(
                must=list(base_filter.must),
                should=text_should,
                min_should=models.MinShould(
                    conditions=text_should,
                    min_count=1,
                ),
            )
        else:
            combined_filter = models.Filter(
                should=text_should,
                min_should=models.MinShould(
                    conditions=text_should,
                    min_count=1,
                ),
            )

        response = self._client.query_points(
            collection_name=self._collection,
            query_filter=combined_filter,
            limit=limit,
            with_payload=True,
        )

        return [_scored_point_to_search_hit(p) for p in response.points]

    def close(self) -> None:
        """Close the underlying client."""
        self._client.close()


class AsyncQdrantVectorStore:
    """Async Qdrant vector store for MCP handlers."""

    def __init__(self, config: QdrantConfig) -> None:
        self._client = AsyncQdrantClient(
            url=config.url,
            grpc_port=config.grpc_port,
            prefer_grpc=config.prefer_grpc,
            check_compatibility=False,
        )
        self._collection = config.collection

    @classmethod
    def from_client(cls, client: AsyncQdrantClient, collection: str) -> AsyncQdrantVectorStore:
        """Create from an existing client."""
        instance = object.__new__(cls)
        instance._client = client
        instance._collection = collection
        return instance

    async def ensure_collection(self) -> None:
        """Create collection with 1024-dim cosine vectors and payload indices.

        Idempotent: creates the collection if missing, then ensures all payload
        indexes exist (safe to call on existing collections to pick up new indexes).
        """
        if not await self._client.collection_exists(self._collection):
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=_VECTOR_DIM,
                    distance=_COLLECTION_DISTANCE,
                ),
            )

        for field_name in _KEYWORD_INDEX_FIELDS:
            await self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

        await self._client.create_payload_index(
            collection_name=self._collection,
            field_name="text",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                min_token_len=3,
                max_token_len=20,
                lowercase=True,
            ),
        )

        await self._client.create_payload_index(
            collection_name=self._collection,
            field_name="generated_questions",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                min_token_len=2,
                max_token_len=20,
                lowercase=True,
            ),
        )

        await self._client.create_payload_index(
            collection_name=self._collection,
            field_name="modified_at",
            field_schema=models.PayloadSchemaType.DATETIME,
        )

    async def upsert_points(self, doc_id: str, points: list[VectorPoint]) -> None:
        """Upsert points with deterministic IDs (overwrite semantics)."""
        if not points:
            return

        qdrant_points = [
            models.PointStruct(
                id=p.point_id,
                vector=p.vector,
                payload=p.payload.model_dump(mode="json"),
            )
            for p in points
        ]

        await self._client.upsert(
            collection_name=self._collection,
            points=qdrant_points,
        )

    async def delete_stale_points(self, doc_id: str, keep_ids: set[str]) -> None:
        """Delete points for doc_id that are NOT in keep_ids."""
        all_ids: list[str] = []
        next_page_offset: Any = None

        while True:
            points, next_page_offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id),
                        )
                    ]
                ),
                limit=100,
                offset=next_page_offset,
                with_payload=False,
                with_vectors=False,
            )
            all_ids.extend(str(p.id) for p in points)
            if next_page_offset is None:
                break

        stale_ids = [pid for pid in all_ids if pid not in keep_ids]
        if stale_ids:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=models.PointIdsList(points=stale_ids),
            )

    async def query_dense(
        self,
        vector: list[float],
        filters: SearchFilters,
        limit: int,
        record_type: str | None = None,
    ) -> list[SearchHit]:
        """Dense vector search using query_points() API."""
        query_filter = _build_filter(filters, record_type=record_type)

        response = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        return [_scored_point_to_search_hit(p) for p in response.points]

    async def query_keyword(
        self, query: str, filters: SearchFilters, limit: int
    ) -> list[SearchHit]:
        """Keyword search using text index via query_points()."""
        text_condition = models.FieldCondition(
            key="text",
            match=models.MatchText(text=query),
        )
        questions_condition = models.FieldCondition(
            key="generated_questions",
            match=models.MatchText(text=query),
        )

        # Text matching uses should (OR) -- match either text or generated_questions
        text_should: list[
            models.FieldCondition
            | models.IsEmptyCondition
            | models.IsNullCondition
            | models.HasIdCondition
            | models.HasVectorCondition
            | models.NestedCondition
            | models.Filter
        ] = [text_condition, questions_condition]

        base_filter = _build_filter(filters)
        if base_filter is not None and base_filter.must is not None:
            combined_filter = models.Filter(
                must=list(base_filter.must),
                should=text_should,
                min_should=models.MinShould(
                    conditions=text_should,
                    min_count=1,
                ),
            )
        else:
            combined_filter = models.Filter(
                should=text_should,
                min_should=models.MinShould(
                    conditions=text_should,
                    min_count=1,
                ),
            )

        response = await self._client.query_points(
            collection_name=self._collection,
            query_filter=combined_filter,
            limit=limit,
            with_payload=True,
        )

        return [_scored_point_to_search_hit(p) for p in response.points]

    async def close(self) -> None:
        """Close the underlying client."""
        await self._client.close()
