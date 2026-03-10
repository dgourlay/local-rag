from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from rag.retrieval.query_analyzer import analyze_query
from rag.types import (
    RetrievalResult,
    SearchFilters,
    SearchHit,
)

if TYPE_CHECKING:
    from rag.protocols import Embedder, Reranker, VectorStore
    from rag.retrieval.citations import CitationAssembler

logger = logging.getLogger(__name__)

RRF_K = 60  # RRF constant


def rrf_fuse(
    dense_hits: list[SearchHit],
    keyword_hits: list[SearchHit],
) -> list[SearchHit]:
    """Reciprocal Rank Fusion: score = Σ 1/(k + rank_i)."""
    scores: dict[str, float] = {}
    hit_map: dict[str, SearchHit] = {}

    for rank, hit in enumerate(dense_hits):
        scores[hit.point_id] = scores.get(hit.point_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        hit_map[hit.point_id] = hit

    for rank, hit in enumerate(keyword_hits):
        scores[hit.point_id] = scores.get(hit.point_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        if hit.point_id not in hit_map:
            hit_map[hit.point_id] = hit

    sorted_ids = sorted(scores, key=lambda pid: scores[pid], reverse=True)

    results: list[SearchHit] = []
    for pid in sorted_ids:
        hit = hit_map[pid]
        results.append(
            SearchHit(
                point_id=hit.point_id,
                score=scores[pid],
                record_type=hit.record_type,
                doc_id=hit.doc_id,
                text=hit.text,
                payload=hit.payload,
            )
        )
    return results


class RetrievalEngine:
    """Multi-stage retrieval: embed -> dense + keyword -> RRF -> rerank -> citations."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        reranker: Reranker,
        citation_assembler: CitationAssembler,
        top_k_candidates: int = 30,
        top_k_final: int = 10,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._reranker = reranker
        self._citations = citation_assembler
        self._top_k_candidates = top_k_candidates
        self._top_k_final = top_k_final

    def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        debug: bool = False,
    ) -> RetrievalResult:
        """Synchronous multi-stage retrieval."""
        start = time.monotonic()
        effective_filters = filters or SearchFilters()
        effective_top_k = top_k or self._top_k_final
        debug_info: dict[str, Any] = {}

        # 1. Analyze query
        analysis = analyze_query(query)
        if debug:
            debug_info["query_classification"] = analysis.classification

        # Apply folder hint if no explicit folder filter
        if analysis.folder_hint and not effective_filters.folder_filter:
            effective_filters = SearchFilters(
                folder_filter=analysis.folder_hint,
                date_filter=effective_filters.date_filter,
                file_type=effective_filters.file_type,
            )

        # 2. Embed query
        t0 = time.monotonic()
        query_vector = self._embedder.embed_query(query)
        if debug:
            debug_info["embed_ms"] = int((time.monotonic() - t0) * 1000)

        # 3. Dense search
        t0 = time.monotonic()
        dense_hits = self._vector_store.query_dense(
            query_vector, effective_filters, self._top_k_candidates
        )
        if debug:
            debug_info["dense_ms"] = int((time.monotonic() - t0) * 1000)
            debug_info["dense_count"] = len(dense_hits)

        # 4. Keyword search
        t0 = time.monotonic()
        keyword_hits = self._vector_store.query_keyword(
            query, effective_filters, self._top_k_candidates
        )
        if debug:
            debug_info["keyword_ms"] = int((time.monotonic() - t0) * 1000)
            debug_info["keyword_count"] = len(keyword_hits)

        # 5. RRF fusion
        fused = rrf_fuse(dense_hits, keyword_hits)
        if debug:
            debug_info["fused_count"] = len(fused)

        # 6. Rerank
        t0 = time.monotonic()
        reranked = self._reranker.rerank(query, fused[: self._top_k_candidates], effective_top_k)
        if debug:
            debug_info["rerank_ms"] = int((time.monotonic() - t0) * 1000)

        # 7. Assemble citations
        cited = self._citations.assemble_citations(reranked)

        if debug:
            debug_info["total_ms"] = int((time.monotonic() - start) * 1000)

        return RetrievalResult(
            hits=cited,
            query_classification=analysis.classification,
            debug_info=debug_info if debug else None,
        )

    async def async_search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        debug: bool = False,
    ) -> RetrievalResult:
        """Async wrapper for MCP handlers -- dispatches CPU-bound ops via to_thread."""
        return await asyncio.to_thread(self.search, query, filters, top_k, debug)
