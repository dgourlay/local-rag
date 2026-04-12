from __future__ import annotations

import asyncio
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from rag.retrieval.query_analyzer import analyze_query
from rag.types import (
    RecordType,
    RetrievalResult,
    SearchFilters,
    SearchHit,
)

if TYPE_CHECKING:
    from rag.config import RetrievalConfig, SummarizationConfig
    from rag.protocols import Embedder, Reranker, VectorStore
    from rag.retrieval.citations import CitationAssembler

logger = logging.getLogger(__name__)

RRF_K = 60  # RRF constant

# Layer weight presets keyed by query classification
LAYER_WEIGHTS: dict[str, dict[str, float]] = {
    "broad": {
        RecordType.DOCUMENT_SUMMARY: 1.5,
        RecordType.SECTION_SUMMARY: 1.3,
        RecordType.CHUNK: 1.0,
    },
    "specific": {
        RecordType.DOCUMENT_SUMMARY: 0.7,
        RecordType.SECTION_SUMMARY: 0.9,
        RecordType.CHUNK: 1.0,
    },
    "navigational": {
        RecordType.DOCUMENT_SUMMARY: 1.0,
        RecordType.SECTION_SUMMARY: 1.4,
        RecordType.CHUNK: 1.0,
    },
}

# Recency boost parameters
RECENCY_HALF_LIFE_DAYS = 90
RECENCY_MAX_BOOST = 0.15


def rrf_fuse(
    ranked_lists: list[list[SearchHit]],
) -> list[SearchHit]:
    """Reciprocal Rank Fusion over multiple ranked lists.

    Each list contributes independently with ranks starting at 1.
    score = Σ 1/(k + rank_i) across all lists containing the hit.
    """
    scores: dict[str, float] = {}
    hit_map: dict[str, SearchHit] = {}

    for ranked_list in ranked_lists:
        for rank, hit in enumerate(ranked_list):
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


def apply_layer_weights(hits: list[SearchHit], classification: str) -> list[SearchHit]:
    """Multiply RRF scores by layer weights based on query classification."""
    weights = LAYER_WEIGHTS.get(classification, LAYER_WEIGHTS["specific"])
    weighted: list[SearchHit] = []
    for hit in hits:
        w = weights.get(hit.record_type, 1.0)
        weighted.append(
            SearchHit(
                point_id=hit.point_id,
                score=hit.score * w,
                record_type=hit.record_type,
                doc_id=hit.doc_id,
                text=hit.text,
                payload=hit.payload,
            )
        )
    weighted.sort(key=lambda h: h.score, reverse=True)
    return weighted


def apply_recency_boost(
    hits: list[SearchHit],
    now: datetime | None = None,
) -> list[SearchHit]:
    """Apply exponential-decay recency boost. 90-day half-life, max 15% influence."""
    if now is None:
        now = datetime.now(tz=UTC)

    boosted: list[SearchHit] = []
    for hit in hits:
        modified_at_str = hit.payload.get("modified_at")
        if modified_at_str:
            try:
                modified_at = datetime.fromisoformat(modified_at_str)
                if modified_at.tzinfo is None:
                    modified_at = modified_at.replace(tzinfo=UTC)
                days_since = max((now - modified_at).total_seconds() / 86400, 0)
                boost = RECENCY_MAX_BOOST * math.pow(2, -days_since / RECENCY_HALF_LIFE_DAYS)
                new_score = hit.score * (1.0 + boost)
            except (ValueError, TypeError):
                new_score = hit.score
        else:
            new_score = hit.score

        boosted.append(
            SearchHit(
                point_id=hit.point_id,
                score=new_score,
                record_type=hit.record_type,
                doc_id=hit.doc_id,
                text=hit.text,
                payload=hit.payload,
            )
        )
    boosted.sort(key=lambda h: h.score, reverse=True)
    return boosted


class RetrievalEngine:
    """Multi-stage retrieval with prefetch lanes, layer weighting, and recency boost."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        reranker: Reranker,
        citation_assembler: CitationAssembler,
        top_k_candidates: int = 30,
        top_k_final: int = 10,
        retrieval_config: RetrievalConfig | None = None,
        summarization_config: SummarizationConfig | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._reranker = reranker
        self._citations = citation_assembler
        self._top_k_candidates = top_k_candidates
        self._top_k_final = top_k_final
        self._retrieval_config = retrieval_config
        self._summarization_config = summarization_config

    def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        debug: bool = False,
    ) -> RetrievalResult:
        """Synchronous multi-stage retrieval.

        Always emits a single structured INFO log line per call with per-stage
        wall-clock timings (``search_timing ...``). When ``debug=True`` the same
        timings (as ints) plus extra diagnostic fields are also returned via
        ``RetrievalResult.debug_info``.
        """
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

        # 2. Embed query (with optional HyDE for broad queries)
        #
        # HyDE timing is measured inside _maybe_apply_hyde via an out-dict so we
        # can report the LLM-CLI cost separately from the raw embedder cost.
        hyde_meta: dict[str, float | bool] = {"hyde_ms": 0.0, "hyde_applied": False}
        t_embed_start = time.monotonic()
        hyde_vector = self._maybe_apply_hyde(query, analysis.classification, hyde_meta)
        hyde_applied = bool(hyde_meta["hyde_applied"])
        query_vector: list[float] = (
            hyde_vector if hyde_vector is not None else self._embedder.embed_query(query)
        )
        # embed_ms is the pure embedding cost: total time minus the HyDE LLM cost.
        embed_ms = (time.monotonic() - t_embed_start) * 1000 - float(hyde_meta["hyde_ms"])
        if embed_ms < 0:
            embed_ms = 0.0
        hyde_ms = float(hyde_meta["hyde_ms"])
        if debug:
            debug_info["embed_ms"] = int(embed_ms)
            debug_info["hyde_ms"] = int(hyde_ms)
            debug_info["hyde_applied"] = hyde_applied

        # 3-4. Parallel prefetch: 3 dense lanes + keyword search
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as executor:
            fut_doc_summaries = executor.submit(
                self._vector_store.query_dense,
                query_vector,
                effective_filters,
                20,
                RecordType.DOCUMENT_SUMMARY,
            )
            fut_section_summaries = executor.submit(
                self._vector_store.query_dense,
                query_vector,
                effective_filters,
                20,
                RecordType.SECTION_SUMMARY,
            )
            fut_chunks = executor.submit(
                self._vector_store.query_dense,
                query_vector,
                effective_filters,
                30,
                RecordType.CHUNK,
            )
            fut_keyword = executor.submit(
                self._vector_store.query_keyword,
                query,
                effective_filters,
                self._top_k_candidates,
            )

            dense_doc_summaries = fut_doc_summaries.result()
            dense_section_summaries = fut_section_summaries.result()
            dense_chunks = fut_chunks.result()
            keyword_hits = fut_keyword.result()

        prefetch_ms = (time.monotonic() - t0) * 1000
        if debug:
            debug_info["prefetch_ms"] = int(prefetch_ms)
            debug_info["dense_doc_summary_count"] = len(dense_doc_summaries)
            debug_info["dense_section_summary_count"] = len(dense_section_summaries)
            debug_info["dense_chunk_count"] = len(dense_chunks)
            debug_info["dense_count"] = (
                len(dense_doc_summaries) + len(dense_section_summaries) + len(dense_chunks)
            )
            debug_info["keyword_count"] = len(keyword_hits)

        # 5. RRF fusion (4 independent ranked lists)
        fused = rrf_fuse(
            [
                dense_doc_summaries,
                dense_section_summaries,
                dense_chunks,
                keyword_hits,
            ]
        )
        if debug:
            debug_info["fused_count"] = len(fused)

        # 6. Layer weighting
        weighted = apply_layer_weights(fused, analysis.classification)
        if debug:
            debug_info["layer_weights"] = LAYER_WEIGHTS.get(
                analysis.classification, LAYER_WEIGHTS["specific"]
            )

        # 7. Recency boost (before reranker so reranker has final say)
        boosted = apply_recency_boost(weighted)
        if debug:
            debug_info["recency_applied"] = True

        # 8. Rerank
        t0 = time.monotonic()
        reranked = self._reranker.rerank(query, boosted[: self._top_k_candidates], effective_top_k)
        rerank_ms = (time.monotonic() - t0) * 1000
        if debug:
            debug_info["rerank_ms"] = int(rerank_ms)

        # 9. Assemble citations (N+1 SQLite reads — worth timing)
        t0 = time.monotonic()
        cited = self._citations.assemble_citations(reranked)
        cite_ms = (time.monotonic() - t0) * 1000
        if debug:
            debug_info["cite_ms"] = int(cite_ms)

        total_ms = (time.monotonic() - start) * 1000
        if debug:
            debug_info["total_ms"] = int(total_ms)

        # Always-on structured timing log line. One line, grep-friendly, INFO
        # level. Goes to the root logger via stderr in stdio-mode MCP servers —
        # never to stdout (which would corrupt JSON-RPC framing).
        logger.info(
            "search_timing total_ms=%.1f hyde_ms=%.1f embed_ms=%.1f "
            "prefetch_ms=%.1f rerank_ms=%.1f cite_ms=%.1f "
            "results=%d classification=%s",
            round(total_ms, 1),
            round(hyde_ms, 1),
            round(embed_ms, 1),
            round(prefetch_ms, 1),
            round(rerank_ms, 1),
            round(cite_ms, 1),
            len(cited),
            analysis.classification,
        )

        return RetrievalResult(
            hits=cited,
            query_classification=analysis.classification,
            debug_info=debug_info if debug else None,
        )

    def _maybe_apply_hyde(
        self,
        query: str,
        classification: str,
        meta: dict[str, float | bool],
    ) -> list[float] | None:
        """Apply HyDE for broad queries if enabled. Returns embedding or None.

        Records wall-clock cost and applied flag into ``meta`` for the caller
        to surface in timing logs. ``meta`` is expected to have keys
        ``hyde_ms`` and ``hyde_applied``.
        """
        if classification != "broad":
            return None
        if self._retrieval_config is None or not self._retrieval_config.hyde_enabled:
            return None
        if self._summarization_config is None:
            return None

        from rag.retrieval.hyde import hyde_embed

        t0 = time.monotonic()
        result = hyde_embed(query, self._embedder, self._summarization_config)
        meta["hyde_ms"] = (time.monotonic() - t0) * 1000
        if result is not None:
            meta["hyde_applied"] = True
            logger.debug("HyDE applied for broad query")
        return result

    async def async_search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        debug: bool = False,
    ) -> RetrievalResult:
        """Async wrapper for MCP handlers -- dispatches CPU-bound ops via to_thread."""
        return await asyncio.to_thread(self.search, query, filters, top_k, debug)
