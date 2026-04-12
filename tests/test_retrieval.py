from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from rag.retrieval.engine import (
    RECENCY_MAX_BOOST,
    RRF_K,
    RetrievalEngine,
    apply_layer_weights,
    apply_recency_boost,
    rrf_fuse,
)
from rag.retrieval.query_analyzer import analyze_query
from rag.types import (
    Citation,
    CitedEvidence,
    RecordType,
    RetrievalResult,
    SearchFilters,
    SearchHit,
)

# --- Helpers ---


def _make_hit(
    point_id: str,
    score: float = 0.9,
    doc_id: str = "doc1",
    text: str = "some text",
    record_type: RecordType = RecordType.CHUNK,
    modified_at: str = "2025-01-01",
) -> SearchHit:
    return SearchHit(
        point_id=point_id,
        score=score,
        record_type=record_type,
        doc_id=doc_id,
        text=text,
        payload={"file_path": "/docs/test.pdf", "title": "Test", "modified_at": modified_at},
    )


def _make_cited(hit: SearchHit) -> CitedEvidence:
    return CitedEvidence(
        text=hit.text,
        citation=Citation(
            title="Test",
            path="/docs/test.pdf",
            section=None,
            pages=None,
            modified="2025-01-01",
            label="test.pdf",
        ),
        score=hit.score,
        record_type=hit.record_type.value,
        doc_id=hit.doc_id,
    )


# --- RRF Fusion Tests ---


class TestRRFFuse:
    def test_single_source_scores(self) -> None:
        """Dense-only hits get correct RRF scores."""
        hits = [_make_hit("a"), _make_hit("b")]
        fused = rrf_fuse([hits])
        assert len(fused) == 2
        assert fused[0].point_id == "a"
        assert fused[0].score == pytest.approx(1.0 / (RRF_K + 1))
        assert fused[1].score == pytest.approx(1.0 / (RRF_K + 2))

    def test_both_sources_combined_score(self) -> None:
        """Hit appearing in both lists gets summed RRF score."""
        dense = [_make_hit("a"), _make_hit("b")]
        keyword = [_make_hit("b"), _make_hit("c")]
        fused = rrf_fuse([dense, keyword])

        scores = {h.point_id: h.score for h in fused}
        # "b" appears at rank 1 in dense (1/(61+1)) and rank 0 in keyword (1/(60+1))
        expected_b = 1.0 / (RRF_K + 2) + 1.0 / (RRF_K + 1)
        assert scores["b"] == pytest.approx(expected_b)
        assert fused[0].point_id == "b"  # highest combined score

    def test_deduplication(self) -> None:
        """Same point_id in both lists produces single entry."""
        dense = [_make_hit("x")]
        keyword = [_make_hit("x")]
        fused = rrf_fuse([dense, keyword])
        assert len(fused) == 1

    def test_empty_inputs(self) -> None:
        """Empty inputs return empty list."""
        assert rrf_fuse([]) == []

    def test_ordering_by_score(self) -> None:
        """Results are sorted descending by RRF score."""
        dense = [_make_hit("a"), _make_hit("b"), _make_hit("c")]
        keyword = [_make_hit("c"), _make_hit("a")]
        fused = rrf_fuse([dense, keyword])
        scores = [h.score for h in fused]
        assert scores == sorted(scores, reverse=True)


# --- Query Analyzer Tests ---


class TestQueryAnalyzer:
    def test_short_noun_query_is_navigational(self) -> None:
        result = analyze_query("machine learning")
        assert result.classification == "navigational"

    def test_long_question_without_broad_keywords_is_specific(self) -> None:
        result = analyze_query("what are the main findings of the research paper")
        assert result.classification == "specific"

    def test_overview_query_is_broad(self) -> None:
        result = analyze_query("give me an overview of the project")
        assert result.classification == "broad"

    def test_specific_query(self) -> None:
        result = analyze_query(
            "implementation details of the gradient descent optimizer in section 3.2"
        )
        assert result.classification == "specific"

    def test_folder_hint_extraction(self) -> None:
        result = analyze_query("find documents in /docs/reports")
        assert result.folder_hint == "/docs/reports"

    def test_date_hint_extraction(self) -> None:
        result = analyze_query("changes since 2025-01-15")
        assert result.date_hint == "2025-01-15"

    def test_no_hints(self) -> None:
        result = analyze_query("how does the system work")
        assert result.folder_hint is None
        assert result.date_hint is None

    def test_frozen_dataclass(self) -> None:
        result = analyze_query("test")
        with pytest.raises(AttributeError):
            result.classification = "specific"  # type: ignore[misc]


# --- Layer Weighting Tests ---


class TestApplyLayerWeights:
    def test_broad_boosts_summaries(self) -> None:
        """Broad classification boosts document and section summaries."""
        hits = [
            _make_hit("c1", score=1.0, record_type=RecordType.CHUNK),
            _make_hit("ds1", score=1.0, record_type=RecordType.DOCUMENT_SUMMARY),
            _make_hit("ss1", score=1.0, record_type=RecordType.SECTION_SUMMARY),
        ]
        weighted = apply_layer_weights(hits, "broad")
        scores = {h.point_id: h.score for h in weighted}
        assert scores["ds1"] == pytest.approx(1.5)
        assert scores["ss1"] == pytest.approx(1.3)
        assert scores["c1"] == pytest.approx(1.0)
        # document_summary should be first after sorting
        assert weighted[0].point_id == "ds1"

    def test_specific_penalizes_summaries(self) -> None:
        """Specific classification reduces summary scores."""
        hits = [
            _make_hit("c1", score=1.0, record_type=RecordType.CHUNK),
            _make_hit("ds1", score=1.0, record_type=RecordType.DOCUMENT_SUMMARY),
            _make_hit("ss1", score=1.0, record_type=RecordType.SECTION_SUMMARY),
        ]
        weighted = apply_layer_weights(hits, "specific")
        scores = {h.point_id: h.score for h in weighted}
        assert scores["ds1"] == pytest.approx(0.7)
        assert scores["ss1"] == pytest.approx(0.9)
        assert scores["c1"] == pytest.approx(1.0)
        # chunk should be first
        assert weighted[0].point_id == "c1"

    def test_unknown_classification_defaults_to_specific(self) -> None:
        """Unknown classification falls back to specific weights."""
        hits = [_make_hit("ds1", score=1.0, record_type=RecordType.DOCUMENT_SUMMARY)]
        weighted = apply_layer_weights(hits, "factual")
        assert weighted[0].score == pytest.approx(0.7)

    def test_preserves_relative_ordering_within_layer(self) -> None:
        """Within the same record_type, relative ordering is preserved."""
        hits = [
            _make_hit("c1", score=0.8, record_type=RecordType.CHUNK),
            _make_hit("c2", score=0.6, record_type=RecordType.CHUNK),
        ]
        weighted = apply_layer_weights(hits, "broad")
        assert weighted[0].point_id == "c1"
        assert weighted[1].point_id == "c2"

    def test_empty_hits(self) -> None:
        """Empty input returns empty output."""
        assert apply_layer_weights([], "broad") == []

    def test_result_sorted_descending(self) -> None:
        """Output is sorted by weighted score descending."""
        hits = [
            _make_hit("a", score=0.5, record_type=RecordType.CHUNK),
            _make_hit("b", score=0.3, record_type=RecordType.DOCUMENT_SUMMARY),
        ]
        weighted = apply_layer_weights(hits, "broad")
        scores = [h.score for h in weighted]
        assert scores == sorted(scores, reverse=True)


# --- Recency Boost Tests ---


class TestApplyRecencyBoost:
    def test_recent_document_gets_max_boost(self) -> None:
        """Document modified today gets ~15% boost."""
        now = datetime(2025, 6, 15, tzinfo=UTC)
        hits = [_make_hit("a", score=1.0, modified_at="2025-06-15T00:00:00+00:00")]
        boosted = apply_recency_boost(hits, now=now)
        # 0 days => boost = 0.15 * 2^0 = 0.15, new_score = 1.0 * 1.15
        assert boosted[0].score == pytest.approx(1.15)

    def test_90_day_old_document_gets_half_boost(self) -> None:
        """Document modified 90 days ago gets ~7.5% boost (half-life)."""
        now = datetime(2025, 6, 15, tzinfo=UTC)
        old_date = now - timedelta(days=90)
        hits = [_make_hit("a", score=1.0, modified_at=old_date.isoformat())]
        boosted = apply_recency_boost(hits, now=now)
        expected = 1.0 * (1.0 + 0.15 * 0.5)  # half-life decay
        assert boosted[0].score == pytest.approx(expected)

    def test_180_day_old_document_gets_quarter_boost(self) -> None:
        """Two half-lives => 25% of max boost."""
        now = datetime(2025, 6, 15, tzinfo=UTC)
        old_date = now - timedelta(days=180)
        hits = [_make_hit("a", score=1.0, modified_at=old_date.isoformat())]
        boosted = apply_recency_boost(hits, now=now)
        expected = 1.0 * (1.0 + 0.15 * 0.25)
        assert boosted[0].score == pytest.approx(expected)

    def test_very_old_document_negligible_boost(self) -> None:
        """Document from years ago gets near-zero boost."""
        now = datetime(2025, 6, 15, tzinfo=UTC)
        old_date = now - timedelta(days=900)  # ~10 half-lives
        hits = [_make_hit("a", score=1.0, modified_at=old_date.isoformat())]
        boosted = apply_recency_boost(hits, now=now)
        assert boosted[0].score > 1.0  # still some boost
        assert boosted[0].score < 1.001  # but negligible

    def test_missing_modified_at_no_boost(self) -> None:
        """Hit without modified_at in payload gets no boost."""
        hit = SearchHit(
            point_id="x",
            score=1.0,
            record_type=RecordType.CHUNK,
            doc_id="doc1",
            text="text",
            payload={},
        )
        boosted = apply_recency_boost([hit])
        assert boosted[0].score == pytest.approx(1.0)

    def test_invalid_date_no_boost(self) -> None:
        """Invalid date string is handled gracefully."""
        hits = [_make_hit("a", score=1.0, modified_at="not-a-date")]
        boosted = apply_recency_boost(hits)
        assert boosted[0].score == pytest.approx(1.0)

    def test_recency_reorders_results(self) -> None:
        """More recent document can overtake older one with same score."""
        now = datetime(2025, 6, 15, tzinfo=UTC)
        hits = [
            _make_hit("old", score=1.0, modified_at="2024-01-01T00:00:00+00:00"),
            _make_hit("new", score=1.0, modified_at="2025-06-15T00:00:00+00:00"),
        ]
        boosted = apply_recency_boost(hits, now=now)
        assert boosted[0].point_id == "new"

    def test_boost_max_30_percent(self) -> None:
        """Boost never exceeds 30% of original score."""
        now = datetime(2025, 6, 15, tzinfo=UTC)
        hits = [_make_hit("a", score=2.0, modified_at="2025-06-15T00:00:00+00:00")]
        boosted = apply_recency_boost(hits, now=now)
        max_allowed = 2.0 * (1.0 + RECENCY_MAX_BOOST)
        assert boosted[0].score <= max_allowed + 1e-10

    def test_naive_datetime_treated_as_utc(self) -> None:
        """Naive ISO datetime is treated as UTC."""
        now = datetime(2025, 6, 15, tzinfo=UTC)
        hits = [_make_hit("a", score=1.0, modified_at="2025-06-15")]
        boosted = apply_recency_boost(hits, now=now)
        assert boosted[0].score == pytest.approx(1.15)

    def test_empty_hits(self) -> None:
        """Empty input returns empty output."""
        assert apply_recency_boost([]) == []


# --- RetrievalEngine Tests ---


class TestRetrievalEngine:
    def _build_engine(self) -> tuple[RetrievalEngine, MagicMock, MagicMock, MagicMock, MagicMock]:
        vector_store = MagicMock()
        embedder = MagicMock()
        reranker = MagicMock()
        citation_assembler = MagicMock()

        engine = RetrievalEngine(
            vector_store=vector_store,
            embedder=embedder,
            reranker=reranker,
            citation_assembler=citation_assembler,
            top_k_candidates=30,
            top_k_final=10,
        )
        return engine, vector_store, embedder, reranker, citation_assembler

    def _setup_mocks(
        self,
        vs: MagicMock,
        embedder: MagicMock,
        reranker: MagicMock,
        citations: MagicMock,
    ) -> None:
        """Common mock setup for tests that don't care about specific results."""
        embedder.embed_query.return_value = [0.1] * 1024
        vs.query_dense.return_value = []
        vs.query_keyword.return_value = []
        reranker.rerank.return_value = []
        citations.assemble_citations.return_value = []

    def test_search_pipeline_order(self) -> None:
        """search() calls embed -> 3 dense lanes -> keyword -> rerank -> citations."""
        engine, vs, embedder, reranker, citations = self._build_engine()

        query_vec = [0.1] * 1024
        embedder.embed_query.return_value = query_vec

        dense_hits = [_make_hit("d1"), _make_hit("d2")]
        keyword_hits = [_make_hit("k1")]
        vs.query_dense.return_value = dense_hits
        vs.query_keyword.return_value = keyword_hits

        reranked = [_make_hit("d1")]
        reranker.rerank.return_value = reranked

        cited = [_make_cited(reranked[0])]
        citations.assemble_citations.return_value = cited

        result = engine.search("test query")

        embedder.embed_query.assert_called_once_with("test query")
        # 3 prefetch lanes
        assert vs.query_dense.call_count == 3
        vs.query_keyword.assert_called_once()
        reranker.rerank.assert_called_once()
        citations.assemble_citations.assert_called_once()

        assert isinstance(result, RetrievalResult)
        assert len(result.hits) == 1

    def test_prefetch_lanes_record_types(self) -> None:
        """Dense search issues 3 calls with correct record_type filters."""
        engine, vs, embedder, reranker, citations = self._build_engine()
        self._setup_mocks(vs, embedder, reranker, citations)

        engine.search("test query")

        calls = vs.query_dense.call_args_list
        assert len(calls) == 3
        # Extract (record_type, limit) pairs — order is nondeterministic due to ThreadPoolExecutor
        lane_specs = {(c[0][3], c[0][2]) for c in calls}
        assert lane_specs == {
            (RecordType.DOCUMENT_SUMMARY, 20),
            (RecordType.SECTION_SUMMARY, 20),
            (RecordType.CHUNK, 30),
        }

    def test_filters_passed_through(self) -> None:
        """Explicit filters are forwarded to vector store."""
        engine, vs, embedder, reranker, citations = self._build_engine()
        self._setup_mocks(vs, embedder, reranker, citations)

        filters = SearchFilters(folder_filter="/my/folder")
        engine.search("test", filters=filters)

        # All 3 dense calls should have the filter
        for call in vs.query_dense.call_args_list:
            assert call[0][1].folder_filter == "/my/folder"

        keyword_call_filters = vs.query_keyword.call_args[0][1]
        assert keyword_call_filters.folder_filter == "/my/folder"

    def test_debug_mode_includes_timing(self) -> None:
        """Debug mode populates debug_info with timing data."""
        engine, vs, embedder, reranker, citations = self._build_engine()
        self._setup_mocks(vs, embedder, reranker, citations)

        result = engine.search("test query", debug=True)

        assert result.debug_info is not None
        assert "embed_ms" in result.debug_info
        assert "prefetch_ms" in result.debug_info
        assert "rerank_ms" in result.debug_info
        assert "total_ms" in result.debug_info
        assert "query_classification" in result.debug_info

    def test_debug_mode_includes_new_fields(self) -> None:
        """Debug mode includes prefetch lane counts, layer weights, and recency flag."""
        engine, vs, embedder, reranker, citations = self._build_engine()
        self._setup_mocks(vs, embedder, reranker, citations)

        result = engine.search("test query", debug=True)

        assert result.debug_info is not None
        assert "dense_doc_summary_count" in result.debug_info
        assert "dense_section_summary_count" in result.debug_info
        assert "dense_chunk_count" in result.debug_info
        assert "layer_weights" in result.debug_info
        assert result.debug_info["recency_applied"] is True

    def test_debug_false_no_debug_info(self) -> None:
        """Without debug, debug_info is None."""
        engine, vs, embedder, reranker, citations = self._build_engine()
        self._setup_mocks(vs, embedder, reranker, citations)

        result = engine.search("test query", debug=False)
        assert result.debug_info is None

    def test_custom_top_k(self) -> None:
        """Custom top_k is passed to reranker."""
        engine, vs, embedder, reranker, citations = self._build_engine()
        self._setup_mocks(vs, embedder, reranker, citations)

        engine.search("test", top_k=5)

        reranker_call_top_k = reranker.rerank.call_args[0][2]
        assert reranker_call_top_k == 5

    def test_query_classification_in_result(self) -> None:
        """Result includes query classification."""
        engine, vs, embedder, reranker, citations = self._build_engine()
        self._setup_mocks(vs, embedder, reranker, citations)

        result = engine.search("what is this")
        assert result.query_classification == "broad"

    def test_async_search_wraps_sync(self) -> None:
        """async_search dispatches to search via asyncio.to_thread."""
        engine, vs, embedder, reranker, citations = self._build_engine()
        self._setup_mocks(vs, embedder, reranker, citations)

        result = asyncio.run(engine.async_search("test query"))

        assert isinstance(result, RetrievalResult)
        embedder.embed_query.assert_called_once()

    def test_search_timing_log_always_emitted(self, caplog: pytest.LogCaptureFixture) -> None:
        """Every search() call emits a structured timing INFO line.

        The line must include total_ms, hyde_ms, embed_ms, prefetch_ms,
        rerank_ms, cite_ms, results, and classification — even when debug=False.
        """
        engine, vs, embedder, reranker, citations = self._build_engine()
        self._setup_mocks(vs, embedder, reranker, citations)
        # Return two cited results so `results=2` shows up in the log line.
        hit = _make_hit("r1")
        citations.assemble_citations.return_value = [_make_cited(hit), _make_cited(hit)]

        with caplog.at_level(logging.INFO, logger="rag.retrieval.engine"):
            engine.search("what is this", debug=False)

        timing_records = [r for r in caplog.records if r.getMessage().startswith("search_timing ")]
        assert len(timing_records) == 1, (
            f"expected exactly one search_timing line, got {len(timing_records)}: "
            f"{[r.getMessage() for r in caplog.records]}"
        )

        msg = timing_records[0].getMessage()
        # All expected fields present.
        for field in (
            "total_ms=",
            "hyde_ms=",
            "embed_ms=",
            "prefetch_ms=",
            "rerank_ms=",
            "cite_ms=",
            "results=2",
            "classification=broad",
        ):
            assert field in msg, f"missing {field!r} in log line: {msg!r}"

        # Values for *_ms fields are floats with one decimal.
        for ms_field in ("total_ms", "hyde_ms", "embed_ms", "prefetch_ms", "rerank_ms", "cite_ms"):
            match = re.search(rf"{ms_field}=(\d+\.\d)", msg)
            assert match is not None, f"{ms_field} value not formatted as float: {msg!r}"

    def test_search_timing_log_emitted_in_debug_mode(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The always-on timing log is also emitted when debug=True (no regression)."""
        engine, vs, embedder, reranker, citations = self._build_engine()
        self._setup_mocks(vs, embedder, reranker, citations)

        with caplog.at_level(logging.INFO, logger="rag.retrieval.engine"):
            result = engine.search("test query", debug=True)

        # Debug info still populated.
        assert result.debug_info is not None
        assert "cite_ms" in result.debug_info
        assert "hyde_ms" in result.debug_info

        # Timing log still emitted.
        timing_lines = [
            r.getMessage() for r in caplog.records if r.getMessage().startswith("search_timing ")
        ]
        assert len(timing_lines) == 1
