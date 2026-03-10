from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from rag.config import RerankerConfig
from rag.retrieval.reranker import OnnxReranker
from rag.types import RecordType, SearchHit


@pytest.fixture()
def config() -> RerankerConfig:
    return RerankerConfig(model_path=Path("/tmp/fake-model"))


@pytest.fixture()
def sample_hits() -> list[SearchHit]:
    return [
        SearchHit(
            point_id=f"p{i}",
            score=0.5,
            record_type=RecordType.CHUNK,
            doc_id=f"doc{i}",
            text=f"text {i}",
            payload={"key": f"val{i}"},
        )
        for i in range(5)
    ]


def _make_mock_session(scores: list[float]) -> MagicMock:
    session = MagicMock()
    input_ids_input = MagicMock()
    input_ids_input.name = "input_ids"
    attention_mask_input = MagicMock()
    attention_mask_input.name = "attention_mask"
    session.get_inputs.return_value = [input_ids_input, attention_mask_input]
    session.run.return_value = [np.array(scores)]
    return session


def _make_mock_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    result = MagicMock()
    result.__contains__ = MagicMock(return_value=True)
    result.__getitem__ = MagicMock(return_value=np.zeros((1, 10)))
    tokenizer.return_value = result
    return tokenizer


def _patch_loading(reranker: OnnxReranker, scores: list[float]) -> None:
    """Inject mock session and tokenizer into the reranker."""
    reranker._session = _make_mock_session(scores)
    reranker._tokenizer = _make_mock_tokenizer()


class TestOnnxReranker:
    def test_rerank_empty_returns_empty(self, config: RerankerConfig) -> None:
        reranker = OnnxReranker(config)
        result = reranker.rerank("query", [], top_k=5)
        assert result == []

    def test_rerank_single_candidate_returns_it(
        self, config: RerankerConfig, sample_hits: list[SearchHit]
    ) -> None:
        reranker = OnnxReranker(config)
        single = [sample_hits[0]]
        result = reranker.rerank("query", single, top_k=5)
        assert len(result) == 1
        assert result[0].point_id == "p0"

    def test_rerank_reorders_by_score(
        self, config: RerankerConfig, sample_hits: list[SearchHit]
    ) -> None:
        reranker = OnnxReranker(config)
        candidates = sample_hits[:3]
        # Logits: higher logit -> higher sigmoid score
        # p0=0.1 -> ~0.525, p1=5.0 -> ~0.993, p2=-2.0 -> ~0.119
        _patch_loading(reranker, [0.1, 5.0, -2.0])

        result = reranker.rerank("query", candidates, top_k=3)
        assert len(result) == 3
        assert result[0].point_id == "p1"
        assert result[1].point_id == "p0"
        assert result[2].point_id == "p2"

    def test_top_k_respected(self, config: RerankerConfig, sample_hits: list[SearchHit]) -> None:
        reranker = OnnxReranker(config)
        candidates = sample_hits[:4]
        _patch_loading(reranker, [1.0, 3.0, 2.0, 0.5])

        result = reranker.rerank("query", candidates, top_k=2)
        assert len(result) == 2
        assert result[0].point_id == "p1"
        assert result[1].point_id == "p2"

    def test_scores_updated_in_results(
        self, config: RerankerConfig, sample_hits: list[SearchHit]
    ) -> None:
        reranker = OnnxReranker(config)
        candidates = sample_hits[:2]
        _patch_loading(reranker, [3.0, -3.0])

        result = reranker.rerank("query", candidates, top_k=2)
        assert result[0].score == pytest.approx(0.9526, abs=0.01)
        assert result[1].score == pytest.approx(0.0474, abs=0.01)
        assert result[0].score != 0.5

    def test_lazy_loading_no_model_loaded_on_init(self, config: RerankerConfig) -> None:
        reranker = OnnxReranker(config)
        assert reranker._session is None
        assert reranker._tokenizer is None

    def test_ensure_loaded_called_on_rerank(
        self, config: RerankerConfig, sample_hits: list[SearchHit]
    ) -> None:
        reranker = OnnxReranker(config)
        candidates = sample_hits[:2]
        _patch_loading(reranker, [1.0, 0.5])

        reranker.rerank("query", candidates, top_k=2)
        reranker._session.run.assert_called_once()  # type: ignore[union-attr]

    def test_2d_scores_squeezed(
        self, config: RerankerConfig, sample_hits: list[SearchHit]
    ) -> None:
        reranker = OnnxReranker(config)
        candidates = sample_hits[:2]
        reranker._tokenizer = _make_mock_tokenizer()
        session = _make_mock_session([])
        session.run.return_value = [np.array([[2.0], [-1.0]])]
        reranker._session = session

        result = reranker.rerank("query", candidates, top_k=2)
        assert len(result) == 2
        assert result[0].point_id == "p0"  # logit 2.0 > -1.0
