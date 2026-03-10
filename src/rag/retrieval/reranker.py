from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import onnxruntime as ort

    from rag.config import RerankerConfig
    from rag.types import SearchHit

logger = logging.getLogger(__name__)


class OnnxReranker:
    """Cross-encoder reranker using bge-reranker-v2-m3 ONNX model."""

    def __init__(self, config: RerankerConfig) -> None:
        self._config = config
        self._session: ort.InferenceSession | None = None
        self._tokenizer: Any = None

    def _ensure_loaded(self) -> None:
        """Lazy-load the ONNX model and tokenizer."""
        if self._session is not None:
            return
        import onnxruntime as ort
        from transformers import AutoTokenizer

        model_path = str(self._config.model_path)
        onnx_path = model_path
        if not onnx_path.endswith(".onnx"):
            for candidate in [
                os.path.join(model_path, "model.onnx"),
                os.path.join(model_path, "model_optimized.onnx"),
            ]:
                if os.path.exists(candidate):
                    onnx_path = candidate
                    break

        self._session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)  # type: ignore[no-untyped-call]
        logger.info("Loaded reranker model from %s", model_path)

    def rerank(self, query: str, candidates: list[SearchHit], top_k: int) -> list[SearchHit]:
        """Rerank candidates by cross-encoder relevance. Returns top_k sorted by score."""
        from rag.types import SearchHit as SearchHitCls

        if not candidates:
            return []

        if len(candidates) <= 1:
            return candidates[:top_k]

        self._ensure_loaded()
        assert self._session is not None
        assert self._tokenizer is not None

        import numpy as np

        pairs = [(query, hit.text) for hit in candidates]

        encoded: Any = self._tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )

        input_feed: dict[str, Any] = {
            name: encoded[name]
            for name in [inp.name for inp in self._session.get_inputs()]
            if name in encoded
        }
        outputs: list[Any] = self._session.run(None, input_feed)
        scores: Any = outputs[0]

        if scores.ndim > 1:
            scores = scores.squeeze(-1)

        # Sigmoid for probability-like scores
        scores = 1.0 / (1.0 + np.exp(-scores))

        scored = list(zip(candidates, scores.tolist(), strict=True))
        scored.sort(key=lambda x: x[1], reverse=True)

        results: list[SearchHit] = []
        for hit, score in scored[:top_k]:
            results.append(
                SearchHitCls(
                    point_id=hit.point_id,
                    score=score,
                    record_type=hit.record_type,
                    doc_id=hit.doc_id,
                    text=hit.text,
                    payload=hit.payload,
                )
            )
        return results
