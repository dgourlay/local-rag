from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import types

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
        """Lazy-load the ONNX model and tokenizer, exporting from HuggingFace if needed."""
        if self._session is not None:
            return
        import onnxruntime as ort
        from transformers import AutoTokenizer

        model_path = str(self._config.model_path)
        onnx_path = self._resolve_onnx_path(model_path)

        if onnx_path is None:
            logger.info("Reranker ONNX model not found, exporting from HuggingFace...")
            self._export_model(model_path)
            onnx_path = self._resolve_onnx_path(model_path)
            if onnx_path is None:
                msg = (
                    f"Failed to export reranker model to {model_path}. "
                    "Install optimum: pip install 'optimum[onnxruntime]'"
                )
                raise FileNotFoundError(msg)

        providers = self._resolve_providers(ort)
        self._session = ort.InferenceSession(
            onnx_path,
            providers=providers,
        )
        # Suppress spurious "incorrect regex pattern" warning from transformers
        # (the fix_mistral_regex flag crashes on this tokenizer type)
        _tf_logger = logging.getLogger("transformers.tokenization_utils_base")
        prev_level = _tf_logger.level
        _tf_logger.setLevel(logging.ERROR)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)  # type: ignore[no-untyped-call]
        finally:
            _tf_logger.setLevel(prev_level)
        logger.info("Loaded reranker model from %s", model_path)

    def _resolve_providers(self, ort: types.ModuleType) -> list[str]:
        """Build the list of ONNX execution providers based on config."""
        providers: list[str] = []
        if self._config.use_coreml:
            available = ort.get_available_providers()
            if "CoreMLExecutionProvider" in available:
                providers.append("CoreMLExecutionProvider")
                logger.info("Using CoreMLExecutionProvider for reranker")
            else:
                logger.info(
                    "CoreML requested but not available, falling back to CPU"
                )
        providers.append("CPUExecutionProvider")
        return providers

    @staticmethod
    def _resolve_onnx_path(model_path: str) -> str | None:
        """Find the ONNX model file in the model directory."""
        if model_path.endswith(".onnx") and os.path.exists(model_path):
            return model_path
        for candidate in [
            os.path.join(model_path, "model.onnx"),
            os.path.join(model_path, "model_optimized.onnx"),
        ]:
            if os.path.exists(candidate):
                return candidate
        return None

    @staticmethod
    def _export_model(model_path: str) -> None:
        """Export the reranker from HuggingFace to ONNX format."""
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
        except ImportError:
            logger.error("optimum not installed. Install with: pip install 'optimum[onnxruntime]'")
            raise

        from transformers import AutoTokenizer

        logger.info("Downloading and converting bge-reranker-v2-m3 to ONNX...")
        model = ORTModelForSequenceClassification.from_pretrained(  # type: ignore[no-untyped-call]
            "BAAI/bge-reranker-v2-m3", export=True
        )
        model.save_pretrained(model_path)  # type: ignore[no-untyped-call]

        _tf_logger = logging.getLogger("transformers.tokenization_utils_base")
        prev_level = _tf_logger.level
        _tf_logger.setLevel(logging.ERROR)
        try:
            tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")  # type: ignore[no-untyped-call]
        finally:
            _tf_logger.setLevel(prev_level)
        tokenizer.save_pretrained(model_path)  # type: ignore[no-untyped-call]
        logger.info("Reranker model exported to %s", model_path)

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
