from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from rag.config import EmbeddingConfig

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    """Resolve device string, detecting MPS availability when device is 'auto'."""
    if device != "auto":
        return device
    try:
        import torch

        if torch.backends.mps.is_available():
            logger.info("Auto-detected MPS (Metal) device")
            return "mps"
    except (ImportError, AttributeError):
        pass
    logger.info("Auto-detected device: cpu")
    return "cpu"


class SentenceTransformerEmbedder:
    """Embedder using sentence-transformers BGE-M3 model."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            resolved_device = _resolve_device(self._config.device)
            model_kwargs: dict[str, object] = {}
            if self._config.fp16:
                import torch

                model_kwargs["dtype"] = torch.float16
            self._model = SentenceTransformer(
                self._config.model,
                cache_folder=str(self._config.cache_dir),
                device=resolved_device,
                model_kwargs=model_kwargs if model_kwargs else None,
            )
        return self._model

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self._config.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [vec.tolist() for vec in embeddings]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_batch([query])[0]

    @property
    def dimensions(self) -> int:
        return self._config.dimensions

    @property
    def model_version(self) -> str:
        return self._config.model
