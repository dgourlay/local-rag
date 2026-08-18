from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

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
    """Embedder using sentence-transformers BGE-M3 model.

    Encoding is serialized behind a lock: the model is a single shared GPU
    resource and the indexing pipeline touches it from both the parser thread
    (semantic chunking) and the main thread (chunk embedding).
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            resolved_device = _resolve_device(self._config.device)
            model_kwargs: dict[str, object] = {}
            if self._config.fp16:
                import torch

                model_kwargs["dtype"] = torch.float16
            model = SentenceTransformer(
                self._config.model,
                cache_folder=str(self._config.cache_dir),
                device=resolved_device,
                model_kwargs=model_kwargs if model_kwargs else None,
            )
            # Clamp the model's own window (8192 for BGE-M3) to our budget.
            if model.max_seq_length > self._config.max_seq_length:
                logger.info(
                    "Capping embedder max_seq_length %d -> %d",
                    model.max_seq_length,
                    self._config.max_seq_length,
                )
                model.max_seq_length = self._config.max_seq_length
            self._model = model
        return self._model

    def _token_lengths(self, model: SentenceTransformer, texts: list[str]) -> list[int]:
        """Padded sequence length each text will occupy, clamped to the window."""
        encoded = model.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=self._config.max_seq_length,
        )
        return [len(ids) for ids in encoded["input_ids"]]

    def _micro_batches(self, order: list[int], lengths: list[int]) -> Iterator[list[int]]:
        """Group text indices so padded tokens per forward pass stay bounded.

        A batch is padded to its longest member, so the cost driver is
        ``len(batch) * max(lengths in batch)``. Keeping that under
        ``max_batch_tokens`` bounds peak attention memory regardless of how
        long an individual input is.
        """
        budget = self._config.max_batch_tokens
        batch: list[int] = []
        batch_max = 0
        for idx in order:
            candidate_max = max(batch_max, lengths[idx])
            if batch and (
                len(batch) + 1 > self._config.batch_size
                or (len(batch) + 1) * candidate_max > budget
            ):
                yield batch
                batch = []
                candidate_max = lengths[idx]
            batch.append(idx)
            batch_max = candidate_max
        if batch:
            yield batch

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = [[] for _ in texts]
        with self._lock:
            model = self._get_model()
            lengths = self._token_lengths(model, texts)
            # Longest first, so each micro-batch groups similar lengths and
            # padding waste stays low.
            order = sorted(range(len(texts)), key=lambda i: lengths[i], reverse=True)
            for batch in self._micro_batches(order, lengths):
                embeddings = model.encode(
                    [texts[i] for i in batch],
                    batch_size=self._config.batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                for idx, vec in zip(batch, embeddings, strict=True):
                    vectors[idx] = vec.tolist()
        return vectors

    def embed_query(self, query: str) -> list[float]:
        return self.embed_batch([query])[0]

    @property
    def dimensions(self) -> int:
        return self._config.dimensions

    @property
    def model_version(self) -> str:
        return self._config.model
