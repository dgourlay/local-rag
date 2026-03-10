from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag.config import EmbeddingConfig
from rag.pipeline.embedder import SentenceTransformerEmbedder


@pytest.fixture
def config() -> EmbeddingConfig:
    return EmbeddingConfig(
        model="BAAI/bge-m3",
        dimensions=1024,
        batch_size=16,
        cache_dir=Path("/tmp/test-models"),
    )


@pytest.fixture
def mock_model() -> MagicMock:
    model = MagicMock()
    model.encode.return_value = np.random.rand(3, 1024).astype(np.float32)
    return model


class TestSentenceTransformerEmbedder:
    def test_embed_batch_returns_correct_count(
        self, config: EmbeddingConfig, mock_model: MagicMock
    ) -> None:
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            embedder = SentenceTransformerEmbedder(config)
            results = embedder.embed_batch(["text one", "text two", "text three"])

        assert len(results) == 3
        assert all(len(vec) == 1024 for vec in results)

    def test_embed_batch_calls_encode_with_config(
        self, config: EmbeddingConfig, mock_model: MagicMock
    ) -> None:
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            embedder = SentenceTransformerEmbedder(config)
            embedder.embed_batch(["hello"])

        mock_model.encode.assert_called_once_with(
            ["hello"],
            batch_size=16,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def test_embed_query_returns_single_vector(self, config: EmbeddingConfig) -> None:
        single_mock = MagicMock()
        single_mock.encode.return_value = np.random.rand(1, 1024).astype(np.float32)
        with patch("sentence_transformers.SentenceTransformer", return_value=single_mock):
            embedder = SentenceTransformerEmbedder(config)
            result = embedder.embed_query("search query")

        assert len(result) == 1024
        assert isinstance(result, list)

    def test_dimensions_returns_config_value(self, config: EmbeddingConfig) -> None:
        embedder = SentenceTransformerEmbedder(config)
        assert embedder.dimensions == 1024

    def test_model_version_returns_config_model(self, config: EmbeddingConfig) -> None:
        embedder = SentenceTransformerEmbedder(config)
        assert embedder.model_version == "BAAI/bge-m3"

    def test_lazy_loading_model_not_loaded_at_init(self, config: EmbeddingConfig) -> None:
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            embedder = SentenceTransformerEmbedder(config)
            mock_cls.assert_not_called()
            assert embedder._model is None

    def test_model_loaded_once_on_first_embed(
        self, config: EmbeddingConfig, mock_model: MagicMock
    ) -> None:
        with patch(
            "sentence_transformers.SentenceTransformer", return_value=mock_model
        ) as mock_cls:
            embedder = SentenceTransformerEmbedder(config)
            embedder.embed_batch(["a"])
            embedder.embed_batch(["b"])

        mock_cls.assert_called_once_with(
            "BAAI/bge-m3",
            cache_folder=str(config.cache_dir),
        )

    def test_embed_batch_returns_plain_floats(
        self, config: EmbeddingConfig, mock_model: MagicMock
    ) -> None:
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            embedder = SentenceTransformerEmbedder(config)
            results = embedder.embed_batch(["text"])

        assert isinstance(results[0][0], float)
