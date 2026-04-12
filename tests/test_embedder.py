from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag.config import EmbeddingConfig
from rag.pipeline.embedder import SentenceTransformerEmbedder, _resolve_device


@pytest.fixture
def config() -> EmbeddingConfig:
    return EmbeddingConfig(
        model="BAAI/bge-m3",
        dimensions=1024,
        batch_size=16,
        cache_dir=Path("/tmp/test-models"),
        device="cpu",
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

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args
        assert call_kwargs[0][0] == "BAAI/bge-m3"
        assert call_kwargs[1]["cache_folder"] == str(config.cache_dir)
        assert call_kwargs[1]["device"] == "cpu"

    def test_embed_batch_returns_plain_floats(
        self, config: EmbeddingConfig, mock_model: MagicMock
    ) -> None:
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            embedder = SentenceTransformerEmbedder(config)
            results = embedder.embed_batch(["text"])

        assert isinstance(results[0][0], float)


class TestDeviceResolution:
    def test_cpu_device_passthrough(self) -> None:
        assert _resolve_device("cpu") == "cpu"

    def test_mps_device_passthrough(self) -> None:
        assert _resolve_device("mps") == "mps"

    def test_auto_resolves_to_mps_when_available(self) -> None:
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert _resolve_device("auto") == "mps"

    def test_auto_resolves_to_cpu_when_mps_unavailable(self) -> None:
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert _resolve_device("auto") == "cpu"

    def test_auto_resolves_to_cpu_when_torch_missing(self) -> None:
        with patch.dict("sys.modules", {"torch": None}):
            assert _resolve_device("auto") == "cpu"

    def test_device_passed_to_sentence_transformer(self) -> None:
        config = EmbeddingConfig(
            model="BAAI/bge-m3",
            dimensions=1024,
            batch_size=16,
            cache_dir=Path("/tmp/test-models"),
            device="mps",
        )
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 1024).astype(np.float32)
        with patch(
            "sentence_transformers.SentenceTransformer", return_value=mock_model
        ) as mock_cls:
            embedder = SentenceTransformerEmbedder(config)
            embedder.embed_batch(["test"])

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args
        assert call_kwargs[1]["device"] == "mps"

    def test_device_default_is_auto(self) -> None:
        config = EmbeddingConfig(
            cache_dir=Path("/tmp/test-models"),
        )
        assert config.device == "auto"


class TestFP16Config:
    def test_fp16_default_is_true(self) -> None:
        config = EmbeddingConfig(cache_dir=Path("/tmp/test-models"))
        assert config.fp16 is True

    def test_fp16_can_be_disabled(self) -> None:
        config = EmbeddingConfig(cache_dir=Path("/tmp/test-models"), fp16=False)
        assert config.fp16 is False

    def test_fp16_passes_dtype_to_model(self) -> None:
        import torch

        config = EmbeddingConfig(
            cache_dir=Path("/tmp/test-models"),
            fp16=True,
        )
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 1024).astype(np.float32)
        with patch(
            "sentence_transformers.SentenceTransformer", return_value=mock_model
        ) as mock_cls:
            embedder = SentenceTransformerEmbedder(config)
            embedder.embed_batch(["test"])

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["model_kwargs"]["dtype"] == torch.float16

    def test_fp16_disabled_no_model_kwargs(self) -> None:
        config = EmbeddingConfig(
            cache_dir=Path("/tmp/test-models"),
            fp16=False,
        )
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 1024).astype(np.float32)
        with patch(
            "sentence_transformers.SentenceTransformer", return_value=mock_model
        ) as mock_cls:
            embedder = SentenceTransformerEmbedder(config)
            embedder.embed_batch(["test"])

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs.get("model_kwargs") is None
