from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag.config import EmbeddingConfig
from rag.pipeline.embedder import SentenceTransformerEmbedder, _resolve_device


def _make_mock_model(max_seq_length: int = 8192) -> MagicMock:
    """Mock SentenceTransformer with the surface the embedder relies on.

    Token length is approximated as one token per whitespace-separated word
    plus two special tokens, which is enough to exercise micro-batching.
    """
    model = MagicMock()
    model.max_seq_length = max_seq_length
    model.tokenizer = MagicMock(
        side_effect=lambda texts, **kwargs: {
            "input_ids": [
                [0] * min(len(t.split()) + 2, kwargs.get("max_length", max_seq_length))
                for t in texts
            ]
        }
    )
    model.encode = MagicMock(
        side_effect=lambda batch, **kwargs: np.random.rand(len(batch), 1024).astype(np.float32)
    )
    return model


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
    return _make_mock_model()


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
        single_mock = _make_mock_model()
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
        mock_model = _make_mock_model()
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
        mock_model = _make_mock_model()
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
        mock_model = _make_mock_model()
        with patch(
            "sentence_transformers.SentenceTransformer", return_value=mock_model
        ) as mock_cls:
            embedder = SentenceTransformerEmbedder(config)
            embedder.embed_batch(["test"])

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs.get("model_kwargs") is None


class TestMemoryBounds:
    """Guards against the MPS abort: an uncapped long input makes the padded
    batch enormous (attention scales with batch * seq_len^2), and Metal raises
    a fatal assertion rather than a catchable exception."""

    def test_max_seq_length_clamped_on_load(self, config: EmbeddingConfig) -> None:
        mock_model = _make_mock_model(max_seq_length=8192)
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            embedder = SentenceTransformerEmbedder(config)
            embedder.embed_batch(["text"])

        assert mock_model.max_seq_length == config.max_seq_length

    def test_model_window_left_alone_when_already_smaller(self, config: EmbeddingConfig) -> None:
        mock_model = _make_mock_model(max_seq_length=512)
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            embedder = SentenceTransformerEmbedder(config)
            embedder.embed_batch(["text"])

        assert mock_model.max_seq_length == 512

    def test_long_text_shrinks_the_batch(self, config: EmbeddingConfig) -> None:
        cfg = config.model_copy(update={"batch_size": 16, "max_batch_tokens": 2048})
        long_text = "word " * 1000  # ~1002 mock tokens
        texts = [long_text, *["short text"] * 15]
        mock_model = _make_mock_model()
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            embedder = SentenceTransformerEmbedder(cfg)
            embedder.embed_batch(texts)

        for call in mock_model.encode.call_args_list:
            batch = call[0][0]
            padded = len(batch) * max(len(t.split()) + 2 for t in batch)
            assert padded <= cfg.max_batch_tokens, (
                f"batch of {len(batch)} padded to {padded} tokens exceeds budget"
            )

    def test_short_texts_still_batch_up_to_batch_size(self, config: EmbeddingConfig) -> None:
        mock_model = _make_mock_model()
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            embedder = SentenceTransformerEmbedder(config)
            embedder.embed_batch(["short text"] * 16)

        assert mock_model.encode.call_count == 1

    def test_batch_size_is_respected(self, config: EmbeddingConfig) -> None:
        mock_model = _make_mock_model()
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            embedder = SentenceTransformerEmbedder(config)
            embedder.embed_batch(["short text"] * 40)

        assert mock_model.encode.call_count == 3
        for call in mock_model.encode.call_args_list:
            assert len(call[0][0]) <= config.batch_size

    def test_vectors_returned_in_input_order(self, config: EmbeddingConfig) -> None:
        cfg = config.model_copy(update={"batch_size": 4, "max_batch_tokens": 512})
        texts = [f"{'word ' * (i * 20)}tail{i}" for i in range(12)]
        counter = {"n": 0}

        def encode(batch: list[str], **_: object) -> np.ndarray:
            rows = []
            for text in batch:
                # Encode the text's identity in the vector so we can verify
                # the returned order maps back to the input order.
                marker = float(int(text.rsplit("tail", 1)[1]))
                rows.append(np.full(1024, marker, dtype=np.float32))
                counter["n"] += 1
            return np.stack(rows)

        mock_model = _make_mock_model()
        mock_model.encode = MagicMock(side_effect=encode)
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            embedder = SentenceTransformerEmbedder(cfg)
            results = embedder.embed_batch(texts)

        assert counter["n"] == len(texts)
        assert mock_model.encode.call_count > 1, "expected multiple micro-batches"
        assert [vec[0] for vec in results] == [float(i) for i in range(12)]

    def test_empty_input_skips_model_load(self, config: EmbeddingConfig) -> None:
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            embedder = SentenceTransformerEmbedder(config)
            assert embedder.embed_batch([]) == []
            mock_cls.assert_not_called()

    def test_encode_is_serialized_across_threads(self, config: EmbeddingConfig) -> None:
        # The semantic chunker embeds from the parser thread while the main
        # thread embeds chunks; concurrent forward passes on one MPS module
        # are unsafe.
        import threading

        active = 0
        overlaps = 0
        guard = threading.Lock()

        def encode(batch: list[str], **_: object) -> np.ndarray:
            nonlocal active, overlaps
            with guard:
                active += 1
                if active > 1:
                    overlaps += 1
            time.sleep(0.01)
            with guard:
                active -= 1
            return np.random.rand(len(batch), 1024).astype(np.float32)

        mock_model = _make_mock_model()
        mock_model.encode = MagicMock(side_effect=encode)
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            embedder = SentenceTransformerEmbedder(config)
            threads = [
                threading.Thread(target=embedder.embed_batch, args=(["text"],)) for _ in range(4)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert overlaps == 0
