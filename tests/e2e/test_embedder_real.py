"""Real-model embedder tests.

Every other test in the suite patches ``sentence_transformers.SentenceTransformer``
or uses ``FakeEmbedder``, so the embedding path -- the one that actually allocates
GPU memory -- had no real coverage. That gap is why the unbounded-sequence abort
shipped: a chunk with no sentence punctuation reached the embedder at ~5k tokens,
padded its whole batch to that length, and made Metal request a ~24 GB attention
buffer. Metal answers that with a fatal assertion and ``abort()``, so `rag watch`
died with no Python traceback and nothing to catch.

These tests load the real BGE-M3 model and are the regression guard for it.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag.config import ChunkingConfig, EmbeddingConfig
from rag.pipeline.chunker import OVERLAP_TOKENS, TARGET_TOKENS, chunk_document
from rag.pipeline.embedder import SentenceTransformerEmbedder
from rag.pipeline.normalizer import normalize
from rag.types import ParsedDocument, ParsedSection

pytestmark = pytest.mark.e2e

# Chunks are capped at the target plus the overlap carried from the previous chunk.
_MAX_CHUNK_TOKENS = TARGET_TOKENS + OVERLAP_TOKENS


def _model_is_cached(config: EmbeddingConfig) -> bool:
    """True when BGE-M3 weights are already on disk.

    Avoids turning a fresh checkout into a multi-gigabyte download.
    """
    slug = config.model.replace("/", "--")
    return any(config.cache_dir.glob(f"models--{slug}/snapshots/*/config.json"))


@pytest.fixture(scope="module")
def real_embedder() -> SentenceTransformerEmbedder:
    config = EmbeddingConfig()
    if not _model_is_cached(config):
        pytest.skip(f"{config.model} not in {config.cache_dir}; run `make download-models`")
    embedder = SentenceTransformerEmbedder(config)
    # Force the load here so a failure is attributed to setup, not to a test.
    embedder.embed_batch(["warmup"])
    return embedder


def _make_doc(text: str, title: str = "table.md") -> ParsedDocument:
    return ParsedDocument(
        doc_id="doc-real",
        title=title,
        file_type="md",
        sections=[ParsedSection(heading=None, order=0, text=text, page_start=None, page_end=None)],
        raw_content_hash="hash-real",
    )


class TestOversizedInputDoesNotAbort:
    def test_long_text_batched_with_short_ones(
        self, real_embedder: SentenceTransformerEmbedder
    ) -> None:
        """The exact shape that aborted: one huge text padding a full batch.

        Before the fix this asked Metal for a ~24 GB buffer and killed the process.
        """
        texts = ["word " * 20_000, *["short text"] * 31]
        vectors = real_embedder.embed_batch(texts)

        assert len(vectors) == len(texts)
        assert all(len(vec) == 1024 for vec in vectors)
        # normalize_embeddings=True, so every vector must be unit length.
        for vec in vectors:
            assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-2)

    def test_many_long_texts_in_one_call(self, real_embedder: SentenceTransformerEmbedder) -> None:
        """Several long texts at once: micro-batching must split them apart.

        Without the token budget these would land in a single padded batch,
        multiplying the attention allocation by the batch size.
        """
        vectors = real_embedder.embed_batch(["word " * 5_000] * 8)

        assert len(vectors) == 8
        assert all(len(vec) == 1024 for vec in vectors)


class TestUnpunctuatedContentEndToEnd:
    def test_markdown_table_chunks_and_embeds(
        self, real_embedder: SentenceTransformerEmbedder
    ) -> None:
        """A table has no `.!?`, so it used to segment into one giant chunk."""
        table = "\n".join(f"| row {i} | value {i} | note {i} |" for i in range(2_000))
        chunks = chunk_document(normalize(_make_doc(table)))

        assert len(chunks) > 1, "unpunctuated table must not collapse into one chunk"
        for chunk in chunks:
            assert chunk.token_count <= _MAX_CHUNK_TOKENS

        vectors = real_embedder.embed_batch([c.text for c in chunks])
        assert len(vectors) == len(chunks)
        assert all(len(vec) == 1024 for vec in vectors)

    def test_semantic_strategy_also_bounded(
        self, real_embedder: SentenceTransformerEmbedder
    ) -> None:
        """Semantic chunking embeds sentences too, so it needs the same cap."""
        config = ChunkingConfig(strategy="semantic")
        table = "\n".join(f"| row {i} | value {i} | note {i} |" for i in range(500))
        chunks = chunk_document(normalize(_make_doc(table)), config, real_embedder)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.token_count <= config.max_chunk_tokens

        vectors = real_embedder.embed_batch([c.text for c in chunks])
        assert len(vectors) == len(chunks)


class TestMicroBatchingCorrectness:
    def test_vectors_returned_in_input_order(
        self, real_embedder: SentenceTransformerEmbedder
    ) -> None:
        """Mixed lengths are reordered internally; results must come back in order."""
        texts = [
            "alpha short",
            "beta " * 400,
            "gamma short",
            "delta " * 200,
            "epsilon short",
        ]
        batched = real_embedder.embed_batch(texts)
        individually = [real_embedder.embed_batch([text])[0] for text in texts]

        for i, (got, want) in enumerate(zip(batched, individually, strict=True)):
            assert np.allclose(got, want, atol=1e-2), f"vector {i} does not match text {i}"

    def test_query_and_batch_agree(self, real_embedder: SentenceTransformerEmbedder) -> None:
        query = "quarterly revenue growth in the retail segment"
        assert np.allclose(
            real_embedder.embed_query(query),
            real_embedder.embed_batch([query])[0],
            atol=1e-2,
        )

    def test_semantically_close_texts_score_higher(
        self, real_embedder: SentenceTransformerEmbedder
    ) -> None:
        """Sanity-check the vectors carry real meaning, not just the right shape."""
        anchor, near, far = real_embedder.embed_batch(
            [
                "The company grew revenue by 12% this quarter.",
                "Quarterly sales increased twelve percent.",
                "Preheat the oven to 200 degrees and butter the tin.",
            ]
        )
        assert np.dot(anchor, near) > np.dot(anchor, far)


class TestConfiguredLimitsApplied:
    def test_model_window_clamped_to_config(
        self, real_embedder: SentenceTransformerEmbedder
    ) -> None:
        """BGE-M3 ships an 8192 window; the config must clamp it down."""
        config = EmbeddingConfig()
        assert config.max_seq_length < 8192
        model = real_embedder._get_model()
        assert model.max_seq_length == config.max_seq_length

    def test_padded_tokens_per_forward_pass_stay_within_budget(
        self, real_embedder: SentenceTransformerEmbedder
    ) -> None:
        """The budget that actually bounds attention memory."""
        config = EmbeddingConfig()
        model = real_embedder._get_model()
        texts = ["word " * 600, *["short text"] * 40]

        lengths = real_embedder._token_lengths(model, texts)
        order = sorted(range(len(texts)), key=lambda i: lengths[i], reverse=True)
        for batch in real_embedder._micro_batches(order, lengths):
            padded = len(batch) * max(lengths[i] for i in batch)
            assert padded <= config.max_batch_tokens
            assert len(batch) <= config.batch_size
