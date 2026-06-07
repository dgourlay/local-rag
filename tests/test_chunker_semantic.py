from __future__ import annotations

import numpy as np
import pytest

from rag.config import ChunkingConfig
from rag.pipeline.chunker import chunk_document, chunk_document_fixed
from rag.pipeline.chunker_semantic import (
    chunk_document_semantic,
    detect_boundaries,
    extract_code_blocks,
    restore_code_blocks,
    segment_sentences,
)
from rag.types import NormalizedDocument, ParsedSection

# --- Mock Embedder ---


class MockEmbedder:
    """Mock embedder that produces deterministic vectors for testing.

    Sentences containing the same topic keyword get similar vectors.
    Different topics get orthogonal vectors.
    """

    def __init__(self, dims: int = 1024) -> None:
        self._dims = dims
        self._rng = np.random.RandomState(42)
        # Pre-generate topic base vectors (orthogonal-ish)
        self._topic_vectors: dict[str, np.ndarray] = {}

    @property
    def dimensions(self) -> int:
        return self._dims

    @property
    def model_version(self) -> str:
        return "mock-v1"

    def _get_topic_vector(self, topic: str) -> np.ndarray:
        if topic not in self._topic_vectors:
            vec = self._rng.randn(self._dims).astype(np.float32)
            vec /= np.linalg.norm(vec)
            self._topic_vectors[topic] = vec
        return self._topic_vectors[topic]

    def _embed_single(self, text: str) -> list[float]:
        """Embed a single text. Detects topic keywords for clustering."""
        text_lower = text.lower()
        ml_kw = ["machine learning", "neural", "algorithm", "model", "training", "ai"]
        cook_kw = ["cooking", "recipe", "ingredient", "bake", "kitchen", "food"]
        astro_kw = ["astronomy", "planet", "star", "galaxy", "orbit", "space"]
        if any(kw in text_lower for kw in ml_kw):
            base = self._get_topic_vector("ml")
        elif any(kw in text_lower for kw in cook_kw):
            base = self._get_topic_vector("cooking")
        elif any(kw in text_lower for kw in astro_kw):
            base = self._get_topic_vector("astronomy")
        else:
            base = self._get_topic_vector("general")

        # Very small perturbation so within-topic similarity stays high (~0.99)
        # In 1024 dims, noise_scale=0.003 gives ~0.99 cosine similarity to base
        noise = self._rng.randn(self._dims).astype(np.float32) * 0.003
        vec = base + noise
        vec /= np.linalg.norm(vec)
        return list(vec.tolist())

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_single(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed_single(query)


# --- Helpers ---


def _make_doc(
    sections: list[ParsedSection],
    doc_id: str = "doc-001",
    title: str = "test.pdf",
) -> NormalizedDocument:
    return NormalizedDocument(
        doc_id=doc_id,
        title=title,
        file_type="pdf",
        sections=sections,
        normalized_content_hash="abc123",
        raw_content_hash="def456",
    )


def _make_section(
    text: str,
    order: int = 0,
    heading: str | None = "Test Section",
    page_start: int | None = 1,
    page_end: int | None = 1,
) -> ParsedSection:
    return ParsedSection(
        heading=heading,
        order=order,
        text=text,
        page_start=page_start,
        page_end=page_end,
    )


def _semantic_config(**overrides: object) -> ChunkingConfig:
    return ChunkingConfig(strategy="semantic", **overrides)


# --- Code Block Tests (spec §12.2) ---


class TestCodeBlockExtractRestore:
    def test_roundtrip_preserves_code_blocks(self) -> None:
        text = "Before code.\n```python\nprint('hello')\n```\nAfter code."
        cleaned, blocks = extract_code_blocks(text)
        assert "```" not in cleaned
        assert len(blocks) == 1
        assert blocks[0] == "```python\nprint('hello')\n```"
        restored = restore_code_blocks(cleaned, blocks)
        assert restored == text

    def test_multiple_code_blocks(self) -> None:
        text = "A.\n```\ncode1\n```\nB.\n```\ncode2\n```\nC."
        cleaned, blocks = extract_code_blocks(text)
        assert len(blocks) == 2
        restored = restore_code_blocks(cleaned, blocks)
        assert restored == text

    def test_nested_code_blocks(self) -> None:
        # Nested triple-backticks — regex matches first ``` to first closing ```
        text = "Before.\n```\nouter\n```\nMiddle.\n```\ninner\n```\nAfter."
        cleaned, blocks = extract_code_blocks(text)
        assert len(blocks) == 2
        restored = restore_code_blocks(cleaned, blocks)
        assert restored == text

    def test_placeholder_in_original(self) -> None:
        # Text containing null bytes should not collide with placeholders
        text = "Normal text with \x00 null bytes.\n```\ncode\n```\nEnd."
        cleaned, blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        restored = restore_code_blocks(cleaned, blocks)
        assert restored == text

    def test_no_code_blocks(self) -> None:
        text = "Just regular text. No code blocks here."
        cleaned, blocks = extract_code_blocks(text)
        assert cleaned == text
        assert blocks == []


# --- Sentence Segmentation Tests ---


class TestSegmentSentences:
    def test_basic_sentences(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        sentences = segment_sentences(text)
        assert len(sentences) >= 2

    def test_empty_text(self) -> None:
        assert segment_sentences("") == []
        assert segment_sentences("   ") == []


# --- Boundary Detection Tests ---


class TestDetectBoundaries:
    def test_always_starts_with_zero(self) -> None:
        # 5 identical vectors — no boundaries beyond start
        embeddings = [np.ones(10).tolist()] * 5
        boundaries = detect_boundaries(embeddings, 0.35, 15)
        assert boundaries[0] == 0

    def test_identical_vectors_no_split(self) -> None:
        vec = (np.ones(10) / np.sqrt(10)).tolist()
        embeddings = [vec] * 5
        boundaries = detect_boundaries(embeddings, 0.35, 15)
        assert boundaries == [0]

    def test_orthogonal_vectors_split(self) -> None:
        # Create clearly orthogonal vectors for different "topics"
        rng = np.random.RandomState(99)
        topic_a = rng.randn(64).astype(np.float32)
        topic_a /= np.linalg.norm(topic_a)
        topic_b = rng.randn(64).astype(np.float32)
        topic_b /= np.linalg.norm(topic_b)
        # Make sure they're actually dissimilar
        assert abs(float(np.dot(topic_a, topic_b))) < 0.5

        embeddings = [topic_a.tolist()] * 4 + [topic_b.tolist()] * 4
        boundaries = detect_boundaries(embeddings, 0.35, 15)
        assert len(boundaries) >= 2
        # Boundary should be at or near index 4
        assert 4 in boundaries


# --- Main Semantic Chunker Tests (spec §12.1) ---


class TestSingleTopicNoSplit:
    def test_coherent_paragraph_single_chunk(self) -> None:
        text = (
            "Machine learning algorithms process data to find patterns. "
            "Neural networks are a type of machine learning model. "
            "Training a neural network involves adjusting model weights. "
            "The algorithm converges when the training loss decreases."
        )
        doc = _make_doc(sections=[_make_section(text)])
        embedder = MockEmbedder()
        chunks = chunk_document_semantic(doc, _semantic_config(), embedder)
        assert len(chunks) == 1


class TestTopicShiftDetected:
    def test_two_topics_produce_two_chunks(self) -> None:
        ml_sents = [
            "Machine learning algorithms process large datasets.",
            "Neural networks are a machine learning model type.",
            "Training a neural network adjusts model weights.",
            "The algorithm converges as training loss decreases.",
            "Deep learning models use multiple neural layers.",
            "Gradient descent drives machine learning training.",
            "Transfer learning reuses pretrained neural networks.",
            "Model evaluation uses cross-validation of algorithms.",
            "Supervised machine learning requires labeled training data.",
        ]
        cook_sents = [
            "Cooking a recipe requires ingredient measurements.",
            "Baking bread involves flour and yeast in a kitchen.",
            "Food preparation needs proper kitchen equipment.",
            "The recipe calls for specific cooking temperatures.",
            "Ingredient quality affects the final food taste.",
            "Kitchen management coordinates food preparation areas.",
            "Cooking techniques require precise temperature control.",
            "Recipe development demands ingredient knowledge.",
            "Professional baking requires calibrated kitchen ovens.",
        ]
        text = " ".join(ml_sents) + " " + " ".join(cook_sents)
        doc = _make_doc(sections=[_make_section(text)])
        embedder = MockEmbedder()
        chunks = chunk_document_semantic(doc, _semantic_config(), embedder)
        assert len(chunks) >= 2


class TestMinChunkMerge:
    def test_small_chunk_merged_with_neighbor(self) -> None:
        # A very short section between two longer ones should be merged
        text = (
            "Machine learning is a field of artificial intelligence. "
            "Neural networks learn from training data. "
            "The algorithm improves with more training examples. "
            "Ok. "  # tiny sentence
            "Cooking requires following a recipe carefully. "
            "Food preparation involves measuring ingredients. "
            "Baking bread needs flour and yeast in the kitchen."
        )
        doc = _make_doc(sections=[_make_section(text)])
        embedder = MockEmbedder()
        chunks = chunk_document_semantic(doc, _semantic_config(), embedder)
        # All chunks should have at least MIN_CHUNK_TOKENS (64)
        # (unless the entire section is smaller)
        for chunk in chunks:
            if chunk.token_count < 64:
                # Only acceptable if this is the only chunk
                assert len(chunks) == 1


class TestMaxChunkSplit:
    def test_oversized_chunk_split(self) -> None:
        # Create a very long monotopic section that exceeds max_chunk_tokens
        sentences = [f"Machine learning algorithm {i} processes data." for i in range(60)]
        text = " ".join(sentences)
        doc = _make_doc(sections=[_make_section(text)])
        embedder = MockEmbedder()
        config = _semantic_config(max_chunk_tokens=256)
        chunks = chunk_document_semantic(doc, config, embedder)
        assert len(chunks) >= 2


class TestFewSentencesSkip:
    def test_single_sentence_no_embedding(self) -> None:
        text = "This is the only sentence in this section."
        doc = _make_doc(sections=[_make_section(text)])
        embedder = MockEmbedder()
        chunks = chunk_document_semantic(doc, _semantic_config(), embedder)
        assert len(chunks) == 1

    def test_two_sentences_no_embedding(self) -> None:
        text = "First sentence here. Second sentence here."
        doc = _make_doc(sections=[_make_section(text)])
        embedder = MockEmbedder()
        chunks = chunk_document_semantic(doc, _semantic_config(), embedder)
        assert len(chunks) == 1


class TestCodeBlockPreserved:
    def test_code_blocks_survive_chunking(self) -> None:
        text = (
            "Machine learning uses algorithms for training. "
            "Neural networks process data in layers. "
            "Here is an example:\n"
            "```python\nimport torch\nmodel = torch.nn.Linear(10, 5)\n```\n"
            "The model above is a simple neural network."
        )
        doc = _make_doc(sections=[_make_section(text)])
        embedder = MockEmbedder()
        chunks = chunk_document_semantic(doc, _semantic_config(), embedder)
        # Code block should appear in one of the chunks
        all_text = " ".join(c.text for c in chunks)
        assert "```python" in all_text
        assert "import torch" in all_text


class TestCodeBlockNotSplit:
    def test_code_block_stays_in_one_chunk(self) -> None:
        lines = "\n".join(f"line_{i} = {i}" for i in range(20))
        code = f"```python\n{lines}\n```"
        text = f"Before the code block. Some intro text here.\n{code}\nAfter the code."
        doc = _make_doc(sections=[_make_section(text)])
        embedder = MockEmbedder()
        chunks = chunk_document_semantic(doc, _semantic_config(), embedder)
        # The code block should not be split across chunks
        for chunk in chunks:
            if "```python" in chunk.text:
                assert "```" in chunk.text
                # Count opening and closing backticks — should be balanced
                backtick_count = chunk.text.count("```")
                assert backtick_count % 2 == 0


class TestDeterministicIds:
    def test_same_input_same_ids(self) -> None:
        text = (
            "Machine learning algorithms process data. "
            "Neural networks learn from training examples. "
            "The model converges after sufficient training."
        )
        doc = _make_doc(sections=[_make_section(text)])
        embedder = MockEmbedder()
        config = _semantic_config()
        chunks_a = chunk_document_semantic(doc, config, embedder)
        # Reset embedder RNG for determinism
        embedder2 = MockEmbedder()
        chunks_b = chunk_document_semantic(doc, config, embedder2)
        assert len(chunks_a) == len(chunks_b)
        for a, b in zip(chunks_a, chunks_b, strict=True):
            assert a.chunk_id == b.chunk_id


class TestSectionBoundaryRespected:
    def test_chunks_never_span_sections(self) -> None:
        ml = "Machine learning processes data. Neural networks are models. The algorithm trains."
        cook = "Cooking requires recipes. Baking needs ingredients. Food preparation is important."
        doc = _make_doc(
            sections=[
                _make_section(
                    ml,
                    order=0,
                    heading="ML Section",
                ),
                _make_section(
                    cook,
                    order=1,
                    heading="Cooking Section",
                ),
            ]
        )
        embedder = MockEmbedder()
        chunks = chunk_document_semantic(doc, _semantic_config(), embedder)
        section_ids = {c.section_id for c in chunks}
        assert len(section_ids) == 2
        # Each chunk should belong to only one section
        for chunk in chunks:
            assert chunk.section_id is not None


class TestEmptySection:
    def test_empty_text_no_chunks(self) -> None:
        doc = _make_doc(sections=[_make_section("", order=0)])
        embedder = MockEmbedder()
        chunks = chunk_document_semantic(doc, _semantic_config(), embedder)
        assert chunks == []

    def test_whitespace_only_no_chunks(self) -> None:
        doc = _make_doc(sections=[_make_section("   \n\t  ", order=0)])
        embedder = MockEmbedder()
        chunks = chunk_document_semantic(doc, _semantic_config(), embedder)
        assert chunks == []


# --- Dispatch Tests ---


class TestStrategyDispatch:
    def test_dispatch_to_fixed_by_default(self) -> None:
        doc = _make_doc(sections=[_make_section("Some content here.")])
        # No config = fixed strategy (backward compatible)
        chunks = chunk_document(doc)
        assert len(chunks) >= 1

    def test_dispatch_to_fixed_explicit(self) -> None:
        doc = _make_doc(sections=[_make_section("Some content here.")])
        config = ChunkingConfig(strategy="fixed")
        chunks = chunk_document(doc, config)
        assert len(chunks) >= 1

    def test_dispatch_to_semantic(self) -> None:
        doc = _make_doc(
            sections=[
                _make_section(
                    "Machine learning algorithms. Neural networks learn. Training improves models."
                )
            ]
        )
        embedder = MockEmbedder()
        config = _semantic_config()
        chunks = chunk_document(doc, config, embedder)
        assert len(chunks) >= 1

    def test_semantic_without_embedder_raises(self) -> None:
        doc = _make_doc(sections=[_make_section("Some content.")])
        config = _semantic_config()
        with pytest.raises(ValueError, match="Embedder required"):
            chunk_document(doc, config)


class TestFixedStrategyUnchanged:
    def test_fixed_output_identical(self) -> None:
        """Fixed strategy via dispatcher produces identical output to direct call."""
        text = " ".join(["This is a test sentence with several words."] * 50)
        doc = _make_doc(sections=[_make_section(text)])
        direct = chunk_document_fixed(doc)
        via_dispatch = chunk_document(doc, ChunkingConfig(strategy="fixed"))
        assert len(direct) == len(via_dispatch)
        for d, v in zip(direct, via_dispatch, strict=True):
            assert d.chunk_id == v.chunk_id
            assert d.text == v.text
            assert d.token_count == v.token_count
