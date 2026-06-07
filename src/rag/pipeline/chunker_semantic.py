from __future__ import annotations

import math
import re
import uuid
from typing import TYPE_CHECKING

import numpy as np

from rag.pipeline.chunker import _make_chunk, count_tokens
from rag.types import NAMESPACE_RAG, Chunk, NormalizedDocument, ParsedSection

if TYPE_CHECKING:
    from rag.config import ChunkingConfig
    from rag.protocols import Embedder

# --- Constants ---
MIN_CHUNK_TOKENS = 64
MAX_CHUNK_SENTENCES = 15
_SENTENCE_BATCH_SIZE = 128

# --- Code block extraction ---
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_PLACEHOLDER_FMT = "\x00CODE_BLOCK_{}\x00"
_PLACEHOLDER_RE = re.compile(r"\x00CODE_BLOCK_(\d+)\x00")


def extract_code_blocks(text: str) -> tuple[str, list[str]]:
    """Replace fenced code blocks with null-delimited placeholders."""
    blocks: list[str] = []

    def replacer(match: re.Match[str]) -> str:
        blocks.append(match.group(0))
        return _PLACEHOLDER_FMT.format(len(blocks) - 1)

    cleaned = _CODE_BLOCK_RE.sub(replacer, text)
    return cleaned, blocks


def restore_code_blocks(chunk_text: str, blocks: list[str]) -> str:
    """Replace placeholders back with original code blocks."""

    def replacer(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        return blocks[idx] if idx < len(blocks) else match.group(0)

    return _PLACEHOLDER_RE.sub(replacer, chunk_text)


# --- Sentence segmentation ---

# Common abbreviations that should not trigger sentence splits
_ABBREVIATIONS = frozenset(
    {
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "sr",
        "jr",
        "st",
        "ave",
        "blvd",
        "dept",
        "est",
        "vol",
        "vs",
        "etc",
        "inc",
        "ltd",
        "corp",
        "govt",
        "approx",
        "fig",
        "gen",
        "no",
        "ref",
        "rev",
        "sgt",
        "pvt",
        "e.g",
        "i.e",
        "al",
        "cf",
    }
)

# Regex: split at sentence-ending punctuation followed by whitespace and
# an uppercase letter or quote, but not after known abbreviations or decimals.
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])"  # lookbehind: sentence-ending punctuation
    r"(?<!\b\d\.)"  # negative lookbehind: not a decimal like "3."
    r"\s+"  # whitespace gap
    r"(?=[A-Z\"\'\u201c\u201d])"  # lookahead: uppercase or opening quote
)


def segment_sentences(text: str) -> list[str]:
    """Split text into sentences using rule-based segmentation.

    Handles abbreviations (Dr., Mr., etc.), decimal numbers, and ellipses
    better than the naive regex in the fixed chunker.
    """
    if not text.strip():
        return []

    # First pass: split on obvious sentence boundaries
    candidates = _SENTENCE_SPLIT_RE.split(text)

    # Second pass: rejoin splits that were after abbreviations
    sentences: list[str] = []
    for candidate in candidates:
        stripped = candidate.strip()
        if not stripped:
            continue

        # Check if previous sentence ended with an abbreviation
        if sentences:
            prev = sentences[-1].rstrip()
            # Get last word before the period
            last_word_match = re.search(r"(\w+)\.\s*$", prev)
            if last_word_match:
                last_word = last_word_match.group(1).lower()
                if last_word in _ABBREVIATIONS:
                    sentences[-1] = prev + " " + stripped
                    continue

        sentences.append(stripped)

    return [s for s in sentences if s.strip()]


# --- Boundary detection ---


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity via dot product (vectors assumed L2-normalized)."""
    return float(np.dot(a, b))


def detect_boundaries(
    sentence_embeddings: list[list[float]],
    similarity_threshold: float,
    max_chunk_sentences: int,
) -> list[int]:
    """Return sentence indices where a new chunk should start.

    Always includes index 0 (first sentence starts first chunk).
    Uses simplified Max-Min algorithm with sigmoid growth pressure.
    """
    boundaries: list[int] = [0]
    current_chunk_start = 0

    for i in range(1, len(sentence_embeddings)):
        chunk_size = i - current_chunk_start

        # Sigmoid growth pressure
        growth_pressure = 1.0 / (1.0 + math.exp(-(chunk_size - max_chunk_sentences) / 2))

        # Dynamic threshold rises with growth pressure
        dynamic_threshold = similarity_threshold + (1.0 - similarity_threshold) * growth_pressure

        # Max similarity: best cosine match to any sentence in current chunk
        max_sim = max(
            cosine_similarity(sentence_embeddings[i], sentence_embeddings[j])
            for j in range(current_chunk_start, i)
        )

        if max_sim < dynamic_threshold:
            boundaries.append(i)
            current_chunk_start = i

    return boundaries


# --- Guardrails ---


def _merge_small_chunks(
    sentence_groups: list[list[str]],
) -> list[list[str]]:
    """Merge chunks with fewer than MIN_CHUNK_TOKENS into neighbors."""
    if len(sentence_groups) <= 1:
        return sentence_groups

    merged: list[list[str]] = []
    for group in sentence_groups:
        group_text = " ".join(group)
        group_tokens = count_tokens(group_text)

        if group_tokens < MIN_CHUNK_TOKENS and merged:
            # Merge with previous
            merged[-1].extend(group)
        elif group_tokens < MIN_CHUNK_TOKENS and not merged:
            # First group is too small — will merge with next
            merged.append(group)
        else:
            merged.append(group)

    # Final pass: if first group is still too small, merge with next
    if len(merged) > 1:
        first_text = " ".join(merged[0])
        if count_tokens(first_text) < MIN_CHUNK_TOKENS:
            merged[1] = merged[0] + merged[1]
            merged.pop(0)

    return merged


def _split_oversized_chunks(
    sentence_groups: list[list[str]],
    max_chunk_tokens: int,
) -> list[list[str]]:
    """Split chunks exceeding max_chunk_tokens at sentence midpoint."""
    result: list[list[str]] = []
    for group in sentence_groups:
        group_text = " ".join(group)
        if count_tokens(group_text) <= max_chunk_tokens or len(group) <= 1:
            result.append(group)
            continue
        # Split at midpoint
        mid = len(group) // 2
        result.append(group[:mid])
        result.append(group[mid:])
    return result


# --- Embedding helpers ---


def _embed_sentences(sentences: list[str], embedder: Embedder) -> list[list[float]]:
    """Embed sentences in batches to avoid memory spikes."""
    if len(sentences) <= _SENTENCE_BATCH_SIZE:
        return embedder.embed_batch(sentences)

    all_embeddings: list[list[float]] = []
    for start in range(0, len(sentences), _SENTENCE_BATCH_SIZE):
        batch = sentences[start : start + _SENTENCE_BATCH_SIZE]
        all_embeddings.extend(embedder.embed_batch(batch))
    return all_embeddings


# --- Main entry point ---


def chunk_document_semantic(
    doc: NormalizedDocument,
    config: ChunkingConfig,
    embedder: Embedder,
) -> list[Chunk]:
    """Chunk a document using semantic boundary detection."""
    chunks: list[Chunk] = []
    chunk_order = 0
    file_name = doc.title or doc.doc_id

    for section in doc.sections:
        section_id = str(uuid.uuid5(NAMESPACE_RAG, f"{doc.doc_id}:section:{section.order}"))
        section_chunks = _chunk_section_semantic(
            text=section.text,
            doc_id=doc.doc_id,
            section_id=section_id,
            section=section,
            file_name=file_name,
            chunk_order_start=chunk_order,
            config=config,
            embedder=embedder,
        )
        chunks.extend(section_chunks)
        chunk_order += len(section_chunks)

    return chunks


def _chunk_section_semantic(
    *,
    text: str,
    doc_id: str,
    section_id: str,
    section: ParsedSection,
    file_name: str,
    chunk_order_start: int,
    config: ChunkingConfig,
    embedder: Embedder,
) -> list[Chunk]:
    """Chunk a single section using semantic boundary detection."""
    if not text.strip():
        return []

    # 1. Extract code blocks
    cleaned_text, code_blocks = extract_code_blocks(text)

    # 2. Segment sentences
    sentences = segment_sentences(cleaned_text)
    if not sentences:
        return []

    # 3. Few sentences: emit as single chunk
    if len(sentences) < 3:
        full_text = restore_code_blocks(" ".join(sentences), code_blocks)
        return [
            _make_chunk(
                text=full_text,
                doc_id=doc_id,
                section_id=section_id,
                section=section,
                file_name=file_name,
                chunk_order=chunk_order_start,
                chunk_idx=0,
                token_count=count_tokens(full_text),
            )
        ]

    # 4. Embed all sentences
    embeddings = _embed_sentences(sentences, embedder)

    # 5. Detect boundaries
    boundary_indices = detect_boundaries(
        embeddings,
        config.similarity_threshold,
        MAX_CHUNK_SENTENCES,
    )

    # 6. Group sentences by boundaries
    sentence_groups: list[list[str]] = []
    for idx, boundary in enumerate(boundary_indices):
        end = boundary_indices[idx + 1] if idx + 1 < len(boundary_indices) else len(sentences)
        sentence_groups.append(sentences[boundary:end])

    # 7. Apply guardrails
    sentence_groups = _merge_small_chunks(sentence_groups)
    sentence_groups = _split_oversized_chunks(sentence_groups, config.max_chunk_tokens)

    # 8. Build Chunk objects
    chunks: list[Chunk] = []
    for chunk_idx, group in enumerate(sentence_groups):
        chunk_text = " ".join(group)
        # Restore code blocks
        chunk_text = restore_code_blocks(chunk_text, code_blocks)
        chunks.append(
            _make_chunk(
                text=chunk_text,
                doc_id=doc_id,
                section_id=section_id,
                section=section,
                file_name=file_name,
                chunk_order=chunk_order_start + chunk_idx,
                chunk_idx=chunk_idx,
                token_count=count_tokens(chunk_text),
            )
        )

    return chunks
