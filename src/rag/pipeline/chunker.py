from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING

import tiktoken

from rag.types import NAMESPACE_RAG, Chunk, NormalizedDocument, ParsedSection

if TYPE_CHECKING:
    from rag.config import ChunkingConfig
    from rag.protocols import Embedder

# Target tokens per chunk and overlap
TARGET_TOKENS = 512
OVERLAP_TOKENS = 64

# Chunker version constants
CHUNKER_VERSION_FIXED = "fixed-v1"
CHUNKER_VERSION_SEMANTIC = "semantic-v1"

# Module-level tokenizer (loaded once)
_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in text using cl100k_base encoding."""
    return len(_encoding.encode(text))


def get_chunker_version(strategy: str) -> str:
    """Return the chunker version string for the given strategy."""
    if strategy == "semantic":
        return CHUNKER_VERSION_SEMANTIC
    return CHUNKER_VERSION_FIXED


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences at sentence boundaries."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p.strip()]


def _split_on_token_windows(text: str, max_tokens: int) -> list[str]:
    """Hard-split ``text`` on token windows, ignoring word boundaries.

    Last-resort splitter for content with no whitespace to break on (base64
    blobs, long URLs).  Decoding a token window can re-encode to a slightly
    different length, so each window is verified and narrowed until it fits.
    """
    tokens = _encoding.encode(text)
    pieces: list[str] = []
    start = 0
    while start < len(tokens):
        width = max_tokens
        piece = _encoding.decode(tokens[start : start + width])
        while width > 1 and count_tokens(piece) > max_tokens:
            width //= 2
            piece = _encoding.decode(tokens[start : start + width])
        pieces.append(piece)
        start += width
    return pieces


def split_long_sentence(sentence: str, max_tokens: int) -> list[str]:
    """Split a sentence into pieces of at most ``max_tokens`` tokens.

    Sentence segmentation only breaks on ``.!?``, so content without sentence
    punctuation (markdown tables, bullet lists, log dumps) segments into a
    single arbitrarily long "sentence".  Left unbounded it becomes one huge
    chunk, which blows up attention memory in the embedder -- a batch is
    padded to its longest member, and attention scales with the square of
    that length.

    Splits on whitespace so pieces stay readable, falling back to token
    windows for runs with no whitespace.
    """
    total = count_tokens(sentence)
    if total <= max_tokens:
        return [sentence]

    words = sentence.split()
    word_tokens = [count_tokens(w) for w in words]
    # Tokens counted per word in isolation drift from the joined count (BPE
    # merges across word boundaries), so scale the per-word budget by the
    # drift measured on this sentence.  Keeps packing O(n) instead of
    # re-tokenizing a growing prefix for every word.
    isolated = sum(word_tokens) or 1
    budget = max(1, max_tokens * isolated // total)

    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word, tokens in zip(words, word_tokens, strict=True):
        if current and current_tokens + tokens > budget:
            pieces.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(word)
        current_tokens += tokens
    if current:
        pieces.append(" ".join(current))

    # Enforce the cap exactly: the scaled budget lands close but a single
    # unbreakable word, or uneven drift, can still overshoot.
    return [
        part
        for piece in pieces
        if piece.strip()
        for part in (
            [piece]
            if count_tokens(piece) <= max_tokens
            else _split_on_token_windows(piece, max_tokens)
        )
    ]


def chunk_document(
    doc: NormalizedDocument,
    config: ChunkingConfig | None = None,
    embedder: Embedder | None = None,
) -> list[Chunk]:
    """Chunk a document using the configured strategy.

    Args:
        doc: Normalized document with sections.
        config: Chunking configuration. Defaults to fixed strategy.
        embedder: Required for semantic strategy. Ignored for fixed.
    """
    if config is not None and config.strategy == "semantic":
        if embedder is None:
            msg = "Embedder required for semantic chunking strategy"
            raise ValueError(msg)
        from rag.pipeline.chunker_semantic import chunk_document_semantic

        return chunk_document_semantic(doc, config, embedder)
    return chunk_document_fixed(doc)


def chunk_document_fixed(doc: NormalizedDocument) -> list[Chunk]:
    """Chunk a normalized document using fixed-size strategy.

    Each section starts a new chunk boundary. Chunks respect sentence
    boundaries and use 64-token overlap between adjacent chunks within
    a section.
    """
    chunks: list[Chunk] = []
    chunk_order = 0
    file_name = doc.title or doc.doc_id

    for section in doc.sections:
        section_id = str(uuid.uuid5(NAMESPACE_RAG, f"{doc.doc_id}:section:{section.order}"))
        section_chunks = _chunk_section(
            text=section.text,
            doc_id=doc.doc_id,
            section_id=section_id,
            section=section,
            file_name=file_name,
            chunk_order_start=chunk_order,
        )
        chunks.extend(section_chunks)
        chunk_order += len(section_chunks)

    return chunks


def _chunk_section(
    *,
    text: str,
    doc_id: str,
    section_id: str,
    section: ParsedSection,
    file_name: str,
    chunk_order_start: int,
) -> list[Chunk]:
    """Chunk a single section, respecting sentence boundaries."""
    sentences = _split_sentences(text)
    if not sentences:
        return []

    # Enforce the token cap per sentence so no single chunk can exceed it.
    sentences = [
        piece for sentence in sentences for piece in split_long_sentence(sentence, TARGET_TOKENS)
    ]

    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_tokens = 0
    chunk_idx = 0

    for sentence in sentences:
        sent_tokens = count_tokens(sentence)

        # If single sentence exceeds target and buffer is empty, emit it alone
        if sent_tokens > TARGET_TOKENS and not current_sentences:
            chunks.append(
                _make_chunk(
                    text=sentence,
                    doc_id=doc_id,
                    section_id=section_id,
                    section=section,
                    file_name=file_name,
                    chunk_order=chunk_order_start + chunk_idx,
                    chunk_idx=chunk_idx,
                    token_count=sent_tokens,
                )
            )
            chunk_idx += 1
            continue

        # Flush current buffer if adding this sentence would exceed target
        if current_tokens + sent_tokens > TARGET_TOKENS and current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                _make_chunk(
                    text=chunk_text,
                    doc_id=doc_id,
                    section_id=section_id,
                    section=section,
                    file_name=file_name,
                    chunk_order=chunk_order_start + chunk_idx,
                    chunk_idx=chunk_idx,
                    token_count=current_tokens,
                )
            )
            chunk_idx += 1

            # Keep overlap sentences from tail of current buffer
            overlap_sentences: list[str] = []
            overlap_tokens = 0
            for s in reversed(current_sentences):
                st = count_tokens(s)
                if overlap_tokens + st > OVERLAP_TOKENS:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += st
            current_sentences = overlap_sentences
            current_tokens = overlap_tokens

        current_sentences.append(sentence)
        current_tokens += sent_tokens

    # Flush remaining sentences
    if current_sentences:
        chunk_text = " ".join(current_sentences)
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


def _make_chunk(
    *,
    text: str,
    doc_id: str,
    section_id: str,
    section: ParsedSection,
    file_name: str,
    chunk_order: int,
    chunk_idx: int,
    token_count: int,
) -> Chunk:
    """Create a Chunk with deterministic UUID5 ID and citation label."""
    chunk_id = str(uuid.uuid5(NAMESPACE_RAG, f"{doc_id}:{section.order}:{chunk_idx}"))
    citation = _build_citation(file_name, section.heading, section.page_start, section.page_end)
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        section_id=section_id,
        chunk_order=chunk_order,
        text=text,
        text_normalized=text.lower(),
        page_start=section.page_start,
        page_end=section.page_end,
        section_heading=section.heading,
        citation_label=citation,
        token_count=token_count,
    )


def _build_citation(
    file_name: str,
    heading: str | None,
    page_start: int | None,
    page_end: int | None,
) -> str:
    """Build citation label: 'filename.pdf § Section Heading, pp. 12-14'."""
    parts = [file_name]
    if heading:
        parts.append(f"§ {heading}")
    if page_start is not None:
        if page_end is not None and page_end != page_start:
            parts.append(f"pp. {page_start}-{page_end}")
        else:
            parts.append(f"p. {page_start}")
    return ", ".join(parts) if len(parts) > 1 else parts[0]
