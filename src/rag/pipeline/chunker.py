from __future__ import annotations

import re
import uuid

import tiktoken

from rag.types import NAMESPACE_RAG, Chunk, NormalizedDocument, ParsedSection

# Target tokens per chunk and overlap
TARGET_TOKENS = 512
OVERLAP_TOKENS = 64

# Module-level tokenizer (loaded once)
_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in text using cl100k_base encoding."""
    return len(_encoding.encode(text))


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences at sentence boundaries."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p.strip()]


def chunk_document(doc: NormalizedDocument) -> list[Chunk]:
    """Chunk a normalized document into sized chunks with deterministic IDs.

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
