from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rag.types import ChunkRow, Citation, CitedEvidence, SearchHit

if TYPE_CHECKING:
    from rag.protocols import MetadataDB


class CitationAssembler:
    """Assemble citations from search hits with optional context expansion."""

    def __init__(self, db: MetadataDB) -> None:
        self._db = db

    def assemble_citations(
        self,
        hits: list[SearchHit],
        expand_context: bool = True,
        context_window: int = 1,
    ) -> list[CitedEvidence]:
        """Convert search hits to cited evidence with formatted citations."""
        results: list[CitedEvidence] = []

        for hit in hits:
            payload = hit.payload

            # Build text with optional context expansion
            text = hit.text
            if expand_context and context_window > 0:
                text = self._expand_context(hit, context_window)

            # Build citation
            file_path = payload.get("file_path", "")
            file_name = Path(file_path).name if file_path else ""
            section: str | None = payload.get("section_heading")
            page_start: int | None = payload.get("page_start")
            page_end: int | None = payload.get("page_end")

            # Format label: "filename.pdf § Section Heading, pp. 12-14"
            label = self._build_label(file_name, section, page_start, page_end)

            # Format pages string
            pages: str | None = None
            if page_start is not None:
                if page_end is not None and page_end != page_start:
                    pages = f"pp. {page_start}-{page_end}"
                else:
                    pages = f"p. {page_start}"

            citation = Citation(
                title=payload.get("title", file_name),
                path=str(file_path),
                section=section,
                pages=pages,
                modified=payload.get("modified_at", ""),
                label=label,
            )

            results.append(
                CitedEvidence(
                    text=text,
                    citation=citation,
                    score=hit.score,
                    record_type=hit.record_type.value,
                )
            )

        return results

    def _expand_context(self, hit: SearchHit, window: int) -> str:
        """Fetch adjacent chunks and merge text, deduplicating overlap."""
        chunk_order: int | None = hit.payload.get("chunk_order")

        if chunk_order is None:
            return hit.text

        adjacent: list[ChunkRow] = self._db.get_adjacent_chunks(hit.doc_id, chunk_order, window)
        if not adjacent:
            return hit.text

        # Sort by chunk_order and combine
        all_chunks = sorted(adjacent, key=lambda c: c.chunk_order)
        texts = [c.chunk_text for c in all_chunks]

        # Deduplicate overlapping content
        return self._merge_overlapping_texts(texts)

    def _merge_overlapping_texts(self, texts: list[str]) -> str:
        """Merge texts that may have overlapping content."""
        if not texts:
            return ""
        if len(texts) == 1:
            return texts[0]

        merged = texts[0]
        for text in texts[1:]:
            overlap = self._find_overlap(merged, text)
            if overlap > 0:
                merged += text[overlap:]
            else:
                merged += "\n\n" + text
        return merged

    def _find_overlap(self, text_a: str, text_b: str) -> int:
        """Find length of overlap between end of text_a and start of text_b."""
        max_overlap = min(len(text_a), len(text_b), 500)
        for i in range(max_overlap, 0, -1):
            if text_a.endswith(text_b[:i]):
                return i
        return 0

    def _build_label(
        self,
        file_name: str,
        section: str | None,
        page_start: int | None,
        page_end: int | None,
    ) -> str:
        """Build citation label: 'filename.pdf § Section, pp. 12-14'."""
        parts: list[str] = [file_name] if file_name else ["Unknown"]
        if section:
            parts.append(f"§ {section}")
        if page_start is not None:
            if page_end is not None and page_end != page_start:
                parts.append(f"pp. {page_start}-{page_end}")
            else:
                parts.append(f"p. {page_start}")
        return ", ".join(parts)
