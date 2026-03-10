from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rag.retrieval.citations import CitationAssembler
from rag.types import ChunkRow, RecordType, SearchHit


def _make_hit(
    *,
    text: str = "Some chunk text.",
    score: float = 0.9,
    doc_id: str = "doc-1",
    file_path: str = "/docs/report.pdf",
    title: str = "Report",
    section_heading: str | None = "Introduction",
    page_start: int | None = 1,
    page_end: int | None = 3,
    chunk_order: int | None = 0,
    modified_at: str = "2025-01-01T00:00:00Z",
    record_type: RecordType = RecordType.CHUNK,
) -> SearchHit:
    payload: dict[str, object] = {
        "file_path": file_path,
        "title": title,
        "section_heading": section_heading,
        "page_start": page_start,
        "page_end": page_end,
        "chunk_order": chunk_order,
        "modified_at": modified_at,
    }
    return SearchHit(
        point_id="pt-1",
        score=score,
        record_type=record_type,
        doc_id=doc_id,
        text=text,
        payload=payload,
    )


def _make_chunk_row(
    chunk_order: int,
    chunk_text: str,
    doc_id: str = "doc-1",
) -> ChunkRow:
    return ChunkRow(
        chunk_id=f"chunk-{chunk_order}",
        doc_id=doc_id,
        chunk_order=chunk_order,
        chunk_text=chunk_text,
        chunk_text_normalized=chunk_text.lower(),
    )


@pytest.fixture
def mock_db() -> MagicMock:
    return MagicMock()


class TestCitationLabel:
    def test_full_label(self, mock_db: MagicMock) -> None:
        assembler = CitationAssembler(mock_db)
        mock_db.get_adjacent_chunks.return_value = []
        hit = _make_hit(section_heading="Introduction", page_start=12, page_end=14)
        results = assembler.assemble_citations([hit], expand_context=False)
        assert len(results) == 1
        assert results[0].citation.label == "report.pdf, § Introduction, pp. 12-14"

    def test_single_page(self, mock_db: MagicMock) -> None:
        assembler = CitationAssembler(mock_db)
        hit = _make_hit(page_start=5, page_end=5)
        results = assembler.assemble_citations([hit], expand_context=False)
        assert results[0].citation.label == "report.pdf, § Introduction, p. 5"
        assert results[0].citation.pages == "p. 5"

    def test_no_section_no_pages(self, mock_db: MagicMock) -> None:
        assembler = CitationAssembler(mock_db)
        hit = _make_hit(section_heading=None, page_start=None, page_end=None)
        results = assembler.assemble_citations([hit], expand_context=False)
        assert results[0].citation.label == "report.pdf"
        assert results[0].citation.section is None
        assert results[0].citation.pages is None

    def test_section_no_pages(self, mock_db: MagicMock) -> None:
        assembler = CitationAssembler(mock_db)
        hit = _make_hit(section_heading="Conclusion", page_start=None, page_end=None)
        results = assembler.assemble_citations([hit], expand_context=False)
        assert results[0].citation.label == "report.pdf, § Conclusion"


class TestCitationFields:
    def test_citation_fields(self, mock_db: MagicMock) -> None:
        assembler = CitationAssembler(mock_db)
        hit = _make_hit(
            title="My Report",
            file_path="/docs/report.pdf",
            modified_at="2025-06-01T12:00:00Z",
        )
        results = assembler.assemble_citations([hit], expand_context=False)
        c = results[0].citation
        assert c.title == "My Report"
        assert c.path == "/docs/report.pdf"
        assert c.modified == "2025-06-01T12:00:00Z"

    def test_score_and_record_type(self, mock_db: MagicMock) -> None:
        assembler = CitationAssembler(mock_db)
        hit = _make_hit(score=0.85, record_type=RecordType.SECTION_SUMMARY)
        results = assembler.assemble_citations([hit], expand_context=False)
        assert results[0].score == 0.85
        assert results[0].record_type == "section_summary"


class TestContextExpansion:
    def test_expands_with_adjacent_chunks(self, mock_db: MagicMock) -> None:
        assembler = CitationAssembler(mock_db)
        mock_db.get_adjacent_chunks.return_value = [
            _make_chunk_row(0, "Before chunk."),
            _make_chunk_row(1, "Main chunk."),
            _make_chunk_row(2, "After chunk."),
        ]
        hit = _make_hit(text="Main chunk.", chunk_order=1)
        results = assembler.assemble_citations([hit], expand_context=True, context_window=1)
        assert "Before chunk." in results[0].text
        assert "Main chunk." in results[0].text
        assert "After chunk." in results[0].text

    def test_no_expansion_when_disabled(self, mock_db: MagicMock) -> None:
        assembler = CitationAssembler(mock_db)
        hit = _make_hit(text="Only this text.")
        results = assembler.assemble_citations([hit], expand_context=False)
        assert results[0].text == "Only this text."
        mock_db.get_adjacent_chunks.assert_not_called()

    def test_no_expansion_when_chunk_order_missing(self, mock_db: MagicMock) -> None:
        assembler = CitationAssembler(mock_db)
        hit = _make_hit(text="Original text.", chunk_order=None)
        results = assembler.assemble_citations([hit], expand_context=True, context_window=1)
        assert results[0].text == "Original text."
        mock_db.get_adjacent_chunks.assert_not_called()

    def test_fallback_when_no_adjacent_chunks(self, mock_db: MagicMock) -> None:
        assembler = CitationAssembler(mock_db)
        mock_db.get_adjacent_chunks.return_value = []
        hit = _make_hit(text="Standalone chunk.", chunk_order=0)
        results = assembler.assemble_citations([hit], expand_context=True, context_window=1)
        assert results[0].text == "Standalone chunk."


class TestOverlapDedup:
    def test_dedup_overlapping_text(self, mock_db: MagicMock) -> None:
        assembler = CitationAssembler(mock_db)
        # Simulate 64-token overlap between chunks
        overlap = "This is the overlapping portion."
        chunk_a = f"First part of chunk A. {overlap}"
        chunk_b = f"{overlap} Second part of chunk B."
        mock_db.get_adjacent_chunks.return_value = [
            _make_chunk_row(0, chunk_a),
            _make_chunk_row(1, chunk_b),
        ]
        hit = _make_hit(text="whatever", chunk_order=0)
        results = assembler.assemble_citations([hit], expand_context=True, context_window=1)
        merged = results[0].text
        # Overlap text should appear exactly once
        assert merged.count(overlap) == 1
        assert "First part of chunk A." in merged
        assert "Second part of chunk B." in merged

    def test_no_overlap_joined_with_separator(self, mock_db: MagicMock) -> None:
        assembler = CitationAssembler(mock_db)
        mock_db.get_adjacent_chunks.return_value = [
            _make_chunk_row(0, "Chunk A content."),
            _make_chunk_row(1, "Chunk B content."),
        ]
        hit = _make_hit(text="whatever", chunk_order=0)
        results = assembler.assemble_citations([hit], expand_context=True, context_window=1)
        assert results[0].text == "Chunk A content.\n\nChunk B content."


class TestEdgeCases:
    def test_empty_hits(self, mock_db: MagicMock) -> None:
        assembler = CitationAssembler(mock_db)
        results = assembler.assemble_citations([])
        assert results == []

    def test_missing_file_path(self, mock_db: MagicMock) -> None:
        assembler = CitationAssembler(mock_db)
        hit = _make_hit(file_path="", section_heading=None, page_start=None, page_end=None)
        results = assembler.assemble_citations([hit], expand_context=False)
        assert results[0].citation.label == "Unknown"

    def test_missing_title_uses_filename(self, mock_db: MagicMock) -> None:
        assembler = CitationAssembler(mock_db)
        payload: dict[str, object] = {
            "file_path": "/docs/notes.md",
            "section_heading": None,
            "page_start": None,
            "page_end": None,
            "chunk_order": 0,
            "modified_at": "2025-01-01T00:00:00Z",
        }
        hit = SearchHit(
            point_id="pt-1",
            score=0.7,
            record_type=RecordType.CHUNK,
            doc_id="doc-2",
            text="Some text.",
            payload=payload,
        )
        results = assembler.assemble_citations([hit], expand_context=False)
        assert results[0].citation.title == "notes.md"

    def test_multiple_hits(self, mock_db: MagicMock) -> None:
        assembler = CitationAssembler(mock_db)
        hits = [
            _make_hit(text="First.", score=0.9),
            _make_hit(text="Second.", score=0.8),
            _make_hit(text="Third.", score=0.7),
        ]
        results = assembler.assemble_citations(hits, expand_context=False)
        assert len(results) == 3
        assert [r.score for r in results] == [0.9, 0.8, 0.7]
