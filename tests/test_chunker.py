from __future__ import annotations

from rag.pipeline.chunker import (
    TARGET_TOKENS,
    _build_citation,
    chunk_document,
    count_tokens,
)
from rag.types import NormalizedDocument, ParsedSection


def _make_doc(
    sections: list[ParsedSection],
    doc_id: str = "doc-001",
    title: str = "report.pdf",
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
    heading: str | None = "Introduction",
    page_start: int | None = 1,
    page_end: int | None = 3,
) -> ParsedSection:
    return ParsedSection(
        heading=heading,
        order=order,
        text=text,
        page_start=page_start,
        page_end=page_end,
    )


class TestChunkDocumentBasic:
    def test_multi_section_produces_chunks(self) -> None:
        doc = _make_doc(
            sections=[
                _make_section("First section has some content.", order=0),
                _make_section("Second section also has content.", order=1),
            ]
        )
        chunks = chunk_document(doc)
        assert len(chunks) >= 2
        # Each section should produce at least one chunk
        section_ids = {c.section_id for c in chunks}
        assert len(section_ids) == 2

    def test_chunk_order_is_sequential(self) -> None:
        doc = _make_doc(
            sections=[
                _make_section("First section content.", order=0),
                _make_section("Second section content.", order=1),
            ]
        )
        chunks = chunk_document(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_order == i


class TestTokenLimits:
    def test_chunks_respect_token_limit(self) -> None:
        # Build a long section with many sentences
        sentences = ["This is a test sentence with several words in it."] * 200
        text = " ".join(sentences)
        doc = _make_doc(sections=[_make_section(text)])
        chunks = chunk_document(doc)

        max_allowed = TARGET_TOKENS + 64  # overlap budget
        for chunk in chunks:
            assert chunk.token_count <= max_allowed + 20, (
                f"Chunk has {chunk.token_count} tokens, exceeds soft limit"
            )


class TestHeadingBoundary:
    def test_new_section_starts_new_chunk(self) -> None:
        doc = _make_doc(
            sections=[
                _make_section("Content of section A.", order=0, heading="Section A"),
                _make_section("Content of section B.", order=1, heading="Section B"),
            ]
        )
        chunks = chunk_document(doc)
        headings = [c.section_heading for c in chunks]
        assert "Section A" in headings
        assert "Section B" in headings
        # First chunk of section B should not contain section A text
        section_b_chunks = [c for c in chunks if c.section_heading == "Section B"]
        for c in section_b_chunks:
            assert "section A" not in c.text.lower()


class TestDeterministicIDs:
    def test_same_input_same_ids(self) -> None:
        doc = _make_doc(sections=[_make_section("Some repeatable content here.", order=0)])
        chunks_a = chunk_document(doc)
        chunks_b = chunk_document(doc)
        assert len(chunks_a) == len(chunks_b)
        for a, b in zip(chunks_a, chunks_b, strict=True):
            assert a.chunk_id == b.chunk_id

    def test_different_doc_id_different_chunk_ids(self) -> None:
        sec = _make_section("Same content.", order=0)
        chunks_a = chunk_document(_make_doc(sections=[sec], doc_id="doc-A"))
        chunks_b = chunk_document(_make_doc(sections=[sec], doc_id="doc-B"))
        assert chunks_a[0].chunk_id != chunks_b[0].chunk_id


class TestCitationLabel:
    def test_full_citation(self) -> None:
        label = _build_citation("report.pdf", "Methods", 5, 8)
        assert label == "report.pdf, § Methods, pp. 5-8"

    def test_single_page(self) -> None:
        label = _build_citation("report.pdf", "Intro", 3, 3)
        assert label == "report.pdf, § Intro, p. 3"

    def test_no_heading_no_pages(self) -> None:
        label = _build_citation("notes.txt", None, None, None)
        assert label == "notes.txt"

    def test_heading_no_pages(self) -> None:
        label = _build_citation("doc.pdf", "Summary", None, None)
        assert label == "doc.pdf, § Summary"

    def test_chunk_citation_label_populated(self) -> None:
        doc = _make_doc(
            sections=[
                _make_section(
                    "Some text here.",
                    order=0,
                    heading="Results",
                    page_start=12,
                    page_end=14,
                )
            ]
        )
        chunks = chunk_document(doc)
        assert chunks[0].citation_label == "report.pdf, § Results, pp. 12-14"


class TestEmptySection:
    def test_empty_text_no_chunks(self) -> None:
        doc = _make_doc(sections=[_make_section("", order=0)])
        chunks = chunk_document(doc)
        assert chunks == []

    def test_whitespace_only_no_chunks(self) -> None:
        doc = _make_doc(sections=[_make_section("   \n\t  ", order=0)])
        chunks = chunk_document(doc)
        assert chunks == []


class TestLongSentence:
    def test_single_long_sentence_own_chunk(self) -> None:
        # Create a sentence longer than TARGET_TOKENS
        long_sentence = "word " * (TARGET_TOKENS + 100)
        long_sentence = long_sentence.strip() + "."
        doc = _make_doc(sections=[_make_section(long_sentence, order=0)])
        chunks = chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].token_count > TARGET_TOKENS


class TestOverlap:
    def test_adjacent_chunks_share_overlap(self) -> None:
        # Build enough sentences to force multiple chunks
        sentence = "The quick brown fox jumps over the lazy dog."
        sentences = [sentence] * 200
        text = " ".join(sentences)
        doc = _make_doc(sections=[_make_section(text, order=0)])
        chunks = chunk_document(doc)

        assert len(chunks) >= 2, "Expected multiple chunks for overlap test"

        # Check that adjacent chunks share some trailing/leading text
        for i in range(len(chunks) - 1):
            current_text = chunks[i].text
            next_text = chunks[i + 1].text
            # The overlap means the end of current chunk should appear
            # at the start of the next chunk
            current_sentences = current_text.split(". ")
            # At least one sentence from end of current should appear in next
            overlap_found = any(
                s.strip() in next_text for s in current_sentences[-3:] if s.strip()
            )
            assert overlap_found, f"No overlap found between chunk {i} and chunk {i + 1}"


class TestTextNormalized:
    def test_text_normalized_is_lowercase(self) -> None:
        doc = _make_doc(sections=[_make_section("Hello World. This Is Mixed Case.", order=0)])
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.text_normalized == chunk.text.lower()


class TestCountTokens:
    def test_empty_string(self) -> None:
        assert count_tokens("") == 0

    def test_known_string(self) -> None:
        tokens = count_tokens("hello world")
        assert tokens == 2
