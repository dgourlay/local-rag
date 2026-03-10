from __future__ import annotations

from rag.pipeline.normalizer import normalize
from rag.types import FileType, ParsedDocument, ParsedSection


def _make_doc(sections: list[ParsedSection]) -> ParsedDocument:
    return ParsedDocument(
        doc_id="test-doc-1",
        title="Test Document",
        file_type=FileType.TXT,
        sections=sections,
        raw_content_hash="abc123",
    )


class TestWhitespaceCleanup:
    def test_collapses_multiple_spaces(self) -> None:
        section = ParsedSection(heading=None, order=0, text="hello   world   foo")
        doc = _make_doc([section])
        result = normalize(doc)
        assert result.sections[0].text == "hello world foo"

    def test_collapses_multiple_blank_lines(self) -> None:
        section = ParsedSection(heading=None, order=0, text="line1\n\n\n\n\nline2")
        doc = _make_doc([section])
        result = normalize(doc)
        assert result.sections[0].text == "line1\n\nline2"

    def test_strips_trailing_whitespace_per_line(self) -> None:
        section = ParsedSection(heading=None, order=0, text="hello   \nworld   ")
        doc = _make_doc([section])
        result = normalize(doc)
        assert result.sections[0].text == "hello\nworld"

    def test_strips_leading_trailing_whitespace(self) -> None:
        section = ParsedSection(heading=None, order=0, text="  \n  hello  \n  ")
        doc = _make_doc([section])
        result = normalize(doc)
        assert result.sections[0].text == "hello"


class TestHeadingPreservation:
    def test_heading_passes_through(self) -> None:
        section = ParsedSection(heading="Chapter 1", order=0, text="content")
        doc = _make_doc([section])
        result = normalize(doc)
        assert result.sections[0].heading == "Chapter 1"

    def test_heading_is_stripped(self) -> None:
        section = ParsedSection(heading="  Chapter 1  ", order=0, text="content")
        doc = _make_doc([section])
        result = normalize(doc)
        assert result.sections[0].heading == "Chapter 1"

    def test_none_heading_preserved(self) -> None:
        section = ParsedSection(heading=None, order=0, text="content")
        doc = _make_doc([section])
        result = normalize(doc)
        assert result.sections[0].heading is None


class TestPageMappingPreservation:
    def test_page_start_end_preserved(self) -> None:
        section = ParsedSection(heading=None, order=0, text="content", page_start=3, page_end=5)
        doc = _make_doc([section])
        result = normalize(doc)
        assert result.sections[0].page_start == 3
        assert result.sections[0].page_end == 5

    def test_none_pages_preserved(self) -> None:
        section = ParsedSection(heading=None, order=0, text="content")
        doc = _make_doc([section])
        result = normalize(doc)
        assert result.sections[0].page_start is None
        assert result.sections[0].page_end is None


class TestHashConsistency:
    def test_same_input_same_hash(self) -> None:
        section = ParsedSection(heading=None, order=0, text="hello world")
        doc1 = _make_doc([section])
        doc2 = _make_doc([section])
        r1 = normalize(doc1)
        r2 = normalize(doc2)
        assert r1.normalized_content_hash == r2.normalized_content_hash

    def test_different_input_different_hash(self) -> None:
        s1 = ParsedSection(heading=None, order=0, text="hello world")
        s2 = ParsedSection(heading=None, order=0, text="goodbye world")
        r1 = normalize(_make_doc([s1]))
        r2 = normalize(_make_doc([s2]))
        assert r1.normalized_content_hash != r2.normalized_content_hash

    def test_hash_is_sha256_hex(self) -> None:
        section = ParsedSection(heading=None, order=0, text="content")
        result = normalize(_make_doc([section]))
        assert len(result.normalized_content_hash) == 64


class TestEmptySectionRemoval:
    def test_empty_text_section_removed(self) -> None:
        sections = [
            ParsedSection(heading="Intro", order=0, text=""),
            ParsedSection(heading="Body", order=1, text="real content"),
        ]
        doc = _make_doc(sections)
        result = normalize(doc)
        assert len(result.sections) == 1
        assert result.sections[0].heading == "Body"

    def test_whitespace_only_section_removed(self) -> None:
        sections = [
            ParsedSection(heading="Empty", order=0, text="   \n\n   "),
            ParsedSection(heading="Body", order=1, text="real content"),
        ]
        doc = _make_doc(sections)
        result = normalize(doc)
        assert len(result.sections) == 1
        assert result.sections[0].heading == "Body"


class TestMultipleSections:
    def test_sections_normalized_independently(self) -> None:
        sections = [
            ParsedSection(heading="A", order=0, text="hello   world"),
            ParsedSection(heading="B", order=1, text="foo\n\n\n\nbar"),
        ]
        doc = _make_doc(sections)
        result = normalize(doc)
        assert len(result.sections) == 2
        assert result.sections[0].text == "hello world"
        assert result.sections[1].text == "foo\n\nbar"

    def test_doc_metadata_preserved(self) -> None:
        section = ParsedSection(heading=None, order=0, text="content")
        doc = _make_doc([section])
        result = normalize(doc)
        assert result.doc_id == "test-doc-1"
        assert result.title == "Test Document"
        assert result.file_type == FileType.TXT
        assert result.raw_content_hash == "abc123"
