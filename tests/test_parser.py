from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from rag.pipeline.parser.base import get_parser
from rag.pipeline.parser.docling_parser import DoclingParser
from rag.results import ParseError, ParseSuccess
from rag.types import FileType


def _make_mock_context(
    result: dict[str, Any],
    *,
    alive: bool = False,
) -> MagicMock:
    """Build a mock multiprocessing spawn context that returns `result` from queue."""
    mock_queue = MagicMock()
    mock_queue.empty.return_value = False
    mock_queue.get.return_value = result

    mock_proc = MagicMock()
    mock_proc.is_alive.return_value = alive

    mock_ctx = MagicMock()
    mock_ctx.Queue.return_value = mock_queue
    mock_ctx.Process.return_value = mock_proc

    return mock_ctx


class TestDoclingParserSupportedTypes:
    def test_supported_types(self) -> None:
        parser = DoclingParser()
        assert parser.supported_types == {FileType.PDF, FileType.DOCX}

    def test_does_not_support_txt(self) -> None:
        parser = DoclingParser()
        assert FileType.TXT not in parser.supported_types

    def test_does_not_support_md(self) -> None:
        parser = DoclingParser()
        assert FileType.MD not in parser.supported_types


class TestDoclingParserFileNotFound:
    def test_returns_parse_error_for_missing_file(self) -> None:
        parser = DoclingParser()
        result = parser.parse("/nonexistent/file.pdf", ocr_enabled=False)
        assert isinstance(result, ParseError)
        assert "not found" in result.error.lower()
        assert result.file_path == "/nonexistent/file.pdf"


class TestDoclingParserSubprocess:
    def test_parse_success_via_mock_subprocess(self, tmp_path: Path) -> None:
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        mock_result = {
            "status": "success",
            "sections": [
                {
                    "heading": "Introduction",
                    "order": 0,
                    "text": "This is the intro text.",
                    "page_start": 1,
                    "page_end": 1,
                },
                {
                    "heading": "Methods",
                    "order": 1,
                    "text": "We used these methods.",
                    "page_start": 2,
                    "page_end": 3,
                },
            ],
            "title": "Test Document",
        }

        mock_ctx = _make_mock_context(mock_result)

        with patch("multiprocessing.get_context", return_value=mock_ctx):
            parser = DoclingParser()
            result = parser.parse(str(pdf_file), ocr_enabled=False)

        assert isinstance(result, ParseSuccess)
        doc = result.document
        assert doc.title == "Test Document"
        assert doc.file_type == FileType.PDF
        assert len(doc.sections) == 2
        assert doc.sections[0].heading == "Introduction"
        assert doc.sections[0].text == "This is the intro text."
        assert doc.sections[0].page_start == 1
        assert doc.sections[1].heading == "Methods"
        assert doc.sections[1].page_start == 2
        assert doc.sections[1].page_end == 3
        assert doc.raw_content_hash  # non-empty hash
        assert doc.doc_id  # non-empty UUID
        assert doc.ocr_required is False

    def test_parse_error_from_subprocess(self, tmp_path: Path) -> None:
        pdf_file = tmp_path / "bad.pdf"
        pdf_file.write_bytes(b"not a real pdf")

        mock_result = {
            "status": "error",
            "error": "Docling could not parse the file",
        }

        mock_ctx = _make_mock_context(mock_result)

        with patch("multiprocessing.get_context", return_value=mock_ctx):
            parser = DoclingParser()
            result = parser.parse(str(pdf_file), ocr_enabled=False)

        assert isinstance(result, ParseError)
        assert "could not parse" in result.error.lower()

    def test_parse_timeout(self, tmp_path: Path) -> None:
        pdf_file = tmp_path / "slow.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 content")

        mock_ctx = _make_mock_context({}, alive=True)

        with patch("multiprocessing.get_context", return_value=mock_ctx):
            parser = DoclingParser()
            result = parser.parse(str(pdf_file), ocr_enabled=False)

        assert isinstance(result, ParseError)
        assert "timed out" in result.error.lower()
        mock_ctx.Process.return_value.terminate.assert_called_once()

    def test_subprocess_empty_queue(self, tmp_path: Path) -> None:
        pdf_file = tmp_path / "empty.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 content")

        mock_queue = MagicMock()
        mock_queue.empty.return_value = True

        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = False

        mock_ctx = MagicMock()
        mock_ctx.Queue.return_value = mock_queue
        mock_ctx.Process.return_value = mock_proc

        with patch("multiprocessing.get_context", return_value=mock_ctx):
            parser = DoclingParser()
            result = parser.parse(str(pdf_file), ocr_enabled=False)

        assert isinstance(result, ParseError)
        assert "no result" in result.error.lower()

    def test_subprocess_is_called_with_spawn(self, tmp_path: Path) -> None:
        """Verify that the parser uses spawn context for subprocess isolation."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 content")

        mock_ctx = _make_mock_context({"status": "error", "error": "test"})

        with patch("multiprocessing.get_context", return_value=mock_ctx) as mock_get_ctx:
            parser = DoclingParser()
            parser.parse(str(pdf_file), ocr_enabled=False)

        mock_get_ctx.assert_called_once_with("spawn")

    def test_docling_not_imported_in_parent(self) -> None:
        """Verify that importing DoclingParser does not import docling at module level."""
        import rag.pipeline.parser.docling_parser as mod

        source = Path(mod.__file__).read_text()  # type: ignore[arg-type]
        lines = source.split("\n")
        top_level_imports = [
            line
            for line in lines
            if line.startswith("import docling") or line.startswith("from docling")
        ]
        assert top_level_imports == [], (
            "docling should only be imported inside the subprocess function"
        )

    def test_ocr_enabled_passed_to_subprocess(self, tmp_path: Path) -> None:
        pdf_file = tmp_path / "ocr.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 content")

        mock_result = {
            "status": "success",
            "sections": [
                {
                    "heading": None,
                    "order": 0,
                    "text": "OCR text.",
                    "page_start": None,
                    "page_end": None,
                }
            ],
            "title": "OCR Doc",
        }

        mock_ctx = _make_mock_context(mock_result)

        with patch("multiprocessing.get_context", return_value=mock_ctx):
            parser = DoclingParser()
            result = parser.parse(str(pdf_file), ocr_enabled=True)

        assert isinstance(result, ParseSuccess)
        assert result.document.ocr_required is True

        # Verify _parse_in_subprocess was called with ocr_enabled=True
        call_args = mock_ctx.Process.call_args
        assert call_args is not None
        proc_kwargs = call_args[1] if call_args[1] else {}
        if "args" in proc_kwargs:
            assert proc_kwargs["args"][1] is True  # ocr_enabled


class TestDoclingParserDocx:
    def test_parse_docx_file_type(self, tmp_path: Path) -> None:
        docx_file = tmp_path / "test.docx"
        docx_file.write_bytes(b"PK fake docx content")

        mock_result = {
            "status": "success",
            "sections": [
                {
                    "heading": None,
                    "order": 0,
                    "text": "Document content.",
                    "page_start": None,
                    "page_end": None,
                }
            ],
            "title": "Test DOCX",
        }

        mock_ctx = _make_mock_context(mock_result)

        with patch("multiprocessing.get_context", return_value=mock_ctx):
            parser = DoclingParser()
            result = parser.parse(str(docx_file), ocr_enabled=False)

        assert isinstance(result, ParseSuccess)
        assert result.document.file_type == FileType.DOCX


class TestGetParser:
    def test_returns_matching_parser(self) -> None:
        parser = DoclingParser()
        found = get_parser(FileType.PDF, [parser])
        assert found is parser

    def test_returns_none_for_unsupported(self) -> None:
        parser = DoclingParser()
        found = get_parser(FileType.TXT, [parser])
        assert found is None

    def test_returns_first_matching_parser(self) -> None:
        parser1 = DoclingParser()
        parser2 = DoclingParser()
        found = get_parser(FileType.PDF, [parser1, parser2])
        assert found is parser1
