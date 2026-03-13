from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from rag.pipeline.parser.base import get_parser
from rag.pipeline.parser.docling_parser import DoclingParser
from rag.results import ParseError, ParseSuccess
from rag.types import FileType


def _patch_worker(
    parser: DoclingParser,
    result: dict[str, Any],
    *,
    timeout: bool = False,
    crash: bool = False,
) -> None:
    """Replace the parser's worker with a mock pipe that returns `result`.

    If timeout=True, pipe.poll() returns False (simulates timeout).
    If crash=True, pipe.recv() raises BrokenPipeError.
    """
    mock_pipe = MagicMock()
    if timeout:
        mock_pipe.poll.return_value = False
    elif crash:
        mock_pipe.poll.return_value = True
        mock_pipe.recv.side_effect = BrokenPipeError("worker died")
    else:
        mock_pipe.poll.return_value = True
        mock_pipe.recv.return_value = result

    mock_proc = MagicMock()
    mock_proc.is_alive.return_value = True

    parser._pipe = mock_pipe
    parser._worker = mock_proc


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


class TestDoclingParserWorker:
    def test_parse_success_via_mock_worker(self, tmp_path: Path) -> None:
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

        parser = DoclingParser()
        _patch_worker(parser, mock_result)
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

    def test_parse_error_from_worker(self, tmp_path: Path) -> None:
        pdf_file = tmp_path / "bad.pdf"
        pdf_file.write_bytes(b"not a real pdf")

        parser = DoclingParser()
        _patch_worker(parser, {"status": "error", "error": "Docling could not parse the file"})
        result = parser.parse(str(pdf_file), ocr_enabled=False)

        assert isinstance(result, ParseError)
        assert "could not parse" in result.error.lower()

    def test_parse_timeout(self, tmp_path: Path) -> None:
        pdf_file = tmp_path / "slow.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 content")

        parser = DoclingParser()
        _patch_worker(parser, {}, timeout=True)
        result = parser.parse(str(pdf_file), ocr_enabled=False)

        assert isinstance(result, ParseError)
        assert "timed out" in result.error.lower()

    def test_worker_crash_returns_error(self, tmp_path: Path) -> None:
        pdf_file = tmp_path / "crash.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 content")

        parser = DoclingParser()
        _patch_worker(parser, {}, crash=True)
        result = parser.parse(str(pdf_file), ocr_enabled=False)

        assert isinstance(result, ParseError)
        assert "crashed" in result.error.lower()

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
            "docling should only be imported inside the worker function"
        )

    def test_ocr_enabled_sent_to_worker(self, tmp_path: Path) -> None:
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

        parser = DoclingParser()
        _patch_worker(parser, mock_result)
        result = parser.parse(str(pdf_file), ocr_enabled=True)

        assert isinstance(result, ParseSuccess)
        assert result.document.ocr_required is True

        # Verify the pipe.send was called with (file_path, True)
        parser._pipe.send.assert_called_once()  # type: ignore[union-attr]
        sent_args = parser._pipe.send.call_args[0][0]  # type: ignore[union-attr]
        assert sent_args[1] is True  # ocr_enabled


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

        parser = DoclingParser()
        _patch_worker(parser, mock_result)
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
