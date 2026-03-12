from __future__ import annotations

import hashlib
import multiprocessing
import uuid
from pathlib import Path
from typing import Any

from rag.results import ParseError, ParseSuccess
from rag.types import FileType, ParsedDocument, ParsedSection

# Resolve forward refs from results.py (caused by `from __future__ import annotations`)
ParseSuccess.model_rebuild()
ParseError.model_rebuild()


def _parse_in_subprocess(
    file_path: str,
    ocr_enabled: bool,
    result_queue: multiprocessing.Queue[dict[str, Any]],
) -> None:
    """Run Docling parsing in a child process for memory isolation.

    All docling imports happen here — never in the parent process.
    """
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(file_path)
        doc = result.document

        sections: list[dict[str, Any]] = []
        current_heading: str | None = None
        current_text_parts: list[str] = []
        current_page_start: int | None = None
        current_page_end: int | None = None
        order = 0

        for item, _level in doc.iterate_items():
            text = getattr(item, "text", "") or ""
            label = str(getattr(item, "label", ""))

            # Extract page info from prov if available
            prov_list = getattr(item, "prov", None)
            if prov_list:
                for prov in prov_list:
                    page_no = getattr(prov, "page_no", None)
                    if page_no is not None:
                        if current_page_start is None:
                            current_page_start = page_no
                        current_page_end = page_no

            if "heading" in label.lower():
                # Flush previous section
                if current_text_parts:
                    sections.append(
                        {
                            "heading": current_heading,
                            "order": order,
                            "text": "\n".join(current_text_parts),
                            "page_start": current_page_start,
                            "page_end": current_page_end,
                        }
                    )
                    order += 1
                current_heading = text
                current_text_parts = []
                current_page_start = None
                current_page_end = None
            elif text.strip():
                current_text_parts.append(text)

        # Flush last section
        if current_text_parts:
            sections.append(
                {
                    "heading": current_heading,
                    "order": order,
                    "text": "\n".join(current_text_parts),
                    "page_start": current_page_start,
                    "page_end": current_page_end,
                }
            )

        # If no sections extracted, treat whole doc as one section
        if not sections:
            full_text = doc.export_to_text() if hasattr(doc, "export_to_text") else str(doc)
            sections.append(
                {
                    "heading": None,
                    "order": 0,
                    "text": full_text,
                    "page_start": None,
                    "page_end": None,
                }
            )

        title = getattr(doc, "name", None) or Path(file_path).stem
        result_queue.put(
            {
                "status": "success",
                "sections": sections,
                "title": title,
            }
        )
    except Exception as e:
        result_queue.put(
            {
                "status": "error",
                "error": str(e),
            }
        )


_SUBPROCESS_TIMEOUT_SECONDS = 600


class DoclingParser:
    """Parser for PDF and DOCX files using Docling, run in a subprocess."""

    @property
    def supported_types(self) -> set[FileType]:
        return {FileType.PDF, FileType.DOCX}

    def parse(self, file_path: str, ocr_enabled: bool) -> ParseSuccess | ParseError:
        path = Path(file_path)
        if not path.is_file():
            return ParseError(error=f"File not found: {file_path}", file_path=file_path)

        # Compute content hash
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                h.update(block)
        raw_hash = h.hexdigest()

        # Run Docling in a subprocess for memory isolation
        ctx = multiprocessing.get_context("spawn")
        result_queue: multiprocessing.Queue[dict[str, Any]] = ctx.Queue()
        proc = ctx.Process(
            target=_parse_in_subprocess,
            args=(file_path, ocr_enabled, result_queue),
        )
        proc.start()
        proc.join(timeout=_SUBPROCESS_TIMEOUT_SECONDS)

        if proc.is_alive():
            proc.terminate()
            proc.join()
            return ParseError(
                error=f"Parsing timed out after {_SUBPROCESS_TIMEOUT_SECONDS} seconds",
                file_path=file_path,
            )

        if result_queue.empty():
            return ParseError(error="Subprocess returned no result", file_path=file_path)

        result: dict[str, Any] = result_queue.get()
        if result["status"] == "error":
            return ParseError(error=result["error"], file_path=file_path)

        sections = [ParsedSection(**s) for s in result["sections"]]
        doc_id = str(uuid.uuid4())

        ext = path.suffix.lower().lstrip(".")
        file_type = FileType(ext)

        return ParseSuccess(
            document=ParsedDocument(
                doc_id=doc_id,
                title=result.get("title"),
                file_type=file_type,
                sections=sections,
                ocr_required=ocr_enabled,
                raw_content_hash=raw_hash,
            )
        )
