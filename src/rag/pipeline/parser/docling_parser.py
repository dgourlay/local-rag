from __future__ import annotations

import contextlib
import hashlib
import logging
import multiprocessing
import uuid
from pathlib import Path
from typing import Any

from rag.results import ParseError, ParseSuccess
from rag.types import FileType, ParsedDocument, ParsedSection

# Resolve forward refs from results.py (caused by `from __future__ import annotations`)
ParseSuccess.model_rebuild()
ParseError.model_rebuild()

logger = logging.getLogger(__name__)

_PARSE_TIMEOUT_SECONDS = 360


def _worker_loop(
    request_pipe: multiprocessing.connection.Connection,
) -> None:
    """Long-lived worker process that loads Docling once and handles parse requests.

    Protocol: receives (file_path, ocr_enabled) tuples, sends back result dicts.
    A None request signals the worker to exit.
    Lazily caches two converters: one with OCR enabled, one without.
    """
    import warnings

    warnings.filterwarnings("ignore", message=".*DrawingML.*")

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except Exception as e:
        # Signal that init failed — send error for any request until we exit
        while True:
            try:
                req = request_pipe.recv()
            except EOFError:
                return
            if req is None:
                return
            request_pipe.send({"status": "error", "error": f"Docling init failed: {e}"})
        return

    # Lazily-created converters: keyed by ocr_enabled bool
    converters: dict[bool, DocumentConverter] = {}

    def _get_converter(ocr_enabled: bool) -> DocumentConverter:
        if ocr_enabled not in converters:
            if ocr_enabled:
                converters[ocr_enabled] = DocumentConverter()
            else:
                pipeline_options = PdfPipelineOptions(do_ocr=False)
                converters[ocr_enabled] = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=pipeline_options,
                        ),
                    },
                )
        return converters[ocr_enabled]

    while True:
        try:
            req = request_pipe.recv()
        except EOFError:
            return
        if req is None:
            return

        file_path, ocr_enabled = req
        try:
            converter = _get_converter(ocr_enabled)
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

                prov_list = getattr(item, "prov", None)
                if prov_list:
                    for prov in prov_list:
                        page_no = getattr(prov, "page_no", None)
                        if page_no is not None:
                            if current_page_start is None:
                                current_page_start = page_no
                            current_page_end = page_no

                if "heading" in label.lower():
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
            request_pipe.send(
                {
                    "status": "success",
                    "sections": sections,
                    "title": title,
                }
            )
        except Exception as e:
            request_pipe.send({"status": "error", "error": str(e)})


class DoclingParser:
    """Parser for PDF and DOCX files using Docling.

    Uses a persistent worker subprocess so Docling's ML models are loaded once
    and reused across parse calls, while keeping memory isolated from the parent.
    """

    def __init__(self) -> None:
        self._worker: multiprocessing.Process | None = None
        self._pipe: multiprocessing.connection.Connection | None = None

    @property
    def supported_types(self) -> set[FileType]:
        return {FileType.PDF, FileType.DOCX}

    def _ensure_worker(self) -> multiprocessing.connection.Connection:
        """Start the worker process if it isn't running."""
        if self._worker is not None and self._worker.is_alive() and self._pipe is not None:
            return self._pipe

        # Clean up any dead worker
        self._shutdown_worker()

        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self._worker = ctx.Process(target=_worker_loop, args=(child_conn,), daemon=True)
        self._worker.start()
        # Close the child end in the parent so only the worker holds it
        child_conn.close()
        self._pipe = parent_conn
        logger.info("Started Docling worker process (pid=%s)", self._worker.pid)
        return self._pipe

    def _shutdown_worker(self) -> None:
        """Gracefully shut down the worker process."""
        if self._pipe is not None:
            with contextlib.suppress(BrokenPipeError, OSError):
                self._pipe.send(None)
            with contextlib.suppress(OSError):
                self._pipe.close()
            self._pipe = None
        if self._worker is not None:
            self._worker.join(timeout=5)
            if self._worker.is_alive():
                self._worker.terminate()
                self._worker.join(timeout=5)
            self._worker = None

    def parse(
        self,
        file_path: str,
        ocr_enabled: bool,
        content_hash: str | None = None,
    ) -> ParseSuccess | ParseError:
        path = Path(file_path)
        if not path.is_file():
            return ParseError(error=f"File not found: {file_path}", file_path=file_path)

        # Use pre-computed hash when available, otherwise compute it
        if content_hash is not None:
            raw_hash = content_hash
        else:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(8192), b""):
                    h.update(block)
            raw_hash = h.hexdigest()

        # Send request to persistent worker
        try:
            pipe = self._ensure_worker()
            pipe.send((file_path, ocr_enabled))

            # Wait for response with timeout
            if not pipe.poll(timeout=_PARSE_TIMEOUT_SECONDS):
                logger.error(
                    "Docling worker timed out after %ds for %s",
                    _PARSE_TIMEOUT_SECONDS, file_path,
                )
                self._shutdown_worker()
                return ParseError(
                    error=f"Parsing timed out after {_PARSE_TIMEOUT_SECONDS} seconds",
                    file_path=file_path,
                )

            result: dict[str, Any] = pipe.recv()
        except (BrokenPipeError, EOFError, OSError) as e:
            logger.error("Docling worker crashed while parsing %s: %s", file_path, e)
            self._shutdown_worker()
            return ParseError(error=f"Worker process crashed: {e}", file_path=file_path)

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

    def __del__(self) -> None:
        self._shutdown_worker()
