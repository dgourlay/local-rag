from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path

from rag.results import ParseError, ParseSuccess
from rag.types import FileType, ParsedDocument, ParsedSection

ParseSuccess.model_rebuild()


class TextParser:
    @property
    def supported_types(self) -> set[FileType]:
        return {FileType.TXT, FileType.MD}

    def parse(
        self,
        file_path: str,
        ocr_enabled: bool,
        content_hash: str | None = None,
    ) -> ParseSuccess | ParseError:
        path = Path(file_path)
        if not path.is_file():
            return ParseError(error=f"File not found: {file_path}", file_path=file_path)

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = path.read_text(encoding="latin-1")
            except Exception as e:
                return ParseError(error=f"Cannot read file: {e}", file_path=file_path)
        except Exception as e:
            return ParseError(error=f"Cannot read file: {e}", file_path=file_path)

        if not content.strip():
            return ParseError(error="File is empty", file_path=file_path)

        # Use pre-computed hash when available, otherwise compute it
        if content_hash is not None:
            raw_hash = content_hash
        else:
            raw_hash = hashlib.sha256(path.read_bytes()).hexdigest()

        ext = path.suffix.lower().lstrip(".")
        file_type = FileType(ext)
        doc_id = str(uuid.uuid4())

        if file_type == FileType.MD:
            sections = _parse_markdown(content)
        else:
            sections = [ParsedSection(heading=None, order=0, text=content)]

        return ParseSuccess(
            document=ParsedDocument(
                doc_id=doc_id,
                title=path.stem,
                file_type=file_type,
                sections=sections,
                raw_content_hash=raw_hash,
            )
        )


def _parse_markdown(content: str) -> list[ParsedSection]:
    """Split markdown on headings (# ## ### etc.) into sections."""
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    matches = list(heading_pattern.finditer(content))

    if not matches:
        return [ParsedSection(heading=None, order=0, text=content)]

    sections: list[ParsedSection] = []

    # Text before first heading (preamble)
    if matches[0].start() > 0:
        preamble = content[: matches[0].start()].strip()
        if preamble:
            sections.append(ParsedSection(heading=None, order=0, text=preamble))

    for i, match in enumerate(matches):
        heading_text = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        text = content[start:end].strip()
        sections.append(
            ParsedSection(
                heading=heading_text,
                order=len(sections),
                text=text,
            )
        )

    return sections
