from __future__ import annotations

from pathlib import Path
from typing import Literal

from rag.types import ClassificationResult, FileType


def classify(file_path: str, folder_path: str) -> ClassificationResult:
    """Classify a file by type and estimate complexity."""
    path = Path(file_path)
    ext = path.suffix.lower().lstrip(".")

    try:
        file_type = FileType(ext)
    except ValueError:
        file_type = FileType.TXT  # fallback

    likely_scanned = False
    ocr_enabled = file_type in (FileType.PDF, FileType.DOCX)

    size = path.stat().st_size if path.exists() else 0
    complexity: Literal["low", "medium", "high"]
    if file_type == FileType.PDF and size > 5_000_000:
        complexity = "high"
    elif file_type in (FileType.PDF, FileType.DOCX):
        complexity = "medium"
    else:
        complexity = "low"

    return ClassificationResult(
        file_type=file_type,
        likely_scanned=likely_scanned,
        ocr_enabled=ocr_enabled,
        folder_context=folder_path,
        complexity_estimate=complexity,
    )
