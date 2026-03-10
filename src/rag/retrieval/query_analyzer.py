from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    """Result of analyzing a search query for classification and filter hints."""

    classification: str  # "broad" or "specific"
    folder_hint: str | None = None
    date_hint: str | None = None


def analyze_query(query: str) -> QueryAnalysis:
    """Analyze a query to determine broad/specific and extract filter hints."""
    words = query.strip().split()

    # Broad queries: short, question-like, overview-seeking
    is_broad = len(words) <= 4 or any(
        w.lower() in ("overview", "summary", "what", "describe", "explain") for w in words
    )

    # Extract folder hint (look for path-like patterns or quoted folder names)
    folder_hint: str | None = None
    # Match explicit "in folder X", "from folder X", or path-like values containing /
    folder_match = re.search(
        r'(?:in\s+folder|from\s+folder|folder)\s+["\']?([/\w.-]+)["\']?',
        query,
        re.IGNORECASE,
    )
    if not folder_match:
        # Match "in/from" only when followed by a path (contains /)
        folder_match = re.search(
            r'(?:in|from)\s+["\']?([/\w.-]*[/][/\w.-]*)["\']?',
            query,
            re.IGNORECASE,
        )
    if folder_match:
        folder_hint = folder_match.group(1)

    # Extract date hint
    date_hint: str | None = None
    date_match = re.search(r"(?:since|after|before|from)\s+(\d{4}[-/]\d{2}(?:[-/]\d{2})?)", query)
    if date_match:
        date_hint = date_match.group(1)

    return QueryAnalysis(
        classification="broad" if is_broad else "specific",
        folder_hint=folder_hint,
        date_hint=date_hint,
    )
