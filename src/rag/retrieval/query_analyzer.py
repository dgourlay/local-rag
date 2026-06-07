from __future__ import annotations

import re
from dataclasses import dataclass

_BROAD_KEYWORDS = frozenset(
    {
        "overview",
        "summary",
        "summarize",
        "general",
        "explain",
        "describe",
        "introduction",
    }
)

_NAVIGATIONAL_KEYWORDS = frozenset(
    {
        "where",
        "find",
        "locate",
        "location",
        "path",
    }
)

# Regex patterns for code identifiers: snake_case, camelCase, dotted.paths
_CODE_IDENT_RE = re.compile(r"[a-z]+_[a-z]|[a-z][A-Z]|[a-zA-Z]+\.[a-zA-Z]+")
_VERSION_RE = re.compile(r"\d+\.\d+")
_PATH_LIKE_RE = re.compile(r"[/\\][\w.-]+")


@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    """Result of analyzing a search query for classification and filter hints."""

    classification: str  # "broad", "specific", or "navigational"
    folder_hint: str | None = None
    date_hint: str | None = None


def _score_broad(query: str, words: list[str]) -> float:
    """Score how broad/overview-seeking the query is."""
    score = 0.0
    lower_words = [w.lower() for w in words]

    # Broad keyword matches
    broad_count = sum(1 for w in lower_words if w in _BROAD_KEYWORDS)
    score += broad_count * 2.0

    # "what is X" with short X (3 words or fewer after "what is")
    if query.lower().startswith("what is") and len(words) <= 5:
        score += 2.0

    # Short queries without specific signals tend to be broad
    if len(words) <= 3:
        score += 1.0

    return score


def _score_specific(query: str, words: list[str]) -> float:
    """Score how specific/targeted the query is."""
    score = 0.0
    lower_words = [w.lower() for w in words]

    # Code identifiers (snake_case, camelCase, dotted.paths)
    code_matches = _CODE_IDENT_RE.findall(query)
    score += len(code_matches) * 2.0

    # Numbers and version strings
    version_matches = _VERSION_RE.findall(query)
    score += len(version_matches) * 1.5

    # Long questions without broad keywords
    has_broad = any(w in _BROAD_KEYWORDS for w in lower_words)
    if len(words) >= 6 and not has_broad:
        score += 1.5

    return score


def _score_navigational(query: str, words: list[str]) -> float:
    """Score how navigational the query is (seeking location/path)."""
    score = 0.0
    lower_words = [w.lower() for w in words]

    # Navigational keywords
    nav_count = sum(1 for w in lower_words if w in _NAVIGATIONAL_KEYWORDS)
    score += nav_count * 2.0

    # Path-like patterns
    path_matches = _PATH_LIKE_RE.findall(query)
    score += len(path_matches) * 2.0

    # Short noun queries (2-4 words, no question words, no broad keywords)
    question_words = {"what", "how", "why", "when", "which"}
    has_question = any(w in question_words for w in lower_words)
    has_broad = any(w in _BROAD_KEYWORDS for w in lower_words)
    if 2 <= len(words) <= 4 and not has_question and not has_broad:
        score += 1.0

    return score


def _classify(query: str, words: list[str]) -> str:
    """Classify query using argmax of scores. Tie-break: specific > navigational > broad."""
    s_broad = _score_broad(query, words)
    s_specific = _score_specific(query, words)
    s_navigational = _score_navigational(query, words)

    # Argmax with tie-break order: specific > navigational > broad
    if s_specific >= s_navigational and s_specific >= s_broad:
        return "specific"
    if s_navigational >= s_broad:
        return "navigational"
    return "broad"


def analyze_query(query: str) -> QueryAnalysis:
    """Analyze a query to determine classification and extract filter hints."""
    words = query.strip().split()

    classification = _classify(query, words)

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
        classification=classification,
        folder_hint=folder_hint,
        date_hint=date_hint,
    )
