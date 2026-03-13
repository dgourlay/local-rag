from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from typing import TYPE_CHECKING

from rag.results import (
    CombinedSectionSummary,
    CombinedSummaryError,
    CombinedSummarySuccess,
    SectionSummaryError,
    SectionSummarySuccess,
    SummaryError,
    SummarySuccess,
)

if TYPE_CHECKING:
    from rag.config import SummarizationConfig
    from rag.results import CombinedSummaryResult, SectionSummaryResult, SummaryResult

logger = logging.getLogger(__name__)

MAX_EXCERPT_CHARS = 5000
_COMBINED_PROMPT_CHAR_LIMIT = 80_000

# Known CLI tool presets: (args, input_mode)
_CLI_PRESETS: dict[str, tuple[list[str], str]] = {
    "claude": (["--print"], "stdin"),
    "kiro-cli": (["chat", "--no-interactive", "--wrap", "never"], "arg"),
    "codex": (
        ["exec", "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral",
         "-o", "/dev/stdout", "-"],
        "stdin",
    ),
}

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def get_cli_preset(command: str) -> tuple[list[str], str] | None:
    """Return (args, input_mode) preset for a known CLI tool, or None."""
    return _CLI_PRESETS.get(command)

DOCUMENT_PROMPT_TEMPLATE = """\
Analyze the following document and return a JSON object with these fields:
- "summary_8w": A short phrase (~8 words) describing the document
- "summary_16w": A single sentence (~16 words) capturing the main point
- "summary_32w": 1-2 sentences (~32 words) summarizing the document
- "summary_64w": A short paragraph (~64 words) covering the key points
- "summary_128w": A detailed paragraph (~128 words) summarizing the document comprehensively
- "key_topics": A list of 3-7 key topic strings
- "doc_type_guess": The document type (e.g. "report", "meeting notes", "readme", "plan")

Each summary level must be independently understandable (not building on the previous level).

Return ONLY the JSON object, no other text.

Title: {title}
File type: {file_type}

Document excerpt:
{excerpt}
"""

SECTION_PROMPT_TEMPLATE = """\
Summarize the following section from a document. Return a JSON object with:
- "section_summary_8w": A short phrase (~8 words) for this section
- "section_summary_32w": 1-2 sentences (~32 words) summarizing this section
- "section_summary_128w": A detailed paragraph (~128 words) covering the section's key points

Each summary level must be independently understandable (not building on the previous level).

Return ONLY the JSON object, no other text.

Document context: {doc_context}
Section heading: {heading}

Section text:
{text}
"""


COMBINED_PROMPT_TEMPLATE = """\
Analyze the following document and its sections. Return a single JSON object with:

Document-level fields:
- "summary_8w": A short phrase (~8 words) describing the document
- "summary_16w": A single sentence (~16 words) capturing the main point
- "summary_32w": 1-2 sentences (~32 words) summarizing the document
- "summary_64w": A short paragraph (~64 words) covering the key points
- "summary_128w": A detailed paragraph (~128 words) summarizing the document comprehensively
- "key_topics": A list of 3-7 key topic strings
- "doc_type_guess": The document type (e.g. "report", "meeting notes", "readme", "plan")
- "sections": An array of section summary objects (one per section, in order)

Each section object must have:
- "heading": The section heading (as given below)
- "section_summary_8w": A short phrase (~8 words) for this section
- "section_summary_32w": 1-2 sentences (~32 words) summarizing this section
- "section_summary_128w": A detailed paragraph (~128 words) covering the section's key points

Each summary level must be independently understandable (not building on the previous level).

Return ONLY the JSON object, no other text.

Title: {title}
File type: {file_type}

Document excerpt:
{excerpt}

Sections:
{sections_text}
"""

BATCH_SECTION_PROMPT_TEMPLATE = """\
Summarize the following sections from a document. Return a JSON object with a single field:
- "sections": An array of section summary objects (one per section, in the same order)

Each section object must have:
- "heading": The section heading (as given below)
- "section_summary_8w": A short phrase (~8 words) for this section
- "section_summary_32w": 1-2 sentences (~32 words) summarizing this section
- "section_summary_128w": A detailed paragraph (~128 words) covering the section's key points

Each summary level must be independently understandable (not building on the previous level).

Return ONLY the JSON object, no other text.

Document context: {doc_context}

Sections:
{sections_text}
"""


def _format_sections_text(sections: list[tuple[str | None, str]]) -> str:
    """Format section (heading, text) pairs into numbered text for prompts."""
    parts: list[str] = []
    for i, (heading, text) in enumerate(sections, 1):
        excerpt = text[:MAX_EXCERPT_CHARS]
        parts.append(f"--- Section {i}: {heading or 'Untitled section'} ---\n{excerpt}")
    return "\n\n".join(parts)


def _extract_json(text: str) -> dict[str, object] | None:
    """Extract JSON from CLI output, handling markdown fences."""
    text = text.strip()
    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Try finding outermost { ... } block (handles nested braces)
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        result = json.loads(text[start : i + 1])
                        if isinstance(result, dict):
                            return result
                    except json.JSONDecodeError:
                        pass
                    break

    return None


class CliSummarizer:
    """Summarizer that shells out to a configured LLM CLI tool."""

    def __init__(self, config: SummarizationConfig) -> None:
        self._config = config
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = (
                self._config.enabled and shutil.which(self._config.command) is not None
            )
        return self._available

    def summarize_document(self, text: str, title: str | None, file_type: str) -> SummaryResult:
        if not self.available:
            return SummaryError(error="Summarizer not available")

        excerpt = text[:MAX_EXCERPT_CHARS]
        prompt = DOCUMENT_PROMPT_TEMPLATE.format(
            title=title or "Untitled",
            file_type=file_type,
            excerpt=excerpt,
        )

        stdout = self._run_cli(prompt)
        if stdout is None:
            return SummaryError(error="CLI call failed or timed out")

        parsed = _extract_json(stdout)
        if parsed is None:
            return SummaryError(error=f"Could not parse JSON from CLI output: {stdout[:200]}")

        try:
            return SummarySuccess.model_validate(parsed)
        except Exception as e:
            return SummaryError(error=f"Validation failed: {e}")

    def summarize_section(
        self, text: str, heading: str | None, doc_context: str
    ) -> SectionSummaryResult:
        if not self.available:
            return SectionSummaryError(error="Summarizer not available")

        excerpt = text[:MAX_EXCERPT_CHARS]
        prompt = SECTION_PROMPT_TEMPLATE.format(
            doc_context=doc_context,
            heading=heading or "Untitled section",
            text=excerpt,
        )

        stdout = self._run_cli(prompt)
        if stdout is None:
            return SectionSummaryError(error="CLI call failed or timed out")

        parsed = _extract_json(stdout)
        if parsed is None:
            return SectionSummaryError(
                error=f"Could not parse JSON from CLI output: {stdout[:200]}"
            )

        try:
            return SectionSummarySuccess.model_validate(parsed)
        except Exception as e:
            return SectionSummaryError(error=f"Validation failed: {e}")

    def summarize_combined(
        self,
        text: str,
        title: str | None,
        file_type: str,
        sections: list[tuple[str | None, str]],
    ) -> CombinedSummaryResult:
        """Summarize document + all sections in as few LLM calls as possible."""
        if not self.available:
            return CombinedSummaryError(error="Summarizer not available")

        excerpt = text[:MAX_EXCERPT_CHARS]
        sections_text = _format_sections_text(sections)
        total_chars = len(excerpt) + len(sections_text)

        if total_chars < _COMBINED_PROMPT_CHAR_LIMIT:
            return self._summarize_combined_single(
                excerpt, title or "Untitled", file_type, sections_text, len(sections)
            )

        # Over threshold: split into doc summary + batched sections
        return self._summarize_combined_split(text, title, file_type, sections)

    def _summarize_combined_single(
        self,
        excerpt: str,
        title: str,
        file_type: str,
        sections_text: str,
        num_sections: int,
    ) -> CombinedSummaryResult:
        """Single-call combined summarization."""
        prompt = COMBINED_PROMPT_TEMPLATE.format(
            title=title,
            file_type=file_type,
            excerpt=excerpt,
            sections_text=sections_text,
        )

        stdout = self._run_cli(prompt)
        if stdout is None:
            return CombinedSummaryError(error="CLI call failed or timed out")

        parsed = _extract_json(stdout)
        if parsed is None:
            return CombinedSummaryError(
                error=f"Could not parse JSON from CLI output: {stdout[:200]}"
            )

        try:
            return CombinedSummarySuccess.model_validate(parsed)
        except Exception as e:
            return CombinedSummaryError(error=f"Validation failed: {e}")

    def _summarize_combined_split(
        self,
        text: str,
        title: str | None,
        file_type: str,
        sections: list[tuple[str | None, str]],
    ) -> CombinedSummaryResult:
        """Split summarization: doc summary first, then batched sections."""
        doc_result = self.summarize_document(text, title, file_type)
        if not isinstance(doc_result, SummarySuccess):
            return CombinedSummaryError(error=f"Document summary failed: {doc_result.error}")

        doc_context = f"{title or 'Untitled'} ({file_type})"
        section_results = self.summarize_sections_batch(sections, doc_context)

        return CombinedSummarySuccess(
            summary_8w=doc_result.summary_8w,
            summary_16w=doc_result.summary_16w,
            summary_32w=doc_result.summary_32w,
            summary_64w=doc_result.summary_64w,
            summary_128w=doc_result.summary_128w,
            key_topics=doc_result.key_topics,
            doc_type_guess=doc_result.doc_type_guess,
            sections=section_results,
        )

    def summarize_sections_batch(
        self,
        sections: list[tuple[str | None, str]],
        doc_context: str,
    ) -> list[CombinedSectionSummary]:
        """Batch sections into groups that fit under the char limit."""
        if not sections:
            return []

        batches = self._group_sections_into_batches(sections)
        all_results: list[CombinedSectionSummary] = []

        for batch in batches:
            sections_text = _format_sections_text(batch)
            prompt = BATCH_SECTION_PROMPT_TEMPLATE.format(
                doc_context=doc_context,
                sections_text=sections_text,
            )

            stdout = self._run_cli(prompt)
            if stdout is None:
                logger.warning("Batch section summarization CLI call failed")
                continue

            parsed = _extract_json(stdout)
            if parsed is None:
                logger.warning("Could not parse JSON from batch section output")
                continue

            raw_sections = parsed.get("sections")
            if not isinstance(raw_sections, list):
                logger.warning("Batch section output missing 'sections' array")
                continue

            for raw_sec in raw_sections:
                if isinstance(raw_sec, dict):
                    try:
                        all_results.append(CombinedSectionSummary.model_validate(raw_sec))
                    except Exception:
                        logger.warning("Failed to validate section summary in batch")

        return all_results

    def _group_sections_into_batches(
        self, sections: list[tuple[str | None, str]]
    ) -> list[list[tuple[str | None, str]]]:
        """Group sections into batches that fit under the char limit."""
        batches: list[list[tuple[str | None, str]]] = []
        current_batch: list[tuple[str | None, str]] = []
        current_size = 0

        for heading, text in sections:
            excerpt = text[:MAX_EXCERPT_CHARS]
            section_size = len(excerpt) + len(heading or "Untitled section") + 50
            if current_batch and current_size + section_size > _COMBINED_PROMPT_CHAR_LIMIT:
                batches.append(current_batch)
                current_batch = []
                current_size = 0
            current_batch.append((heading, text))
            current_size += section_size

        if current_batch:
            batches.append(current_batch)

        return batches

    def _cli_env(self) -> dict[str, str]:
        """Build env for subprocess, stripping vars that prevent nested CLI sessions."""
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        return env

    def _run_cli(self, prompt: str) -> str | None:
        """Run the LLM CLI with the given prompt. Returns cleaned stdout or None."""
        input_mode = self._config.input_mode

        extra_args = self._config.args or []
        if input_mode == "arg":
            cmd = [self._config.command, *extra_args, prompt]
            stdin_text = None
        else:
            cmd = [self._config.command, *extra_args]
            stdin_text = prompt

        env = self._cli_env()

        try:
            result = subprocess.run(
                cmd,
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=self._config.timeout_seconds,
                env=env,
            )
            if result.returncode != 0:
                logger.warning(
                    "CLI %s exited with code %d. stderr: %s",
                    self._config.command,
                    result.returncode,
                    result.stderr[:500],
                )
                # Retry once
                result = subprocess.run(
                    cmd,
                    input=stdin_text,
                    capture_output=True,
                    text=True,
                    timeout=self._config.timeout_seconds * 2,
                    env=env,
                )
                if result.returncode != 0:
                    logger.error(
                        "CLI retry failed with code %d. stderr: %s",
                        result.returncode,
                        result.stderr[:500],
                    )
                    return None

            return _clean_cli_output(result.stdout)

        except subprocess.TimeoutExpired:
            logger.error(
                "CLI %s timed out after %ds", self._config.command, self._config.timeout_seconds
            )
            return None
        except FileNotFoundError:
            logger.error("CLI %s not found", self._config.command)
            self._available = False
            return None


def _clean_cli_output(text: str) -> str:
    """Strip ANSI escape codes, leading '> ' prefix, and bare format hints from CLI output."""
    text = _ANSI_ESCAPE_RE.sub("", text)
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        line = line.lstrip("> ") if line.startswith("> ") else line
        # Skip bare format hint lines (e.g. "json" before a code block)
        stripped = line.strip()
        if stripped in ("json", "```json", "```"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)
