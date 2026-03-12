from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from typing import TYPE_CHECKING

from rag.results import (
    SectionSummaryError,
    SectionSummarySuccess,
    SummaryError,
    SummarySuccess,
)

if TYPE_CHECKING:
    from rag.config import SummarizationConfig
    from rag.results import SectionSummaryResult, SummaryResult

logger = logging.getLogger(__name__)

MAX_EXCERPT_CHARS = 5000

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
- "summary_l1": A short phrase (3-8 words) describing the document
- "summary_l2": 1-2 sentences summarizing the document
- "summary_l3": A paragraph (3-5 sentences) summarizing the key points
- "key_topics": A list of 3-7 key topic strings
- "doc_type_guess": The document type (e.g. "report", "meeting notes", "readme", "plan")

Return ONLY the JSON object, no other text.

Title: {title}
File type: {file_type}

Document excerpt:
{excerpt}
"""

SECTION_PROMPT_TEMPLATE = """\
Summarize the following section from a document. Return a JSON object with:
- "section_summary": 1-2 sentences summarizing this section
- "section_summary_l2": A short phrase (3-8 words) for this section (optional, null ok)

Return ONLY the JSON object, no other text.

Document context: {doc_context}
Section heading: {heading}

Section text:
{text}
"""


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

    # Try finding first { ... } block
    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        try:
            result = json.loads(brace_match.group(0))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

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

    def _cli_env(self) -> dict[str, str]:
        """Build env for subprocess, stripping vars that prevent nested CLI sessions."""
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        return env

    def _run_cli(self, prompt: str) -> str | None:
        """Run the LLM CLI with the given prompt. Returns cleaned stdout or None."""
        input_mode = self._config.input_mode

        if input_mode == "arg":
            cmd = [self._config.command, *self._config.args, prompt]
            stdin_text = None
        else:
            cmd = [self._config.command, *self._config.args]
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
