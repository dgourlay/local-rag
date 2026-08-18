from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from rag.config import SummarizationConfig
from rag.pipeline.summarizer import (
    _COMBINED_PROMPT_CHAR_LIMIT,
    BATCH_SECTION_PROMPT_TEMPLATE,
    COMBINED_PROMPT_TEMPLATE,
    MAX_EXCERPT_CHARS,
    CliSummarizer,
    _extract_json,
    _format_sections_text,
)
from rag.results import (
    CombinedSummaryError,
    CombinedSummarySuccess,
)

# --- Helpers ---


def _make_summarizer(*, command: str = "echo", enabled: bool = True) -> CliSummarizer:
    """Create a CliSummarizer with a basic config."""
    config = SummarizationConfig(
        enabled=enabled,
        command=command,
        args=[],
        input_mode="stdin",
        timeout_seconds=10,
    )
    return CliSummarizer(config)


def _make_combined_json(num_sections: int = 2) -> str:
    """Build a valid combined JSON response."""
    data = {
        "summary_8w": "Short doc summary phrase here",
        "summary_16w": "A single sentence capturing the main point of the document.",
        "summary_32w": "A moderate summary covering the key ideas in this test document.",
        "summary_64w": "An extended summary that goes into more detail about the content.",
        "summary_128w": "A comprehensive detailed summary of the document.",
        "key_topics": ["topic1", "topic2"],
        "doc_type_guess": "report",
        "sections": [
            {
                "heading": f"Section {i}",
                "section_summary_8w": f"Short summary for section {i}",
                "section_summary_32w": f"Moderate summary for section {i}.",
                "section_summary_128w": f"Detailed summary for section {i}.",
            }
            for i in range(num_sections)
        ],
    }
    return json.dumps(data)


def _make_batch_sections_json(num_sections: int = 2) -> str:
    """Build a valid batch sections JSON response."""
    data = {
        "sections": [
            {
                "heading": f"Section {i}",
                "section_summary_8w": f"Short summary for section {i}",
                "section_summary_32w": f"Moderate summary for section {i}.",
                "section_summary_128w": f"Detailed summary for section {i}.",
            }
            for i in range(num_sections)
        ],
    }
    return json.dumps(data)


# --- Tests for _extract_json with nested structures ---


class TestExtractJsonNested:
    def test_nested_json_with_arrays(self) -> None:
        text = _make_combined_json(2)
        result = _extract_json(text)
        assert result is not None
        assert "sections" in result
        assert isinstance(result["sections"], list)
        assert len(result["sections"]) == 2

    def test_nested_json_in_markdown_fence(self) -> None:
        text = f"```json\n{_make_combined_json(1)}\n```"
        result = _extract_json(text)
        assert result is not None
        assert "sections" in result

    def test_nested_json_with_surrounding_text(self) -> None:
        text = f"Here is the result:\n{_make_combined_json(2)}\nDone."
        result = _extract_json(text)
        assert result is not None
        assert len(result["sections"]) == 2


# --- Tests for _format_sections_text ---


class TestFormatSectionsText:
    def test_basic_formatting(self) -> None:
        sections = [("Intro", "Hello world"), ("Body", "Main content")]
        result = _format_sections_text(sections)
        assert "--- Section 1: Intro ---" in result
        assert "Hello world" in result
        assert "--- Section 2: Body ---" in result

    def test_none_heading(self) -> None:
        sections = [(None, "Some text")]
        result = _format_sections_text(sections)
        assert "Untitled section" in result

    def test_truncates_to_max_excerpt(self) -> None:
        long_text = "x" * (MAX_EXCERPT_CHARS + 1000)
        sections = [("Long", long_text)]
        result = _format_sections_text(sections)
        # The section text in the result should be truncated
        assert len(result) < len(long_text)


# --- Tests for combined prompt generation ---


class TestCombinedPromptGeneration:
    def test_combined_prompt_has_all_fields(self) -> None:
        prompt = COMBINED_PROMPT_TEMPLATE.format(
            title="Test Doc",
            file_type="txt",
            excerpt="Document content here.",
            sections_text="--- Section 1: Intro ---\nHello",
        )
        assert "summary_8w" in prompt
        assert "section_summary_8w" in prompt
        assert "Test Doc" in prompt
        assert "Document content here." in prompt

    def test_batch_section_prompt_has_all_fields(self) -> None:
        prompt = BATCH_SECTION_PROMPT_TEMPLATE.format(
            doc_context="Test Doc (txt)",
            sections_text="--- Section 1: Intro ---\nHello",
        )
        assert "section_summary_8w" in prompt
        assert "Test Doc (txt)" in prompt


# --- Tests for summarize_combined ---


class TestSummarizeCombined:
    def test_not_available_returns_error(self) -> None:
        summarizer = _make_summarizer(enabled=False)
        result = summarizer.summarize_combined("text", "Title", "txt", [])
        assert isinstance(result, CombinedSummaryError)
        assert "not available" in result.error

    @patch.object(CliSummarizer, "_run_cli")
    @patch.object(CliSummarizer, "available", new_callable=lambda: property(lambda self: True))
    def test_single_call_under_threshold(self, _avail: object, mock_cli: MagicMock) -> None:
        """Under char limit, uses single combined prompt."""
        mock_cli.return_value = _make_combined_json(2)
        summarizer = _make_summarizer()

        sections = [("Intro", "Hello"), ("Body", "World")]
        result = summarizer.summarize_combined("Full text", "Doc", "txt", sections)

        assert isinstance(result, CombinedSummarySuccess)
        assert len(result.sections) == 2
        assert result.summary_8w == "Short doc summary phrase here"
        mock_cli.assert_called_once()

    @patch.object(CliSummarizer, "_run_cli")
    @patch.object(CliSummarizer, "available", new_callable=lambda: property(lambda self: True))
    def test_split_call_over_threshold(self, _avail: object, mock_cli: MagicMock) -> None:
        """Over char limit, uses doc summary + batch sections."""
        doc_json = json.dumps({
            "summary_8w": "Short",
            "summary_16w": "Medium sentence.",
            "summary_32w": "Moderate summary.",
            "summary_64w": "Extended summary.",
            "summary_128w": "Detailed summary.",
            "key_topics": ["t1"],
            "doc_type_guess": "report",
        })
        batch_json = _make_batch_sections_json(2)
        # Return doc_json for first call, then batch_json for all subsequent
        mock_cli.side_effect = [doc_json] + [batch_json] * 10

        summarizer = _make_summarizer()

        # Create sections large enough to exceed the threshold
        big_text = "x" * MAX_EXCERPT_CHARS
        sections = [(f"Section {i}", big_text) for i in range(20)]
        result = summarizer.summarize_combined("Full text", "Doc", "txt", sections)

        assert isinstance(result, CombinedSummarySuccess)
        assert result.summary_8w == "Short"
        assert len(result.sections) >= 2
        assert mock_cli.call_count >= 2

    @patch.object(CliSummarizer, "_run_cli")
    @patch.object(CliSummarizer, "available", new_callable=lambda: property(lambda self: True))
    def test_oversized_document_keeps_every_prompt_bounded(
        self, _avail: object, mock_cli: MagicMock,
    ) -> None:
        """An oversized document must never reach the CLI as one unbounded prompt."""
        doc_json = json.dumps({
            "summary_8w": "Short",
            "summary_16w": "Medium sentence.",
            "summary_32w": "Moderate summary.",
            "summary_64w": "Extended summary.",
            "summary_128w": "Detailed summary.",
            "key_topics": ["t1"],
            "doc_type_guess": "report",
        })
        batch_json = _make_batch_sections_json(2)
        prompts: list[str] = []

        def _record(prompt: str) -> str:
            # Section batches run on the summarizer's pool; list.append is atomic.
            prompts.append(prompt)
            return batch_json if "Sections:" in prompt else doc_json

        mock_cli.side_effect = _record
        summarizer = _make_summarizer()

        # ~200k chars of sections plus a 250k-char document: far over the limit
        big_text = "x" * MAX_EXCERPT_CHARS
        sections = [(f"Section {i}", big_text) for i in range(40)]
        result = summarizer.summarize_combined("y" * 250_000, "Doc", "txt", sections)

        assert isinstance(result, CombinedSummarySuccess)
        # Doc summary call plus at least two section batches
        assert len(prompts) >= 3
        # Each prompt may carry at most the char limit plus one section's worth
        # of prompt-template and heading overhead.
        for prompt in prompts:
            assert len(prompt) <= _COMBINED_PROMPT_CHAR_LIMIT + MAX_EXCERPT_CHARS, (
                f"Prompt of {len(prompt)} chars exceeds the combined prompt char limit"
            )

    @patch.object(CliSummarizer, "_run_cli")
    @patch.object(CliSummarizer, "available", new_callable=lambda: property(lambda self: True))
    def test_cli_failure_returns_error(self, _avail: object, mock_cli: MagicMock) -> None:
        mock_cli.return_value = None
        summarizer = _make_summarizer()

        result = summarizer.summarize_combined("text", "Doc", "txt", [("S", "t")])
        assert isinstance(result, CombinedSummaryError)
        assert "failed" in result.error

    @patch.object(CliSummarizer, "_run_cli")
    @patch.object(CliSummarizer, "available", new_callable=lambda: property(lambda self: True))
    def test_malformed_json_returns_error(self, _avail: object, mock_cli: MagicMock) -> None:
        mock_cli.return_value = "not json at all"
        summarizer = _make_summarizer()

        result = summarizer.summarize_combined("text", "Doc", "txt", [("S", "t")])
        assert isinstance(result, CombinedSummaryError)

    @patch.object(CliSummarizer, "_run_cli")
    @patch.object(CliSummarizer, "available", new_callable=lambda: property(lambda self: True))
    def test_partial_json_missing_fields_returns_error(
        self, _avail: object, mock_cli: MagicMock,
    ) -> None:
        """JSON is valid but missing required fields."""
        mock_cli.return_value = json.dumps({"summary_8w": "Short"})
        summarizer = _make_summarizer()

        result = summarizer.summarize_combined("text", "Doc", "txt", [("S", "t")])
        assert isinstance(result, CombinedSummaryError)
        assert "Validation failed" in result.error

    @patch.object(CliSummarizer, "_run_cli")
    @patch.object(CliSummarizer, "available", new_callable=lambda: property(lambda self: True))
    def test_combined_json_in_markdown_fence(
        self, _avail: object, mock_cli: MagicMock,
    ) -> None:
        mock_cli.return_value = f"```json\n{_make_combined_json(1)}\n```"
        summarizer = _make_summarizer()

        result = summarizer.summarize_combined("text", "Doc", "txt", [("S", "t")])
        assert isinstance(result, CombinedSummarySuccess)
        assert len(result.sections) == 1


# --- Tests for batch section grouping ---


class TestBatchSectionGrouping:
    def test_single_batch_under_limit(self) -> None:
        summarizer = _make_summarizer()
        sections = [("S1", "short"), ("S2", "short")]
        batches = summarizer._group_sections_into_batches(sections)
        assert len(batches) == 1
        assert len(batches[0]) == 2

    def test_multiple_batches_over_limit(self) -> None:
        summarizer = _make_summarizer()
        big_text = "x" * (MAX_EXCERPT_CHARS)
        # Create enough sections to exceed the limit
        sections = [(f"Section {i}", big_text) for i in range(20)]
        batches = summarizer._group_sections_into_batches(sections)
        assert len(batches) > 1
        # All sections should be present across batches
        total = sum(len(b) for b in batches)
        assert total == 20

    def test_empty_sections_returns_empty(self) -> None:
        summarizer = _make_summarizer()
        batches = summarizer._group_sections_into_batches([])
        assert batches == []

    def test_single_oversized_section_gets_own_batch(self) -> None:
        summarizer = _make_summarizer()
        # Even a single section exceeding the limit should be in its own batch
        big_text = "x" * MAX_EXCERPT_CHARS
        sections = [("Big", big_text)]
        batches = summarizer._group_sections_into_batches(sections)
        assert len(batches) == 1
        assert len(batches[0]) == 1


# --- Tests for summarize_sections_batch ---


class TestSummarizeSectionsBatch:
    @patch.object(CliSummarizer, "_run_cli")
    @patch.object(CliSummarizer, "available", new_callable=lambda: property(lambda self: True))
    def test_basic_batch(self, _avail: object, mock_cli: MagicMock) -> None:
        mock_cli.return_value = _make_batch_sections_json(2)
        summarizer = _make_summarizer()

        sections = [("S1", "text1"), ("S2", "text2")]
        results = summarizer.summarize_sections_batch(sections, "Doc (txt)")

        assert len(results) == 2
        assert results[0].section_summary_8w == "Short summary for section 0"

    @patch.object(CliSummarizer, "_run_cli")
    @patch.object(CliSummarizer, "available", new_callable=lambda: property(lambda self: True))
    def test_cli_failure_skips_batch(self, _avail: object, mock_cli: MagicMock) -> None:
        mock_cli.return_value = None
        summarizer = _make_summarizer()

        sections = [("S1", "text1")]
        results = summarizer.summarize_sections_batch(sections, "Doc (txt)")
        assert results == []

    @patch.object(CliSummarizer, "_run_cli")
    @patch.object(CliSummarizer, "available", new_callable=lambda: property(lambda self: True))
    def test_empty_sections(self, _avail: object, mock_cli: MagicMock) -> None:
        summarizer = _make_summarizer()
        results = summarizer.summarize_sections_batch([], "Doc (txt)")
        assert results == []
        mock_cli.assert_not_called()
