from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from rag.config import AppConfig, QuestionsConfig, SummarizationConfig
from rag.pipeline.summarizer import (
    CHUNK_QUESTIONS_BATCH_PROMPT_TEMPLATE,
    MAX_EXCERPT_CHARS,
    CliSummarizer,
    _format_chunks_text,
    build_augmented_text,
)
from rag.types import Chunk

# --- Helpers ---


def _make_chunk(
    chunk_order: int = 0,
    text: str = "Some chunk text.",
    doc_id: str = "doc-1",
    generated_questions: list[str] | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=f"chunk-{chunk_order}",
        doc_id=doc_id,
        chunk_order=chunk_order,
        text=text,
        text_normalized=text.lower(),
        token_count=len(text.split()),
        generated_questions=generated_questions,
    )


def _make_summarizer(*, command: str = "echo", enabled: bool = True) -> CliSummarizer:
    config = SummarizationConfig(
        enabled=enabled,
        command=command,
        args=[],
        input_mode="stdin",
    )
    return CliSummarizer(config)


# --- QuestionsConfig ---


class TestQuestionsConfig:
    def test_defaults(self) -> None:
        config = QuestionsConfig()
        assert config.enabled is True

    def test_disabled(self) -> None:
        config = QuestionsConfig(enabled=False)
        assert config.enabled is False

    def test_in_app_config(self) -> None:
        """QuestionsConfig is available on AppConfig with defaults."""
        from rag.config import FoldersConfig

        app = AppConfig(folders=FoldersConfig(paths=["/tmp"]))
        assert app.questions.enabled is True


# --- Chunk model with generated_questions ---


class TestChunkModelQuestions:
    def test_default_none(self) -> None:
        chunk = _make_chunk()
        assert chunk.generated_questions is None

    def test_with_questions(self) -> None:
        chunk = _make_chunk(generated_questions=["Q1", "Q2", "Q3"])
        assert chunk.generated_questions == ["Q1", "Q2", "Q3"]

    def test_serialization_roundtrip(self) -> None:
        chunk = _make_chunk(generated_questions=["Q1", "Q2"])
        data = chunk.model_dump()
        restored = Chunk.model_validate(data)
        assert restored.generated_questions == ["Q1", "Q2"]


# --- build_augmented_text ---


class TestBuildAugmentedText:
    def test_basic(self) -> None:
        result = build_augmented_text("Original text.", ["Q1", "Q2", "Q3"])
        expected = "Questions this content answers:\n- Q1\n- Q2\n- Q3\n\nOriginal text."
        assert result == expected

    def test_single_question(self) -> None:
        result = build_augmented_text("Text.", ["Only question"])
        assert "- Only question" in result
        assert result.endswith("Text.")

    def test_preserves_original_text(self) -> None:
        original = "The original chunk text is preserved."
        result = build_augmented_text(original, ["Q1"])
        assert original in result


# --- _format_chunks_text ---


class TestFormatChunksText:
    def test_single_chunk(self) -> None:
        chunks = [_make_chunk(chunk_order=0, text="Hello world")]
        result = _format_chunks_text(chunks)
        assert "--- Chunk 0 ---" in result
        assert "Hello world" in result

    def test_multiple_chunks(self) -> None:
        chunks = [
            _make_chunk(chunk_order=0, text="First chunk"),
            _make_chunk(chunk_order=1, text="Second chunk"),
        ]
        result = _format_chunks_text(chunks)
        assert "--- Chunk 0 ---" in result
        assert "--- Chunk 1 ---" in result
        assert "First chunk" in result
        assert "Second chunk" in result

    def test_truncates_long_text(self) -> None:
        long_text = "x" * (MAX_EXCERPT_CHARS + 1000)
        chunks = [_make_chunk(text=long_text)]
        result = _format_chunks_text(chunks)
        # Should be truncated to MAX_EXCERPT_CHARS
        assert len(result) < len(long_text) + 100

    def test_empty_list(self) -> None:
        result = _format_chunks_text([])
        assert result == ""


# --- Prompt template ---


class TestPromptTemplate:
    def test_has_placeholders(self) -> None:
        assert "{title}" in CHUNK_QUESTIONS_BATCH_PROMPT_TEMPLATE
        assert "{chunks_text}" in CHUNK_QUESTIONS_BATCH_PROMPT_TEMPLATE

    def test_format_works(self) -> None:
        result = CHUNK_QUESTIONS_BATCH_PROMPT_TEMPLATE.format(
            title="Test Doc",
            chunks_text="--- Chunk 0 ---\nSome text",
        )
        assert "Test Doc" in result
        assert "--- Chunk 0 ---" in result


# --- generate_chunk_questions ---


class TestGenerateChunkQuestions:
    def test_empty_chunks(self) -> None:
        summarizer = _make_summarizer()
        result = summarizer.generate_chunk_questions([], "Title")
        assert result == []

    @patch.object(CliSummarizer, "_run_cli")
    def test_basic_question_generation(self, mock_cli: MagicMock) -> None:
        mock_cli.return_value = json.dumps(
            {
                "chunks": [
                    {
                        "chunk_order": 0,
                        "questions": [
                            "What is chunk zero about?",
                            "Chunk zero details",
                            "How does chunk zero work?",
                        ],
                    },
                    {
                        "chunk_order": 1,
                        "questions": [
                            "What is chunk one about?",
                            "Chunk one details",
                            "How does chunk one work?",
                        ],
                    },
                ]
            }
        )

        summarizer = _make_summarizer()
        chunks = [_make_chunk(chunk_order=0), _make_chunk(chunk_order=1)]
        result = summarizer.generate_chunk_questions(chunks, "Test Doc")

        assert len(result) == 2
        assert result[0].generated_questions is not None
        assert len(result[0].generated_questions) == 3
        assert result[1].generated_questions is not None
        assert len(result[1].generated_questions) == 3
        mock_cli.assert_called_once()

    @patch.object(CliSummarizer, "_run_cli")
    def test_cli_failure_graceful(self, mock_cli: MagicMock) -> None:
        """When CLI returns None, chunks should have generated_questions=None."""
        mock_cli.return_value = None

        summarizer = _make_summarizer()
        chunks = [_make_chunk(chunk_order=0), _make_chunk(chunk_order=1)]
        result = summarizer.generate_chunk_questions(chunks, "Title")

        assert len(result) == 2
        assert result[0].generated_questions is None
        assert result[1].generated_questions is None

    @patch.object(CliSummarizer, "_run_cli")
    def test_invalid_json_graceful(self, mock_cli: MagicMock) -> None:
        """Invalid JSON output should not crash."""
        mock_cli.return_value = "This is not JSON at all"

        summarizer = _make_summarizer()
        chunks = [_make_chunk(chunk_order=0)]
        result = summarizer.generate_chunk_questions(chunks, "Title")

        assert len(result) == 1
        assert result[0].generated_questions is None

    @patch.object(CliSummarizer, "_run_cli")
    def test_partial_json_response(self, mock_cli: MagicMock) -> None:
        """When only some chunks have questions, matched ones get them."""
        mock_cli.return_value = json.dumps(
            {
                "chunks": [
                    {
                        "chunk_order": 0,
                        "questions": ["Q1", "Q2", "Q3"],
                    },
                    # chunk_order=1 is missing (truncated response)
                ]
            }
        )

        summarizer = _make_summarizer()
        chunks = [_make_chunk(chunk_order=0), _make_chunk(chunk_order=1)]
        result = summarizer.generate_chunk_questions(chunks, "Title")

        assert result[0].generated_questions == ["Q1", "Q2", "Q3"]
        assert result[1].generated_questions is None

    @patch.object(CliSummarizer, "_run_cli")
    def test_batch_grouping(self, mock_cli: MagicMock) -> None:
        """Many chunks should be split into multiple batches."""
        # Create chunks with text large enough to force multiple batches
        # Each chunk ~5000 chars, limit is 80K, so ~16 per batch
        big_text = "x" * 5000
        chunks = [_make_chunk(chunk_order=i, text=big_text) for i in range(30)]

        def fake_cli(prompt: str) -> str:
            # Parse which chunk orders are in this batch from the prompt
            import re

            orders = [int(m) for m in re.findall(r"--- Chunk (\d+) ---", prompt)]
            return json.dumps(
                {
                    "chunks": [
                        {"chunk_order": o, "questions": [f"Q{o}-1", f"Q{o}-2", f"Q{o}-3"]}
                        for o in orders
                    ]
                }
            )

        mock_cli.side_effect = fake_cli

        summarizer = _make_summarizer()
        result = summarizer.generate_chunk_questions(chunks, "Title")

        # All chunks should have questions
        assert len(result) == 30
        for chunk in result:
            assert chunk.generated_questions is not None
            assert len(chunk.generated_questions) == 3

        # Should have been called multiple times (batches)
        assert mock_cli.call_count >= 2

    def test_unavailable_summarizer(self) -> None:
        summarizer = _make_summarizer(command="nonexistent_tool_xyz")
        chunks = [_make_chunk(chunk_order=0)]
        result = summarizer.generate_chunk_questions(chunks, "Title")

        # Should return chunks unchanged (no questions)
        assert len(result) == 1
        assert result[0].generated_questions is None

    @patch.object(CliSummarizer, "_run_cli")
    def test_questions_not_strings_filtered(self, mock_cli: MagicMock) -> None:
        """Non-string items in questions list should be filtered out."""
        mock_cli.return_value = json.dumps(
            {
                "chunks": [
                    {
                        "chunk_order": 0,
                        "questions": ["Valid question", 42, None, "Another valid"],
                    },
                ]
            }
        )

        summarizer = _make_summarizer()
        chunks = [_make_chunk(chunk_order=0)]
        result = summarizer.generate_chunk_questions(chunks, "Title")

        # Should only have the string questions
        assert result[0].generated_questions is not None
        for q in result[0].generated_questions:
            assert isinstance(q, str)

    @patch.object(CliSummarizer, "_run_cli")
    def test_original_text_preserved(self, mock_cli: MagicMock) -> None:
        """chunk.text should remain unchanged after question generation."""
        original_text = "This is the original chunk text."
        mock_cli.return_value = json.dumps(
            {
                "chunks": [
                    {"chunk_order": 0, "questions": ["Q1", "Q2", "Q3"]},
                ]
            }
        )

        summarizer = _make_summarizer()
        chunks = [_make_chunk(chunk_order=0, text=original_text)]
        result = summarizer.generate_chunk_questions(chunks, "Title")

        assert result[0].text == original_text
        assert result[0].generated_questions is not None


# --- Integration: augmented text not stored as chunk text ---


class TestAugmentedTextIntegration:
    @patch.object(CliSummarizer, "_run_cli")
    def test_augmented_text_differs_from_chunk_text(self, mock_cli: MagicMock) -> None:
        """Augmented text should include questions, but chunk.text should not."""
        mock_cli.return_value = json.dumps(
            {
                "chunks": [
                    {"chunk_order": 0, "questions": ["Q1", "Q2", "Q3"]},
                ]
            }
        )

        summarizer = _make_summarizer()
        chunks = [_make_chunk(chunk_order=0, text="Original.")]
        result = summarizer.generate_chunk_questions(chunks, "Title")

        assert result[0].generated_questions is not None
        augmented = build_augmented_text(result[0].text, result[0].generated_questions)

        # chunk.text is original, augmented has questions prepended
        assert result[0].text == "Original."
        assert "Questions this content answers:" in augmented
        assert "Original." in augmented


# --- ChunkRow with generated_questions ---


class TestChunkRowQuestions:
    def test_default_none(self) -> None:
        from rag.types import ChunkRow

        row = ChunkRow(
            chunk_id="c1",
            doc_id="d1",
            chunk_order=0,
            chunk_text="text",
            chunk_text_normalized="text",
        )
        assert row.generated_questions is None

    def test_json_string(self) -> None:
        from rag.types import ChunkRow

        row = ChunkRow(
            chunk_id="c1",
            doc_id="d1",
            chunk_order=0,
            chunk_text="text",
            chunk_text_normalized="text",
            generated_questions=json.dumps(["Q1", "Q2"]),
        )
        assert row.generated_questions is not None
        parsed = json.loads(row.generated_questions)
        assert parsed == ["Q1", "Q2"]


# --- QdrantPayloadModel with generated_questions ---


class TestQdrantPayloadQuestions:
    def test_default_none(self) -> None:
        from rag.types import QdrantPayloadModel, RecordType

        payload = QdrantPayloadModel(
            record_type=RecordType.CHUNK,
            doc_id="d1",
            title="Test",
            file_path="/test.txt",
            folder_path="/",
            folder_ancestors=[],
            file_type="txt",
            modified_at="2024-01-01T00:00:00Z",
            text="content",
        )
        assert payload.generated_questions is None

    def test_with_questions(self) -> None:
        from rag.types import QdrantPayloadModel, RecordType

        payload = QdrantPayloadModel(
            record_type=RecordType.CHUNK,
            doc_id="d1",
            title="Test",
            file_path="/test.txt",
            folder_path="/",
            folder_ancestors=[],
            file_type="txt",
            modified_at="2024-01-01T00:00:00Z",
            generated_questions=["Q1", "Q2"],
            text="content",
        )
        assert payload.generated_questions == ["Q1", "Q2"]
        # Verify it serializes properly
        data = payload.model_dump(mode="json")
        assert data["generated_questions"] == ["Q1", "Q2"]


# --- SyncStatusOutput with question fields ---


class TestSyncStatusQuestions:
    def test_defaults(self) -> None:
        from rag.types import SyncStatusOutput

        output = SyncStatusOutput(
            total_files=10,
            indexed_count=8,
            pending_count=1,
            error_count=1,
            folders=[],
        )
        assert output.questions_enabled is False
        assert output.chunks_with_questions == 0

    def test_with_questions(self) -> None:
        from rag.types import SyncStatusOutput

        output = SyncStatusOutput(
            total_files=10,
            indexed_count=8,
            pending_count=1,
            error_count=1,
            questions_enabled=True,
            chunks_with_questions=500,
            folders=[],
        )
        assert output.questions_enabled is True
        assert output.chunks_with_questions == 500
