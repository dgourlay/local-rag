from __future__ import annotations

from rag.mcp.prompts import (
    _PROMPTS,
    _build_catchup_messages,
    _build_discover_messages,
    _build_research_messages,
)


class TestPromptDefinitions:
    def test_three_prompts_defined(self) -> None:
        """_PROMPTS contains exactly 3 prompts."""
        assert len(_PROMPTS) == 3

    def test_prompt_names(self) -> None:
        """Prompts have the expected names."""
        names = {p.name for p in _PROMPTS}
        assert names == {"research", "discover", "catch-up"}

    def test_research_has_topic_argument(self) -> None:
        """research prompt requires a 'topic' argument."""
        prompt = next(p for p in _PROMPTS if p.name == "research")
        assert prompt.arguments is not None
        topic_arg = next(a for a in prompt.arguments if a.name == "topic")
        assert topic_arg.required is True

    def test_research_has_optional_folder(self) -> None:
        """research prompt has optional 'folder' argument."""
        prompt = next(p for p in _PROMPTS if p.name == "research")
        assert prompt.arguments is not None
        folder_arg = next(a for a in prompt.arguments if a.name == "folder")
        assert folder_arg.required is False

    def test_discover_has_optional_folder(self) -> None:
        """discover prompt has optional 'folder' argument."""
        prompt = next(p for p in _PROMPTS if p.name == "discover")
        assert prompt.arguments is not None
        folder_arg = next(a for a in prompt.arguments if a.name == "folder")
        assert folder_arg.required is False

    def test_catchup_has_optional_folder(self) -> None:
        """catch-up prompt has optional 'folder' argument."""
        prompt = next(p for p in _PROMPTS if p.name == "catch-up")
        assert prompt.arguments is not None
        folder_arg = next(a for a in prompt.arguments if a.name == "folder")
        assert folder_arg.required is False

    def test_research_has_two_arguments(self) -> None:
        """research prompt has exactly 2 arguments (topic, folder)."""
        prompt = next(p for p in _PROMPTS if p.name == "research")
        assert prompt.arguments is not None
        assert len(prompt.arguments) == 2

    def test_discover_has_one_argument(self) -> None:
        """discover prompt has exactly 1 argument (folder)."""
        prompt = next(p for p in _PROMPTS if p.name == "discover")
        assert prompt.arguments is not None
        assert len(prompt.arguments) == 1

    def test_catchup_has_one_argument(self) -> None:
        """catch-up prompt has exactly 1 argument (folder)."""
        prompt = next(p for p in _PROMPTS if p.name == "catch-up")
        assert prompt.arguments is not None
        assert len(prompt.arguments) == 1

    def test_all_prompts_have_descriptions(self) -> None:
        """Every prompt has a non-empty description."""
        for prompt in _PROMPTS:
            assert prompt.description, f"Prompt {prompt.name} has no description"


class TestResearchMessages:
    def test_contains_topic(self) -> None:
        """Research messages include the provided topic."""
        messages = _build_research_messages("quarterly revenue", None)
        text = messages[0].content.text  # type: ignore[union-attr]
        assert "quarterly revenue" in text

    def test_contains_workflow_steps(self) -> None:
        """Research messages reference expected tools."""
        messages = _build_research_messages("test topic", None)
        text = messages[0].content.text  # type: ignore[union-attr]
        assert "quick_search" in text
        assert "get_document_context" in text
        assert "search_documents" in text

    def test_folder_clause_present(self) -> None:
        """Research messages include folder clause when folder provided."""
        messages = _build_research_messages("test", "/docs/work")
        text = messages[0].content.text  # type: ignore[union-attr]
        assert "Filter to folder: /docs/work" in text

    def test_folder_clause_absent(self) -> None:
        """Research messages omit folder clause when no folder."""
        messages = _build_research_messages("test", None)
        text = messages[0].content.text  # type: ignore[union-attr]
        assert "Filter to folder" not in text

    def test_single_user_message(self) -> None:
        """Research returns exactly one user message."""
        messages = _build_research_messages("test", None)
        assert len(messages) == 1
        assert messages[0].role == "user"


class TestDiscoverMessages:
    def test_contains_workflow_steps(self) -> None:
        """Discover messages reference expected tools."""
        messages = _build_discover_messages(None)
        text = messages[0].content.text  # type: ignore[union-attr]
        assert "get_sync_status" in text
        assert "list_recent_documents" in text

    def test_folder_clause_present(self) -> None:
        """Discover messages include folder clause when folder provided."""
        messages = _build_discover_messages("/docs")
        text = messages[0].content.text  # type: ignore[union-attr]
        assert "Filter to folder: /docs" in text

    def test_folder_clause_absent(self) -> None:
        """Discover messages omit folder clause when no folder."""
        messages = _build_discover_messages(None)
        text = messages[0].content.text  # type: ignore[union-attr]
        assert "Filter to folder" not in text

    def test_single_user_message(self) -> None:
        """Discover returns exactly one user message."""
        messages = _build_discover_messages(None)
        assert len(messages) == 1
        assert messages[0].role == "user"


class TestCatchupMessages:
    def test_contains_workflow_steps(self) -> None:
        """Catch-up messages reference expected tools."""
        messages = _build_catchup_messages(None)
        text = messages[0].content.text  # type: ignore[union-attr]
        assert "list_recent_documents" in text

    def test_folder_clause_present(self) -> None:
        """Catch-up messages include folder clause when folder provided."""
        messages = _build_catchup_messages("/docs")
        text = messages[0].content.text  # type: ignore[union-attr]
        assert "/docs" in text

    def test_folder_clause_absent(self) -> None:
        """Catch-up messages omit folder clause when no folder."""
        messages = _build_catchup_messages(None)
        text = messages[0].content.text  # type: ignore[union-attr]
        assert "Filter to folder" not in text

    def test_single_user_message(self) -> None:
        """Catch-up returns exactly one user message."""
        messages = _build_catchup_messages(None)
        assert len(messages) == 1
        assert messages[0].role == "user"
