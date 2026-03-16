from __future__ import annotations

from typing import TYPE_CHECKING

import mcp.types as types

if TYPE_CHECKING:
    from mcp.server import Server

    from rag.config import AppConfig

def _folder_clause(folder: str | None) -> str:
    return f" Filter to folder: {folder}." if folder else ""


def _user_message(text: str) -> list[types.PromptMessage]:
    return [
        types.PromptMessage(
            role="user",
            content=types.TextContent(type="text", text=text),
        )
    ]


_PROMPTS: list[types.Prompt] = [
    types.Prompt(
        name="research",
        description=(
            "Deep research on a topic across all indexed documents. "
            "Scouts for relevant documents, extracts key evidence, "
            "and synthesizes findings."
        ),
        arguments=[
            types.PromptArgument(
                name="topic",
                description="The topic or question to research",
                required=True,
            ),
            types.PromptArgument(
                name="folder",
                description="Restrict research to a specific folder path",
                required=False,
            ),
        ],
    ),
    types.Prompt(
        name="discover",
        description=(
            "Browse and summarize what documents are available in the "
            "indexed collection, optionally filtered by folder."
        ),
        arguments=[
            types.PromptArgument(
                name="folder",
                description="Restrict to a specific folder path",
                required=False,
            ),
        ],
    ),
    types.Prompt(
        name="catch-up",
        description="Summarize what documents have been added or changed recently.",
        arguments=[
            types.PromptArgument(
                name="folder",
                description="Restrict to a specific folder path",
                required=False,
            ),
        ],
    ),
]

_PROMPT_MAP: dict[str, types.Prompt] = {p.name: p for p in _PROMPTS}


def _build_research_messages(
    topic: str, folder: str | None
) -> list[types.PromptMessage]:
    fc = _folder_clause(folder)
    return _user_message(
        f"Research the following topic across my indexed documents: {topic}\n\n"
        "Follow these steps:\n"
        "1. Use quick_search to find which documents are relevant to this topic."
        f"{fc}\n"
        "2. For the top 3-5 most relevant documents, use get_document_context "
        "with their doc_ids to read their full summaries and section structure.\n"
        "3. Use search_documents with 2-3 specific queries targeting different "
        "aspects of the topic to extract cited evidence passages.\n"
        "4. Synthesize your findings, citing the specific documents and sections "
        "where each piece of information was found."
    )


def _build_discover_messages(folder: str | None) -> list[types.PromptMessage]:
    fc = _folder_clause(folder)
    return _user_message(
        "Give me an overview of what documents are in my indexed collection."
        f"{fc}\n\n"
        "Follow these steps:\n"
        "1. Use get_sync_status to see how many documents are indexed "
        "and which folders are tracked.\n"
        "2. Use list_recent_documents with detail '32w' to see what's available."
        f"{fc}\n"
        "3. Summarize the collection: what types of documents are there, "
        "what topics do they cover, how many per folder, and when they were "
        "last updated."
    )


def _build_catchup_messages(folder: str | None) -> list[types.PromptMessage]:
    fc = _folder_clause(folder)
    return _user_message(
        "Summarize what's changed recently in my indexed documents."
        f"{fc}\n\n"
        "Follow these steps:\n"
        "1. Use list_recent_documents with detail '32w' to see recently "
        f"modified documents.{fc}\n"
        "2. Review the modification dates and identify which documents are new "
        "or recently updated.\n"
        "3. For each new or significantly changed document, briefly describe "
        "what it covers and highlight anything that looks important or actionable."
    )


def register_prompts(server: Server, config: AppConfig) -> None:
    _ = config

    @server.list_prompts()  # type: ignore[no-untyped-call, untyped-decorator]
    async def handle_list_prompts() -> list[types.Prompt]:
        return _PROMPTS

    @server.get_prompt()  # type: ignore[no-untyped-call, untyped-decorator]
    async def handle_get_prompt(
        name: str, arguments: dict[str, str] | None
    ) -> types.GetPromptResult:
        args = arguments or {}
        if name == "research":
            messages = _build_research_messages(args["topic"], args.get("folder"))
        elif name == "discover":
            messages = _build_discover_messages(args.get("folder"))
        elif name == "catch-up":
            messages = _build_catchup_messages(args.get("folder"))
        else:
            raise ValueError(f"Unknown prompt: {name}")

        return types.GetPromptResult(
            description=_PROMPT_MAP[name].description, messages=messages
        )
