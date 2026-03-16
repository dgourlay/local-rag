from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from mcp.server import Server

from rag.mcp.prompts import register_prompts
from rag.mcp.tools import register_tools

if TYPE_CHECKING:
    from rag.config import AppConfig

logger = logging.getLogger(__name__)

_INSTRUCTIONS_TEMPLATE = """\
local-rag is a local document search system that indexes files (PDF, DOCX, TXT, MD) \
from configured folders on this machine. It provides hybrid semantic + keyword search \
with cross-encoder reranking over the indexed collection.

## Indexed Folders

{folders_block}

## Recommended Workflow

1. **Scout** — Start with quick_search or list_recent_documents to understand what \
documents exist and which are relevant. Use the returned doc_ids for drill-down.
2. **Search** — Use search_documents with specific natural-language queries to extract \
cited evidence passages. Prefer multi-word queries with domain terminology.
3. **Drill down** — Use get_document_context with a doc_id (for full document overview) \
or chunk_id (for surrounding passage context) to get deeper information.

## Query Tips

- Use 3-8 word natural-language queries, not single keywords.
- Include domain-specific terms that would appear in the documents.
- If initial results are weak, try rephrasing with different terminology or synonyms.
- Use folder_filter when you know which folder contains the target content.
- Use date_filter on search_documents when the user asks about recent information.

## Important Notes

- This system returns evidence passages, not answers. Synthesize answers from the \
returned citations.
- Scores range from 0 to 1. Results above 0.5 are typically strong matches; below 0.3 \
may be tangential.
- The detail parameter on summary tools controls verbosity: "8w" for lists, "32w" for \
overviews, "128w" for deep reading.
- get_sync_status can verify the index is current before searching.\
"""


def _build_instructions(config: AppConfig) -> str:
    """Build server instructions with configured folder paths."""
    if config.folders.paths:
        folders_block = "\n".join(f"- {p}" for p in config.folders.paths)
    else:
        folders_block = "(No folders configured)"
    return _INSTRUCTIONS_TEMPLATE.format(folders_block=folders_block)


def create_server(config: AppConfig) -> Server:
    """Create and configure the MCP server with all tools registered."""
    instructions = _build_instructions(config)
    server = Server("local-rag", instructions=instructions)
    register_tools(server, config)
    register_prompts(server, config)
    return server


async def run_stdio_server(config: AppConfig) -> None:
    """Run the MCP server using stdio transport.

    All logging goes to stderr so stdout remains clean for JSON-RPC.
    """
    from mcp.server.stdio import stdio_server

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    server = create_server(config)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


async def run_http_server(config: AppConfig) -> None:
    """Run the MCP server using Streamable HTTP transport."""
    import uvicorn
    from mcp.server.streamable_http import StreamableHTTPServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Mount

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    server = create_server(config)

    transport = StreamableHTTPServerTransport(
        mcp_session_id=None,
        is_json_response_enabled=True,
    )

    async def handle_mcp() -> None:
        async with transport.connect() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    app = Starlette(
        routes=[Mount("/mcp", app=transport.handle_request)],
        on_startup=[lambda: None],
    )

    config_uvicorn = uvicorn.Config(
        app,
        host=config.mcp.host,
        port=config.mcp.port,
        log_level="info",
    )
    uvicorn_server = uvicorn.Server(config_uvicorn)
    await uvicorn_server.serve()
