from __future__ import annotations

import contextlib
import ipaddress
import logging
import sys
from typing import TYPE_CHECKING

from mcp.server import Server

from rag.mcp.prompts import register_prompts
from rag.mcp.tools import register_tools

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.applications import Starlette

    from rag.config import AppConfig

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure logging to stderr (stdout reserved for JSON-RPC)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )


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

    _configure_logging()
    server = create_server(config)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def _require_loopback(host: str) -> None:
    """Refuse a non-loopback bind: the HTTP transport has no authentication."""
    if host == "localhost":
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    msg = (
        f"Refusing to serve MCP over HTTP on {host!r}: the HTTP transport has no "
        "authentication, so mcp.host must be a loopback address (127.0.0.1, ::1, "
        "or localhost). To reach it from another machine, use an SSH tunnel "
        "(ssh -L <port>:127.0.0.1:<port> <this-host>)."
    )
    raise ValueError(msg)


def _loopback_security(port: int) -> TransportSecuritySettings:
    """Accept only loopback Host/Origin headers, blocking DNS-rebinding pages."""
    from mcp.server.transport_security import TransportSecuritySettings

    hosts = ("127.0.0.1", "localhost", "[::1]")
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[f"{h}:{port}" for h in hosts],
        allowed_origins=[f"http://{h}:{port}" for h in hosts],
    )


def create_http_app(config: AppConfig) -> Starlette:
    """Build the Streamable HTTP ASGI app, served at /mcp/."""
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.routing import Mount

    _require_loopback(config.mcp.host)
    server = create_server(config)
    session_manager = StreamableHTTPSessionManager(
        app=server,
        json_response=True,
        stateless=True,
        security_settings=_loopback_security(config.mcp.port),
    )

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with session_manager.run():
            yield

    return Starlette(
        routes=[Mount("/mcp", app=session_manager.handle_request)],
        lifespan=lifespan,
    )


async def run_http_server(config: AppConfig) -> None:
    """Run the MCP server using Streamable HTTP transport."""
    import uvicorn

    _configure_logging()
    app = create_http_app(config)

    config_uvicorn = uvicorn.Config(
        app,
        host=config.mcp.host,
        port=config.mcp.port,
        log_level="info",
    )
    uvicorn_server = uvicorn.Server(config_uvicorn)
    await uvicorn_server.serve()
