from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from mcp.server import Server

from rag.mcp.tools import register_tools

if TYPE_CHECKING:
    from rag.config import AppConfig

logger = logging.getLogger(__name__)


def create_server(config: AppConfig) -> Server:
    """Create and configure the MCP server with all tools registered."""
    server = Server("dropbox-rag")
    register_tools(server, config)
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
