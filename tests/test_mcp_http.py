from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx
import pytest

from rag.config import AppConfig, FoldersConfig, MCPConfig
from rag.mcp.server import create_http_app

if TYPE_CHECKING:
    from starlette.applications import Starlette


_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "0"},
    },
}
_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


def _post(app: Starlette, base_url: str, headers: dict[str, str] | None = None) -> httpx.Response:
    """POST an MCP initialize request through the app, with its lifespan running."""

    async def go() -> httpx.Response:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url=base_url) as client:
                return await client.post(
                    "/mcp/", json=_INITIALIZE, headers={**_HEADERS, **(headers or {})}
                )

    return asyncio.run(go())


def _config(host: str = "127.0.0.1", port: int = 8080) -> AppConfig:
    return AppConfig(folders=FoldersConfig(paths=[]), mcp=MCPConfig(host=host, port=port))


def test_http_app_answers_initialize() -> None:
    app = create_http_app(_config())

    response = _post(app, "http://127.0.0.1:8080")

    assert response.status_code == 200
    assert response.json()["result"]["serverInfo"]["name"] == "local-rag"


def test_http_app_accepts_localhost_host_header() -> None:
    app = create_http_app(_config())

    response = _post(app, "http://localhost:8080")

    assert response.status_code == 200


def test_http_app_rejects_foreign_host_header() -> None:
    """A DNS-rebinding page reaches 127.0.0.1 but still sends its own Host."""
    app = create_http_app(_config())

    response = _post(app, "http://attacker.example:8080")

    assert response.status_code == 421


def test_http_app_rejects_foreign_origin() -> None:
    app = create_http_app(_config())

    response = _post(app, "http://127.0.0.1:8080", {"Origin": "http://attacker.example"})

    assert response.status_code == 403


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20"])
def test_http_app_refuses_non_loopback_host(host: str) -> None:
    """The HTTP server has no authentication, so it must stay on loopback."""
    with pytest.raises(ValueError, match="loopback"):
        create_http_app(_config(host=host))


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_http_app_allows_loopback_host(host: str) -> None:
    create_http_app(_config(host=host))
