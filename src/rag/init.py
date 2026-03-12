from __future__ import annotations

import json
import shutil
import sys
import urllib.request
from pathlib import Path


def detect_llm_clis() -> list[str]:
    """Auto-detect all available LLM CLI tools."""
    return [tool for tool in ["claude", "kiro-cli", "codex"] if shutil.which(tool)]


def check_docker_available() -> bool:
    """Check if Docker CLI is on PATH."""
    return shutil.which("docker") is not None


def check_qdrant_running(url: str = "http://localhost:6333") -> bool:
    """Check if Qdrant is reachable."""
    try:
        req = urllib.request.urlopen(f"{url}/healthz", timeout=5)
        return bool(req.status == 200)
    except Exception:
        return False


DEFAULT_EXTENSIONS = ["pdf", "docx", "txt", "md"]
DEFAULT_IGNORE = ["**/node_modules", "**/.git", "**/venv", "**/__pycache__"]


def create_config(
    folders: list[str],
    llm_command: str | None = None,
    config_path: Path | None = None,
    extensions: list[str] | None = None,
    ignore: list[str] | None = None,
) -> Path:
    """Create config.toml with the given settings."""
    if config_path is None:
        config_path = Path("~/.config/local-rag/config.toml").expanduser()

    config_path.parent.mkdir(parents=True, exist_ok=True)

    ext_list = extensions if extensions is not None else DEFAULT_EXTENSIONS
    ign_list = ignore if ignore is not None else DEFAULT_IGNORE

    lines = ["[folders]"]
    # Format paths as TOML array
    formatted = [f'"{p}"' for p in folders]
    lines.append(f"paths = [{', '.join(formatted)}]")
    ext_formatted = [f'"{e}"' for e in ext_list]
    lines.append(f"extensions = [{', '.join(ext_formatted)}]")
    ign_formatted = [f'"{i}"' for i in ign_list]
    lines.append(f"ignore = [{', '.join(ign_formatted)}]")
    lines.append("")

    if llm_command:
        from rag.pipeline.summarizer import get_cli_preset

        lines.append("[summarization]")
        lines.append("enabled = true")
        lines.append(f'command = "{llm_command}"')
        preset = get_cli_preset(llm_command)
        if preset is not None:
            args, input_mode = preset
            args_formatted = [f'"{a}"' for a in args]
            lines.append(f"args = [{', '.join(args_formatted)}]")
            lines.append(f'input_mode = "{input_mode}"')
        lines.append("")

    config_path.write_text("\n".join(lines) + "\n")
    return config_path


def generate_mcp_config(transport: str = "stdio") -> dict[str, object]:
    """Generate MCP server config JSON for Claude Desktop / Claude Code."""
    python_path = sys.executable
    if transport == "stdio":
        return {
            "mcpServers": {
                "local-rag": {
                    "command": python_path,
                    "args": ["-m", "rag.cli", "serve"],
                }
            }
        }
    return {
        "mcpServers": {
            "local-rag": {
                "command": python_path,
                "args": ["-m", "rag.cli", "serve", "--http"],
            }
        }
    }


_MCP_CONFIG_PATHS: dict[str, str] = {
    "claude-desktop": "~/Library/Application Support/Claude/claude_desktop_config.json",
    "claude-code": "~/.claude.json",
    "kiro": "~/.kiro/settings/mcp.json",
}


def install_mcp_config(target: str) -> bool:
    """Install MCP config for the given target.

    Supported targets: claude-desktop, claude-code, kiro.
    Returns True on success.
    """
    config = generate_mcp_config()

    path_template = _MCP_CONFIG_PATHS.get(target)
    if path_template is None:
        return False

    config_path = Path(path_template).expanduser()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, object] = {}
    if config_path.is_file():
        existing = json.loads(config_path.read_text())

    if "mcpServers" not in existing:
        existing["mcpServers"] = {}

    servers = existing["mcpServers"]
    if isinstance(servers, dict):
        mcp_servers = config.get("mcpServers", {})
        if isinstance(mcp_servers, dict):
            servers.update(mcp_servers)

    config_path.write_text(json.dumps(existing, indent=2) + "\n")
    return True
