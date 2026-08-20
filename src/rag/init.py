from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import tomllib
import urllib.request
from dataclasses import dataclass
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


MCP_SERVER_NAME = "local-rag"
_MCP_STDIO_ARGS = ["-m", "rag.cli", "serve"]
_MCP_HTTP_ARGS = ["-m", "rag.cli", "serve", "--http"]


def generate_mcp_config(transport: str = "stdio") -> dict[str, object]:
    """Generate MCP server config JSON for Claude Desktop / Claude Code."""
    args = _MCP_STDIO_ARGS if transport == "stdio" else _MCP_HTTP_ARGS
    return {
        "mcpServers": {
            MCP_SERVER_NAME: {
                "command": sys.executable,
                "args": list(args),
            }
        }
    }


_MCP_CONFIG_PATHS: dict[str, str] = {
    "claude-desktop": "~/Library/Application Support/Claude/claude_desktop_config.json",
    "claude-code": "~/.claude.json",
    "kiro": "~/.kiro/settings/mcp.json",
}

_CODEX_TARGET = "codex"
_CODEX_CONFIG_PATH = "~/.codex/config.toml"
# Codex uses snake_case table names, unlike the JSON targets' "mcpServers".
_CODEX_SERVERS_TABLE = "mcp_servers"


def install_mcp_config(target: str, config_path: Path | None = None) -> bool:
    """Install MCP config for the given target.

    Supported targets: claude-desktop, claude-code, kiro (JSON), codex (TOML).
    ``config_path`` overrides the target's default config location (for tests).
    Returns True on success.
    """
    if target == _CODEX_TARGET:
        path = config_path if config_path is not None else Path(_CODEX_CONFIG_PATH).expanduser()
        return _install_codex_mcp_config(path)

    path_template = _MCP_CONFIG_PATHS.get(target)
    if path_template is None:
        return False

    json_path = config_path if config_path is not None else Path(path_template).expanduser()
    json_path.parent.mkdir(parents=True, exist_ok=True)

    config = generate_mcp_config()

    existing: dict[str, object] = {}
    if json_path.is_file():
        existing = json.loads(json_path.read_text())

    if "mcpServers" not in existing:
        existing["mcpServers"] = {}

    servers = existing["mcpServers"]
    if isinstance(servers, dict):
        mcp_servers = config.get("mcpServers", {})
        if isinstance(mcp_servers, dict):
            servers.update(mcp_servers)

    json_path.write_text(json.dumps(existing, indent=2) + "\n")
    return True


# --- Codex (TOML) install -------------------------------------------------
#
# Codex's config.toml is hand-maintained and holds model providers, project
# trust levels and secrets, so it is edited as text rather than parsed and
# re-serialized: reserializing would drop the user's comments and reorder
# their file. Only the lines of the `[mcp_servers.local-rag]` table (and its
# sub-tables) are rewritten; everything else is passed through byte for byte.


@dataclass(frozen=True, slots=True)
class _CodexServerEntry:
    """The `[mcp_servers.<name>]` block local-rag owns in Codex's config."""

    name: str
    command: str
    args: tuple[str, ...]

    @property
    def table_path(self) -> tuple[str, str]:
        return (_CODEX_SERVERS_TABLE, self.name)

    def render(self) -> list[str]:
        header = ".".join(_render_toml_key(part) for part in self.table_path)
        args = ", ".join(_render_toml_string(a) for a in self.args)
        return [
            f"[{header}]",
            f"command = {_render_toml_string(self.command)}",
            f"args = [{args}]",
        ]


def _codex_server_entry() -> _CodexServerEntry:
    return _CodexServerEntry(
        name=MCP_SERVER_NAME,
        command=sys.executable,
        args=tuple(_MCP_STDIO_ARGS),
    )


_BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_TABLE_HEADER_RE = re.compile(r"^\s*\[\s*(?P<key>[^\[\]]+?)\s*\]\s*(?:#.*)?$")


def _render_toml_string(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _render_toml_key(key: str) -> str:
    return key if _BARE_KEY_RE.match(key) else _render_toml_string(key)


def _split_toml_key(key: str) -> list[str] | None:
    """Split a dotted TOML key into its unquoted parts, or None if unparseable."""
    parts: list[str] = []
    current: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(key):
        char = key[index]
        if quote is not None:
            if char == quote:
                quote = None
            elif char == "\\" and quote == '"' and index + 1 < len(key):
                current.append(key[index + 1])
                index += 2
                continue
            else:
                current.append(char)
        elif char in ('"', "'"):
            quote = char
        elif char == ".":
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        index += 1
    if quote is not None:
        return None
    parts.append("".join(current).strip())
    if any(part == "" for part in parts):
        return None
    return parts


def _toml_table_path(line: str) -> list[str] | None:
    """Return the table path for a `[a.b]` header line, else None."""
    match = _TABLE_HEADER_RE.match(line)
    if match is None:
        return None
    return _split_toml_key(match.group("key"))


def _find_table_regions(lines: list[str], table_path: tuple[str, ...]) -> list[tuple[int, int]]:
    """Find `[start, end)` line ranges covering a table and its sub-tables."""
    regions: list[tuple[int, int]] = []
    index = 0
    while index < len(lines):
        path = _toml_table_path(lines[index])
        if path is None or tuple(path) != table_path:
            index += 1
            continue
        end = index + 1
        while end < len(lines):
            if not lines[end].lstrip().startswith("["):
                end += 1
                continue
            nested = _toml_table_path(lines[end])
            is_child = (
                nested is not None
                and len(nested) > len(table_path)
                and tuple(nested[: len(table_path)]) == table_path
            )
            if not is_child:
                break
            end += 1
        regions.append((index, end))
        index = end
    return regions


def _splice_codex_entry(text: str, entry: _CodexServerEntry) -> str:
    """Return ``text`` with the entry's block replaced in place, or appended."""
    eol = "\r" if "\r\n" in text else ""
    block = [line + eol for line in entry.render()]

    lines = text.split("\n")
    regions = _find_table_regions(lines, entry.table_path)

    if not regions:
        prefix = lines[:]
        # Keep exactly one blank line between existing content and our block.
        while prefix and prefix[-1].strip() == "":
            prefix.pop()
        separator = [""] if prefix else []
        return "\n".join([*prefix, *separator, *block, ""])

    # Replace the first block; drop any duplicates (illegal TOML anyway).
    for start, end in reversed(regions):
        region = lines[start:end]
        trailing: list[str] = []
        while region and region[-1].strip() == "":
            trailing.insert(0, region.pop())
        replacement = [*block, *trailing] if (start, end) == regions[0] else trailing
        lines[start:end] = replacement
    return "\n".join(lines)


def _install_codex_mcp_config(config_path: Path) -> bool:
    """Add or update local-rag in Codex's TOML config, preserving everything else.

    Returns False (without writing) if the existing file cannot be read, does
    not parse as TOML, already defines the server in a form this text editor
    cannot rewrite (e.g. an inline table), or if the edited text would not
    parse back to the expected entry.
    """
    entry = _codex_server_entry()

    text = ""
    if config_path.is_file():
        try:
            text = config_path.read_text(encoding="utf-8")
            existing = tomllib.loads(text)
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
            return False
        if not _codex_entry_is_editable(text, existing, entry):
            return False

    updated = _splice_codex_entry(text, entry)
    if not _codex_text_has_entry(updated, entry):
        return False

    _atomic_write_text(config_path, updated)
    return True


def _codex_entry_is_editable(
    text: str, existing: dict[str, object], entry: _CodexServerEntry
) -> bool:
    """True unless the server is defined somewhere we cannot safely rewrite."""
    servers = existing.get(_CODEX_SERVERS_TABLE)
    already_defined = isinstance(servers, dict) and entry.name in servers
    if not already_defined:
        return True
    return bool(_find_table_regions(text.split("\n"), entry.table_path))


def _codex_text_has_entry(text: str, entry: _CodexServerEntry) -> bool:
    """Validate that rendered TOML parses and yields exactly the entry we wrote."""
    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return False
    servers = parsed.get(_CODEX_SERVERS_TABLE)
    if not isinstance(servers, dict):
        return False
    server = servers.get(entry.name)
    if not isinstance(server, dict):
        return False
    return bool(server.get("command") == entry.command and server.get("args") == list(entry.args))


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` via a same-directory temp file, preserving the file mode."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    mode = path.stat().st_mode & 0o777 if path.is_file() else 0o600

    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
