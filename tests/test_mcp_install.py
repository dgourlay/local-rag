from __future__ import annotations

import json
import stat
import sys
import tomllib
from typing import TYPE_CHECKING, Any

from rag.init import install_mcp_config

if TYPE_CHECKING:
    from pathlib import Path


def _load(path: Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _server(path: Path) -> dict[str, Any]:
    servers = _load(path)["mcp_servers"]
    assert isinstance(servers, dict)
    entry = servers["local-rag"]
    assert isinstance(entry, dict)
    return entry


def _file_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


class TestCodexInstall:
    def test_creates_missing_file(self, tmp_path: Path) -> None:
        config_path = tmp_path / "codex" / "config.toml"

        assert install_mcp_config("codex", config_path=config_path) is True

        entry = _server(config_path)
        assert entry == {"command": sys.executable, "args": ["-m", "rag.cli", "serve"]}
        assert _file_mode(config_path) == 0o600
        assert config_path.read_text().startswith("[mcp_servers.local-rag]\n")

    def test_preserves_existing_content_and_comments(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"
        original = (
            "# my hand-written codex config\n"
            "model = 'gpt-5'\n"
            "\n"
            "[projects]\n"
            '"/Users/me/work" = { trust_level = "trusted" }  # inline comment\n'
            "\n"
            "[mcp_servers.builder-mcp]\n"
            'command = "builder-mcp"\n'
            'args = ["--include-tools", "*"]\n'
            "\n"
            "[mcp_servers.builder-mcp.env]\n"
            'TOOL_PERSONALIZATION_ENABLED = "false"\n'
        )
        config_path.write_text(original)

        assert install_mcp_config("codex", config_path=config_path) is True

        text = config_path.read_text()
        assert text.startswith(original)
        assert "# my hand-written codex config" in text
        assert "# inline comment" in text

        data = _load(config_path)
        assert data["model"] == "gpt-5"
        assert data["projects"]["/Users/me/work"] == {"trust_level": "trusted"}
        assert data["mcp_servers"]["builder-mcp"] == {
            "command": "builder-mcp",
            "args": ["--include-tools", "*"],
            "env": {"TOOL_PERSONALIZATION_ENABLED": "false"},
        }
        assert _server(config_path)["command"] == sys.executable

    def test_install_is_idempotent(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text('model = "gpt-5"\n')

        assert install_mcp_config("codex", config_path=config_path) is True
        first = config_path.read_text()
        assert install_mcp_config("codex", config_path=config_path) is True
        second = config_path.read_text()

        assert first == second
        assert second.count("[mcp_servers.local-rag]") == 1

    def test_updates_stale_block_in_place(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            "[mcp_servers.local-rag]\n"
            'command = "/old/python"\n'
            'args = ["-m", "rag.cli", "serve"]\n'
            "\n"
            "[mcp_servers.zzz-last]\n"
            'command = "zzz"\n'
        )

        assert install_mcp_config("codex", config_path=config_path) is True

        text = config_path.read_text()
        assert "/old/python" not in text
        assert text.index("[mcp_servers.local-rag]") < text.index("[mcp_servers.zzz-last]")
        assert _server(config_path)["command"] == sys.executable
        assert _load(config_path)["mcp_servers"]["zzz-last"] == {"command": "zzz"}

    def test_replaces_managed_subtables(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            "[mcp_servers.local-rag]\n"
            'command = "/old/python"\n'
            "\n"
            "[mcp_servers.local-rag.env]\n"
            'STALE = "1"\n'
            "\n"
            "[mcp_servers.other]\n"
            'command = "other"\n'
        )

        assert install_mcp_config("codex", config_path=config_path) is True

        assert "STALE" not in config_path.read_text()
        assert _server(config_path) == {
            "command": sys.executable,
            "args": ["-m", "rag.cli", "serve"],
        }
        assert _load(config_path)["mcp_servers"]["other"] == {"command": "other"}

    def test_handles_quoted_table_name(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text('[mcp_servers."local-rag"]\ncommand = "/old/python"\n')

        assert install_mcp_config("codex", config_path=config_path) is True

        text = config_path.read_text()
        assert text.count("mcp_servers") == 1
        assert _server(config_path)["command"] == sys.executable

    def test_preserves_file_mode(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text('model = "gpt-5"\n')
        config_path.chmod(0o600)

        assert install_mcp_config("codex", config_path=config_path) is True
        assert _file_mode(config_path) == 0o600

        config_path.chmod(0o644)
        assert install_mcp_config("codex", config_path=config_path) is True
        assert _file_mode(config_path) == 0o644

    def test_leaves_no_temp_files(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"

        assert install_mcp_config("codex", config_path=config_path) is True
        assert [p.name for p in tmp_path.iterdir()] == ["config.toml"]

    def test_refuses_inline_table_definition(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"
        original = "[mcp_servers]\n'local-rag' = { command = \"/old/python\" }\n"
        config_path.write_text(original)

        assert install_mcp_config("codex", config_path=config_path) is False
        assert config_path.read_text() == original

    def test_refuses_invalid_toml(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"
        original = "this is not = = toml\n"
        config_path.write_text(original)

        assert install_mcp_config("codex", config_path=config_path) is False
        assert config_path.read_text() == original


class TestJsonInstall:
    def test_merges_into_existing_json_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / ".claude.json"
        config_path.write_text(
            json.dumps({"someOtherKey": 1, "mcpServers": {"other": {"command": "other"}}})
        )

        assert install_mcp_config("claude-code", config_path=config_path) is True

        data = json.loads(config_path.read_text())
        assert data["someOtherKey"] == 1
        assert data["mcpServers"]["other"] == {"command": "other"}
        assert data["mcpServers"]["local-rag"]["command"] == sys.executable

    def test_creates_missing_json_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / "settings" / "mcp.json"

        assert install_mcp_config("kiro", config_path=config_path) is True

        data = json.loads(config_path.read_text())
        assert data["mcpServers"]["local-rag"]["args"] == ["-m", "rag.cli", "serve"]

    def test_unknown_target_returns_false(self, tmp_path: Path) -> None:
        assert install_mcp_config("nope", config_path=tmp_path / "x.json") is False
        assert list(tmp_path.iterdir()) == []
