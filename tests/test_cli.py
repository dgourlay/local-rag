from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from rag.cli import main

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


class TestHelp:
    def test_main_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        for cmd in [
            "init",
            "index",
            "serve",
            "watch",
            "status",
            "doctor",
            "search",
            "mcp-config",
        ]:
            assert cmd in result.output

    def test_init_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["init", "--help"])
        assert result.exit_code == 0
        assert "--add-folder" in result.output

    def test_search_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["search", "--help"])
        assert result.exit_code == 0
        assert "--debug" in result.output
        assert "--top-k" in result.output


class TestMcpConfig:
    def test_print_config(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["mcp-config", "--print"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "mcpServers" in data
        assert "dropbox-rag" in data["mcpServers"]

    def test_no_flags(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["mcp-config"])
        assert result.exit_code == 0
        assert "--print" in result.output


class TestDoctor:
    @patch("rag.config.load_config")
    @patch("rag.init.check_qdrant_running", return_value=True)
    @patch("rag.db.connection.get_connection")
    def test_doctor_all_pass(
        self,
        mock_conn: MagicMock,
        mock_qdrant: MagicMock,
        mock_config: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        folder = tmp_path / "docs"
        folder.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "model.bin").touch()

        config = MagicMock()
        config.qdrant.url = "http://localhost:6333"
        config.database.path = str(tmp_path / "test.db")
        config.folders.paths = [folder]
        config.embedding.cache_dir = cache_dir
        mock_config.return_value = config

        conn_mock = MagicMock()
        mock_conn.return_value = conn_mock

        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 0
        assert "Config:    PASS" in result.output
        assert "Qdrant:    PASS" in result.output

    @patch("rag.config.load_config", side_effect=FileNotFoundError("not found"))
    def test_doctor_no_config(self, mock_config: MagicMock, runner: CliRunner) -> None:
        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 0
        assert "FAIL" in result.output


class TestStatus:
    def test_status_json(self, runner: CliRunner, tmp_path: Path) -> None:
        import sqlite3

        from rag.db.migrations import run_migrations

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        run_migrations(conn)

        # Insert test data
        conn.execute(
            """INSERT INTO sync_state
            (id, file_path, file_name, folder_path, folder_ancestors,
             file_type, modified_at, content_hash, process_status, retry_count, is_deleted)
            VALUES ('1', '/docs/a.txt', 'a.txt', '/docs', '[]',
                    'txt', '2024-01-01', 'abc', 'done', 0, 0)"""
        )
        conn.execute(
            """INSERT INTO documents
            (doc_id, file_path, folder_path, folder_ancestors,
             file_type, modified_at, raw_content_hash)
            VALUES ('d1', '/docs/a.txt', '/docs', '[]',
                    'txt', '2024-01-01', 'abc')"""
        )
        conn.execute(
            """INSERT INTO chunks
            (chunk_id, doc_id, chunk_order, chunk_text,
             chunk_text_normalized)
            VALUES ('c1', 'd1', 0, 'hello', 'hello')"""
        )
        conn.commit()
        conn.close()

        with patch("rag.config.load_config") as mock_config:
            config = MagicMock()
            config.database.path = str(db_path)
            mock_config.return_value = config

            result = runner.invoke(main, ["status", "--json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["documents"] == 1
            assert data["chunks"] == 1
            assert data["errors"] == 0
            assert len(data["folders"]) == 1
            assert data["folders"][0]["folder_path"] == "/docs"


class TestSearch:
    @patch("rag.cli._init_components")
    @patch("rag.config.load_config")
    def test_search_no_results(
        self,
        mock_config: MagicMock,
        mock_init: MagicMock,
        runner: CliRunner,
    ) -> None:
        config = MagicMock()
        mock_config.return_value = config

        engine = MagicMock()
        engine.search.return_value = MagicMock(hits=[], debug_info=None)
        mock_init.return_value = (MagicMock(), MagicMock(), engine)

        result = runner.invoke(main, ["search", "test query"])
        assert result.exit_code == 0
        assert "No results found" in result.output

    @patch("rag.cli._init_components")
    @patch("rag.config.load_config")
    def test_search_with_results(
        self,
        mock_config: MagicMock,
        mock_init: MagicMock,
        runner: CliRunner,
    ) -> None:
        from rag.types import Citation, CitedEvidence

        config = MagicMock()
        mock_config.return_value = config

        hit = CitedEvidence(
            text="some content here",
            citation=Citation(
                title="doc.pdf",
                path="/docs/doc.pdf",
                section="Intro",
                pages="p. 1",
                modified="2024-01-01",
                label="doc.pdf, § Intro, p. 1",
            ),
            score=0.95,
            record_type="chunk",
        )

        engine = MagicMock()
        engine.search.return_value = MagicMock(hits=[hit], debug_info=None)
        mock_init.return_value = (MagicMock(), MagicMock(), engine)

        result = runner.invoke(main, ["search", "test query"])
        assert result.exit_code == 0
        assert "Result 1" in result.output
        assert "0.95" in result.output
        assert "some content" in result.output


class TestInit:
    def test_add_folder(self, runner: CliRunner, tmp_path: Path) -> None:
        folder = tmp_path / "docs"
        folder.mkdir()
        config_file = tmp_path / "config.toml"

        from rag.init import create_config

        result_path = create_config(
            folders=[str(folder)],
            llm_command="claude",
            config_path=config_file,
        )
        assert result_path == config_file
        content = config_file.read_text()
        assert str(folder) in content
        assert "claude" in content

    def test_add_folder_cli(self, runner: CliRunner, tmp_path: Path) -> None:
        folder = tmp_path / "docs"
        folder.mkdir()
        config_file = tmp_path / "config.toml"

        with patch("rag.init.create_config") as mock_create:
            mock_create.return_value = config_file
            result = runner.invoke(main, ["init", "--add-folder", str(folder)])
            assert result.exit_code == 0


class TestInitModule:
    def test_detect_llm_cli_found(self) -> None:
        from rag.init import detect_llm_cli

        def _which(x: str) -> str | None:
            return "/usr/bin/claude" if x == "claude" else None

        with patch("shutil.which", side_effect=_which):
            assert detect_llm_cli() == "claude"

    def test_detect_llm_cli_not_found(self) -> None:
        from rag.init import detect_llm_cli

        with patch("shutil.which", return_value=None):
            assert detect_llm_cli() is None

    def test_check_qdrant_running_down(self) -> None:
        from rag.init import check_qdrant_running

        assert check_qdrant_running("http://localhost:19999") is False

    def test_generate_mcp_config(self) -> None:
        from rag.init import generate_mcp_config

        config = generate_mcp_config()
        assert "mcpServers" in config
        assert "dropbox-rag" in config["mcpServers"]  # type: ignore[operator]

    def test_generate_mcp_config_http(self) -> None:
        from rag.init import generate_mcp_config

        config = generate_mcp_config(transport="http")
        servers = config["mcpServers"]
        assert isinstance(servers, dict)
        assert "--http" in servers["dropbox-rag"]["args"]  # type: ignore[index]

    def test_create_config(self, tmp_path: Path) -> None:
        from rag.init import create_config

        config_path = tmp_path / "config.toml"
        result = create_config(
            folders=["/home/user/docs", "/home/user/notes"],
            llm_command="claude",
            config_path=config_path,
        )
        assert result == config_path
        content = config_path.read_text()
        assert "/home/user/docs" in content
        assert "/home/user/notes" in content
        assert 'command = "claude"' in content
