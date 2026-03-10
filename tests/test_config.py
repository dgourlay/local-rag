from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag.config import AppConfig, FoldersConfig, RerankerConfig, load_config

MINIMAL_TOML = """\
[folders]
paths = ["/tmp/docs"]
"""

FULL_TOML = """\
[folders]
paths = ["~/Documents", "/tmp/other"]
extensions = ["pdf", "md"]
ignore = ["**/node_modules"]

[database]
path = "/tmp/test-rag/metadata.db"

[qdrant]
url = "http://localhost:6333"
collection = "test_docs"

[embedding]
model = "BAAI/bge-m3"
dimensions = 1024
batch_size = 16

[reranker]
top_k_candidates = 20
top_k_final = 5

[summarization]
enabled = false
command = "kiro-cli"

[mcp]
transport = "streamable-http"
port = 9090

[watcher]
poll_interval_seconds = 10
use_polling = true
"""


class TestLoadConfig:
    def test_load_minimal_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text(MINIMAL_TOML)
        cfg = load_config(config_file)
        assert len(cfg.folders.paths) == 1
        assert cfg.folders.paths[0] == Path("/tmp/docs").resolve()

    def test_load_full_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text(FULL_TOML)
        cfg = load_config(config_file)
        assert cfg.qdrant.collection == "test_docs"
        assert cfg.embedding.batch_size == 16
        assert cfg.reranker.top_k_final == 5
        assert cfg.summarization.enabled is False
        assert cfg.mcp.transport == "streamable-http"
        assert cfg.mcp.port == 9090
        assert cfg.watcher.use_polling is True

    def test_missing_config_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/path/config.toml")

    def test_no_config_anywhere_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("RAG_CONFIG_PATH", raising=False)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="No config file found"):
            load_config()

    def test_env_var_precedence(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        env_config = tmp_path / "env_config.toml"
        env_config.write_text(MINIMAL_TOML)
        local_config = tmp_path / "config.toml"
        local_config.write_text(FULL_TOML)

        monkeypatch.setenv("RAG_CONFIG_PATH", str(env_config))
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        # env config has minimal TOML with defaults, local has collection = "test_docs"
        assert cfg.qdrant.collection == "documents"  # default, from minimal TOML

    def test_local_config_toml_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("RAG_CONFIG_PATH", raising=False)
        monkeypatch.chdir(tmp_path)
        local_config = tmp_path / "config.toml"
        local_config.write_text(FULL_TOML)
        cfg = load_config()
        assert cfg.qdrant.collection == "test_docs"


class TestDefaults:
    def test_defaults_applied_with_minimal_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text(MINIMAL_TOML)
        cfg = load_config(config_file)
        assert cfg.qdrant.url == "http://localhost:6333"
        assert cfg.qdrant.collection == "documents"
        assert cfg.embedding.model == "BAAI/bge-m3"
        assert cfg.embedding.dimensions == 1024
        assert cfg.reranker.top_k_candidates == 30
        assert cfg.reranker.top_k_final == 10
        assert cfg.summarization.enabled is True
        assert cfg.summarization.command == "claude"
        assert cfg.mcp.transport == "stdio"
        assert cfg.watcher.debounce_seconds == 2

    def test_default_extensions(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text(MINIMAL_TOML)
        cfg = load_config(config_file)
        ext_values = [e.value for e in cfg.folders.extensions]
        assert ext_values == ["pdf", "docx", "txt", "md"]


class TestPathExpansion:
    def test_folders_paths_expand_tilde(self) -> None:
        fc = FoldersConfig(paths=["~/Documents"])  # type: ignore[arg-type]
        assert fc.paths[0] == Path.home() / "Documents"
        assert fc.paths[0].is_absolute()

    def test_folders_paths_resolve_relative(self) -> None:
        fc = FoldersConfig(paths=["./relative/path"])  # type: ignore[arg-type]
        assert fc.paths[0].is_absolute()

    def test_database_path_expands_tilde(self) -> None:
        from rag.config import DatabaseConfig

        db = DatabaseConfig(path="~/mydb.db")  # type: ignore[arg-type]
        assert db.path == Path.home() / "mydb.db"

    def test_embedding_cache_dir_expands_tilde(self) -> None:
        from rag.config import EmbeddingConfig

        ec = EmbeddingConfig(cache_dir="~/models")  # type: ignore[arg-type]
        assert ec.cache_dir == Path.home() / "models"


class TestValidation:
    def test_reranker_top_k_final_exceeds_candidates(self) -> None:
        with pytest.raises(ValidationError, match="top_k_final"):
            RerankerConfig(top_k_final=50, top_k_candidates=10)

    def test_reranker_top_k_equal_is_valid(self) -> None:
        rc = RerankerConfig(top_k_final=10, top_k_candidates=10)
        assert rc.top_k_final == 10

    def test_folders_requires_paths(self) -> None:
        with pytest.raises(ValidationError):
            AppConfig.model_validate({})
