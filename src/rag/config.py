from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

from rag.types import FileType


class FoldersConfig(BaseModel):
    paths: list[Path]
    extensions: list[FileType] = [FileType.PDF, FileType.DOCX, FileType.TXT, FileType.MD]
    ignore: list[str] = ["**/node_modules", "**/.git", "**/venv", "**/__pycache__"]

    @field_validator("paths", mode="before")
    @classmethod
    def expand_paths(cls, v: list[str | Path]) -> list[Path]:
        return [Path(p).expanduser().resolve() for p in v]


class DatabaseConfig(BaseModel):
    path: Path = Path("~/.local/share/dropbox-rag/metadata.db")

    @field_validator("path", mode="before")
    @classmethod
    def expand_path(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()


class QdrantConfig(BaseModel):
    url: str = "http://localhost:6333"
    collection: str = "documents"


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-m3"
    dimensions: int = 1024
    batch_size: int = 32
    cache_dir: Path = Path("~/.cache/dropbox-rag/models")

    @field_validator("cache_dir", mode="before")
    @classmethod
    def expand_cache_dir(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()


class RerankerConfig(BaseModel):
    model_path: Path = Path("~/.cache/dropbox-rag/models/bge-reranker-v2-m3")
    top_k_candidates: int = 30
    top_k_final: int = 10

    @field_validator("model_path", mode="before")
    @classmethod
    def expand_model_path(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()

    @model_validator(mode="after")
    def validate_top_k(self) -> RerankerConfig:
        if self.top_k_final > self.top_k_candidates:
            msg = (
                f"top_k_final ({self.top_k_final}) must be "
                f"<= top_k_candidates ({self.top_k_candidates})"
            )
            raise ValueError(msg)
        return self


class SummarizationConfig(BaseModel):
    enabled: bool = True
    provider: str = "cli"
    command: str = "claude"
    args: list[str] = ["--print", "--max-tokens", "2048"]
    timeout_seconds: int = 60


class MCPConfig(BaseModel):
    transport: Literal["stdio", "streamable-http"] = "stdio"
    host: str = "127.0.0.1"
    port: int = 8080


class WatcherConfig(BaseModel):
    poll_interval_seconds: int = 5
    debounce_seconds: int = 2
    use_polling: bool = False
    batch_window_seconds: int = 10


class AppConfig(BaseModel):
    folders: FoldersConfig
    database: DatabaseConfig = DatabaseConfig()
    qdrant: QdrantConfig = QdrantConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    reranker: RerankerConfig = RerankerConfig()
    summarization: SummarizationConfig = SummarizationConfig()
    mcp: MCPConfig = MCPConfig()
    watcher: WatcherConfig = WatcherConfig()


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load config from TOML file.

    Precedence: explicit path > RAG_CONFIG_PATH env > ./config.toml
    > ~/.config/dropbox-rag/config.toml.
    """
    path: Path
    if config_path is not None:
        path = Path(config_path).expanduser().resolve()
    elif env_path := os.environ.get("RAG_CONFIG_PATH"):
        path = Path(env_path).expanduser().resolve()
    elif Path("config.toml").exists():
        path = Path("config.toml").resolve()
    elif Path("~/.config/dropbox-rag/config.toml").expanduser().exists():
        path = Path("~/.config/dropbox-rag/config.toml").expanduser().resolve()
    else:
        msg = (
            "No config file found. Searched:\n"
            "  1. RAG_CONFIG_PATH environment variable\n"
            "  2. ./config.toml\n"
            "  3. ~/.config/dropbox-rag/config.toml\n\n"
            "Run 'rag init' to create a config file."
        )
        raise FileNotFoundError(msg)

    if not path.is_file():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    return AppConfig.model_validate(raw)
