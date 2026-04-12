from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from rag.types import ChunkingStrategy, FileType


class FoldersConfig(BaseModel):
    paths: list[Path]
    extensions: list[FileType] = [FileType.PDF, FileType.DOCX, FileType.TXT, FileType.MD]
    ignore: list[str] = ["**/node_modules", "**/.git", "**/venv", "**/__pycache__"]

    @field_validator("paths", mode="before")
    @classmethod
    def expand_paths(cls, v: list[str | Path]) -> list[Path]:
        return [Path(p).expanduser().resolve() for p in v]


class DatabaseConfig(BaseModel):
    path: Path = Path("~/.local/share/local-rag/metadata.db")

    @model_validator(mode="after")
    def expand_paths(self) -> DatabaseConfig:
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve())
        return self


class QdrantConfig(BaseModel):
    url: str = "http://localhost:6333"
    collection: str = "documents"
    grpc_port: int = 6334
    prefer_grpc: bool = True


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-m3"
    dimensions: int = 1024
    batch_size: int = 32
    cache_dir: Path = Path("~/.cache/local-rag/models")
    device: Literal["cpu", "mps", "auto"] = "auto"
    fp16: bool = True

    @model_validator(mode="after")
    def expand_paths(self) -> EmbeddingConfig:
        object.__setattr__(self, "cache_dir", Path(self.cache_dir).expanduser().resolve())
        return self


class ChunkingConfig(BaseModel):
    strategy: ChunkingStrategy = "fixed"
    similarity_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    max_chunk_tokens: int = Field(default=768, ge=128, le=2048)


class RerankerConfig(BaseModel):
    model_path: Path = Path("~/.cache/local-rag/models/bge-reranker-v2-m3")
    top_k_candidates: int = 30
    top_k_final: int = 10
    use_coreml: bool = False

    @model_validator(mode="after")
    def validate_and_expand(self) -> RerankerConfig:
        object.__setattr__(self, "model_path", Path(self.model_path).expanduser().resolve())
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
    args: list[str] | None = None
    input_mode: Literal["stdin", "arg"] | None = None
    timeout_seconds: int = 300
    max_concurrent_llm: int = Field(default=3, ge=1, le=4)

    @model_validator(mode="after")
    def _apply_preset_defaults(self) -> SummarizationConfig:
        """Auto-detect args and input_mode from command if not explicitly set."""
        from rag.pipeline.summarizer import get_cli_preset

        preset = get_cli_preset(self.command)
        if preset is not None:
            if self.args is None:
                self.args = preset[0]
            if self.input_mode is None:
                self.input_mode = preset[1]  # type: ignore[assignment]
        # Fallback for unknown tools
        if self.args is None:
            self.args = []
        if self.input_mode is None:
            self.input_mode = "stdin"
        return self


class QuestionsConfig(BaseModel):
    enabled: bool = True


class RetrievalConfig(BaseModel):
    hyde_enabled: bool = True


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
    chunking: ChunkingConfig = ChunkingConfig()
    reranker: RerankerConfig = RerankerConfig()
    summarization: SummarizationConfig = SummarizationConfig()
    questions: QuestionsConfig = QuestionsConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    mcp: MCPConfig = MCPConfig()
    watcher: WatcherConfig = WatcherConfig()


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load config from TOML file.

    Precedence: explicit path > RAG_CONFIG_PATH env > ./config.toml
    > ~/.config/local-rag/config.toml.
    """
    path: Path
    if config_path is not None:
        path = Path(config_path).expanduser().resolve()
    elif env_path := os.environ.get("RAG_CONFIG_PATH"):
        path = Path(env_path).expanduser().resolve()
    elif Path("config.toml").exists():
        path = Path("config.toml").resolve()
    elif Path("~/.config/local-rag/config.toml").expanduser().exists():
        path = Path("~/.config/local-rag/config.toml").expanduser().resolve()
    else:
        msg = (
            "No config file found. Searched:\n"
            "  1. RAG_CONFIG_PATH environment variable\n"
            "  2. ./config.toml\n"
            "  3. ~/.config/local-rag/config.toml\n\n"
            "Run 'rag init' to create a config file."
        )
        raise FileNotFoundError(msg)

    if not path.is_file():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    return AppConfig.model_validate(raw)
