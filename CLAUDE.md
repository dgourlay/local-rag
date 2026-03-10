# CLAUDE.md

## Project

Local RAG system that indexes documents from configured filesystem folders, builds a hybrid search index (dense vectors + keyword), and exposes retrieval via MCP to local LLM tools (Claude Desktop, Claude Code). Single Python process + Qdrant Docker container. No cloud infrastructure required.

Full spec: `plan/local/local-rag-spec.md`
Cloud variant spec (future): `plan/cloud/dropbox-rag-final-spec.md`

## Status

Pre-implementation. Building the local version first (Phase 1).

## Architecture

Single Python process handles: filesystem watching (watchdog) → Docling parsing (in subprocess for memory isolation) → normalization → dedup → chunking → embedding (local BGE-M3, 1024-dim) → Qdrant indexing. Summarization shells out to a user-configured LLM CLI tool (claude, kiro-cli, etc.). MCP server runs via stdio for Claude Desktop or Streamable HTTP. MCP handlers are async; CPU-bound ops dispatched via asyncio.to_thread(). Retrieval pipeline: hybrid dense+keyword search via query_points() → RRF fusion → ONNX cross-encoder reranker → cited evidence returned to calling LLM.

## Tech Stack

- Python 3.11+, Pydantic v2, SQLite (WAL mode), Qdrant v1.17 (Docker)
- Docling (parsing, OcrAutoOptions), sentence-transformers/BGE-M3 (embeddings), onnxruntime (reranker)
- MCP SDK `mcp>=1.25,<2` (stdio + Streamable HTTP transport)

## Typing & Code Quality (Mandatory)

- **mypy strict mode** — `strict = true`, `plugins = ["pydantic.mypy"]`. Full annotations. No `Any` except behind typed wrappers.
- **Pydantic v2 models** at every module boundary. No raw dicts crossing boundaries (except within typed wrapper functions around external clients).
- **dataclasses** (`frozen=True, slots=True`) for stage-internal value objects only. Cross-stage data uses Pydantic.
- **Literal** for small field-annotation sets (ProcessStatus, SummaryLevel). **StrEnum** for sets used in iteration/runtime logic (RecordType, FileType).
- **Protocol classes** for all pluggable backends (Embedder, Summarizer, MetadataDB, VectorStore, Reranker, Parser).
- **Discriminated union Result types** for fallible operations — no `success: bool` + optional fields pattern.
- **ruff** for linting and formatting (ANN, TC rule sets enabled). Note: `TCH` was renamed to `TC` in ruff v0.8.0.
- **Async/sync**: indexing pipeline is sync. MCP handlers are async. CPU-bound ops via `asyncio.to_thread()`. Qdrant queries via `AsyncQdrantClient`.

See `plan/local/local-rag-spec.md` §2.1 for full typing rules and examples.

## Project Structure

```
src/rag/
  cli.py              # Entry points: rag index, rag serve, rag watch, rag status
  config.py           # TOML config loader (Pydantic)
  sync/               # Filesystem watcher + scanner
  pipeline/           # classify → parse → normalize → dedup → chunk → embed → summarize → index
    parser/            # Docling wrapper (PDF/DOCX) + text fallback
  retrieval/           # Multi-stage search: dense + keyword + RRF + reranker + citations
  mcp/                 # MCP server + tool definitions (4 tools)
  db/                  # SQLite connection, models, migrations
migrations/            # SQL schema files
tests/                 # Unit + integration tests with fixtures
```

## Key Commands

```bash
rag init               # Interactive setup wizard (folders, LLM CLI, Qdrant, MCP)
rag index              # Full scan + process all documents
rag serve              # Start MCP server (stdio)
rag watch              # Filesystem watcher (auto-index on changes)
rag status             # Dashboard: docs/chunks/errors, per-folder breakdown
rag doctor             # Health check: Qdrant, OCR, models, folders
rag search "query"     # CLI search for testing
rag mcp-config --print # Print MCP config JSON snippet
```

## Conventions

- Target chunk size: 512 tokens (tiktoken cl100k_base), 64-token overlap
- Embedding dimensions: 1024 (BGE-M3)
- Qdrant: single collection "documents", cosine distance, record_type payload field
- UUIDs stored as TEXT in SQLite
- All timestamps ISO 8601
- Config file: TOML at `~/.config/dropbox-rag/config.toml`
