# CLAUDE.md

## Project

Local RAG system that indexes documents from configured filesystem folders, builds a hybrid search index (dense vectors + keyword), and exposes retrieval via MCP to local LLM tools (Claude Desktop, Claude Code). Single Python process + Qdrant Docker container. No cloud infrastructure required.

Full spec: `plan/local/local-rag-spec.md`
Cloud variant spec (future): `plan/cloud/dropbox-rag-final-spec.md`

## Status

Pre-implementation. Building the local version first (Phase 1).

## Architecture

Single Python process handles: filesystem watching (watchdog) → Docling parsing → normalization → dedup → chunking → embedding (local bge-base-en-v1.5) → Qdrant indexing. Summarization shells out to a user-configured LLM CLI tool (claude, kiro-cli, etc.). MCP server runs via stdio for Claude Desktop or localhost HTTP. Retrieval pipeline: hybrid dense+keyword search → RRF fusion → ONNX cross-encoder reranker → cited evidence returned to calling LLM.

## Tech Stack

- Python 3.11+, Pydantic v2, SQLite, Qdrant (Docker)
- Docling (parsing), sentence-transformers (embeddings), onnxruntime (reranker)
- MCP SDK (stdio + HTTP transport)

## Typing & Code Quality (Mandatory)

- **mypy strict mode** — `strict = true` in pyproject.toml. Full annotations on all functions. No `Any` except behind typed wrappers for untyped third-party libs.
- **Pydantic v2 models** at every module boundary. No raw dicts crossing boundaries.
- **dataclasses** (`frozen=True, slots=True`) for internal value objects.
- **Literal/StrEnum** for all constrained string values (process_status, record_type, file_type, etc.).
- **Protocol classes** for pluggable backends (embedder, summarizer, database).
- **Result types** for fallible operations — no exceptions for expected failures.
- **ruff** for linting and formatting (ANN, TCH rule sets enabled).

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
rag index              # Full scan + process all documents
rag serve              # Start MCP server (stdio)
rag watch              # Filesystem watcher (auto-index on changes)
rag status             # Show index stats
rag search "query"     # CLI search for testing
```

## Conventions

- Target chunk size: 512 tokens (tiktoken cl100k_base), 64-token overlap
- Embedding dimensions: 768 (bge-base-en-v1.5)
- Qdrant: single collection "documents", cosine distance, record_type payload field
- UUIDs stored as TEXT in SQLite
- All timestamps ISO 8601
- Config file: TOML at `~/.config/dropbox-rag/config.toml`
