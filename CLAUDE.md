# CLAUDE.md

## Project

Local RAG system that indexes documents from configured filesystem folders, builds a hybrid search index (dense vectors + keyword), and exposes retrieval via MCP to local LLM tools (Claude Desktop, Claude Code). Single Python process + Qdrant Docker container. No cloud infrastructure required.

## Quality Checks

Always run the full test suite (`pytest`) and lint/type checks (`ruff check`, `mypy`) after any code changes before committing. All tests must pass and lint must be clean.

## Prerequisites / Pre-flight Checks

Before running searches or indexing commands, verify that Docker and Qdrant are running (`docker ps | grep qdrant`). If infrastructure is down, inform the user immediately rather than retrying.

## Workflow Conventions

When editing code, always re-read the file after editing to confirm changes stuck. Do not assume edits were applied successfully, especially during multi-file changes.

## Plans

- Spec: `plan/local/local-rag-spec.md`
- Agent team plan: `plan/local/agent-team-plan.md`
- Spec review: `plan/local/spec-review.md`
- Cloud variant (future): `plan/cloud/local-rag-final-spec.md`

## Architecture

Single Python process handles: filesystem watching (watchdog) → Docling parsing (in subprocess for memory isolation) → normalization → dedup → chunking (fixed 512-token or opt-in semantic via BGE-M3 sentence embeddings, Max-Min algorithm) → auto-question generation (LLM generates 3 questions per chunk, prepended before embedding for enriched vectors) → embedding (local BGE-M3, 1024-dim) → summarization (LLM CLI) → Qdrant indexing. Summarization shells out to a user-configured LLM CLI tool (claude, kiro-cli, etc.) and generates geometric pyramid summaries — 5 document levels (8w/16w/32w/64w/128w) and 3 section levels (8w/32w/128w). Only the 128w summaries are embedded as vectors in Qdrant; shorter levels serve display and enumeration. MCP server runs via stdio for Claude Desktop or Streamable HTTP. MCP handlers are async; CPU-bound ops dispatched via asyncio.to_thread(). Server provides enriched tool descriptions, ~600-word server instructions (scout→search→drill-down workflow), and 3 prompts (research, discover, catch-up). Retrieval pipeline: 3-lane dense prefetch (document_summary top 20, section_summary top 20, chunks top 30) + keyword search (queries both `text` and `generated_questions` fields), run in parallel via ThreadPoolExecutor → RRF fusion over 4 separate ranked lists (doc_summaries, section_summaries, chunks, keywords) with layer weighting (broad/specific/navigational query classification, multi-signal scoring) → recency boost (90-day half-life, max 15% influence) → reranker enrichment (summary hits get title+topics prepended) → ONNX cross-encoder reranker → citation expansion (summary hits expand to grounded source chunks) → cited evidence returned to calling LLM. HyDE (Hypothetical Document Embeddings) generates a hypothetical answer via LLM CLI for broad queries, embedding that instead of the raw query (configurable via `hyde_enabled`).

## Tech Stack

- Python 3.11+, Pydantic v2, SQLite (WAL mode), Qdrant v1.17 (Docker)
- Docling (parsing, OcrAutoOptions), sentence-transformers/BGE-M3 (embeddings), onnxruntime (reranker)
- MCP SDK `mcp>=1.25,<2` (stdio + Streamable HTTP transport)

This project is primarily Python. Use Pydantic models for data types. The project uses ruff for linting, mypy for type checking, and pytest for tests. Do not use spaCy (incompatible with Python 3.14) — use custom sentencizer instead.

## Typing & Code Quality (Mandatory)

- **mypy strict mode** — `strict = true`, `plugins = ["pydantic.mypy"]`. Full annotations. No `Any` except behind typed wrappers.
- **Pydantic v2 models** at every module boundary. No raw dicts crossing boundaries (except within typed wrapper functions around external clients).
- **dataclasses** (`frozen=True, slots=True`) for stage-internal value objects only. Cross-stage data uses Pydantic.
- **Literal** for small field-annotation sets (ProcessStatus, SummaryLevel). **StrEnum** for sets used in iteration/runtime logic (RecordType, FileType).
- **Protocol classes** for all pluggable backends (Embedder, Summarizer, MetadataDB, VectorStore, Reranker, Parser).
- **Discriminated union Result types** for fallible operations — no `success: bool` + optional fields pattern.
- **ruff** for linting and formatting (ANN, TC rule sets enabled). Note: `TCH` was renamed to `TC` in ruff v0.8.0.
- **Async/sync**: indexing pipeline is sync. MCP handlers are async. CPU-bound ops via `asyncio.to_thread()`. Qdrant queries via `AsyncQdrantClient`.

- **`from __future__ import annotations`** in every module. Use `model_rebuild()` for cross-module Pydantic forward refs.

See `plan/local/local-rag-spec.md` §2.1 for full typing rules and examples.

## Project Structure

```
src/rag/
  types.py             # All Pydantic models, StrEnums, Literals, type aliases
  protocols.py         # All Protocol classes (Embedder, Summarizer, MetadataDB, VectorStore, Reranker, Parser)
  results.py           # Discriminated union Result types (ParseResult, SummaryResult, etc.)
  cli.py               # Entry points: rag init, rag index, rag serve, rag watch, rag status, rag search
  config.py            # TOML config loader (AppConfig Pydantic model)
  init.py              # Interactive setup wizard (rag init)
  sync/                # Filesystem watcher + scanner
  pipeline/            # classify → parse → normalize → dedup → chunk → embed → questions → summarize → index
    parser/            # Docling wrapper (PDF/DOCX, subprocess) + text fallback (TXT/MD)
    chunker_semantic.py # Semantic chunking (opt-in, Max-Min algorithm with BGE-M3)
    summarizer.py      # CliSummarizer: LLM CLI for doc/section summaries + question generation
  retrieval/           # 3-lane prefetch + RRF fusion + layer weighting + recency boost + reranker + citations
    hyde.py            # HyDE: hypothetical document embeddings for broad queries
  mcp/                 # MCP server (async) + 5 tools + 3 prompts + server instructions
    prompts.py         # MCP prompts: research, discover, catch-up
  db/                  # SQLite connection + Qdrant client, models, migrations
migrations/            # SQL schema files
tests/                 # Unit tests per module
tests/e2e/             # End-to-end tests (real Qdrant, real models, real MCP server)
tests/fixtures/        # Real sample documents with known searchable content
```

## Key Commands

```bash
rag init                          # Interactive setup wizard (folders, LLM CLI, Qdrant, MCP)
rag index                         # Full scan + process all documents
rag index --reindex               # Purge all index data, re-process everything (prompts confirmation)
rag index --reindex /path/to/file # Clear + re-process a single file
rag serve                         # Start MCP server (stdio)
rag watch                         # Filesystem watcher (auto-index on changes)
rag status                        # Dashboard: docs/chunks/errors, MCP health, liveness checks
rag doctor                        # Health check: Qdrant, OCR, models, folders
rag search "query"                # CLI search for testing
rag search "query" --debug        # Search with per-lane counts, weights, timing
rag search "query" --top-k 5      # Limit number of results
rag mcp-config --print            # Print MCP config JSON snippet
```

## Conventions

- Chunking: fixed strategy (512 tokens, 64-token overlap) or opt-in semantic strategy (Max-Min algorithm, BGE-M3 sentence embeddings, similarity_threshold 0.35, max 768 tokens, no overlap). Configured via `[chunking]` section. `chunker_version` = "semantic-v1" triggers re-index on strategy change.
- Auto-generated questions: LLM generates 3 questions per chunk at index time (`[questions]` config). Questions prepended to chunk text before embedding. Stored in `generated_questions` Qdrant payload field (keyword-indexed). Graceful degradation if LLM unavailable.
- Embedding dimensions: 1024 (BGE-M3)
- Qdrant: single collection "documents", cosine distance, record_type payload field, all search via `query_points()` API (not removed `search()`)
- Qdrant indexing: deterministic UUID5 point IDs for overwrite semantics (no delete+upsert)
- UUID5 namespace: `NAMESPACE_RAG` constant, format `f"{doc_id}:{section_order}:{chunk_order}"`
- UUIDs stored as TEXT in SQLite
- All timestamps ISO 8601
- SQLite: WAL mode, busy_timeout=30000
- Config file: TOML at `~/.config/local-rag/config.toml`
- Docling parsing runs in child subprocess (multiprocessing) for memory isolation
- MCP stdio servers: never write to stdout (corrupts JSON-RPC); use stderr for logging
- SQLite: `check_same_thread=False` for async MCP handler access
- Pyramid summaries: document levels summary_8w/16w/32w/64w/128w, section levels section_summary_8w/32w/128w. Only 128w embedded as vectors.
- Retrieval: 3-lane dense prefetch + keyword search (queries both `text` and `generated_questions` fields) in parallel (ThreadPoolExecutor), RRF fusion over 4 ranked lists, query classification (broad/specific/navigational), recency boost (90-day half-life, max 15%), reranker enrichment for summary hits, citation expansion for summary hits, HyDE for broad queries
- key_topics stored in Qdrant payload and keyword-indexed
- MCP tools: 5 tools (search_documents, quick_search, get_document_context, list_recent_documents, get_sync_status) with enriched descriptions (3-5 sentences, under 500 chars)
- MCP tools `detail` parameter for progressive disclosure: list_recent_documents default "8w", quick_search default "32w", get_document_context default "128w"
- search_documents `format` parameter: "text" (default, LLM-friendly) or "json" (raw structured data)
- MCP server instructions: ~600 words guiding scout→search→drill-down workflow, query tips, configured folder list
- MCP prompts: 3 prompts (research, discover, catch-up) registered as slash commands in Claude Code. Defined in `src/rag/mcp/prompts.py`.

## Testing

- `make lint` — ruff check + mypy strict
- `make test` — unit tests (fast, no Docker)
- `make test-e2e` — end-to-end with real Qdrant Docker, real BGE-M3 model, real `claude` CLI for summarization, real MCP server subprocess
- E2e tests use fixture documents with known query-answer pairs — asserts on specific content, not just "something returned"
- `make test-e2e` passing = system works. No ambiguity.
- When fixing tests, be careful with patch paths — always verify the import path used in the module under test, not the original definition path. After fixing lint, re-run to confirm no new unused import warnings were introduced.
