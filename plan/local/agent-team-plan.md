# Agent Team Plan: Local RAG Implementation

## Overview

This document defines the agent team strategy for implementing the local RAG system per `plan/local/local-rag-spec.md`. Agents are organized by the dependency graph — types and infrastructure first, then pipeline stages, then retrieval, then MCP server, then CLI/UX.

Each agent runs in an isolated worktree and produces a PR-ready branch. Agents that share no file dependencies can run in parallel.

---

## Dependency Graph

```
A1 (Scaffolding)
 │
 ├──► A2 (Types & Protocols)
 │     │
 │     ├──► A3 (Config)
 │     │     │
 │     │     ├──► A4 (SQLite DB) ─────────────────────────────┐
 │     │     │                                                 │
 │     │     ├──► A5 (Qdrant Client) ─────────────────────────┤
 │     │     │                                                 │
 │     │     └──► A6 (Scanner) ───────────────────────────────┤
 │     │                                                       │
 │     ├──► A7 (Docling Parser) ──────────────────────────────┤
 │     │                                                       │
 │     ├──► A8 (Text Parser) ─────────────────────────────────┤
 │     │                                                       │
 │     ├──► A9 (Normalizer) ──────────────────────────────────┤
 │     │                                                       │
 │     ├──► A10 (Chunker) ────────────────────────────────────┤
 │     │                                                       │
 │     └──► A11 (Embedder) ───────────────────────────────────┤
 │                                                             │
 │     ┌───────────────────────────────────────────────────────┘
 │     ▼
 │   A12 (Pipeline Runner) ── orchestrates A4-A11
 │     │
 │     ├──► A13 (Dedup)
 │     │
 │     └──► A14 (Indexer)
 │           │
 │           ▼
 │         A15 (Retrieval Engine)
 │           │
 │           ├──► A16 (Reranker)
 │           │
 │           └──► A17 (Citations)
 │                 │
 │                 ▼
 │               A18 (MCP Server)
 │                 │
 │                 ▼
 │               A19 (CLI + Init)
 │                 │
 │                 ▼
 │               A20 (Integration Tests)
```

---

## Parallelism Windows

Agents within the same window can run concurrently. Each window must complete before the next starts.

| Window | Agents | Description |
|--------|--------|-------------|
| **W1** | A1 | Scaffolding (solo — everything depends on it) |
| **W2** | A2 | Types & Protocols (solo — everything imports from it) |
| **W3** | A3 | Config (solo — most agents need AppConfig) |
| **W4** | A4, A5, A6, A7, A8, A9, A10, A11 | All pipeline components + infra (parallel — no file overlaps) |
| **W5** | A12, A13 | Pipeline runner + dedup (runner wires up W4 components; dedup is a pipeline stage) |
| **W6** | A14 | Indexer (needs runner + Qdrant client) |
| **W7** | A15, A16, A17 | Retrieval engine + reranker + citations (parallel — separate files) |
| **W8** | A18 | MCP server (needs retrieval engine) |
| **W9** | A19 | CLI + init wizard (needs everything above) |
| **W10** | A20 | Integration tests (end-to-end validation) |

---

## Agent Definitions

### A1: Scaffolding

**Responsibility:** Create the project skeleton — directory structure, pyproject.toml with all dependencies and entry points, tool configuration, Makefile, and config.example.toml.

**Files owned:**
- `pyproject.toml`
- `Makefile`
- `config.example.toml`
- `src/rag/__init__.py`
- `src/rag/py.typed` (PEP 561 marker)
- All `__init__.py` files in subdirectories
- `tests/__init__.py`, `tests/conftest.py`
- `tests/fixtures/` (sample.pdf, sample.docx, sample.txt — placeholder empty files)
- `migrations/` directory

**Spec sections:** §2.1 (pyproject.toml config), §4.1 (config.example.toml), §9 (project structure), §10 (prerequisites)

**Dependencies:** None

**Acceptance criteria:**
- `pip install -e ".[dev]"` succeeds
- `ruff check src/` runs (no files to check yet, but config loads)
- `mypy src/` runs (no files to check yet, but config loads)
- `pytest` runs (no tests yet, but collection works)
- Directory structure matches §9 exactly
- `pyproject.toml` includes all dependencies: docling, sentence-transformers, onnxruntime, watchdog, tiktoken, qdrant-client, mcp, pydantic, tomli/tomllib, click (or typer for CLI)
- `pyproject.toml` includes dev dependencies: pytest, mypy, ruff, pydantic[mypy]
- `[project.scripts] rag = "rag.cli:main"` entry point defined
- mypy and ruff config matches §2.1 exactly (strict=true, plugins, TC not TCH)

---

### A2: Types & Protocols

**Responsibility:** Define all shared types, Pydantic models, Protocol classes, StrEnums, Literals, TypedDicts, and discriminated union Result types. This is the foundational module that every other agent imports from.

**Files owned:**
- `src/rag/types.py` — all Pydantic models, enums, literals, type aliases
- `src/rag/protocols.py` — all Protocol classes
- `src/rag/results.py` — all discriminated union Result types

**Spec sections:** §2.1 (all type patterns), §5.3 (Qdrant payload), §8.2 (MCP tool input/output types)

**Dependencies:** A1 (needs pyproject.toml for pydantic dependency)

**Acceptance criteria:**
- `mypy --strict src/rag/types.py src/rag/protocols.py src/rag/results.py` passes with zero errors
- `ruff check` passes
- All types from §2.1 example patterns are implemented:
  - Constrained values: `ProcessStatus`, `SummaryLevel` (Literal), `RecordType`, `FileType` (StrEnum)
  - Pipeline boundary models: `ParsedSection`, `ParsedDocument`, `ClassificationResult`, `NormalizedDocument`, `Chunk`, `EmbeddedChunk`, `VectorPoint`, `FileEvent`, `SearchFilters`, `SearchHit`, `RetrievalResult`, `CitedEvidence`
  - Qdrant types: `QdrantPayloadModel` (construction), `QdrantPayloadReadBack` (TypedDict)
  - DB row models: `SyncStateRow`, `DocumentRow`, `ChunkRow`, `SectionRow`, `ProcessingLogEntry`
  - MCP tool models: input/output for `search_documents`, `get_document_context`, `list_recent_documents`, `get_sync_status`
  - Frozen dataclasses: `ChunkWindow`, `RRFCandidate`
- All Protocols implemented: `Embedder`, `Summarizer`, `MetadataDB`, `VectorStore`, `Reranker`, `Parser`
- All Result types implemented as discriminated unions: `ParseResult`, `SummaryResult`, `SectionSummaryResult`, `EmbedResult`, `IndexResult`
- Every module uses `from __future__ import annotations`
- `NAMESPACE_RAG` UUID constant defined for deterministic chunk IDs
- No business logic — pure type definitions only

---

### A3: Config

**Responsibility:** TOML config loading, validation, path expansion, precedence resolution, and the `AppConfig` Pydantic model with all nested config sections.

**Files owned:**
- `src/rag/config.py`
- `tests/test_config.py`

**Spec sections:** §4.1 (config file), §4.2 (env vars), §4.3 (precedence), §4.4 (Pydantic model)

**Dependencies:** A2 (imports `FileType` from types)

**Acceptance criteria:**
- `AppConfig` model with all nested sections: `FoldersConfig`, `DatabaseConfig`, `QdrantConfig`, `EmbeddingConfig`, `RerankerConfig`, `SummarizationConfig`, `MCPConfig`, `WatcherConfig`
- `load_config()` function implements precedence: `RAG_CONFIG_PATH` > `./config.toml` > `~/.config/dropbox-rag/config.toml`
- Path expansion (`~` → home dir) on all path fields
- `folders.paths` validated as existing directories (with clear error if not)
- `reranker.top_k_final <= reranker.top_k_candidates` validated
- Sensible defaults for all optional sections
- Clear error message if no config file found
- Tests: load from TOML string, missing file error, path expansion, validation errors, precedence with temp files
- `mypy --strict` and `ruff check` pass

---

### A4: SQLite Database

**Responsibility:** SQLite connection management, WAL mode setup, schema migration runner, and all CRUD operations implementing the `MetadataDB` Protocol.

**Files owned:**
- `src/rag/db/__init__.py`
- `src/rag/db/connection.py`
- `src/rag/db/models.py`
- `src/rag/db/migrations.py`
- `migrations/001_initial.sql`
- `tests/test_db.py`

**Spec sections:** §5.1 (SQLite schema), §2.1 Rule 6 (`MetadataDB` Protocol)

**Dependencies:** A2 (imports Protocol and row types), A3 (imports `DatabaseConfig`)

**Acceptance criteria:**
- `get_connection()` returns a connection with `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=30000`
- Migration runner applies `001_initial.sql` idempotently (creates all tables from §5.1)
- `SqliteMetadataDB` class satisfies the `MetadataDB` Protocol (mypy verified)
- CRUD for all tables: sync_state, documents, sections, chunks, document_hashes, processing_log
- All DB operations use parameterized queries (no SQL injection)
- UUIDs stored as TEXT, timestamps as ISO 8601
- Tests: create tables, insert/read/update rows, WAL mode verified, migration idempotency, concurrent reads (WAL allows)
- `mypy --strict` and `ruff check` pass

---

### A5: Qdrant Client

**Responsibility:** Qdrant connection, collection creation with full schema (1024-dim cosine, all payload indices, text index), and all vector operations implementing the `VectorStore` Protocol. Both sync and async clients.

**Files owned:**
- `src/rag/db/qdrant.py`
- `scripts/setup-qdrant.sh`
- `tests/test_qdrant.py`

**Spec sections:** §5.2 (Qdrant collection), §5.3 (payload), §6.9 (indexing strategy), §7.1 (query_points API)

**Dependencies:** A2 (imports Protocol, `VectorPoint`, `SearchHit`, `SearchFilters`, `QdrantPayloadModel`), A3 (imports `QdrantConfig`)

**Acceptance criteria:**
- `QdrantVectorStore` class satisfies the `VectorStore` Protocol (mypy verified)
- `ensure_collection()` creates collection with: 1024-dim cosine vectors, all payload keyword indices (record_type, summary_level, doc_id, folder_path, file_type, modified_at, file_path, doc_type_guess), full-text index on `text` field
- `upsert_points()` — upserts with deterministic point IDs (overwrite semantics)
- `delete_stale_points()` — deletes only point IDs not in `keep_ids` set for a given `doc_id`
- `query_dense()` — uses `query_points()` with cosine similarity + filters
- `query_keyword()` — uses `query_points()` with text index + filters
- `AsyncQdrantVectorStore` — async variant using `AsyncQdrantClient` for MCP handlers
- `setup-qdrant.sh` — pulls and starts `qdrant/qdrant:v1.17` with persistent volume
- Tests: collection creation (with in-memory Qdrant), upsert + query round-trip, stale point deletion, filter queries
- `mypy --strict` and `ruff check` pass

---

### A6: Filesystem Scanner

**Responsibility:** Walk configured folders, hash files, detect changes against sync_state, produce `FileEvent` objects. Initial scan and startup re-scan logic.

**Files owned:**
- `src/rag/sync/__init__.py`
- `src/rag/sync/scanner.py`
- `tests/test_scanner.py`

**Spec sections:** §6.1 (file discovery — initial scan, re-scan on startup)

**Dependencies:** A2 (imports `FileEvent`, `FileType`), A3 (imports `FoldersConfig`), A4 (imports `MetadataDB` for sync_state comparison)

**Acceptance criteria:**
- `scan_folders()` — walks all configured paths, respects `extensions` filter and `ignore` globs, computes SHA-256 content hash, returns list of `FileEvent`
- `rescan_for_changes()` — compares current filesystem state against sync_state table, uses mtime as fast pre-filter (only hashes files whose mtime changed), returns list of `FileEvent` for new/modified/deleted files
- Handles `~` expansion in paths
- Skips files matching ignore patterns
- Tests: scan a temp directory with known files, detect new/modified/deleted files, ignore pattern filtering, mtime pre-filter logic
- `mypy --strict` and `ruff check` pass

---

### A7: Docling Parser

**Responsibility:** Parse PDF and DOCX files via Docling, running in a child subprocess for memory isolation. Implements the `Parser` Protocol.

**Files owned:**
- `src/rag/pipeline/parser/__init__.py`
- `src/rag/pipeline/parser/base.py` (re-exports from types, parser registry)
- `src/rag/pipeline/parser/docling_parser.py`
- `tests/test_parser.py` (Docling-specific tests)

**Spec sections:** §6.3 (Docling parsing), §3.2 (subprocess memory isolation)

**Dependencies:** A2 (imports `Parser` Protocol, `ParseResult`, `ParsedDocument`, `ParsedSection`, `FileType`)

**Acceptance criteria:**
- `DoclingParser` class satisfies the `Parser` Protocol
- `supported_types` returns `{FileType.PDF, FileType.DOCX}`
- `parse()` runs Docling in a child subprocess via `multiprocessing` — the parent process never loads Docling models directly
- Uses `OcrAutoOptions` for OCR auto-detection
- Extracts: heading hierarchy, page boundaries, tables, reading order
- Returns `ParseSuccess` with populated `ParsedDocument` or `ParseError` with descriptive error
- Handles corrupted/unreadable files gracefully (returns `ParseError`, never raises)
- Tests: parse a fixture PDF (if available), parse error on nonexistent file, subprocess isolation verified (Docling not importable in parent after parse)
- `mypy --strict` and `ruff check` pass

---

### A8: Text Parser

**Responsibility:** Parse TXT and MD files as fallback for non-Docling file types. Implements the `Parser` Protocol.

**Files owned:**
- `src/rag/pipeline/parser/text_parser.py`
- `tests/test_text_parser.py`

**Spec sections:** §9 (text_parser.py in project structure)

**Dependencies:** A2 (imports `Parser` Protocol, `ParseResult`, `ParsedDocument`, `ParsedSection`, `FileType`)

**Acceptance criteria:**
- `TextParser` class satisfies the `Parser` Protocol
- `supported_types` returns `{FileType.TXT, FileType.MD}`
- For MD files: splits on headings (`#`, `##`, etc.) into `ParsedSection` objects with heading text and order
- For TXT files: treats entire file as a single section
- Reads files as UTF-8 with fallback encoding detection
- Returns `ParseError` for unreadable files (permission denied, binary files, etc.)
- Tests: parse a markdown file with multiple headings, parse a plain text file, encoding handling
- `mypy --strict` and `ruff check` pass

---

### A9: Normalizer

**Responsibility:** Clean parsed document text — whitespace normalization, header/footer suppression, paragraph reflow, heading hierarchy preservation, table boundary markers.

**Files owned:**
- `src/rag/pipeline/normalizer.py`
- `tests/test_normalizer.py`

**Spec sections:** §6.4 (normalization)

**Dependencies:** A2 (imports `ParsedDocument`, `ParsedSection`, `NormalizedDocument`)

**Acceptance criteria:**
- `normalize()` takes `ParsedDocument`, returns `NormalizedDocument`
- Whitespace cleanup: collapse multiple newlines/spaces, strip leading/trailing
- Heading hierarchy preservation: section headings pass through intact
- Page mapping preservation: `page_start`/`page_end` survive normalization
- Table boundary markers preserved (if Docling produces them)
- Computes `normalized_content_hash` (SHA-256 on normalized full text)
- Tests: whitespace cleanup, heading preservation, hash consistency (same input → same hash)
- `mypy --strict` and `ruff check` pass

---

### A10: Chunker

**Responsibility:** Structure-aware chunking — 512-token target, 64-token overlap, sentence-boundary aware, heading-boundary respecting, deterministic UUID5 IDs.

**Files owned:**
- `src/rag/pipeline/chunker.py`
- `tests/test_chunker.py`

**Spec sections:** §6.7 (chunking)

**Dependencies:** A2 (imports `NormalizedDocument`, `Chunk`, `NAMESPACE_RAG`)

**Acceptance criteria:**
- `chunk_document()` takes `NormalizedDocument`, returns `list[Chunk]`
- Target chunk size: 512 tokens via tiktoken `cl100k_base`
- Overlap: 64 tokens between adjacent chunks
- Respects sentence boundaries (doesn't split mid-sentence)
- Respects heading boundaries (new section = new chunk)
- Preserves table blocks (doesn't split tables across chunks)
- Deterministic IDs: `UUID5(NAMESPACE_RAG, f"{doc_id}:{section_order}:{chunk_order}")`
- Each `Chunk` has: text, text_normalized, page_start/end, section_heading, citation_label, token_count
- Citation label format: `"filename.pdf § Section Heading, pp. 12-14"`
- Tests: chunk a multi-section document, verify token counts ≤ 576 (512 + overlap tolerance), heading boundary respected, deterministic IDs stable across runs
- `mypy --strict` and `ruff check` pass

---

### A11: Embedder

**Responsibility:** Load BGE-M3 via sentence-transformers, batch embed texts, implement the `Embedder` Protocol.

**Files owned:**
- `src/rag/pipeline/embedder.py`
- `tests/test_embedder.py`

**Spec sections:** §6.8 (embedding)

**Dependencies:** A2 (imports `Embedder` Protocol), A3 (imports `EmbeddingConfig`)

**Acceptance criteria:**
- `SentenceTransformerEmbedder` class satisfies the `Embedder` Protocol
- Loads `BAAI/bge-m3` model from configured cache directory
- `embed_batch()` — encodes list of texts, returns list of 1024-dim float vectors
- `embed_query()` — encodes single query, returns 1024-dim float vector
- `dimensions` property returns `1024`
- `model_version` property returns the model identifier string
- Batch size configurable (default 32)
- Tests: embed a short text, verify vector dimensions = 1024, batch embedding returns correct count (can use a small model or mock for CI speed)
- `mypy --strict` and `ruff check` pass

---

### A12: Pipeline Runner

**Responsibility:** Orchestrate the full indexing pipeline: classify → parse → normalize → dedup → chunk → embed → index. Per-document error handling. Progress reporting.

**Files owned:**
- `src/rag/pipeline/__init__.py`
- `src/rag/pipeline/runner.py`
- `src/rag/pipeline/classifier.py`

**Spec sections:** §6.1-§6.9 (full pipeline), §6.2 (classification)

**Dependencies:** A4 (MetadataDB), A5 (VectorStore), A6 (Scanner — FileEvent input), A7+A8 (Parsers), A9 (Normalizer), A10 (Chunker), A11 (Embedder), A13 (Dedup), A14 (Indexer)

**Acceptance criteria:**
- `PipelineRunner` class wired with all dependencies (parsers, normalizer, dedup, chunker, embedder, indexer, db)
- `process_file()` — runs full pipeline for a single file, returns success/error result
- `process_batch()` — processes a list of `FileEvent` objects, wraps each in try/except, logs errors to processing_log, sets `process_status` in sync_state, continues on failure
- `classify()` in `classifier.py` — routes by file type, detects scanned PDFs (sample pages, avg chars < 50), produces `ClassificationResult`
- Parser selection: Docling for PDF/DOCX, TextParser for TXT/MD
- Pipeline flow: classify → parse → normalize → dedup check → chunk → embed → index
- If dedup detects duplicate: link to canonical, skip remaining stages
- Progress callback support (for `rag status` / CLI output)
- Prints summary on completion: "Indexed 487/500 documents. 13 errors."
- Tests: mock all dependencies, verify pipeline flow, verify error handling continues on failure
- `mypy --strict` and `ruff check` pass

---

### A13: Dedup

**Responsibility:** Exact-hash deduplication — SHA-256 on raw bytes and normalized text. Check document_hashes table, link duplicates to canonical.

**Files owned:**
- `src/rag/pipeline/dedup.py`
- `tests/test_dedup.py`

**Spec sections:** §6.5 (deduplication)

**Dependencies:** A2 (imports types), A4 (imports MetadataDB for document_hashes table)

**Acceptance criteria:**
- `check_duplicate()` — takes raw_hash and normalized_hash, queries document_hashes table
- Returns canonical doc_id if duplicate found, None if unique
- `register_hash()` — inserts raw_hash + normalized_hash + canonical_doc_id
- Tests: register a hash then detect duplicate, unique document passes through
- `mypy --strict` and `ruff check` pass

---

### A14: Qdrant Indexer

**Responsibility:** Construct `VectorPoint` objects from embedded chunks, upsert to Qdrant with deterministic IDs, clean up stale points.

**Files owned:**
- `src/rag/pipeline/indexer.py`
- `tests/test_indexer.py`

**Spec sections:** §6.9 (Qdrant indexing strategy)

**Dependencies:** A2 (imports `VectorPoint`, `QdrantPayloadModel`, `EmbeddedChunk`), A5 (imports `VectorStore`)

**Acceptance criteria:**
- `index_document()` — takes doc metadata + list of `EmbeddedChunk`, constructs `VectorPoint` objects with full `QdrantPayloadModel` payload, calls `VectorStore.upsert_points()`
- After upsert, calls `VectorStore.delete_stale_points()` with the set of new point IDs (removes points from previous indexing that no longer exist)
- Handles both chunk points and summary points (for Phase 2)
- Tests: mock VectorStore, verify correct point construction, verify stale cleanup called with correct IDs
- `mypy --strict` and `ruff check` pass

---

### A15: Retrieval Engine

**Responsibility:** Multi-stage retrieval pipeline — dense search, keyword search, RRF fusion, filter application. Orchestrates the full query flow.

**Files owned:**
- `src/rag/retrieval/__init__.py`
- `src/rag/retrieval/engine.py`
- `src/rag/retrieval/query_analyzer.py`
- `tests/test_retrieval.py`

**Spec sections:** §7.1 (retrieval pipeline), §7.2 (layer weighting), §7.3 (debug mode)

**Dependencies:** A5 (VectorStore for queries), A11 (Embedder for query embedding), A16 (Reranker), A17 (Citations)

**Acceptance criteria:**
- `RetrievalEngine` class with `search()` method (sync for CLI, `async_search()` for MCP)
- Query flow: embed query → dense search via VectorStore → keyword search via VectorStore → RRF fusion → rerank → citation assembly
- RRF fusion: `score = Σ 1/(60 + rank_i)`, merges dense + keyword results, deduplicates by point_id
- `async_search()` dispatches embedding and reranking via `asyncio.to_thread()`, uses AsyncQdrantClient for Qdrant queries
- `QueryAnalyzer` — classifies broad vs specific, extracts folder/date filter intent
- Folder filtering passed through to VectorStore queries
- Debug mode returns: query classification, layer weights, per-stage scores, timing
- Tests: mock VectorStore + Embedder + Reranker, verify RRF score calculation, verify filter passthrough, verify async wrapper
- `mypy --strict` and `ruff check` pass

---

### A16: Reranker

**Responsibility:** Load bge-reranker-v2-m3 ONNX model, rerank candidates by query-document relevance. Implements the `Reranker` Protocol.

**Files owned:**
- `src/rag/retrieval/reranker.py`
- `tests/test_reranker.py`

**Spec sections:** §7.1 step 6 (cross-encoder rerank)

**Dependencies:** A2 (imports `Reranker` Protocol, `SearchHit`), A3 (imports `RerankerConfig`)

**Acceptance criteria:**
- `OnnxReranker` class satisfies the `Reranker` Protocol
- Loads bge-reranker-v2-m3 ONNX model from configured path via `onnxruntime.InferenceSession`
- `rerank()` — takes query + list of `SearchHit`, scores each (query, hit.text) pair, returns top_k sorted by relevance score
- Handles empty candidate list gracefully
- Tests: rerank a small set of candidates (can mock ONNX session for CI), verify ordering changes, verify top_k respected
- `mypy --strict` and `ruff check` pass

---

### A17: Citations

**Responsibility:** Assemble citation objects from search hits — format file paths, section headings, page numbers, labels. Context expansion (adjacent chunks).

**Files owned:**
- `src/rag/retrieval/citations.py`
- `tests/test_citations.py`

**Spec sections:** §8.3 (citation format), §7.1 step 7 (post-processing)

**Dependencies:** A2 (imports `SearchHit`, `CitedEvidence`, `RetrievalResult`), A4 (MetadataDB for adjacent chunk lookup)

**Acceptance criteria:**
- `assemble_citations()` — takes reranked `SearchHit` list, returns `list[CitedEvidence]`
- Each `CitedEvidence` includes: text, title, path, section, pages, modified date, label, score, record_type
- Citation label format: `"filename.pdf § Section Heading, pp. 12-14"`
- Context expansion: fetches ±1 adjacent chunks from DB, merges text, deduplicates overlapping content
- Tests: format citations from mock hits, context expansion with adjacent chunks, dedup overlapping text
- `mypy --strict` and `ruff check` pass

---

### A18: MCP Server

**Responsibility:** MCP server setup (stdio + Streamable HTTP), tool definitions for all 4 tools, async handlers wrapping the retrieval engine.

**Files owned:**
- `src/rag/mcp/__init__.py`
- `src/rag/mcp/server.py`
- `src/rag/mcp/tools.py`
- `tests/test_mcp.py`

**Spec sections:** §8.1 (transport), §8.2 (4 tools), §2.1 Rule 10 (async strategy)

**Dependencies:** A15 (RetrievalEngine), A4 (MetadataDB for status/listing tools), A3 (MCPConfig)

**Acceptance criteria:**
- MCP server using `mcp` SDK (`>=1.25,<2`)
- stdio transport (primary) — no stdout writes except JSON-RPC (logging to stderr only)
- Streamable HTTP transport (alternative) on configured host:port
- 4 tools implemented:
  - `search_documents(query, folder_filter?, date_filter?, top_k?, debug?)` — calls `RetrievalEngine.async_search()`
  - `get_document_context(doc_id?, chunk_id?, window?)` — queries MetadataDB + VectorStore
  - `list_recent_documents(folder_filter?, limit?)` — queries MetadataDB
  - `get_sync_status()` — queries MetadataDB for counts
- All handlers are `async def`
- CPU-bound ops (embedding, reranking) dispatched via `asyncio.to_thread()`
- Tool input/output validated via Pydantic models from A2
- Returns structured error responses on failure (never crashes)
- Tests: mock RetrievalEngine + MetadataDB, verify tool schemas, verify async dispatch
- `mypy --strict` and `ruff check` pass

---

### A19: CLI + Init Wizard

**Responsibility:** All CLI commands — `rag init`, `rag index`, `rag serve`, `rag watch`, `rag status`, `rag doctor`, `rag search`, `rag mcp-config`. The user-facing entry point.

**Files owned:**
- `src/rag/cli.py`
- `src/rag/init.py`
- `tests/test_cli.py`

**Spec sections:** §10.3 (`rag init`), §10.5 (`rag mcp-config`), §11 (all CLI commands)

**Dependencies:** A3 (Config), A4 (MetadataDB), A5 (VectorStore), A6 (Scanner), A12 (PipelineRunner), A15 (RetrievalEngine), A18 (MCP Server)

**Acceptance criteria:**
- CLI framework: click or typer with subcommands
- `rag init` — interactive wizard: folder prompts, LLM CLI auto-detection (`which`), Qdrant Docker management, model download, MCP auto-config (Claude Desktop + Claude Code), config file creation. Re-runnable safely.
- `rag init --add-folder PATH` and `rag init --set-llm TOOL` non-interactive modes
- `rag index` — runs PipelineRunner.process_batch() with progress output, prints summary
- `rag index --folder PATH` and `rag index --file PATH` variants
- `rag serve` — starts MCP server (stdio). `rag serve --http` for Streamable HTTP.
- `rag watch` — starts filesystem watcher. `rag watch --daemon` for background.
- `rag status` — dashboard output per §11 (docs/chunks/errors, per-folder, recent errors). `--json` for machine-readable.
- `rag doctor` — health checks: Qdrant reachable, OCR available, models cached, SQLite writable, folders exist
- `rag search "query"` — wraps RetrievalEngine, prints results. `--debug` for debug output.
- `rag mcp-config --print` and `rag mcp-config --install TARGET`
- Tests: test CLI invocations with mocked dependencies, test init wizard flow, test status output formatting
- `mypy --strict` and `ruff check` pass

---

### A20: Integration & Acceptance Tests

**Responsibility:** End-to-end tests that spin up the real system — real Qdrant Docker container, real embedding model, real parsers, real MCP server — and validate the entire flow with real documents. No mocks. This is the definitive answer to "does it work?"

**Files owned:**
- `tests/conftest.py` (shared fixtures: Qdrant container, temp config, model cache)
- `tests/e2e/test_index_pipeline.py`
- `tests/e2e/test_search.py`
- `tests/e2e/test_mcp_server.py`
- `tests/e2e/test_watch_mode.py`
- `tests/e2e/test_cli.py`
- `tests/e2e/test_dedup.py`
- `tests/e2e/test_error_handling.py`
- `tests/e2e/test_summarization.py`
- `tests/fixtures/` — real sample documents (see fixture corpus below)

**Prerequisites:** Docker (for Qdrant), `claude` CLI installed (for summarization tests). The `claude` CLI is the project's standard local LLM for testing — it's available on all dev machines.

**Spec sections:** §15 (acceptance criteria), §12 Phase 1a/1b acceptance

**Dependencies:** All agents (tests the assembled system)

**Fixture corpus — real documents with known, searchable content:**

```
tests/fixtures/
├── quarterly-report.pdf        # 3-page PDF with headings: "Revenue", "Expenses", "Outlook"
│                                 Contains known text: "Q3 revenue increased 12% year-over-year"
├── project-plan.docx           # 2-page DOCX with headings: "Timeline", "Resources", "Risks"
│                                 Contains known text: "Phase 2 delivery is targeted for March"
├── meeting-notes.md            # Markdown with ## headings: "Attendees", "Decisions", "Action Items"
│                                 Contains known text: "Decided to migrate to PostgreSQL by Q4"
├── readme.txt                  # Plain text, no structure
│                                 Contains known text: "This system processes documents for retrieval"
├── scanned-invoice.pdf         # Single-page scanned PDF (image-only, needs OCR)
│                                 Contains known text: "Invoice #INV-2024-0847" (OCR must extract this)
├── duplicate-report.pdf        # Byte-identical copy of quarterly-report.pdf (different filename)
├── corrupted.pdf               # Truncated/invalid PDF (for error handling tests)
└── empty.txt                   # Zero-byte file (for edge case handling)
```

Every fixture has **known query-answer pairs** so tests can assert on specific search results, not just "something came back."

**Test suites:**

#### `test_index_pipeline.py` — Does indexing work end-to-end?

```
test_index_pdf:
  - Index quarterly-report.pdf
  - Assert: sync_state has entry with process_status='done'
  - Assert: documents table has entry with correct file_type, title
  - Assert: chunks table has >0 chunks with token_count <= 576
  - Assert: Qdrant has points with doc_id matching, record_type='chunk'
  - Assert: each Qdrant point has 1024-dim vector
  - Assert: chunk text is non-empty and contains recognizable content

test_index_docx:
  - Same as above for project-plan.docx

test_index_markdown:
  - Index meeting-notes.md
  - Assert: sections split on ## headings
  - Assert: chunk for "Decisions" section contains "PostgreSQL"

test_index_plaintext:
  - Index readme.txt
  - Assert: treated as single section
  - Assert: chunks created

test_index_scanned_pdf:
  - Index scanned-invoice.pdf
  - Assert: ocr_required=True in documents table
  - Assert: chunk text contains "INV-2024-0847" (OCR extraction verified)

test_index_corrupted_file:
  - Index corrupted.pdf
  - Assert: process_status='error' in sync_state
  - Assert: error_message is descriptive
  - Assert: other files in same batch still indexed successfully

test_index_empty_file:
  - Index empty.txt
  - Assert: handled gracefully (error or skip, not crash)

test_index_full_folder:
  - Point at fixtures/ folder
  - Run rag index
  - Assert: all valid files indexed, corrupted file errored
  - Assert: status counts match expected (N indexed, 1-2 errors)
```

#### `test_search.py` — Does search return the right results?

```
test_search_finds_known_content:
  - Index all fixtures
  - Search: "Q3 revenue year-over-year"
  - Assert: top result is from quarterly-report.pdf
  - Assert: result text contains "12%"
  - Assert: citation has correct file path, section "Revenue"

test_search_finds_across_file_types:
  - Search: "PostgreSQL migration"
  - Assert: result is from meeting-notes.md
  - Assert: section heading is "Decisions"

test_search_folder_filter:
  - Place fixtures in two subdirectories (work/, personal/)
  - Index both
  - Search with folder_filter="work/"
  - Assert: only results from work/ folder returned

test_search_returns_citations:
  - Search any query
  - Assert: every result has citation.title, citation.path, citation.label
  - Assert: citation.path is a real filesystem path
  - Assert: citation.label matches format "filename.ext § Section, pp. X-Y"

test_search_respects_top_k:
  - Search with top_k=3
  - Assert: at most 3 results returned

test_search_debug_mode:
  - Search with debug=True
  - Assert: result includes query_classification, debug_info with timing

test_search_no_results:
  - Search: "xyzzy quantum flux capacitor"
  - Assert: empty results list, no crash

test_search_hybrid:
  - Search a query that matches keyword but not semantically
  - Search a query that matches semantically but not keyword
  - Assert: hybrid search surfaces both types of matches
```

#### `test_mcp_server.py` — Does the MCP server respond correctly to tool calls?

```
test_mcp_search_documents:
  - Start MCP server as subprocess (stdio transport)
  - Send JSON-RPC tool call: search_documents(query="revenue growth")
  - Assert: valid JSON-RPC response with cited results
  - Assert: results match what rag search returns for same query

test_mcp_get_document_context:
  - Get a doc_id from a previous search
  - Send: get_document_context(doc_id=...)
  - Assert: returns document summary + section summaries (or sections if no summaries yet)

test_mcp_get_chunk_context:
  - Get a chunk_id from a previous search
  - Send: get_document_context(chunk_id=..., window=1)
  - Assert: returns chunk text + ±1 adjacent chunks

test_mcp_list_recent_documents:
  - Send: list_recent_documents(limit=5)
  - Assert: returns list of recently indexed documents with metadata
  - Assert: count <= 5

test_mcp_get_sync_status:
  - Send: get_sync_status()
  - Assert: returns counts (total, indexed, pending, errors)
  - Assert: counts match actual database state

test_mcp_invalid_tool_call:
  - Send malformed tool call
  - Assert: returns structured JSON-RPC error, server doesn't crash

test_mcp_concurrent_queries:
  - Send 5 search queries concurrently
  - Assert: all return valid results, no deadlocks or crashes

test_mcp_server_no_stdout_pollution:
  - Start MCP server, capture all stdout
  - Assert: only JSON-RPC messages on stdout (no print statements, no logging)
```

#### `test_watch_mode.py` — Does the file watcher detect and index changes?

```
test_watch_detects_new_file:
  - Start watcher on a temp directory
  - Copy a fixture file into the directory
  - Wait up to 15 seconds
  - Assert: file appears in sync_state with process_status='done'
  - Assert: searchable via rag search

test_watch_detects_modified_file:
  - Index a file, then modify its content
  - Wait for watcher to process
  - Assert: chunks updated (old content gone, new content searchable)

test_watch_detects_deleted_file:
  - Index a file, then delete it
  - Wait for watcher to process
  - Assert: is_deleted=1 in sync_state
  - Assert: Qdrant points for that doc_id removed
  - Assert: not returned in search results

test_watch_debounce:
  - Write to a file 5 times rapidly (within 1 second)
  - Assert: only processed once (not 5 times)

test_watch_ignores_excluded_patterns:
  - Create a file matching ignore pattern (e.g., inside .git/)
  - Assert: not indexed
```

#### `test_cli.py` — Do all CLI commands work?

```
test_cli_init_creates_config:
  - Run rag init with piped input (folder path, accept defaults)
  - Assert: config.toml created with correct folder path
  - Assert: Qdrant collection exists

test_cli_index_with_progress:
  - Run rag index on fixture folder
  - Assert: exit code 0
  - Assert: stdout contains progress/summary ("Indexed N/M documents")

test_cli_status_output:
  - Index fixtures, then run rag status
  - Assert: output contains doc counts, chunk count, folder breakdown
  - Assert: rag status --json returns valid JSON with same data

test_cli_doctor_healthy:
  - With everything configured, run rag doctor
  - Assert: all checks pass (Qdrant ✓, models ✓, folders ✓)

test_cli_doctor_unhealthy:
  - Stop Qdrant, run rag doctor
  - Assert: reports Qdrant unreachable

test_cli_search:
  - Run rag search "revenue" on indexed fixtures
  - Assert: prints results with citations to stdout

test_cli_mcp_config_print:
  - Run rag mcp-config --print
  - Assert: outputs valid JSON with mcpServers.dropbox-rag entry
```

#### `test_dedup.py` — Does deduplication work?

```
test_exact_duplicate_suppressed:
  - Index quarterly-report.pdf and duplicate-report.pdf (byte-identical)
  - Assert: only one set of Qdrant points exists (canonical)
  - Assert: duplicate document linked via duplicate_of_doc_id
  - Search for content: returns only one result (not two duplicates)

test_modified_file_not_duplicate:
  - Index a file, modify one word, re-index
  - Assert: treated as update (new chunks), not duplicate
```

#### `test_error_handling.py` — Does the system handle failures gracefully?

```
test_single_bad_file_doesnt_crash_batch:
  - Index a folder containing corrupted.pdf + valid files
  - Assert: valid files indexed successfully
  - Assert: corrupted.pdf has process_status='error'
  - Assert: rag index exit code is 0 (completed, not crashed)

test_qdrant_down_returns_error:
  - Stop Qdrant, attempt a search
  - Assert: returns structured error, not a stack trace

test_missing_folder_in_config:
  - Config points to a nonexistent folder
  - Assert: rag index reports clear error for that folder
  - Assert: other configured folders still indexed

test_poison_after_retries:
  - Index a file that fails 3+ times (simulate via fixture or mock)
  - Assert: process_status transitions to 'poison'
  - Assert: not retried again
```

#### `test_summarization.py` — Does LLM summarization work end-to-end?

Uses `claude` CLI as the local LLM (available on dev machines).

```
test_document_summary_generated:
  - Index quarterly-report.pdf with summarization enabled (command="claude")
  - Assert: documents table has non-null summary_l1, summary_l2, summary_l3
  - Assert: summary_l1 is a short phrase (< 10 words)
  - Assert: summary_l3 is a paragraph (> 50 words)
  - Assert: key_topics is a non-empty list of strings

test_section_summaries_generated:
  - Index quarterly-report.pdf with summarization enabled
  - Assert: sections table has non-null section_summary for each major section
  - Assert: summary text is relevant to section content

test_summary_vectors_in_qdrant:
  - Index with summarization enabled
  - Assert: Qdrant has points with record_type='document_summary'
  - Assert: Qdrant has points with record_type='section_summary'
  - Assert: summary points have 1024-dim vectors

test_pyramid_retrieval:
  - Index with summarization enabled
  - Search a broad query (e.g., "financial performance overview")
  - Assert: results include document_summary or section_summary hits (not just chunks)

test_summarization_caching:
  - Index a document, note summaries
  - Re-index the same document (content unchanged)
  - Assert: summaries not regenerated (cached by content hash)
  - Assert: no CLI calls made on re-index (verify via processing_log)

test_summarization_disabled_gracefully:
  - Set summarization.enabled = false in config
  - Index quarterly-report.pdf
  - Assert: document indexed successfully (chunks only, no summaries)
  - Assert: search still works (chunk-level retrieval)

test_summarization_cli_unavailable:
  - Set summarization.command = "nonexistent-tool"
  - Index quarterly-report.pdf
  - Assert: document indexed successfully (graceful degradation)
  - Assert: summaries are null, but chunks exist
```

**Test infrastructure (in `conftest.py`):**

```python
@pytest.fixture(scope="session")
def qdrant_container():
    """Start a real Qdrant Docker container for the test session.
    Tear down after all tests complete."""
    # docker run -d -p 6340:6333 qdrant/qdrant:v1.17
    # Use port 6340 to avoid conflicting with user's dev instance
    # yield the URL
    # docker stop + rm on teardown

@pytest.fixture(scope="session")
def embedding_model():
    """Load the real BGE-M3 model once for the entire test session.
    Cached in a temp dir. Slow first run (~60s), fast after."""

@pytest.fixture
def temp_config(tmp_path, qdrant_container):
    """Create a temporary AppConfig pointing at tmp_path for
    fixtures, the test Qdrant container, and a fresh SQLite DB.
    Summarization configured with command='claude'."""

@pytest.fixture
def indexed_fixtures(temp_config, embedding_model):
    """Index all fixture documents and return the config.
    Most search/MCP tests use this as their starting state."""

@pytest.fixture
def indexed_with_summaries(temp_config, embedding_model):
    """Index fixtures with summarization enabled (claude CLI).
    Used by pyramid retrieval and summarization tests."""

@pytest.fixture
def mcp_server_process(temp_config):
    """Start a real MCP server as a subprocess (stdio transport).
    Returns a client that can send JSON-RPC tool calls.
    Kills the process on teardown."""
```

**Acceptance criteria — the system works if and only if all of the following pass:**
- All 9 test suites pass: `pytest tests/e2e/ -v`
- Zero mocks in e2e tests — everything is real (real Qdrant Docker, real models, real files, real MCP server process)
- Total e2e suite runs in < 5 minutes (model loading cached across session)
- Any team member can run `make test-e2e` on a machine with Docker and Python 3.11+ and get an unambiguous pass/fail

---

## Testing Strategy

### Three tiers, all mandatory

**Tier 1: Static analysis (every agent, every commit)**
- `ruff check src/ tests/` — formatting and linting
- `mypy --strict src/` — type checking
- Fast, no external dependencies, catches type errors before runtime

**Tier 2: Unit tests (each agent writes its own)**
- Test module logic in isolation with mocked external dependencies
- Fast (< 30 seconds total), no Docker or model downloads needed
- Run with: `pytest tests/ -k "not e2e" -x`
- Purpose: catch regressions within individual modules during development

**Tier 3: End-to-end tests (A20, the real proof)**
- **No mocks.** Real Qdrant Docker container, real BGE-M3 model, real Docling parser, real MCP server subprocess, real fixture documents with known content.
- Every test asserts on specific, known content from fixture documents — not just "did something return" but "did the right thing return."
- MCP tests send actual JSON-RPC calls to a real stdio server process and validate responses.
- Run with: `make test-e2e` (which does `docker start`, `pytest tests/e2e/ -v`)
- First run downloads models (~60 seconds). Subsequent runs use cached models.
- **This is the single source of truth.** If e2e tests pass, the system works. If they fail, it doesn't. No ambiguity.

### CI pipeline

```bash
make lint        # Tier 1: ruff check + mypy strict
make test        # Tier 2: unit tests (fast, no Docker)
make test-e2e    # Tier 3: end-to-end (requires Docker + claude CLI, downloads models on first run)
make test-all    # All three tiers
```

### Local LLM for testing

The project uses `claude` CLI as the local LLM for summarization tests. All dev machines have it installed. The e2e test `conftest.py` configures `summarization.command = "claude"` in temp configs. Tests that exercise summarization are in `test_summarization.py` and `test_search.py` (pyramid retrieval). Tests that don't need summaries use `summarization.enabled = false` for speed.

### Fixture strategy

Fixtures are **real documents** created by A20 with known, specific, searchable content. Every fixture has documented query-answer pairs so any test can assert: "search for X → top result from file Y containing text Z." The fixture corpus is checked into the repo under `tests/fixtures/` and never changes after initial creation (stable test baseline).

### What "done" means for the whole project

The project is shippable when:
1. `make lint` passes (zero type errors, zero lint violations)
2. `make test` passes (all unit tests green)
3. `make test-e2e` passes (all end-to-end tests green on a clean machine with Docker)
4. A team member who has never seen the project can run `pip install -e . && rag init && rag index && rag search "revenue"` and get cited results from their own documents

---

## Merge Strategy

Agents work in isolated worktree branches. Merge order follows the dependency graph:

1. Merge A1 → main
2. Merge A2 → main (rebased on A1)
3. Merge A3 → main
4. Merge A4-A11 in any order (parallel, no conflicts)
5. Merge A12, A13 (may need minor conflict resolution if A12 references A13)
6. Merge A14
7. Merge A15, A16, A17 in any order
8. Merge A18
9. Merge A19
10. Merge A20

After each merge, run `mypy --strict src/` and `pytest` to verify no regressions.

---

## Summary

| Agent | Name | Files | Parallel Window | Est. Complexity |
|-------|------|-------|----------------|-----------------|
| A1 | Scaffolding | pyproject.toml, Makefile, directory structure | W1 (solo) | Low |
| A2 | Types & Protocols | types.py, protocols.py, results.py | W2 (solo) | Medium |
| A3 | Config | config.py | W3 (solo) | Medium |
| A4 | SQLite DB | db/connection.py, db/models.py, db/migrations.py | W4 (parallel) | Medium |
| A5 | Qdrant Client | db/qdrant.py, setup-qdrant.sh | W4 (parallel) | Medium-High |
| A6 | Scanner | sync/scanner.py | W4 (parallel) | Low |
| A7 | Docling Parser | pipeline/parser/docling_parser.py | W4 (parallel) | High |
| A8 | Text Parser | pipeline/parser/text_parser.py | W4 (parallel) | Low |
| A9 | Normalizer | pipeline/normalizer.py | W4 (parallel) | Medium |
| A10 | Chunker | pipeline/chunker.py | W4 (parallel) | Medium-High |
| A11 | Embedder | pipeline/embedder.py | W4 (parallel) | Low-Medium |
| A12 | Pipeline Runner | pipeline/runner.py, pipeline/classifier.py | W5 | Medium-High |
| A13 | Dedup | pipeline/dedup.py | W5 (parallel) | Low |
| A14 | Indexer | pipeline/indexer.py | W6 | Medium |
| A15 | Retrieval Engine | retrieval/engine.py, retrieval/query_analyzer.py | W7 (parallel) | High |
| A16 | Reranker | retrieval/reranker.py | W7 (parallel) | Medium |
| A17 | Citations | retrieval/citations.py | W7 (parallel) | Medium |
| A18 | MCP Server | mcp/server.py, mcp/tools.py | W8 | Medium-High |
| A19 | CLI + Init | cli.py, init.py | W9 | High |
| A20 | Integration Tests | test_integration.py, fixtures/ | W10 | Medium |

**Total: 20 agents, 10 parallelism windows, ~8 max concurrent agents (W4)**
