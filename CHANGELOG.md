# Changelog

All notable changes to local-rag are documented here.

## [v0.5.0] - 2026-03-16

Intelligence upgrade: semantic chunking, auto-generated questions, and MCP tool guidance make search smarter and more precise.

### Added
- **Semantic chunking** (opt-in): Max-Min algorithm with BGE-M3 sentence embeddings splits documents at natural topic boundaries instead of fixed token windows
- **Auto-generated questions**: LLM generates 3 questions per chunk at index time, prepended before embedding for richer vector representations
- **MCP tool guidance**: enriched tool descriptions (3-5 sentences each), ~600-word server instructions teaching scout-search-drill-down workflow, 3 prompts (research, discover, catch-up)
- Hero pipeline SVG on GitHub Pages
- Stale file detection: `rag index` removes vectors for deleted files

### Changed
- MCP module refactored: shared schemas extracted, helper deduplication
- GitHub Pages: responsive mobile layout, polished overview with Quick Start, hero visualization

---

## [v0.4.0] - 2026-03-12

Performance and reliability: 11 pipeline optimizations for Apple Silicon, geometric pyramid summaries, and a hardened filesystem watcher.

### Added
- **Geometric pyramid summaries**: 5 document levels (8w/16w/32w/64w/128w) and 3 section levels (8w/32w/128w) — only 128w embedded as vectors
- **Batch summarization**: single LLM call per document instead of per-chunk, with elapsed time in progress output
- **Persistent Docling worker**: reuses a single subprocess across files instead of spawning per-file
- **Early content-hash skip**: unchanged files bypass the full pipeline
- **Watcher improvements**: deletion handling, startup re-scan, retry queue with poison-file quarantine
- Progress output during model loading and file parsing
- Truncated JSON repair for LLM output

### Changed
- 11 indexing pipeline optimizations for Apple Silicon (dtype fix, persistent worker, hash skip, batch summarize)
- Docling parse timeout bumped from 300s to 360s
- CLI timeout bumped to 5 minutes for large summarization batches
- Progress display: simple line-by-line output replacing ANSI rolling display
- GitHub Pages updated for pyramid summaries and retrieval improvements, Codex added to MCP Integration section

### Fixed
- `--reindex` detecting empty state after cancelled reindex
- Garbled/interleaved progress output
- Dashboard file count reflecting actual files on disk
- Docling DrawingML warnings suppressed

---

## [v0.3.0] - 2026-03-12

Public release: renamed to local-rag, multi-LLM CLI support, Apache 2.0 license, and polished setup experience.

### Added
- **Apache 2.0 license**
- **Multi-LLM CLI support**: claude, kiro-cli, and codex auto-detected during `rag init`
- Cleanup & Uninstall tab on GitHub Pages and README section
- `rag mcp-config` auto-detects installed MCP clients and shows per-target config
- Model download during `make setup`
- HTTPS clone URL for external users

### Changed
- Project renamed from dropbox-rag to local-rag
- `rag init` detects all available LLM CLIs and lets user choose
- Page title and subtitle updated with project name and supported clients
- Setup streamlined: graceful Docker check, correct Qdrant tag, quiet pip

### Fixed
- **Critical**: `INSERT OR REPLACE` triggered CASCADE deleting all chunks — switched to safe upsert
- Chunk persistence: skip unchanged files, prevent cascade on upsert
- Summarizer: pass prompts via stdin instead of temp files
- Blank GitHub Pages site: restored `window.__App` assignment

---

## [v0.2.0] - 2026-03-10

Real-world hardening: full retrieval pipeline with 3-lane dense search, RRF fusion, and reranking. Rich terminal dashboard. Multi-client MCP support.

### Added
- **Full RAG retrieval pipeline**: 3-lane dense prefetch (doc summaries, section summaries, chunks) + keyword search, RRF fusion, cross-encoder reranking, citation expansion
- **LLM-friendly output format** for search results
- **LLM summarization** via CLI tool (claude, kiro-cli, codex)
- **Rich terminal dashboard** for `rag status` with MCP health and liveness checks
- `--reindex` CLI flag for full or single-file re-indexing
- Interactive architecture visualizations for GitHub Pages (Overview, Architecture tabs)
- Kiro MCP support
- File extension selection in `rag init` wizard
- MCP Integration docs with step-by-step setup

### Fixed
- SQLite threading error in MCP handlers (`check_same_thread=False`)
- Three bugs found during real-world testing

---

## [v0.1.0] - 2026-03-10

Initial working system: complete indexing pipeline from filesystem scan through Qdrant storage, MCP server with 4 tools, full CLI with 8 subcommands, and end-to-end test suite.

### Added
- **13-stage indexing pipeline**: classify, parse (Docling for PDF/DOCX, text fallback for TXT/MD), normalize, dedup (SHA-256), chunk (512 tokens, 64-token overlap), embed (BGE-M3, 1024-dim), index to Qdrant
- **MCP server**: stdio + Streamable HTTP transport, 4 tools (search_documents, quick_search, get_document_context, list_recent_documents)
- **CLI**: `rag init`, `rag index`, `rag serve`, `rag watch`, `rag status`, `rag doctor`, `rag search`, `rag mcp-config`
- **Interactive setup wizard** (`rag init`): folder selection, LLM CLI config, Qdrant setup, MCP registration
- **Config system**: TOML loader with Pydantic AppConfig model
- **Type layer**: Pydantic v2 models, Protocol classes, discriminated union Result types
- **SQLite metadata store** (WAL mode) with Qdrant vector store
- **End-to-end test suite** with fixture corpus and real query-answer validation
- Ruff formatting and linting, mypy strict mode
- Docs and setup scripts

[v0.5.0]: https://github.com/dgourlay/local-rag/compare/v0.4.0...v0.5.0
[v0.4.0]: https://github.com/dgourlay/local-rag/compare/v0.3.0...v0.4.0
[v0.3.0]: https://github.com/dgourlay/local-rag/compare/v0.2.0...v0.3.0
[v0.2.0]: https://github.com/dgourlay/local-rag/compare/v0.1.0...v0.2.0
[v0.1.0]: https://github.com/dgourlay/local-rag/commits/v0.1.0
