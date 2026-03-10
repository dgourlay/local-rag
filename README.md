# dropbox-rag

A local RAG system that indexes documents from filesystem folders, builds a
hybrid search index (dense vectors + keyword), and exposes retrieval via MCP to
Claude Desktop and Claude Code. Everything runs locally -- a single Python
process plus a Qdrant Docker container. No cloud infrastructure required.


## Features

- **Hybrid search** -- dense vector (BGE-M3) + keyword search with RRF fusion
- **ONNX cross-encoder reranking** for high-precision results
- **MCP integration** -- works as a tool for Claude Desktop and Claude Code
- **Multi-format parsing** -- PDF, DOCX, TXT, MD via Docling with OCR support
- **Filesystem watching** -- automatic re-indexing when documents change
- **Document summarization** -- shells out to any LLM CLI tool (claude, etc.)
- **Fully local** -- all models run on-device, no API keys needed for core search


## Prerequisites

- Python 3.11+
- Docker (for Qdrant)
- 4GB+ RAM for embedding and reranker models
- macOS or Linux


## Quick Start

```bash
# 1. Clone and install
git clone git@github.com:dgourlay/dropbox-rag.git
cd dropbox-rag
make setup           # creates venv, installs deps, starts Qdrant

# 2. Activate venv and configure
source .venv/bin/activate
rag init             # interactive wizard: pick folders, detect LLM CLI

# 3. Index and search
rag index            # scans folders, parses, chunks, embeds (downloads models on first run)
rag search "your query here"
```

That's it. Models (BGE-M3 embeddings ~1.5GB, BGE reranker ~1.2GB) download
automatically on first use.

To pre-download models so the first search is instant:

```bash
make download-models
```


## MCP Integration

The main use case is as an MCP tool server for Claude Desktop or Claude Code.

```bash
# Auto-install into Claude Desktop or Claude Code
rag mcp-config --install claude-desktop
rag mcp-config --install claude-code

# Or view the JSON config to add manually
rag mcp-config --print
```

After installing, restart Claude Desktop. It launches `rag serve` automatically
and gains access to 4 tools: `search_documents`, `get_document_context`,
`list_recent_documents`, `get_sync_status`.


## CLI Reference

| Command | Description |
|---|---|
| `rag init` | Interactive setup wizard -- configures folders, LLM CLI, Qdrant |
| `rag index` | Full scan and process all documents in configured folders |
| `rag serve` | Start the MCP server (stdio transport by default) |
| `rag watch` | Filesystem watcher -- auto-indexes on document changes |
| `rag status` | Dashboard showing document/chunk/error counts per folder |
| `rag doctor` | Health check -- verifies Qdrant, models, folders |
| `rag search "query"` | CLI search for testing (`--debug` for timing info) |
| `rag mcp-config` | Print or install MCP config (`--print`, `--install`) |


## Configuration

Config is TOML, resolved in this order (first match wins):

1. `RAG_CONFIG_PATH` environment variable
2. `./config.toml` in the current directory
3. `~/.config/dropbox-rag/config.toml` (default, created by `rag init`)

Only `[folders].paths` is required. Everything else has sensible defaults.
See [config.example.toml](config.example.toml) for all options with defaults.

Minimal config:

```toml
[folders]
paths = ["~/Documents"]
```

Full config with all defaults shown:

```toml
[folders]
paths = ["~/Documents", "~/Dropbox"]
extensions = ["pdf", "docx", "txt", "md"]
ignore = ["**/node_modules", "**/.git", "**/venv", "**/__pycache__"]

[database]
path = "~/.local/share/dropbox-rag/metadata.db"

[qdrant]
url = "http://localhost:6333"
collection = "documents"

[embedding]
model = "BAAI/bge-m3"
dimensions = 1024
batch_size = 32
cache_dir = "~/.cache/dropbox-rag/models"

[reranker]
model_path = "~/.cache/dropbox-rag/models/bge-reranker-v2-m3"
top_k_candidates = 30
top_k_final = 10

[summarization]
enabled = true
command = "claude"
args = ["--print", "--max-tokens", "2048"]
timeout_seconds = 60

[mcp]
transport = "stdio"
host = "127.0.0.1"
port = 8080

[watcher]
poll_interval_seconds = 5
debounce_seconds = 2
use_polling = false
batch_window_seconds = 10
```


## Architecture

Single Python process handles: filesystem watching (watchdog) -> Docling
parsing (in subprocess for memory isolation) -> normalization -> dedup ->
chunking (512 tokens, 64 overlap) -> embedding (BGE-M3, 1024-dim) -> Qdrant
indexing. Retrieval: hybrid dense + keyword search -> RRF fusion -> ONNX
cross-encoder reranker -> cited evidence returned to the calling LLM.

```
src/rag/
  cli.py               # CLI entry points (click)
  config.py            # TOML config loader
  init.py              # Setup wizard
  types.py             # Pydantic models, enums, type aliases
  protocols.py         # Protocol classes (Embedder, Summarizer, etc.)
  results.py           # Discriminated union Result types
  sync/                # Filesystem scanner + watcher
  pipeline/            # classify -> parse -> normalize -> dedup -> chunk -> embed -> index
    parser/            # Docling (PDF/DOCX) + text fallback (TXT/MD)
  retrieval/           # Hybrid search + RRF + reranker + citations
  mcp/                 # MCP server (stdio + HTTP) + 4 tool definitions
  db/                  # SQLite + Qdrant clients
migrations/            # SQL schema
tests/                 # Unit + e2e tests
```


## Development

```bash
make lint       # ruff check + format check + mypy strict
make test       # unit tests (fast, no Docker)
make test-e2e   # end-to-end (requires Qdrant + models)
make test-all   # lint + test + test-e2e
make format     # auto-format with ruff
```


## Troubleshooting

**"No config file found"** -- Run `rag init`, or copy `config.example.toml` to
`~/.config/dropbox-rag/config.toml` and edit it.

**Qdrant connection refused** -- Ensure Qdrant is running: `docker compose up -d`

**Model download fails** -- Check internet. Clear cache and retry:
```bash
rm -rf ~/.cache/dropbox-rag/models/
rag index              # re-downloads everything
```

**Search returns no results** -- Check `rag status`. If zero documents, run
`rag index`. If documents are indexed, try `rag search "query" --debug` to see
what's happening at each stage.

**MCP not working in Claude Desktop** -- Run `rag mcp-config --print` and
verify the Python path points to your venv. Restart Claude Desktop after
config changes.
