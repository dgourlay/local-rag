# local-rag

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
git clone https://github.com/dgourlay/local-rag.git
cd local-rag
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

The main use case is as an MCP tool server so your LLM can search your documents.

### Claude Code

**Option A -- auto-install:**

```bash
rag mcp-config --install claude-code
```

**Option B -- manual:** Add this to your `~/.claude.json` (or project-level
`.mcp.json`), replacing the Python path with your venv path:

```json
{
  "mcpServers": {
    "local-rag": {
      "command": "/path/to/local-rag/.venv/bin/python",
      "args": ["-m", "rag.cli", "serve"]
    }
  }
}
```

To find your exact Python path, run `rag mcp-config --print`.

After adding the config, restart Claude Code. The tools will be available
immediately -- Claude will automatically use them when you ask about your
documents.

### Claude Desktop

**Option A -- auto-install:**

```bash
rag mcp-config --install claude-desktop
```

This writes to `~/Library/Application Support/Claude/claude_desktop_config.json`.

**Option B -- manual:** Open Claude Desktop settings, go to Developer > MCP
Servers, and add:

```json
{
  "local-rag": {
    "command": "/path/to/local-rag/.venv/bin/python",
    "args": ["-m", "rag.cli", "serve"]
  }
}
```

Restart Claude Desktop after adding the config.

### Kiro

**Option A -- auto-install:**

```bash
rag mcp-config --install kiro
```

This writes to `~/.kiro/settings/mcp.json` (user-level). For project-level
config, use Option B with `.kiro/settings/mcp.json` in your project root.

**Option B -- manual:** Add to `~/.kiro/settings/mcp.json`:

```json
{
  "mcpServers": {
    "local-rag": {
      "command": "/path/to/local-rag/.venv/bin/python",
      "args": ["-m", "rag.cli", "serve"]
    }
  }
}
```

**Option C -- kiro-cli:**

```bash
kiro-cli mcp add \
  --name "local-rag" \
  --scope global \
  --command "/path/to/local-rag/.venv/bin/python" \
  --args "-m rag.cli serve"
```

Run `rag mcp-config --print` to get the exact Python path for your venv.

### Available MCP tools

Once connected, your LLM has access to these tools:

| Tool | Description |
|---|---|
| `search_documents` | Hybrid search with reranking. Accepts `query`, optional `folder_filter`, `date_filter`, `top_k`, `format` ("text" or "json"). Returns cited evidence passages with scores. |
| `quick_search` | Lightweight document-level scan returning document summaries. Faster than `search_documents` for broad queries. |
| `get_document_context` | Get document overview (summary + sections) by `doc_id`, or a chunk with surrounding context by `chunk_id`. |
| `list_recent_documents` | List recently indexed documents, optionally filtered by folder. |
| `get_sync_status` | Check indexing health: total files, indexed count, errors, per-folder breakdown. |

### Verifying the connection

After setup, ask your LLM something like:

> "What documents do I have indexed?" (uses `list_recent_documents`)
>
> "Search my documents for gate operations procedures" (uses `search_documents`)

If it responds with content from your documents, the MCP connection is working.
If not, check `rag doctor` and verify the Python path in your MCP config points
to the correct venv.


## CLI Reference

| Command | Description |
|---|---|
| `rag init` | Interactive setup wizard -- configures folders, LLM CLI, Qdrant |
| `rag index` | Full scan and process all documents in configured folders |
| `rag index --reindex` | Purge all index data and re-process everything from scratch |
| `rag index --reindex FILE` | Clear index state for a single file and re-process it |
| `rag serve` | Start the MCP server (stdio transport by default) |
| `rag watch` | Filesystem watcher -- auto-indexes on document changes |
| `rag status` | Dashboard showing document/chunk/error counts, MCP health, liveness |
| `rag doctor` | Health check -- verifies Qdrant, models, folders |
| `rag search "query"` | CLI search for testing (`--debug` for lane/weight details, `--top-k N`) |
| `rag mcp-config` | Print or install MCP config (`--print`, `--install`) |


## Configuration

Config is TOML, resolved in this order (first match wins):

1. `RAG_CONFIG_PATH` environment variable
2. `./config.toml` in the current directory
3. `~/.config/local-rag/config.toml` (default, created by `rag init`)

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
paths = ["~/Documents", "~/Work"]
extensions = ["pdf", "docx", "txt", "md"]
ignore = ["**/node_modules", "**/.git", "**/venv", "**/__pycache__"]

[database]
path = "~/.local/share/local-rag/metadata.db"

[qdrant]
url = "http://localhost:6333"
collection = "documents"

[embedding]
model = "BAAI/bge-m3"
dimensions = 1024
batch_size = 32
cache_dir = "~/.cache/local-rag/models"

[reranker]
model_path = "~/.cache/local-rag/models/bge-reranker-v2-m3"
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
chunking (512 tokens, 64 overlap) -> embedding (BGE-M3, 1024-dim) ->
summarization (LLM CLI) -> Qdrant indexing. Retrieval: 3-lane prefetch
(document summaries, section summaries, chunks) -> RRF fusion with layer
weighting -> recency boost -> ONNX cross-encoder reranker -> cited evidence
returned to the calling LLM.

```
src/rag/
  cli.py               # CLI entry points (click)
  config.py            # TOML config loader
  init.py              # Setup wizard
  types.py             # Pydantic models, enums, type aliases
  protocols.py         # Protocol classes (Embedder, Summarizer, etc.)
  results.py           # Discriminated union Result types
  sync/                # Filesystem scanner + watcher
  pipeline/            # classify -> parse -> normalize -> dedup -> chunk -> embed -> summarize -> index
    parser/            # Docling (PDF/DOCX) + text fallback (TXT/MD)
  retrieval/           # 3-lane prefetch + RRF + layer weighting + reranker + citations
  mcp/                 # MCP server (stdio + HTTP) + 5 tool definitions
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


## Cleanup & Uninstall

local-rag stores ML models, config, and data across several directories (~6.5 GB total, mostly models).

### Storage locations

| What | Path | Size |
|------|------|------|
| Embedding model (BGE-M3) | `~/.cache/local-rag/models/models--BAAI--bge-m3/` | ~4.3 GB |
| Reranker model (ONNX) | `~/.cache/local-rag/models/bge-reranker-v2-m3/` | ~2.1 GB |
| SQLite database | `~/.local/share/local-rag/metadata.db` | ~1 MB |
| Config file | `~/.config/local-rag/config.toml` | < 1 KB |
| Qdrant data | Docker volume `local-rag_qdrant_data` | Varies |

### Full uninstall

```bash
# 1. Stop and remove Qdrant container + data
docker compose down -v

# 2. Remove cached models (~6.4 GB)
rm -rf ~/.cache/local-rag

# 3. Remove database and application data
rm -rf ~/.local/share/local-rag

# 4. Remove config
rm -rf ~/.config/local-rag

# 5. Remove MCP config entries (if installed)
#    Edit the relevant file and remove the "local-rag" entry:
#    Claude Code:    ~/.claude.json
#    Claude Desktop: ~/Library/Application Support/Claude/claude_desktop_config.json
#    Kiro:           ~/.kiro/settings/mcp.json

# 6. Uninstall the Python package
pip uninstall local-rag
```

### Free disk space (keep local-rag installed)

```bash
# Remove models only (~6.4 GB, re-downloads on next use)
rm -rf ~/.cache/local-rag/models

# Re-download when ready
make download-models
```


## Troubleshooting

**"No config file found"** -- Run `rag init`, or copy `config.example.toml` to
`~/.config/local-rag/config.toml` and edit it.

**Qdrant connection refused** -- Ensure Qdrant is running: `docker compose up -d`

**Model download fails** -- Check internet. Clear cache and retry:
```bash
rm -rf ~/.cache/local-rag/models/
rag index              # re-downloads everything
```

**Search returns no results** -- Check `rag status`. If zero documents, run
`rag index`. If documents are indexed, try `rag search "query" --debug` to see
what's happening at each stage.

**MCP not working in Claude Desktop** -- Run `rag mcp-config --print` and
verify the Python path points to your venv. Restart Claude Desktop after
config changes.
