# Local Dropbox RAG System — Build Specification

## 1. Document Purpose

This is the build specification for a local, laptop-hosted variant of the Dropbox RAG system. It indexes documents from configured local folders, builds a searchable vector index with multi-level summaries, and exposes retrieval to local LLM tools (Claude Desktop, Claude Code, etc.) via a local MCP server. No cloud infrastructure required — everything runs on the developer's machine.

This spec is the local counterpart to `plan/cloud/dropbox-rag-final-spec.md`. The core pipeline logic (classify → parse → normalize → dedup → chunk → embed → index) and retrieval engine (hybrid search + RRF + reranker) are shared designs. The differences are in infrastructure, deployment, and model choices.

---

## 2. Design Decisions

| Decision | Resolution | Rationale |
|---|---|---|
| Parser | **Docling** (IBM open-source) | Same as cloud — structured output, heading hierarchy, table detection, single pipeline for PDF + DOCX |
| Metadata DB | **SQLite** | No need for LISTEN/NOTIFY in single-process architecture; zero config, stdlib, file-based |
| Vector DB | **Qdrant** (Docker container) | Same as cloud — single collection, Prefetch API, built-in text index; lightweight Docker footprint (~200MB RAM for small collections) |
| Embedding model | **bge-base-en-v1.5** via sentence-transformers (768-dim) | Runs locally on CPU, no API calls, fast batch performance (~50-200ms per batch of 32), good English retrieval quality |
| Summarization LLM | **Local CLI tool** (claude, kiro-cli, codex, or similar) | User provides their preferred CLI; avoids running a large local model in RAM; prompt piped to stdin, JSON response from stdout |
| OCR engine | **Tesseract** via Docling | Same as cloud — local, open source |
| Qdrant topology | **Single collection** with `record_type` payload field | Same as cloud |
| Hybrid search | **Qdrant built-in text index** on chunk text | Same as cloud — BM25-equivalent keyword matching, zero additional infra |
| Score fusion | **Reciprocal Rank Fusion (RRF)** with k=60 | Same as cloud |
| Reranker | **bge-reranker-v2-m3** via ONNX Runtime on CPU | Same as cloud — ~100ms for 30 candidates |
| MCP transport | **stdio** (primary) for Claude Desktop; **localhost HTTP** as alternative | stdio is the standard local MCP transport; no TLS/auth needed |
| File sync | **watchdog** filesystem watcher + content hash dedup | Replaces Dropbox webhook/cursor; watches configured folder list for changes |
| Deployment | **Single Python process** + Qdrant Docker container | No Docker Compose orchestra needed for the Python services; simple `pip install` + `docker run` |
| Answer synthesis | **None — RAG returns evidence only** | The calling LLM (Claude Desktop, etc.) synthesizes answers from returned citations |
| Language | **Python 3.11+ with strict typing** | ML ecosystem (Docling, sentence-transformers, onnxruntime) is Python-only; strict typing via Pydantic v2 + mypy strict mode + dataclasses enables reliable AI-assisted development |

---

## 2.1 Typing & Code Quality Mandate

All Python code in this project must follow strict typing conventions to enable reliable AI-assisted code generation and left-shift testing. These are not guidelines — they are enforced by CI.

### Rules

1. **mypy strict mode** — `mypy.ini` or `pyproject.toml` sets `strict = true`. Every function has full type annotations for all parameters and return values. No `Any` types except where wrapping untyped third-party libraries, and those must be isolated behind typed wrapper functions.

2. **Pydantic v2 models at every boundary** — all data flowing between pipeline stages, all config, all MCP tool inputs/outputs, all database row mappings, and all Qdrant payloads are Pydantic `BaseModel` subclasses with explicit field types and validators. No raw dicts crossing module boundaries.

3. **dataclasses for internal value objects** — lightweight internal structures (e.g. `ParsedSection`, `ChunkWindow`, `RRFCandidate`) use `@dataclass(frozen=True, slots=True)` with full type annotations.

4. **Enum and Literal for constrained values** — `process_status`, `record_type`, `summary_level`, `file_type`, and similar fields use `Literal` types or `StrEnum`. No bare strings for values that have a known set of options.

5. **TypedDict for JSON payloads** — when interfacing with Qdrant payloads or LLM CLI JSON responses, use `TypedDict` (not `dict[str, Any]`) to type the expected shape.

6. **Protocol classes for pluggable backends** — the embedder, summarizer, and database interfaces are defined as `typing.Protocol` classes. This allows swapping backends (e.g. local bge vs Bedrock Titan) without inheritance, and mypy verifies structural compatibility.

7. **No untyped containers** — `list`, `dict`, `set`, `tuple` always have type parameters. `list[str]` not `list`. `dict[str, float]` not `dict`.

8. **Result types for fallible operations** — operations that can fail (LLM CLI calls, file parsing, Qdrant queries) return explicit result types rather than raising exceptions for expected failures. Use a `Result[T, E]` pattern or Pydantic models with success/error variants.

9. **ruff for linting and formatting** — `ruff check` and `ruff format` enforced. Ruff's type-checking rules enabled (ANN, TCH rule sets).

### Example Type Patterns

```python
# Pydantic model at pipeline boundary
class ParsedDocument(BaseModel):
    doc_id: str
    title: str | None
    file_type: Literal["pdf", "docx", "txt", "md"]
    sections: list["ParsedSection"]
    ocr_required: bool
    ocr_confidence: float | None
    raw_content_hash: str

# Frozen dataclass for internal value object
@dataclass(frozen=True, slots=True)
class ChunkWindow:
    center: "Chunk"
    before: list["Chunk"]
    after: list["Chunk"]

# Protocol for pluggable backend
class Embedder(Protocol):
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, query: str) -> list[float]: ...
    @property
    def dimensions(self) -> int: ...
    @property
    def model_version(self) -> str: ...

# Literal types for constrained values
ProcessStatus = Literal["pending", "processing", "done", "error", "poison"]
RecordType = Literal["chunk", "section_summary", "document_summary"]
SummaryLevel = Literal["l1", "l2", "l3"]

# TypedDict for Qdrant payload
class QdrantPayload(TypedDict):
    record_type: RecordType
    summary_level: SummaryLevel | None
    doc_id: str
    section_id: str | None
    chunk_id: str | None
    title: str
    file_path: str
    folder_path: str
    folder_ancestors: list[str]
    file_type: str
    modified_at: str
    text: str

# Result type for fallible operations
class SummaryResult(BaseModel):
    success: bool
    summary_l1: str | None = None
    summary_l2: str | None = None
    summary_l3: str | None = None
    key_topics: list[str] = []
    doc_type_guess: str | None = None
    error: str | None = None
```

### pyproject.toml Type Config

```toml
[tool.mypy]
strict = true
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_generics = true
check_untyped_defs = true
no_implicit_reexport = true

[[tool.mypy.overrides]]
module = ["docling.*", "onnxruntime.*"]
ignore_missing_imports = true

[tool.ruff]
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "ANN", "B", "A", "TCH", "UP", "RUF"]
```

## 3. Architecture

### 3.1 System Overview

```
Configured local folders
  /Users/you/Dropbox/Work
  /Users/you/Documents/Reports
  ~/projects/notes
       │
       ▼
  Filesystem watcher (watchdog)
       │
       ▼
  ┌─── Processing Pipeline (single Python process) ──────────┐
  │                                                           │
  │  classify → Docling parse (+OCR) → normalize → dedup     │
  │  → chunk → embed (local bge model)                       │
  │  → summarize (shell out to LLM CLI)                      │
  │  → Qdrant upsert                                         │
  │                                                           │
  └───────────────────────────────────────────────────────────┘
       │              │              │
       ▼              ▼              ▼
    SQLite        Qdrant          LLM CLI
   (metadata)   (vectors)     (summaries only,
                               during indexing)
       │
       ▼
  MCP Server (stdio / localhost HTTP)
    → embed query (local bge)
    → hybrid search (dense + keyword)
    → RRF fusion
    → ONNX reranker
    → citation assembly
    → return evidence to calling LLM
```

### 3.2 What Runs Where

| Component | How It Runs | Memory Footprint |
|---|---|---|
| Qdrant | Docker container (`qdrant/qdrant:v1.12`) | ~200MB base + index size |
| Embedding model (bge-base-en-v1.5) | Loaded in Python process via sentence-transformers | ~500MB |
| ONNX reranker (bge-reranker-v2-m3) | Loaded in Python process via onnxruntime | ~500MB |
| Docling parser | Loaded in Python process (on-demand during indexing) | 4-6GB during parsing, released after |
| SQLite | In-process (stdlib) | Negligible |
| LLM CLI (claude, kiro, etc.) | Spawned as subprocess during indexing only | External process, not resident |

**Minimum system requirements:** 16GB RAM, modern multi-core CPU (Apple Silicon or recent Intel/AMD). Parsing is the peak memory operation — Docling's ML models load during indexing, then release. At query time, only the embedding model, reranker, and Qdrant are active (~1.2GB total).

---

## 4. Configuration

### 4.1 Config File

Single TOML config file at `~/.config/dropbox-rag/config.toml` (or project-local `config.toml`):

```toml
[folders]
# List of absolute paths to watch and index
paths = [
    "/Users/you/Dropbox/Work",
    "/Users/you/Documents/Reports",
    "~/projects/notes"
]
# File types to index
extensions = ["pdf", "docx", "txt", "md"]
# Paths to ignore (glob patterns)
ignore = ["**/node_modules", "**/.git", "**/venv", "**/__pycache__"]

[database]
# SQLite database location
path = "~/.local/share/dropbox-rag/metadata.db"

[qdrant]
# Qdrant connection
url = "http://localhost:6333"
collection = "documents"

[embedding]
# Local embedding model
model = "BAAI/bge-base-en-v1.5"
dimensions = 768
batch_size = 32
# Cache directory for downloaded model weights
cache_dir = "~/.cache/dropbox-rag/models"

[reranker]
# ONNX reranker
model_path = "~/.cache/dropbox-rag/models/bge-reranker-v2-m3"
top_k_candidates = 30
top_k_final = 10

[summarization]
# LLM CLI tool for generating summaries during indexing
enabled = true
provider = "cli"
command = "claude"
args = ["--print", "--max-tokens", "2048"]
# Prompt is piped to stdin, JSON response read from stdout
# Set to false to skip summaries entirely (chunks + section headings only)
timeout_seconds = 60

[mcp]
# MCP server configuration
transport = "stdio"
# Alternative: transport = "http", host = "127.0.0.1", port = 8080

[watcher]
# Filesystem watcher settings
poll_interval_seconds = 5
# Debounce rapid file changes (e.g. during saves)
debounce_seconds = 2
```

### 4.2 Environment Variables

Minimal — most config lives in the TOML file. Environment variables only for secrets or overrides:

```bash
# Only needed if your LLM CLI requires auth
ANTHROPIC_API_KEY=sk-...  # If using claude CLI
# Override config file location
RAG_CONFIG_PATH=/path/to/config.toml
```

---

## 5. Data Model

### 5.1 SQLite Schema

The schema mirrors the cloud PostgreSQL schema with simplifications (no LISTEN/NOTIFY, no advisory locks, UUIDs stored as TEXT).

```sql
-- SYNC STATE
CREATE TABLE sync_state (
    id                  TEXT PRIMARY KEY,  -- UUID as text
    file_path           TEXT UNIQUE NOT NULL,
    file_name           TEXT NOT NULL,
    folder_path         TEXT NOT NULL,
    folder_ancestors    TEXT NOT NULL,  -- JSON array, e.g. ["/", "/Work", "/Work/Reports"]
    file_type           TEXT NOT NULL,
    size_bytes          INTEGER,
    modified_at         TEXT NOT NULL,  -- ISO 8601
    content_hash        TEXT NOT NULL,  -- SHA-256 of raw file bytes
    synced_at           TEXT DEFAULT (datetime('now')),
    process_status      TEXT DEFAULT 'pending'
                        CHECK (process_status IN ('pending','processing','done','error','poison')),
    error_message       TEXT,
    retry_count         INTEGER DEFAULT 0,
    is_deleted          INTEGER DEFAULT 0,
    created_at          TEXT DEFAULT (datetime('now'))
);
CREATE INDEX idx_sync_status ON sync_state (process_status) WHERE NOT is_deleted;
CREATE INDEX idx_sync_path ON sync_state (file_path);

-- DOCUMENTS
CREATE TABLE documents (
    doc_id                  TEXT PRIMARY KEY,  -- UUID as text
    file_path               TEXT UNIQUE NOT NULL REFERENCES sync_state(file_path),
    folder_path             TEXT NOT NULL,
    folder_ancestors        TEXT NOT NULL,  -- JSON array
    title                   TEXT,
    file_type               TEXT NOT NULL,
    modified_at             TEXT NOT NULL,
    indexed_at              TEXT DEFAULT (datetime('now')),
    parser_version          TEXT,
    raw_content_hash        TEXT NOT NULL,
    normalized_content_hash TEXT,
    duplicate_of_doc_id     TEXT REFERENCES documents(doc_id),
    ocr_required            INTEGER DEFAULT 0,
    ocr_confidence          REAL,
    doc_type_guess          TEXT,
    key_topics              TEXT,  -- JSON array
    summary_l1              TEXT,
    summary_l2              TEXT,
    summary_l3              TEXT,
    summary_content_hash    TEXT,
    embedding_model_version TEXT
);
CREATE INDEX idx_doc_folder ON documents (folder_path);
CREATE INDEX idx_doc_content_hash ON documents (normalized_content_hash);

-- SECTIONS
CREATE TABLE sections (
    section_id          TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    section_heading     TEXT,
    section_order       INTEGER NOT NULL,
    page_start          INTEGER,
    page_end            INTEGER,
    section_summary     TEXT,
    section_summary_l2  TEXT,
    embedding_model_version TEXT
);
CREATE INDEX idx_section_doc ON sections (doc_id);

-- CHUNKS
CREATE TABLE chunks (
    chunk_id                TEXT PRIMARY KEY,
    doc_id                  TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    section_id              TEXT REFERENCES sections(section_id) ON DELETE CASCADE,
    chunk_order             INTEGER NOT NULL,
    chunk_text              TEXT NOT NULL,
    chunk_text_normalized   TEXT NOT NULL,
    page_start              INTEGER,
    page_end                INTEGER,
    section_heading         TEXT,
    citation_label          TEXT,
    token_count             INTEGER,
    embedding_model_version TEXT
);
CREATE INDEX idx_chunk_doc ON chunks (doc_id);
CREATE INDEX idx_chunk_section ON chunks (section_id);

-- DEDUPLICATION
CREATE TABLE document_hashes (
    file_path           TEXT PRIMARY KEY REFERENCES sync_state(file_path),
    raw_hash            TEXT NOT NULL,
    normalized_hash     TEXT,
    canonical_doc_id    TEXT REFERENCES documents(doc_id),
    created_at          TEXT DEFAULT (datetime('now'))
);

-- PROCESSING LOG
CREATE TABLE processing_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id          TEXT REFERENCES documents(doc_id),
    file_path       TEXT,
    stage           TEXT NOT NULL,
    status          TEXT NOT NULL,
    duration_ms     INTEGER,
    details         TEXT,  -- JSON
    created_at      TEXT DEFAULT (datetime('now'))
);
CREATE INDEX idx_proclog_doc ON processing_log (doc_id);
```

### 5.2 Qdrant Collection

Identical to cloud spec. Single collection `documents` with cosine vectors, except **768-dim** (bge-base) instead of 1024-dim (Titan).

Payload indices: record_type (keyword), summary_level (keyword), doc_id (keyword), folder_path (keyword), file_type (keyword), modified_at (datetime), file_path (keyword), doc_type_guess (keyword).

Full-text index on `text` field (word tokenizer, min 3 / max 20 chars, lowercase).

### 5.3 Qdrant Point Payload

```json
{
  "record_type": "chunk | section_summary | document_summary",
  "summary_level": "l1 | l2 | l3 | null",
  "doc_id": "uuid",
  "section_id": "uuid | null",
  "chunk_id": "uuid | null",
  "title": "Q3 Operational Review",
  "file_path": "/Users/you/Dropbox/Work/Reports/Q3-review.pdf",
  "folder_path": "/Users/you/Dropbox/Work/Reports",
  "folder_ancestors": ["/", "/Users/you/Dropbox/Work", "/Users/you/Dropbox/Work/Reports"],
  "file_type": "pdf",
  "modified_at": "2025-03-01T10:30:00Z",
  "page_start": 12,
  "page_end": 14,
  "section_heading": "Revenue Analysis",
  "chunk_order": 7,
  "doc_type_guess": "quarterly_review",
  "ocr_confidence": null,
  "token_count": 487,
  "citation_label": "Q3-review.pdf § Revenue Analysis, pp. 12-14",
  "text": "The chunk or summary text..."
}
```

---

## 6. Processing Pipeline

### 6.1 File Discovery & Sync

Instead of Dropbox API webhooks and cursor-based delta sync, the local system uses filesystem watching.

**Initial scan:** On first run, walk all configured folders, hash every matching file, record in sync_state, queue all for processing.

**Ongoing watch:** `watchdog` library monitors configured folders for file create/modify/delete events. Debounce rapid changes (2-second window) to avoid processing mid-save. On change detection: compute SHA-256 of file, compare to sync_state.content_hash. If unchanged, skip. If changed or new, queue for processing. If deleted, mark is_deleted=1 and remove vectors from Qdrant.

**Re-scan on startup:** Walk all folders on process start to catch changes that happened while the system was stopped. Compare file modification times and hashes against sync_state.

### 6.2 Classification

Same as cloud: route by file type (PDF/DOCX/TXT/MD), detect scanned PDFs (sample 3 pages, if avg chars < 50 → scanned), infer folder context from path, estimate complexity.

### 6.3 Parsing (Docling)

Same as cloud. Docling handles PDF + DOCX. Enable OCR (Tesseract) when likely_scanned=True. Produces structured output with heading hierarchy, page boundaries, tables, reading order.

**Local consideration:** Docling loads ML models on first parse (~4-6GB). For laptop use, load models on demand when processing starts, release after the batch completes. Don't keep Docling models resident — they're too large to hold in memory alongside the embedding model and reranker.

### 6.4 Normalization

Same as cloud: whitespace cleanup, header/footer suppression, paragraph reflow, heading hierarchy preservation, page mapping preservation, table boundary markers.

### 6.5 Deduplication

Same as cloud Phase 1: exact content hash match (SHA-256 on raw bytes), normalized content hash match (SHA-256 on normalized text). If duplicate, link to canonical document, skip indexing.

SimHash near-dedup deferred (same as cloud Phase 4).

### 6.6 Summarization (LLM CLI)

Same structured output as cloud, different backend. Instead of Bedrock Haiku API call, shell out to configured LLM CLI tool.

**Per-document summary call:**
1. Construct prompt requesting structured JSON: summary_l1 (phrase), summary_l2 (1-2 sentences), summary_l3 (paragraph), key_topics (list), doc_type_guess
2. Pipe prompt + document excerpt to CLI stdin
3. Parse JSON response from stdout
4. Timeout after configured seconds (default 60), retry once, then skip summaries for this doc

**Section summaries:** One CLI call per major section (H1/H2 boundary).

**Caching:** All summaries cached by normalized_content_hash in SQLite. If document content hasn't changed, skip summarization entirely.

**Graceful degradation:** If summarization is disabled in config or the CLI tool isn't available, the system still works — you just get chunk-level and section-heading-level retrieval without the pyramid summaries. This is a valid operating mode.

### 6.7 Chunking

Same as cloud: 512-token target (tiktoken cl100k_base), 64-token overlap, sentence-boundary aware, heading-boundary respecting, table-preserving. Deterministic IDs via UUID5(doc_id + section_order + chunk_order).

### 6.8 Embedding (Local bge Model)

Load `bge-base-en-v1.5` via sentence-transformers. Batch size 32 texts per encode call. Produces 768-dim vectors. Record model version in metadata.

**Performance on laptop:** ~50-200ms per batch of 32 on Apple Silicon, ~200-500ms on Intel. A 500-document corpus with ~10,000 chunks takes roughly 2-5 minutes to embed from scratch.

### 6.9 Qdrant Indexing

Same as cloud: atomic per-document swap (delete all existing points for doc_id, then upsert new points). Pyramid summary points + section summary points + chunk points.

---

## 7. Retrieval Engine

Identical to cloud spec. All stages run behind a single `search_documents` MCP tool call.

### 7.1 Pipeline

1. **Query analysis** — broad vs specific classification, extract folder/date filter intent, extract keywords
2. **Embed query** — bge-base-en-v1.5 (same model as indexing)
3. **Dense search** — Qdrant cosine similarity via Prefetch API: doc summaries (top 20), section summaries (top 20), chunks (top 30), with metadata filters
4. **Keyword search** — Qdrant built-in text index on "text" field, chunks top 30
5. **RRF fusion** — score = Σ 1/(60 + rank_i), merge dense + keyword, apply layer weighting
6. **Cross-encoder rerank** — bge-reranker-v2-m3 ONNX, top 30 → top 10
7. **Post-processing** — recency boost (90-day half-life, max 30% influence), context expansion (±1 adjacent chunks), dedup overlapping chunks, assemble citations

### 7.2 Layer Weighting

Same as cloud: broad queries boost summaries, specific queries boost chunks.

### 7.3 Debug Mode

Same as cloud: returns query classification, layer weights, scores at each stage, timing breakdown.

---

## 8. MCP Server

### 8.1 Transport

**Primary: stdio** — Claude Desktop and Claude Code launch the MCP server as a subprocess. This is the standard local MCP pattern. No network configuration, no auth needed.

**Alternative: localhost HTTP** — For tools that need a network MCP endpoint, run on `127.0.0.1:8080`. No TLS (it's localhost). Optional bearer token for defense-in-depth, but not strictly necessary on localhost.

### 8.2 Tools

Same four tools as cloud:

**`search_documents(query, folder_filter?, date_filter?, top_k?, debug?)`**
Full multi-stage hybrid retrieval. Returns ranked evidence with citations.

**`get_document_context(doc_id?, chunk_id?, window?)`**
Drill-down: doc_id returns full summary + all section summaries. chunk_id returns chunk ± window adjacent chunks.

**`list_recent_documents(folder_filter?, limit?)`**
Recently modified/indexed documents with metadata.

**`get_sync_status()`**
Total files tracked, indexed count, pending count, error count, last sync time.

### 8.3 Citation Format

Same as cloud:

```json
{
  "text": "Merged chunk text with context...",
  "citation": {
    "title": "Q3 Operational Review",
    "path": "/Users/you/Dropbox/Work/Reports/Q3-review.pdf",
    "section": "Revenue Analysis",
    "pages": "12-14",
    "modified": "2025-03-01",
    "label": "Q3-review.pdf § Revenue Analysis, pp. 12-14"
  },
  "score": 0.847,
  "record_type": "chunk"
}
```

---

## 9. Project Structure

```
dropbox-rag-local/
├── pyproject.toml                  # Package definition, dependencies, CLI entry points
├── config.example.toml             # Example configuration
├── Makefile                        # Common commands (setup, index, serve, test)
│
├── src/
│   └── rag/
│       ├── __init__.py
│       ├── cli.py                  # CLI entry points: rag index, rag serve, rag status, rag reindex
│       ├── config.py               # TOML config loader + validation
│       │
│       ├── sync/
│       │   ├── __init__.py
│       │   ├── watcher.py          # watchdog filesystem watcher + debounce
│       │   ├── scanner.py          # Initial full scan + startup re-scan
│       │   └── db.py               # sync_state CRUD (SQLite)
│       │
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── runner.py           # Pipeline orchestrator: classify → parse → ... → index
│       │   ├── classifier.py       # File type routing, OCR detection, folder context
│       │   ├── parser/
│       │   │   ├── __init__.py
│       │   │   ├── base.py         # ParsedDocument, ParsedSection dataclasses
│       │   │   ├── docling_parser.py   # Docling wrapper for PDF + DOCX
│       │   │   └── text_parser.py      # TXT/MD fallback
│       │   ├── normalizer.py       # Whitespace, headers, reflow
│       │   ├── dedup.py            # SHA-256 exact + normalized hash
│       │   ├── summarizer.py       # LLM CLI wrapper for structured summaries
│       │   ├── chunker.py          # Structure-aware chunking (512 tokens, 64 overlap)
│       │   ├── embedder.py         # sentence-transformers bge-base-en-v1.5
│       │   └── indexer.py          # Qdrant upsert with atomic swap
│       │
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── engine.py           # Multi-stage: dense + keyword + RRF + rerank
│       │   ├── query_analyzer.py   # Broad vs specific, filter extraction
│       │   ├── reranker.py         # ONNX bge-reranker-v2-m3
│       │   └── citations.py        # Citation assembly + formatting
│       │
│       ├── mcp/
│       │   ├── __init__.py
│       │   ├── server.py           # MCP server setup (stdio + HTTP)
│       │   └── tools.py            # Tool definitions (4 tools)
│       │
│       └── db/
│           ├── __init__.py
│           ├── connection.py       # SQLite connection management
│           ├── models.py           # Document, Section, Chunk CRUD
│           └── migrations.py       # Schema creation + migration runner
│
├── migrations/
│   └── 001_initial.sql             # Full schema (single file for local)
│
├── scripts/
│   ├── download-models.sh          # Download embedding model + reranker ONNX
│   └── setup-qdrant.sh             # docker run qdrant/qdrant with volume mount
│
└── tests/
    ├── test_watcher.py
    ├── test_parser.py
    ├── test_normalizer.py
    ├── test_chunker.py
    ├── test_dedup.py
    ├── test_retrieval.py
    ├── test_summarizer.py
    ├── test_citations.py
    └── fixtures/
        ├── sample.pdf
        ├── sample-scanned.pdf
        ├── sample.docx
        └── sample.txt
```

---

## 10. Installation & Setup

### 10.1 Prerequisites

- Python 3.11+
- Docker (for Qdrant)
- An LLM CLI tool installed (claude, kiro-cli, codex, etc.) — optional, only for summaries
- Tesseract OCR (`brew install tesseract` on macOS, `apt install tesseract-ocr` on Linux)

### 10.2 Setup Steps

```bash
# 1. Clone the repo
git clone <repo> && cd dropbox-rag-local

# 2. Install Python package
pip install -e ".[dev]"

# 3. Download models (embedding + reranker)
./scripts/download-models.sh

# 4. Start Qdrant
./scripts/setup-qdrant.sh
# Or manually: docker run -d -p 6333:6333 -v ~/.local/share/dropbox-rag/qdrant:/qdrant/storage qdrant/qdrant:v1.12

# 5. Create config
cp config.example.toml ~/.config/dropbox-rag/config.toml
# Edit to add your folder paths

# 6. Initial index
rag index          # Full scan + process all documents

# 7. Start MCP server (for Claude Desktop integration)
rag serve          # Starts stdio MCP server
```

### 10.3 Claude Desktop Integration

Add to Claude Desktop's MCP config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "dropbox-rag": {
      "command": "rag",
      "args": ["serve"],
      "env": {}
    }
  }
}
```

### 10.4 Claude Code Integration

Add to `.mcp.json` in your project or `~/.claude/mcp.json` globally:

```json
{
  "mcpServers": {
    "dropbox-rag": {
      "command": "rag",
      "args": ["serve"]
    }
  }
}
```

---

## 11. CLI Commands

```bash
rag index                     # Full scan: discover + process all files in configured folders
rag index --folder ~/Work     # Index only one folder
rag index --file report.pdf   # Index a single file

rag serve                     # Start MCP server (stdio mode, for Claude Desktop)
rag serve --http              # Start MCP server (HTTP mode, localhost:8080)

rag watch                     # Start filesystem watcher (runs in foreground, indexes changes as they happen)
rag watch --daemon            # Start watcher as background process

rag status                    # Show index stats: total docs, indexed, pending, errors
rag status --verbose          # Include per-folder breakdown

rag reindex                   # Force re-process all documents (re-parse, re-chunk, re-embed)
rag reindex --embeddings-only # Re-embed without re-parsing (for model upgrades)
rag reindex --folder ~/Work   # Re-process one folder

rag search "query text"       # Quick CLI search (useful for testing)
rag search "query" --debug    # Search with full debug output
```

---

## 12. Build Phases

### Phase 1 — Foundation + Usable Search (Weeks 1-2)

1.1 Project scaffolding: pyproject.toml, src layout, config loader
1.2 SQLite schema + migration runner
1.3 Qdrant setup script + collection creation (768-dim)
1.4 Filesystem scanner: walk folders, hash files, populate sync_state
1.5 Docling parser: PDF + DOCX → ParsedDocument
1.6 Normalizer: whitespace, headers, reflow
1.7 Chunker: 512-token, sentence-boundary, heading-boundary
1.8 Embedder: bge-base-en-v1.5 via sentence-transformers
1.9 Qdrant indexer: chunk vectors with payload, atomic per-doc swap
1.10 `rag index` CLI command: full pipeline end-to-end
1.11 Basic retrieval: dense search via Qdrant
1.12 Keyword search: Qdrant text index
1.13 RRF fusion
1.14 ONNX reranker
1.15 MCP server with `search_documents` tool (stdio)
1.16 Claude Desktop integration tested

**Phase 1 acceptance:** Claude Desktop can search local documents via MCP. Dense + keyword hybrid search with RRF + reranking returns cited chunk results.

### Phase 2 — Summaries + Watch Mode (Week 3)

2.1 LLM CLI summarizer: document pyramid summaries (l1/l2/l3)
2.2 Section summaries via LLM CLI
2.3 Summary vectors indexed in Qdrant
2.4 Multi-stage retrieval (doc summaries → sections → chunks via Prefetch)
2.5 Filesystem watcher (`rag watch`): auto-index on file changes
2.6 File deletion handling (tombstone + vector removal)
2.7 `get_document_context` MCP tool
2.8 `list_recent_documents` MCP tool
2.9 Summary caching by content hash

**Phase 2 acceptance:** Full pyramid retrieval with summaries. File changes auto-detected and indexed. Drill-down and listing tools working.

### Phase 3 — Polish + Robustness (Week 4)

3.1 Deduplication (exact + normalized hash)
3.2 Folder filtering in retrieval
3.3 Recency boost
3.4 Context expansion (adjacent chunks)
3.5 Citation formatting polish
3.6 Query analysis (broad vs specific layer weighting)
3.7 Debug mode
3.8 `get_sync_status` MCP tool
3.9 Error handling: retry logic, poison document quarantine
3.10 `rag status` CLI command
3.11 OCR routing + confidence metadata
3.12 Startup re-scan (catch changes while stopped)

**Phase 3 acceptance:** Production-quality local RAG. Robust error handling, dedup, folder filters, recency, debug mode, full CLI tooling.

### Phase 4 — Future (Unscheduled)

4.1 SimHash near-duplicate detection
4.2 Embedding model migration tooling (`rag reindex --embeddings-only`)
4.3 HTTP transport mode for non-stdio MCP clients
4.4 LLM-based reranker option (use CLI tool instead of ONNX)
4.5 PPTX/XLSX/HTML support via Docling
4.6 Daemon mode with system service integration (launchd on macOS)
4.7 Web UI for status monitoring and manual search

---

## 13. Differences from Cloud Spec

| Aspect | Cloud | Local |
|---|---|---|
| Infrastructure | AWS CDK, EC2, ALB, VPC | Laptop, Docker (Qdrant only) |
| Metadata DB | PostgreSQL 16 | SQLite |
| Embedding model | Bedrock Titan Embed v2 (1024-dim) | bge-base-en-v1.5 (768-dim) |
| Summarization | Bedrock Claude 3.5 Haiku | LLM CLI tool (claude, kiro, etc.) |
| File sync | Dropbox webhooks + cursor API | watchdog filesystem watcher |
| IPC | PostgreSQL LISTEN/NOTIFY | Single process (no IPC needed) |
| Deployment | Docker Compose (6 services) | pip install + docker run qdrant |
| MCP transport | Remote SSE over HTTPS + bearer token | stdio (local) or localhost HTTP |
| TLS | ALB + ACM certificate | None needed |
| Auth | Bearer token required | None needed (local only) |
| Reverse proxy | ALB with path routing | None needed |
| Cost | ~$275-480/month | $0 (runs on existing hardware) |
| Processing speed | 1-3 sec/doc | 5-15 sec/doc |
| Remote access | Yes (any computer via HTTPS) | No (local machine only) |

---

## 14. Shared Code Strategy

The pipeline logic (classify, parse, normalize, dedup, chunk) and retrieval engine (query analysis, RRF, reranker, citations) are the same between cloud and local. The differences are in:

- **Embedding backend:** Bedrock API call vs local sentence-transformers
- **Summarization backend:** Bedrock API call vs CLI subprocess
- **Database backend:** asyncpg (Postgres) vs sqlite3
- **Sync backend:** Dropbox API vs filesystem watcher
- **MCP transport:** Remote SSE vs stdio

If both variants are maintained, these backends should be abstracted behind interfaces so the core pipeline and retrieval code is shared. The `embedder.py`, `summarizer.py`, `db.py`, and `sync/` modules would have pluggable backends; everything else is identical.

---

## 15. Acceptance Criteria

1. New/changed PDF/DOCX in watched folders is indexed automatically when watcher is running
2. Manual `rag index` processes all unindexed files
3. Deleted files are removed from search results
4. All parsing occurs locally via Docling + Tesseract
5. Claude Desktop can query via MCP stdio and receive cited results
6. Citations include file path, section heading, page numbers
7. Hybrid search (dense + keyword) active from v1
8. Cross-encoder reranking active from v1
9. Pyramid summaries generated via LLM CLI (when enabled)
10. System gracefully degrades if summarization is disabled or CLI unavailable
11. Duplicate documents detected and suppressed
12. Folder filters work
13. No cloud services required at runtime (embedding + reranking are fully local)
14. Qdrant data persists across restarts (Docker volume)
15. SQLite database persists across restarts
16. System recovers cleanly from restart (startup re-scan catches missed changes)
