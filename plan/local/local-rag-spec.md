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
| Embedding model | **BGE-M3** via sentence-transformers (1024-dim) | Runs locally on CPU, no API calls, 8192-token context (vs 512 for bge-base), native sparse retrieval, 100+ languages, higher retrieval accuracy. ~1-1.5GB RAM. |
| Summarization LLM | **Local CLI tool** (claude, kiro-cli, codex, or similar) | User provides their preferred CLI; avoids running a large local model in RAM; prompt piped to stdin, JSON response from stdout |
| OCR engine | **Docling auto-detect** (`OcrAutoOptions`) | Docling auto-selects best available engine: `OcrMacOptions` (macOS native, no install needed), `EasyOCR`, or `TesseractOcrOptions` as fallback |
| Qdrant topology | **Single collection** with `record_type` payload field | Same as cloud |
| Hybrid search | **Qdrant built-in text index** on chunk text | Same as cloud — BM25-equivalent keyword matching, zero additional infra |
| Score fusion | **Reciprocal Rank Fusion (RRF)** with k=60 | Same as cloud |
| Reranker | **bge-reranker-v2-m3** via ONNX Runtime on CPU | Same as cloud — ~200-350ms for 30 candidates on CPU |
| MCP transport | **stdio** (primary) for Claude Desktop; **Streamable HTTP** as alternative | stdio is the standard local MCP transport; no TLS/auth needed. HTTP uses Streamable HTTP transport (`mcp>=1.25,<2`). |
| File sync | **watchdog** filesystem watcher + content hash dedup | Replaces Dropbox webhook/cursor; watches configured folder list for changes |
| Deployment | **Single Python process** + Qdrant Docker container | No Docker Compose orchestra needed for the Python services; simple `pip install` + `docker run` |
| Answer synthesis | **None — RAG returns evidence only** | The calling LLM (Claude Desktop, etc.) synthesizes answers from returned citations |
| Language | **Python 3.11+ with strict typing** | ML ecosystem (Docling, sentence-transformers, onnxruntime) is Python-only; strict typing via Pydantic v2 + mypy strict mode + dataclasses enables reliable AI-assisted development |

---

## 2.1 Typing & Code Quality Mandate

All Python code in this project must follow strict typing conventions to enable reliable AI-assisted code generation and left-shift testing. These are not guidelines — they are enforced by CI.

### Rules

1. **mypy strict mode** — `mypy.ini` or `pyproject.toml` sets `strict = true`. Every function has full type annotations for all parameters and return values. No `Any` types except where wrapping untyped third-party libraries, and those must be isolated behind typed wrapper functions.

2. **Pydantic v2 models at every boundary** — all data flowing between pipeline stages, all config, all MCP tool inputs/outputs, all database row mappings, and all Qdrant payload construction are Pydantic `BaseModel` subclasses with explicit field types and validators. No raw dicts crossing module boundaries, except within typed wrapper functions around external clients (Qdrant, JSON parsing) that convert between raw dicts and typed models at the boundary.

3. **dataclasses for internal value objects** — lightweight internal structures that stay within a single pipeline stage (e.g. `ChunkWindow`, `RRFCandidate`) use `@dataclass(frozen=True, slots=True)` with full type annotations. Data that crosses stage boundaries (e.g. `ParsedSection`) should be a Pydantic model instead. Prefer composition over inheritance for slotted frozen dataclasses (they cannot be subclassed without also using slots).

4. **Enum and Literal for constrained values** — No bare strings for values that have a known set of options. Use `Literal` for Pydantic field annotations and type narrowing when the set is small and used in 1-2 places (e.g., `ProcessStatus`, `SummaryLevel`). Use `StrEnum` when you need iteration, runtime logic, or the set is referenced in 3+ places (e.g., `RecordType`, `FileType`).

5. **TypedDict for read-back typing only** — use `TypedDict` to type plain dicts returned by external libraries (e.g., Qdrant payload read-back, stdlib JSON). For constructing outbound payloads, use a Pydantic model and call `.model_dump()`. For parsing untrusted input (LLM CLI JSON responses), always use Pydantic `model_validate()` for runtime validation.

6. **Protocol classes for pluggable backends** — the embedder, summarizer, database, vector store, reranker, and parser interfaces are defined as `typing.Protocol` classes. This allows swapping backends (e.g. local bge vs Bedrock Titan) without inheritance, and mypy verifies structural compatibility. All Protocols must be defined before implementation begins.

7. **No untyped containers** — `list`, `dict`, `set`, `tuple` always have type parameters. `list[str]` not `list`. `dict[str, float]` not `dict`.

8. **Result types for fallible operations** — operations that can fail (LLM CLI calls, file parsing, Qdrant queries) return explicit result types rather than raising exceptions for expected failures. Use Pydantic v2 discriminated unions (tagged unions) so mypy can narrow types correctly. Never use a `success: bool` flag with optional fields — this allows inconsistent states.

9. **ruff for linting and formatting** — `ruff check` and `ruff format` enforced. Ruff's type-checking rules enabled (ANN, TC rule sets). Note: the `TCH` prefix was renamed to `TC` in ruff v0.8.0.

10. **Async/sync strategy** — The indexing pipeline is synchronous. MCP server handlers are `async def` (required by the MCP SDK). CPU-bound operations (embedding, reranking) are dispatched via `asyncio.to_thread()` from async handlers. Qdrant queries from MCP handlers use `AsyncQdrantClient`. Protocols are defined as sync (for the indexing pipeline); the retrieval engine wraps them in async for MCP use.

11. **`from __future__ import annotations`** — All modules use this import to enable lazy annotation evaluation and eliminate string-quoted forward references. For Pydantic models with cross-module forward references, call `model_rebuild()` after all referenced types are available (typically in the module's `__init__.py`).

### Example Type Patterns

```python
# --- Constrained value types ---

# Literal for small sets used in Pydantic field annotations
ProcessStatus = Literal["pending", "processing", "done", "error", "poison"]
SummaryLevel = Literal["l1", "l2", "l3"]

# StrEnum for sets used in iteration, runtime logic, or 3+ places
class RecordType(StrEnum):
    CHUNK = "chunk"
    SECTION_SUMMARY = "section_summary"
    DOCUMENT_SUMMARY = "document_summary"

class FileType(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"

# --- Pydantic models at pipeline boundaries ---

class ParsedSection(BaseModel):
    heading: str | None
    order: int
    text: str
    page_start: int | None
    page_end: int | None

class ParsedDocument(BaseModel):
    doc_id: str
    title: str | None
    file_type: FileType
    sections: list[ParsedSection]
    ocr_required: bool
    ocr_confidence: float | None
    raw_content_hash: str

class ClassificationResult(BaseModel):
    file_type: FileType
    likely_scanned: bool
    ocr_enabled: bool
    folder_context: str
    complexity_estimate: Literal["low", "medium", "high"]

class NormalizedDocument(BaseModel):
    doc_id: str
    title: str | None
    file_type: FileType
    sections: list[ParsedSection]
    normalized_content_hash: str
    raw_content_hash: str

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    section_id: str | None
    chunk_order: int
    text: str
    text_normalized: str
    page_start: int | None
    page_end: int | None
    section_heading: str | None
    citation_label: str | None
    token_count: int

class EmbeddedChunk(BaseModel):
    chunk: Chunk
    vector: list[float]

class VectorPoint(BaseModel):
    point_id: str
    vector: list[float]
    payload: "QdrantPayloadModel"

class FileEvent(BaseModel):
    file_path: str
    content_hash: str
    file_type: FileType
    event_type: Literal["created", "modified", "deleted"]
    modified_at: str

class SearchFilters(BaseModel):
    folder_filter: str | None = None
    date_filter: str | None = None
    file_type: FileType | None = None

class SearchHit(BaseModel):
    point_id: str
    score: float
    record_type: RecordType
    doc_id: str
    text: str
    payload: dict[str, Any]  # raw from Qdrant read-back

class RetrievalResult(BaseModel):
    hits: list["CitedEvidence"]
    query_classification: str | None = None
    debug_info: dict[str, Any] | None = None

# --- Frozen dataclass for stage-internal value objects ---

@dataclass(frozen=True, slots=True)
class ChunkWindow:
    center: Chunk
    before: list[Chunk]
    after: list[Chunk]

@dataclass(frozen=True, slots=True)
class RRFCandidate:
    point_id: str
    rrf_score: float
    source_ranks: dict[str, int]

# --- Protocols for pluggable backends ---

class Embedder(Protocol):
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, query: str) -> list[float]: ...
    @property
    def dimensions(self) -> int: ...
    @property
    def model_version(self) -> str: ...

class Summarizer(Protocol):
    def summarize_document(
        self, text: str, title: str | None, file_type: str
    ) -> "SummaryResult": ...
    def summarize_section(
        self, text: str, heading: str | None, doc_context: str
    ) -> "SectionSummaryResult": ...
    @property
    def available(self) -> bool: ...  # for graceful degradation

class MetadataDB(Protocol):
    def upsert_sync_state(self, state: "SyncStateRow") -> None: ...
    def get_pending_files(self, limit: int) -> list["SyncStateRow"]: ...
    def upsert_document(self, doc: "DocumentRow") -> None: ...
    def get_document_by_hash(self, content_hash: str) -> "DocumentRow | None": ...
    def insert_chunks(self, chunks: list["ChunkRow"]) -> None: ...
    def log_processing(self, entry: "ProcessingLogEntry") -> None: ...

class VectorStore(Protocol):
    def upsert_points(self, doc_id: str, points: list[VectorPoint]) -> None: ...
    def delete_stale_points(self, doc_id: str, keep_ids: set[str]) -> None: ...
    def query_dense(
        self, vector: list[float], filters: SearchFilters, limit: int
    ) -> list[SearchHit]: ...
    def query_keyword(
        self, query: str, filters: SearchFilters, limit: int
    ) -> list[SearchHit]: ...

class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[SearchHit], top_k: int
    ) -> list[SearchHit]: ...

class Parser(Protocol):
    def parse(self, file_path: str, ocr_enabled: bool) -> "ParseResult": ...
    @property
    def supported_types(self) -> set[FileType]: ...

# --- TypedDict for Qdrant read-back typing ---

class QdrantPayloadReadBack(TypedDict):
    record_type: str
    summary_level: str | None
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

# --- Pydantic model for Qdrant payload construction ---

class QdrantPayloadModel(BaseModel):
    record_type: RecordType
    summary_level: SummaryLevel | None = None
    doc_id: str
    section_id: str | None = None
    chunk_id: str | None = None
    title: str
    file_path: str
    folder_path: str
    folder_ancestors: list[str]
    file_type: FileType
    modified_at: str
    text: str

# --- Discriminated union Result types ---

class SummarySuccess(BaseModel):
    status: Literal["success"] = "success"
    summary_l1: str
    summary_l2: str
    summary_l3: str
    key_topics: list[str]
    doc_type_guess: str | None = None

class SummaryError(BaseModel):
    status: Literal["error"] = "error"
    error: str

SummaryResult = Annotated[
    Annotated[SummarySuccess, Tag("success")]
    | Annotated[SummaryError, Tag("error")],
    Discriminator("status"),
]

class ParseSuccess(BaseModel):
    status: Literal["success"] = "success"
    document: ParsedDocument

class ParseError(BaseModel):
    status: Literal["error"] = "error"
    error: str
    file_path: str

ParseResult = Annotated[
    Annotated[ParseSuccess, Tag("success")]
    | Annotated[ParseError, Tag("error")],
    Discriminator("status"),
]
```

### pyproject.toml Type Config

```toml
[tool.mypy]
strict = true
python_version = "3.11"
plugins = ["pydantic.mypy"]
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_generics = true
check_untyped_defs = true
no_implicit_reexport = true

[tool.pydantic-mypy]
init_forbid_extra = true
init_typed = true
warn_required_dynamic_aliases = true

[[tool.mypy.overrides]]
module = ["docling.*", "onnxruntime.*"]
ignore_missing_imports = true

[tool.ruff]
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "ANN", "B", "A", "TC", "UP", "RUF"]
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
  MCP Server (stdio / Streamable HTTP)
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
| Qdrant | Docker container (`qdrant/qdrant:v1.17`) | ~200MB base + index size |
| Embedding model (BGE-M3) | Loaded in Python process via sentence-transformers | ~1-1.5GB |
| ONNX reranker (bge-reranker-v2-m3) | Loaded in Python process via onnxruntime | ~500MB |
| Docling parser | Run in **subprocess** via `multiprocessing` (on-demand during indexing) | 4-6GB in child process, fully reclaimed by OS on exit |
| SQLite | In-process (stdlib), WAL mode | Negligible |
| LLM CLI (claude, kiro, etc.) | Spawned as subprocess during indexing only | External process, not resident |

**Memory isolation:** Docling is run in a child subprocess so the OS fully reclaims its 4-6GB when parsing completes (Python GC cannot reliably release PyTorch model memory within the same process). A memory semaphore prevents query-time model loading (embedder + reranker) from overlapping with active Docling parsing.

**Minimum system requirements:** 16GB RAM for sequential operation (parse, then query). 24GB RAM recommended for concurrent indexing + querying. Modern multi-core CPU (Apple Silicon or recent Intel/AMD). At query time, only the embedding model, reranker, and Qdrant are active (~2GB total).

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
model = "BAAI/bge-m3"
dimensions = 1024
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
# Alternative: transport = "streamable-http", host = "127.0.0.1", port = 8080

[watcher]
# Filesystem watcher settings
poll_interval_seconds = 5
# Debounce rapid file changes (e.g. during saves)
debounce_seconds = 2
# Use polling mode for network-mounted folders (FSEvents/inotify unreliable)
use_polling = false
# Batch coalescing window for bulk operations (e.g. copying many files)
batch_window_seconds = 10
```

### 4.2 Environment Variables

Minimal — most config lives in the TOML file. Environment variables only for secrets or overrides:

```bash
# Only needed if your LLM CLI requires auth
ANTHROPIC_API_KEY=sk-...  # If using claude CLI
# Override config file location
RAG_CONFIG_PATH=/path/to/config.toml
```

### 4.3 Config Precedence

1. `RAG_CONFIG_PATH` environment variable (highest priority)
2. Project-local `./config.toml`
3. User-global `~/.config/dropbox-rag/config.toml`

Deep merge: more-specific config overrides per key. `folders.paths` is the only required field; all other sections have sensible defaults. If no config file is found, print a clear error with setup instructions.

### 4.4 Config Pydantic Model

All config is validated via a Pydantic model at load time:

```python
class FoldersConfig(BaseModel):
    paths: list[Path]  # required, must have at least one entry
    extensions: list[FileType] = [FileType.PDF, FileType.DOCX, FileType.TXT, FileType.MD]
    ignore: list[str] = ["**/node_modules", "**/.git", "**/venv", "**/__pycache__"]

class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-m3"
    dimensions: int = 1024  # must match model
    batch_size: int = 32
    cache_dir: Path = Path("~/.cache/dropbox-rag/models")

class RerankerConfig(BaseModel):
    model_path: Path = Path("~/.cache/dropbox-rag/models/bge-reranker-v2-m3")
    top_k_candidates: int = 30
    top_k_final: int = 10  # validated: must be <= top_k_candidates

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
    folders: FoldersConfig  # required
    database: DatabaseConfig = DatabaseConfig()
    qdrant: QdrantConfig = QdrantConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    reranker: RerankerConfig = RerankerConfig()
    summarization: SummarizationConfig = SummarizationConfig()
    mcp: MCPConfig = MCPConfig()
    watcher: WatcherConfig = WatcherConfig()
```

---

## 5. Data Model

### 5.1 SQLite Schema

The schema mirrors the cloud PostgreSQL schema with simplifications (no LISTEN/NOTIFY, no advisory locks, UUIDs stored as TEXT). SQLite is configured with WAL mode and busy timeout at connection time:

```sql
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=30000;
```

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
    embedding_model_version TEXT,
    chunker_version         TEXT
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

Single collection `documents` with cosine vectors, **1024-dim** (BGE-M3).

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

**Ongoing watch:** `watchdog` library monitors configured folders for file create/modify/delete events. Uses FSEvents on macOS, inotify on Linux. For network-mounted folders, use polling mode (`use_polling = true` in config). Debounce rapid changes (2-second window per file) and batch-coalesce bulk operations (configurable `batch_window_seconds`, default 10s) — collect all events in the window, deduplicate by file path, then process the batch. On change detection: compute SHA-256 of file, compare to sync_state.content_hash. If unchanged, skip. If changed or new, queue for processing. If deleted, mark is_deleted=1 and remove vectors from Qdrant.

**Re-scan on startup:** Walk all folders on process start to catch changes that happened while the system was stopped. Use file mtime as a fast pre-filter: only hash files whose mtime differs from stored `modified_at` (~10x faster than hashing every file). Re-scan runs in a background thread so the MCP server can start serving queries immediately from the existing index.

### 6.2 Classification

Same as cloud: route by file type (PDF/DOCX/TXT/MD), detect scanned PDFs (sample 3 pages, if avg chars < 50 → scanned), infer folder context from path, estimate complexity.

### 6.3 Parsing (Docling)

Same as cloud. Docling handles PDF + DOCX. Use `OcrAutoOptions` to auto-detect and apply OCR when text extraction yields little content — on macOS this uses native Apple OCR (`OcrMacOptions`, no install needed), falling back to EasyOCR or Tesseract elsewhere. Produces structured output with heading hierarchy, page boundaries, tables, reading order.

**Local consideration:** Docling loads ML models on first parse (~4-6GB). Run Docling in a **child subprocess** via `multiprocessing` so the OS fully reclaims memory when parsing completes. PyTorch models do not reliably release memory via Python garbage collection. A memory semaphore prevents query-time model loading from overlapping with active Docling parsing.

### 6.4 Normalization

Same as cloud: whitespace cleanup, header/footer suppression, paragraph reflow, heading hierarchy preservation, page mapping preservation, table boundary markers.

### 6.5 Deduplication

Same as cloud Phase 1: exact content hash match (SHA-256 on raw bytes), normalized content hash match (SHA-256 on normalized text). If duplicate, link to canonical document, skip indexing.

SimHash near-dedup deferred (same as cloud Phase 4).

### 6.6 Summarization (LLM CLI)

Same structured output as cloud, different backend. Instead of Bedrock Haiku API call, shell out to configured LLM CLI tool.

**Per-document summary call:**
1. Construct prompt requesting structured JSON: summary_l1 (phrase), summary_l2 (1-2 sentences), summary_l3 (paragraph), key_topics (list), doc_type_guess
2. Truncate document excerpt to <5000 characters to avoid known CLI stdin size bugs (e.g., anthropics/claude-code#7263). For larger documents, write prompt to a temporary file and pass as argument.
3. Pipe prompt to CLI stdin, capture both stdout and stderr
4. Parse JSON from stdout with robust extraction: try direct `json.loads()`, fall back to regex extraction of JSON from markdown-fenced output, then skip summarization for this doc
5. Check exit code. Timeout after configured seconds (default 60), retry once with exponential backoff, then skip summaries for this doc
6. Validate parsed JSON via `SummarySuccess.model_validate()` (Pydantic runtime validation on untrusted CLI output)

**Section summaries:** One CLI call per major section (H1/H2 boundary).

**Caching:** All summaries cached by normalized_content_hash in SQLite. If document content hasn't changed, skip summarization entirely.

**Graceful degradation:** If summarization is disabled in config or the CLI tool isn't available, the system still works — you just get chunk-level and section-heading-level retrieval without the pyramid summaries. This is a valid operating mode.

### 6.7 Chunking

Same as cloud: 512-token target (tiktoken cl100k_base), 64-token overlap, sentence-boundary aware, heading-boundary respecting, table-preserving.

**Deterministic IDs:** `UUID5(NAMESPACE_RAG, f"{doc_id}:{section_order}:{chunk_order}")` where `NAMESPACE_RAG` is a project-specific constant UUID. The separator-delimited format prevents ambiguous concatenation. If the chunking algorithm changes (e.g., different sentence boundaries from a library update), all chunk IDs change — store `chunker_version` in the documents table so this can be detected and a targeted re-chunk triggered.

### 6.8 Embedding (Local BGE-M3)

Load `BGE-M3` via sentence-transformers. Batch size 32 texts per encode call. Produces 1024-dim vectors. 8192-token context window (vs 512 for bge-base-en-v1.5). Record model version in metadata.

**Performance on laptop:** Slightly slower than bge-base due to larger model (~568M vs ~109M params). A 500-document corpus with ~10,000 chunks takes roughly 5-10 minutes to embed from scratch on CPU.

### 6.9 Qdrant Indexing

**Per-document upsert with deterministic IDs.** Since chunk/summary point IDs are deterministic (UUID5), upserting with the same ID automatically overwrites existing points. After upserting all new points for a document, delete only stale point IDs that no longer exist (i.e., when a document now has fewer chunks than before). This avoids the non-atomic delete+upsert pattern, which risks data loss if the process crashes between delete and upsert (Qdrant has no multi-operation transactions). Use `wait=true` on delete calls to ensure completion before proceeding.

---

## 7. Retrieval Engine

Identical to cloud spec. All stages run behind a single `search_documents` MCP tool call.

### 7.1 Pipeline

1. **Query analysis** — broad vs specific classification, extract folder/date filter intent, extract keywords
2. **Embed query** — BGE-M3 (same model as indexing)
3. **Dense search** — Qdrant cosine similarity via `query_points()` with `prefetch` parameter: doc summaries (top 20), section summaries (top 20), chunks (top 30), with metadata filters. Note: `search()` was removed in qdrant-client v1.17; all search operations use the unified `query_points()` API.
4. **Keyword search** — Qdrant built-in text index on "text" field, chunks top 30, also via `query_points()`
5. **RRF fusion** — score = Σ 1/(60 + rank_i), merge dense + keyword, apply layer weighting
6. **Cross-encoder rerank** — bge-reranker-v2-m3 ONNX, top 30 → top 10 (~200-350ms on CPU)
7. **Post-processing** — recency boost (90-day half-life, max 30% influence), context expansion (±1 adjacent chunks), dedup overlapping chunks, assemble citations

**Multi-prefetch caution:** A reported bug (qdrant/qdrant-client#1072) affects multiple prefetches with different filter conditions. Integration tests must verify all three record types appear in results. As a fallback, issue three separate `query_points()` calls and perform RRF fusion in Python.

### 7.2 Layer Weighting

Same as cloud: broad queries boost summaries, specific queries boost chunks.

### 7.3 Debug Mode

Same as cloud: returns query classification, layer weights, scores at each stage, timing breakdown.

---

## 8. MCP Server

### 8.1 Transport

**Primary: stdio** — Claude Desktop and Claude Code launch the MCP server as a subprocess. This is the standard local MCP pattern. No network configuration, no auth needed. **Critical: stdio servers must never write to stdout** — this corrupts JSON-RPC framing. Use stderr for logging.

**Alternative: Streamable HTTP** — For tools that need a network MCP endpoint, run on `127.0.0.1:8080/mcp` using the `streamable-http` transport (SSE transport is deprecated). No TLS (it's localhost). Optional bearer token for defense-in-depth, but not strictly necessary on localhost.

**MCP SDK version:** Pin to `mcp>=1.25,<2`. A v2 release is planned with significant transport layer changes.

**Async handlers:** All MCP tool handlers are `async def` (required by the SDK). CPU-bound operations (embedding, reranking) are dispatched via `asyncio.to_thread()`. Qdrant queries use `AsyncQdrantClient`.

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
│       ├── cli.py                  # CLI entry points: rag init, rag index, rag serve, rag status, etc.
│       ├── config.py               # TOML config loader + validation (AppConfig Pydantic model)
│       ├── init.py                 # Interactive setup wizard (rag init)
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
│       │   ├── embedder.py         # sentence-transformers BGE-M3
│       │   └── indexer.py          # Qdrant upsert with deterministic IDs + stale point cleanup
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
- OCR: no install needed on macOS (Docling uses native Apple OCR). On Linux, install Tesseract (`apt install tesseract-ocr`) or EasyOCR (pip, included in dev dependencies).

### 10.2 Quick Start (3 commands)

```bash
# 1. Install
pipx install dropbox-rag          # or: pip install dropbox-rag

# 2. Interactive setup — configures folders, LLM CLI, Qdrant, MCP integration
rag init

# 3. Index and go
rag index
```

That's it. `rag init` handles everything: config file creation, Qdrant Docker container, model downloads, and MCP integration.

### 10.3 `rag init` — Interactive Setup Wizard

`rag init` walks the user through first-time setup with sensible defaults:

```
$ rag init

Welcome to Dropbox RAG! Let's get you set up.

Folders to index
  Add a folder path (tab completion, ~ expansion):
  > ~/Dropbox/Work
  > ~/Documents/Reports
  > (empty to finish)

File types to include [pdf, docx, txt, md]:
  > (enter to accept defaults)

LLM CLI for summaries (optional, for document summaries):
  Detected: claude ✓, kiro ✗, codex ✗
  Use claude? [Y/n]: y

Qdrant
  Docker detected ✓
  Starting qdrant/qdrant:v1.17 container... done
  Listening on localhost:6333 ✓

Downloading models
  BGE-M3 embedding model (1024-dim)... done
  bge-reranker-v2-m3 ONNX... done

MCP Integration
  Claude Desktop config found. Add dropbox-rag MCP server? [Y/n]: y
  → Updated ~/Library/Application Support/Claude/claude_desktop_config.json ✓
  Claude Code config found. Add dropbox-rag MCP server? [Y/n]: y
  → Updated ~/.claude/mcp.json ✓

Config written to ~/.config/dropbox-rag/config.toml

Ready! Run `rag index` to index your documents.
```

**What `rag init` does:**

1. **Folder selection** — prompts for folder paths with `~` expansion and validation (checks path exists). Supports multiple folders. Writes to `folders.paths` in config.
2. **File type defaults** — defaults to `[pdf, docx, txt, md]` with option to customize. Writes to `folders.extensions`.
3. **LLM CLI auto-detection** — runs `which` for known CLI tools (`claude`, `kiro`, `codex`). If found, offers to configure. If none found, disables summarization (graceful degradation). Writes to `summarization.*`.
4. **Qdrant management** — checks if Docker is available and if a Qdrant container is already running. If not, pulls `qdrant/qdrant:v1.17` and starts it with a persistent volume at `~/.local/share/dropbox-rag/qdrant`. Creates the collection with correct schema (1024-dim cosine, all payload indices, text index).
5. **Model download** — downloads BGE-M3 embedding model and bge-reranker-v2-m3 ONNX weights to `~/.cache/dropbox-rag/models`. Shows progress.
6. **MCP auto-config** — detects Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS) and Claude Code config (`~/.claude/mcp.json`). Offers to add the `dropbox-rag` MCP server entry. Non-destructive: reads existing config, merges, writes back. User can skip with `n`.
7. **Config file** — writes `~/.config/dropbox-rag/config.toml` with all settings. If config already exists, offers to update or keep existing.

**Re-running `rag init`** is safe — it detects existing config and offers to update individual sections. Useful for adding folders or changing the LLM CLI tool.

### 10.4 Manual Setup

For users who prefer manual configuration:

```bash
# Install
pip install dropbox-rag

# Start Qdrant manually
docker run -d --name qdrant -p 6333:6333 \
  -v ~/.local/share/dropbox-rag/qdrant:/qdrant/storage \
  qdrant/qdrant:v1.17

# Create config manually
mkdir -p ~/.config/dropbox-rag
cp config.example.toml ~/.config/dropbox-rag/config.toml
# Edit config.toml to add your folder paths

# Index
rag index

# Start MCP server
rag serve
```

### 10.5 MCP Integration

**Auto-configured by `rag init`.** To configure manually or inspect:

```bash
# Print the MCP config JSON snippet (for manual setup)
rag mcp-config --print

# Print for a specific target
rag mcp-config --print --target claude-desktop
rag mcp-config --print --target claude-code

# Install into Claude Desktop config (same as rag init does)
rag mcp-config --install claude-desktop

# Install into Claude Code config
rag mcp-config --install claude-code
```

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
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

**Claude Code** (`~/.claude/mcp.json` globally, or `.mcp.json` per project):
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

### Setup & Config

```bash
rag init                      # Interactive setup wizard (folders, LLM CLI, Qdrant, MCP)
rag init --add-folder ~/Work  # Add a folder to existing config (non-interactive)
rag init --set-llm kiro       # Change LLM CLI tool (non-interactive)
```

### Indexing

```bash
rag index                     # Full scan: discover + process all files in configured folders
rag index --folder ~/Work     # Index only one folder
rag index --file report.pdf   # Index a single file

rag reindex                   # Force re-process all documents (re-parse, re-chunk, re-embed)
rag reindex --embeddings-only # Re-embed without re-parsing (for model upgrades)
rag reindex --folder ~/Work   # Re-process one folder
```

### Serving & Watching

```bash
rag serve                     # Start MCP server (stdio mode, for Claude Desktop)
rag serve --http              # Start MCP server (Streamable HTTP mode, localhost:8080)

rag watch                     # Start filesystem watcher (runs in foreground, indexes changes as they happen)
rag watch --daemon            # Start watcher as background process
```

### Status & Diagnostics

```bash
rag status                    # Dashboard: docs found/pending/indexed/errors, chunk count, Qdrant stats, watcher state
rag status --json             # Machine-readable JSON output
rag doctor                    # Health check: Qdrant reachable, OCR available, models cached, folders exist
```

### Search & MCP Config

```bash
rag search "query text"       # Quick CLI search (useful for testing)
rag search "query" --debug    # Search with full debug output

rag mcp-config --print        # Print MCP config JSON snippet
rag mcp-config --install claude-desktop   # Install MCP entry into Claude Desktop config
rag mcp-config --install claude-code      # Install MCP entry into Claude Code config
```

### `rag status` Output

```
Dropbox RAG Status
──────────────────────────────
Documents:  487 found · 472 indexed · 12 pending · 3 errors
Chunks:     9,841 indexed (1,024-dim BGE-M3)
Qdrant:     connected (localhost:6333, 47 MB)
Watcher:    running (3 folders)
LLM CLI:    claude (available)

Folders:
  ~/Dropbox/Work          312 docs  ✓
  ~/Documents/Reports      98 docs  ✓
  ~/projects/notes          77 docs  ✓

Recent errors:
  ~/Dropbox/Work/corrupt.pdf    parse failed (retry 2/3)
  ~/Documents/Reports/old.docx  parse failed (retry 1/3)
  ~/projects/notes/broken.md    normalize failed (retry 3/3 → poisoned)
```

---

## 12. Build Phases

### Phase 1a — Indexing Pipeline (Weeks 1-2)

1.1 Project scaffolding: pyproject.toml (with `[project.scripts] rag = "rag.cli:main"`), src layout, config Pydantic model (AppConfig), pytest + CI setup (mypy + ruff + pytest), test fixtures (1 PDF, 1 DOCX, 1 TXT, 1 scanned PDF), config.example.toml
1.2 Define all Protocol interfaces (Embedder, Summarizer, MetadataDB, VectorStore, Reranker, Parser) and inter-stage types (FileEvent, ClassificationResult, ParsedDocument, ParsedSection, NormalizedDocument, Chunk, EmbeddedChunk, VectorPoint, discriminated union Result types)
1.3 SQLite schema + migration runner (WAL mode, busy_timeout=30000)
1.4 Qdrant setup script + collection creation (1024-dim cosine, all payload keyword indices, full-text index on `text` field), model download script (embedding + reranker ONNX)
1.5 Docling parser: PDF + DOCX → ParsedDocument (run in subprocess via multiprocessing, OCR via OcrAutoOptions)
1.6 Text/Markdown parser: TXT + MD → ParsedDocument (fallback for non-Docling file types)
1.7 Normalizer: whitespace, headers, reflow
1.8 Exact-hash deduplication: SHA-256 on raw bytes, check document_hashes table, link to canonical if match, skip indexing
1.9 Chunker: 512-token, sentence-boundary, heading-boundary, deterministic UUID5 IDs with NAMESPACE_RAG
1.10 Embedder: BGE-M3 via sentence-transformers (1024-dim)
1.11 Qdrant indexer: upsert with deterministic IDs, delete only stale points (no delete+upsert pattern)
1.12 `rag index` CLI command: full pipeline end-to-end with per-document error handling (try/except, log error, set process_status='error', continue to next doc, print summary)
1.13 `rag init` setup wizard: interactive folder selection, LLM CLI auto-detection, Qdrant Docker management, model download, MCP auto-config (Claude Desktop + Claude Code), config file creation
1.14 `rag status` dashboard: docs found/pending/indexed/errors, chunk count, Qdrant stats, per-folder breakdown, recent errors
1.15 `rag doctor` health check: Qdrant reachable, OCR available, models cached, SQLite writable, configured folders exist
1.16 `rag mcp-config` command: print MCP JSON snippet, install into Claude Desktop / Claude Code config files

**Phase 1a acceptance:** `rag init` → `rag index` is a complete first-run experience. Scans configured folders, parses all supported file types (PDF, DOCX, TXT, MD) including scanned PDFs with OCR, chunks, embeds, and indexes into Qdrant. `rag status` shows progress. Errors on individual documents don't crash the run. Unit tests pass for each component.

### Phase 1b — Retrieval + MCP (Week 3)

1.17 Basic retrieval: dense search via Qdrant `query_points()` API
1.18 Keyword search: Qdrant text index via `query_points()`
1.19 RRF fusion
1.20 ONNX reranker: bge-reranker-v2-m3

*Note: 1.17→1.18→1.19→1.20 is a sequential dependency chain (RRF needs dense + keyword outputs, reranker needs RRF output). Estimate as a single 2-3 day unit, not four independent items.*

1.21 Folder filtering in retrieval (Qdrant filter condition on folder_path)
1.22 `rag search` CLI command: wrap retrieval engine, print results to stdout (essential for debugging without Claude Desktop)
1.23 MCP server with `search_documents` tool (stdio, async handlers, asyncio.to_thread for CPU-bound ops, AsyncQdrantClient)
1.24 Claude Desktop integration tested end-to-end (rag init → rag index → Claude Desktop searches via MCP)
1.25 Integration test: index fixture documents, search, verify results from all file types

**Phase 1b acceptance:** Claude Desktop can search local documents via MCP. Dense + keyword hybrid search with RRF + reranking + folder filtering returns cited chunk results. `rag search` works from CLI.

### Phase 2 — Summaries + Watch Mode (Weeks 4-5)

*Three independent workstreams — can be completed in any order.*

**2A: Watch mode**
2.1 Filesystem watcher (`rag watch`): auto-index on file changes, batch-coalescing event queue
2.2 File deletion handling (tombstone + vector removal)
2.3 Startup re-scan in background thread (mtime pre-filter, non-blocking MCP startup)

**2B: Summarization**
2.4 LLM CLI summarizer: document pyramid summaries (l1/l2/l3), robust JSON extraction, stdin size limit, stderr capture
2.5 Section summaries via LLM CLI
2.6 Summary vectors indexed in Qdrant
2.7 Multi-stage retrieval (doc summaries → sections → chunks via `query_points()` prefetch)
2.8 Query analysis (broad vs specific layer weighting) — prerequisite for effective pyramid retrieval
2.9 Summary caching by content hash

**2C: Additional MCP tools**
2.10 `get_document_context` MCP tool
2.11 `list_recent_documents` MCP tool
2.12 `get_sync_status` MCP tool

**Phase 2 acceptance:** Full pyramid retrieval with summaries. File changes auto-detected and indexed. Drill-down, listing, and status tools working.

### Phase 3 — Polish + Robustness (Week 6)

3.1 Normalized content hash dedup (SHA-256 on normalized text, in addition to raw-byte dedup from Phase 1a)
3.2 Recency boost (90-day half-life, max 30% influence)
3.3 Context expansion (adjacent chunks)
3.4 Citation formatting polish
3.5 Debug mode (query classification, layer weights, scores, timing)
3.6 Error handling: retry logic with exponential backoff, poison document quarantine (3 retries → poison)
3.7 `rag status` CLI command (including per-folder breakdown, error listing)
3.8 `rag doctor` health check command: verify Qdrant reachable, OCR available, models cached, SQLite writable, folders exist

**Phase 3 acceptance:** Production-quality local RAG. Robust error handling, full dedup, recency, debug mode, health checks, full CLI tooling.

### Phase 4 — Future (Unscheduled)

4.1 SimHash near-duplicate detection
4.2 Embedding model migration tooling (`rag reindex --embeddings-only`)
4.3 Streamable HTTP transport mode for non-stdio MCP clients
4.4 LLM-based reranker option (use CLI tool instead of ONNX)
4.5 PPTX/XLSX/HTML support via Docling (already supported by Docling)
4.6 Daemon mode with system service integration (launchd on macOS)
4.7 Web UI for status monitoring and manual search

---

## 13. Differences from Cloud Spec

| Aspect | Cloud | Local |
|---|---|---|
| Infrastructure | AWS CDK, EC2, ALB, VPC | Laptop, Docker (Qdrant only) |
| Metadata DB | PostgreSQL 16 | SQLite |
| Embedding model | Bedrock Titan Embed v2 (1024-dim) | BGE-M3 (1024-dim) |
| Summarization | Bedrock Claude 3.5 Haiku | LLM CLI tool (claude, kiro, etc.) |
| File sync | Dropbox webhooks + cursor API | watchdog filesystem watcher |
| IPC | PostgreSQL LISTEN/NOTIFY | Single process (no IPC needed) |
| Deployment | Docker Compose (6 services) | pip install + docker run qdrant |
| MCP transport | Remote SSE over HTTPS + bearer token | stdio (local) or Streamable HTTP |
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
4. All parsing occurs locally via Docling + auto-detected OCR (Apple native on macOS, Tesseract/EasyOCR on Linux)
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
