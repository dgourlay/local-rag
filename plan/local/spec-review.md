# Spec Review: Local RAG System (`local-rag-spec.md`)

Review date: 2026-03-10
Reviewers: Dependency Auditor, Architecture Reviewer, Typing & Interface Reviewer, Build Order Reviewer

---

## Severity Legend

- **Blocker** — Must fix before implementation begins; will cause build failures, data loss, or fundamental design rework
- **Should-fix** — Fix before or during Phase 1; will cause bugs, poor UX, or avoidable rework if deferred
- **Nice-to-have** — Can address later; improves quality but doesn't block progress

---

## 1. Dependency Auditor

### 1.1 qdrant-client API removal — `search()` is gone [BLOCKER]

**Spec ref:** SS7.1
**Detail:** `qdrant-client` v1.17 (current) has removed `search()`, `search_batch()`, `recommend()`, `recommend_batch()`, `discovery()`, and `discovery_batch()`. All search operations must use the unified `query_points()` API with `prefetch` parameter.
**Severity:** Blocker
**Fix:** Update the spec and all future retrieval code to reference `query_points()`. The Prefetch-based multi-stage query design (SS7.1) is still valid, but must be expressed through `query_points()`.

### 1.2 Qdrant Docker tag is outdated

**Spec ref:** SS3.2, SS10.2
**Detail:** Spec references `qdrant/qdrant:v1.12`. Current release is v1.17.0. v1.12 is 5 minor versions behind and misses native BM25, MMR reranking, weighted RRF, and other features.
**Severity:** Should-fix
**Fix:** Update Docker tag to `qdrant/qdrant:v1.17` (or `latest` for development).

### 1.3 ruff `TCH` rule prefix renamed to `TC`

**Spec ref:** SS2.1 (pyproject.toml config)
**Detail:** Since ruff v0.8.0, the `TCH` rule prefix has been renamed to `TC`. The spec's config line `select = ["E", "F", "W", "I", "N", "ANN", "B", "A", "TCH", "UP", "RUF"]` will produce a warning or error.
**Severity:** Should-fix
**Fix:** Change `"TCH"` to `"TC"` in the ruff lint select list.

### 1.4 mypy config missing Pydantic plugin

**Spec ref:** SS2.1 (pyproject.toml config)
**Detail:** Without `plugins = ["pydantic.mypy"]` and `[tool.pydantic-mypy]` config, mypy strict mode produces false positives on Pydantic model `__init__` calls.
**Severity:** Should-fix
**Fix:** Add to pyproject.toml:
```toml
[tool.mypy]
plugins = ["pydantic.mypy"]
# ... existing options ...

[tool.pydantic-mypy]
init_forbid_extra = true
init_typed = true
warn_required_dynamic_aliases = true
```

### 1.5 MCP HTTP transport — SSE deprecated, use Streamable HTTP

**Spec ref:** SS8.1
**Detail:** The MCP SDK now uses "Streamable HTTP" transport (not plain HTTP or SSE). Endpoint pattern is `http://localhost:8000/mcp`. A v2 SDK release is planned for Q1 2026 with major transport layer changes.
**Severity:** Should-fix
**Fix:** Update SS8.1 to reference `streamable-http` transport. Pin dependency to `mcp>=1.25,<2` until v2 stabilizes. Add note that stdio servers must never write to stdout (corrupts JSON-RPC framing).

### 1.6 Tesseract is no longer Docling's default OCR

**Spec ref:** SS2, SS6.3, SS10.1
**Detail:** Docling now defaults to EasyOCR and supports multiple backends: `EasyOcrOptions`, `TesseractOcrOptions`, `RapidOcrOptions`, `OcrMacOptions` (macOS native), and `OcrAutoOptions` (auto-selects best available).
**Severity:** Should-fix
**Fix:** Remove the hard Tesseract prerequisite. Use `OcrMacOptions` on macOS (no external install needed) or `OcrAutoOptions` for cross-platform. Eliminates a setup step for macOS users.

### 1.7 Embedding model: bge-base-en-v1.5 vs BGE-M3

**Spec ref:** SS2, SS4.1, SS5.2, SS6.8
**Detail:** bge-base-en-v1.5 still works but is no longer top-tier. Key limitation: 512-token context window matches chunk size exactly, leaving zero room for query prefix and `[CLS]` token. BGE-M3 offers 8,192-token context, native sparse retrieval, 100+ languages, higher retrieval accuracy. Tradeoff: ~1-1.5GB RAM vs ~500MB, 1024-dim vs 768-dim.
**Severity:** Should-fix
**Fix:** Evaluate BGE-M3 as the default model. If adopted, update dimensions from 768 to 1024 throughout the spec. If staying with bge-base, document the 512-token context limitation explicitly.

### 1.8 Reranker CPU latency estimate is optimistic

**Spec ref:** SS2
**Detail:** Spec claims "~100ms for 30 candidates." Benchmarks show ~200-350ms for bge-reranker-v2-m3 on CPU. Still acceptable for interactive search.
**Severity:** Nice-to-have
**Fix:** Adjust estimate to "~200-350ms." Consider GTE-ModernBERT reranker (149M params, 77.67% Hit@1) as a smaller/faster alternative.

### 1.9 Qdrant native BM25

**Spec ref:** SS7.1
**Detail:** Qdrant v1.17 adds native `Bm25Config` for keyword search, which could replace or supplement the text index approach.
**Severity:** Nice-to-have
**Fix:** Evaluate native BM25 during implementation. No spec change required yet.

---

## 2. Architecture Reviewer

### 2.1 Non-atomic per-document swap in Qdrant [BLOCKER]

**Spec ref:** SS6.9
**Detail:** "Delete all existing points for doc_id, then upsert new points" is NOT atomic. Qdrant has no multi-operation transactions. If the process crashes between delete and upsert, all vectors for that document are lost. Additionally, a confirmed bug (qdrant/qdrant#6556) shows delete-by-filter can remove concurrently upserted points. Filter-based deletes can also cause extreme server load (qdrant/qdrant#6401).
**Severity:** Blocker
**Fix:** Use deterministic point IDs (UUID5, per SS6.7) for overwrite-based upserts. Upserting with the same ID automatically overwrites existing points. Only explicitly delete point IDs that no longer exist (when a document has fewer chunks after re-processing). This also makes the UUID5 deterministic ID design actually useful.

### 2.2 Memory pressure — models sharing single process

**Spec ref:** SS3.2, SS6.3
**Detail:** Docling (4-6GB) + embedding model (~500MB) + reranker (~500MB) can coexist poorly. PyTorch models don't reliably release memory after `del`. If a search triggers during a parse batch, the process could hit 5-7GB on a 16GB machine. sentence-transformers has a documented memory leak during the first ~10,000 predictions.
**Severity:** Should-fix
**Fix:** (a) Run Docling parsing in a subprocess via `multiprocessing` so OS reclaims memory when it exits. (b) Add a memory semaphore to prevent query-time model loading during active parsing. (c) Consider raising minimum RAM to 24GB, or document that 16GB requires sequential-only operation.

### 2.3 SQLite missing WAL mode

**Spec ref:** SS5.1
**Detail:** Without WAL mode, writers block readers and readers block writers. `rag watch` (writing) + `rag search` (reading) = readers blocked. Two writers (e.g., `rag index` + `rag watch`) = `database is locked` errors. No `busy_timeout` specified either.
**Severity:** Should-fix
**Fix:** Enable WAL mode at database creation: `PRAGMA journal_mode=WAL;`. Set `PRAGMA busy_timeout=30000;`. Document that `rag index` and `rag watch` should not run simultaneously (or add a lockfile).

### 2.4 Qdrant multi-prefetch filter bug

**Spec ref:** SS7.1
**Detail:** Reported bug (qdrant/qdrant-client#1072) where multiple prefetches with different filter conditions may not surface all matching points. Tested against Qdrant v1.15.1.
**Severity:** Should-fix
**Fix:** (a) Test multi-prefetch with filters against the target Qdrant version. (b) Add integration tests verifying all three record types appear in results. (c) As fallback, issue three separate `query_points()` calls and perform RRF fusion in Python.

### 2.5 Watchdog reliability gaps

**Spec ref:** SS6.1
**Detail:** (a) FSEvents/inotify don't reliably detect changes on network-mounted volumes. (b) Bulk copy of 1000 files overwhelms the 2-second per-event debounce. (c) macOS kqueue fallback requires one FD per watched file. (d) Watchdog's FSEvents interface hasn't had a full thread safety audit.
**Severity:** Should-fix
**Fix:** (a) Add `watcher.use_polling = true` config option for network mounts. (b) Implement batch-aware event coalescing (collect all events in a 10-second window, dedup by path, process batch). (c) Document startup re-scan as the reliability backstop. (d) Consider periodic re-scans (e.g., hourly) for high-reliability use.

### 2.6 LLM CLI subprocess model issues

**Spec ref:** SS6.6, SS4.1
**Detail:** (a) `claude --print` outputs natural language, not guaranteed JSON. (b) Documented bug (anthropics/claude-code#7263): empty output when stdin exceeds ~7000 chars. (c) No stderr/exit-code handling specified. (d) Different CLIs (claude, kiro, codex) have different argument and output formats.
**Severity:** Should-fix
**Fix:** (a) Robust JSON extraction: try parse, fallback to regex for fenced JSON blocks, then skip. (b) Truncate piped excerpts to <5000 chars or write to temp file. (c) Capture and log stderr, check exit codes. (d) Define a `SummarizerAdapter` protocol with concrete implementations per CLI tool.

### 2.7 Missing error paths

**Spec ref:** Multiple
**Detail:** Not specified: Qdrant unreachable at startup/query time, Tesseract not installed, corrupted PDF/DOCX, embedding model download failure, Qdrant collection doesn't exist, disk full. The `poison` status exists in the schema but transition rules (retry count threshold) are undefined.
**Severity:** Should-fix
**Fix:** (a) Add a `rag doctor` startup health check: Qdrant reachable, OCR available, models cached, SQLite writable, folders exist. (b) Define retry-to-poison threshold (e.g., 3 retries). (c) Return structured error responses from MCP tools, never crash.

### 2.8 Config file precedence undefined

**Spec ref:** SS4.1
**Detail:** Spec says "`~/.config/dropbox-rag/config.toml` (or project-local `config.toml`)" but doesn't specify precedence, merge semantics, behavior when no config exists, or required vs optional fields.
**Severity:** Nice-to-have
**Fix:** Define: `RAG_CONFIG_PATH` env > `./config.toml` > `~/.config/dropbox-rag/config.toml`. Deep merge with more-specific winning. `folders.paths` is the only required field. If no config found, print setup instructions.

### 2.9 UUID5 deterministic IDs under-specified

**Spec ref:** SS6.7
**Detail:** (a) No UUID5 namespace specified. (b) Concatenation format ambiguous (`doc_id + section_order + chunk_order` — what separator?). (c) If tiktoken or sentence-boundary logic changes, all chunk IDs change, triggering full re-index.
**Severity:** Should-fix
**Fix:** (a) Define a project-specific namespace UUID constant. (b) Use `f"{doc_id}:{section_order}:{chunk_order}"`. (c) Store `chunker_version` in documents table alongside `parser_version` and `embedding_model_version`.

### 2.10 Startup re-scan blocks MCP server

**Spec ref:** SS6.1
**Detail:** Walking 10,000+ files with SHA-256 hashing takes 30-90 seconds. If the MCP server is in the same process, it can't serve queries during re-scan.
**Severity:** Should-fix
**Fix:** (a) Start MCP server immediately, serve from existing index. (b) Run re-scan in background thread. (c) Use mtime as fast pre-filter: only hash files whose mtime changed (10x faster).

---

## 3. Typing & Interface Reviewer

### 3.1 Missing Protocol definitions [BLOCKER]

**Spec ref:** SS2.1 Rule 6, SS14
**Detail:** The spec mandates Protocol classes for "embedder, summarizer, and database interfaces" but only defines `Embedder`. Missing: `Summarizer`, `MetadataDB`, `VectorStore`, `Reranker`, `Parser`. The `Parser` Protocol is especially important since SS9 shows two parser implementations that must share an interface.
**Severity:** Blocker
**Fix:** Define upfront before implementation:
```python
class Summarizer(Protocol):
    def summarize_document(self, text: str, title: str | None, file_type: str) -> SummaryResult: ...
    def summarize_section(self, text: str, heading: str | None, doc_context: str) -> SectionSummaryResult: ...
    @property
    def available(self) -> bool: ...

class MetadataDB(Protocol):
    def upsert_sync_state(self, state: SyncStateRow) -> None: ...
    def get_pending_files(self, limit: int) -> list[SyncStateRow]: ...
    def upsert_document(self, doc: DocumentRow) -> None: ...
    def get_document_by_hash(self, content_hash: str) -> DocumentRow | None: ...
    # ... CRUD methods ...

class VectorStore(Protocol):
    def upsert_points(self, doc_id: str, points: list[VectorPoint]) -> None: ...
    def delete_by_doc_id(self, doc_id: str) -> None: ...
    def search_dense(self, vector: list[float], filters: SearchFilters, limit: int) -> list[SearchHit]: ...
    def search_keyword(self, query: str, filters: SearchFilters, limit: int) -> list[SearchHit]: ...

class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[RerankCandidate]) -> list[RerankCandidate]: ...

class Parser(Protocol):
    def parse(self, file_path: str, ocr_enabled: bool) -> ParseResult: ...
    @property
    def supported_types(self) -> set[str]: ...
```

### 3.2 Result type pattern is unsound [BLOCKER]

**Spec ref:** SS2.1 Rule 8
**Detail:** The example `SummaryResult` allows inconsistent states: `SummaryResult(success=True)` with no summaries, or `SummaryResult(success=False, summary_l1="hello")`. Under mypy strict, every consumer must check `is not None` regardless of `success`, defeating the purpose.
**Severity:** Blocker
**Fix:** Use Pydantic v2 discriminated unions:
```python
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
    Annotated[SummarySuccess, Tag("success")] | Annotated[SummaryError, Tag("error")],
    Discriminator("status"),
]
```
mypy narrows correctly with `match`/`isinstance`, and Pydantic validates at runtime.

### 3.3 Missing inter-stage type definitions [BLOCKER]

**Spec ref:** SS6, SS9
**Detail:** Many types flowing between pipeline stages are undefined:

| Stage transition | Missing type |
|---|---|
| Scanner/Watcher -> Runner | `FileEvent` (path, hash, event_type, modified_at) |
| Classifier -> Parser | `ClassificationResult` (file_type, ocr_enabled, folder_context) |
| Parser -> Normalizer | `ParseResult` (success/error wrapper around `ParsedDocument`) |
| Normalizer -> Dedup | `NormalizedDocument` (normalized hash, cleaned text) |
| Dedup -> Chunker | `DedupResult` (is_duplicate, canonical_doc_id, document) |
| Chunker output | `Chunk` (id, text, section_id, order, pages, token_count) |
| Embedder output | `EmbeddedChunk` (chunk + vector) |
| Indexer input | `VectorPoint` (point_id, vector, payload) |
| Retrieval | `SearchFilters`, `SearchHit`, `RetrievalResult` |
| MCP tools | Input/output models for all 4 tools |

**Severity:** Blocker
**Fix:** Define these types before implementation. They are the contracts between modules and determine the shape of every function signature.

### 3.4 Async/sync strategy unaddressed [BLOCKER]

**Spec ref:** SS8, SS3.1
**Detail:** The MCP SDK requires async handlers (`async def`). But sentence-transformers and the reranker are sync/CPU-bound. Qdrant has `AsyncQdrantClient`. The spec never mentions async.
**Severity:** Blocker
**Fix:** Add to spec: "Indexing pipeline is synchronous. MCP server handlers are async. CPU-bound operations (embedding, reranking) dispatched via `asyncio.to_thread()`. Qdrant queries use `AsyncQdrantClient`." Protocols should be sync (for indexing); the retrieval engine wraps them in async.

### 3.5 TypedDict vs Pydantic boundary unclear

**Spec ref:** SS2.1 Rules 2, 5
**Detail:** Rule 2 says "Pydantic at every boundary." Rule 5 says "TypedDict for JSON payloads." These partially contradict. TypedDict has no runtime validation — dangerous for untrusted LLM CLI output.
**Severity:** Should-fix
**Fix:** Clarify the rule:

| Boundary | Use | Reason |
|---|---|---|
| Between pipeline stages | Pydantic BaseModel | Runtime validation |
| LLM CLI JSON response | Pydantic BaseModel | Untrusted input, must validate |
| Qdrant payload construction | Pydantic model, `.model_dump()` for upsert | Validate before sending |
| Qdrant payload read-back | TypedDict | Qdrant returns plain dicts, TypedDict documents shape |
| SQLite row mappings | Pydantic BaseModel | Validate on read |

### 3.6 Literal vs StrEnum guidance missing

**Spec ref:** SS2.1 Rule 4
**Detail:** All examples use Literal; no guidance on when StrEnum is better.
**Severity:** Should-fix
**Fix:** Use Literal for Pydantic field annotations and type narrowing. Use StrEnum when you need iteration (e.g., Qdrant index setup loops), runtime logic (CLI arg choices), or the same set is referenced in 3+ places. Concrete: `ProcessStatus` and `SummaryLevel` stay Literal. `RecordType` and `FileType` become StrEnum.

### 3.7 Config Pydantic model not defined

**Spec ref:** SS4.1, SS2.1 Rule 2
**Detail:** Rule 2 mandates Pydantic for "all config" but SS4.1 only shows the TOML structure, not the validation model. Implicit validation rules (paths must be absolute, `top_k_final <= top_k_candidates`, `dimensions` must match model) are scattered.
**Severity:** Should-fix
**Fix:** Define `AppConfig` Pydantic model with nested config sections and field validators. `folders.paths` is the only required field; everything else has defaults.

### 3.8 Frozen dataclass + Pydantic interop

**Spec ref:** SS2.1 Rule 3
**Detail:** `@dataclass(frozen=True, slots=True)` cannot be inherited (slots constraint). Pydantic serialization of arbitrary dataclass fields requires `arbitrary_types_allowed=True`, which disables validation for that field.
**Severity:** Should-fix
**Fix:** Clarify: "Frozen dataclasses for data within a single pipeline stage. At stage boundaries, use Pydantic models." Example: `ChunkWindow` (internal to chunker) = dataclass. `ParsedSection` (crosses parser -> normalizer -> chunker) = Pydantic model.

### 3.9 Raw dict prohibition vs external library reality

**Spec ref:** SS2.1 Rules 1, 2, 5
**Detail:** Qdrant client returns `payload` as `dict[str, Any]`. `json.loads()` returns `dict[str, Any]`. Raw dicts are unavoidable at external boundaries.
**Severity:** Should-fix
**Fix:** Amend Rule 2: "No raw dicts crossing module boundaries, except within typed wrapper functions around external clients. These wrappers convert between raw dicts and typed models at the boundary."

### 3.10 Forward references and `from __future__ import annotations`

**Spec ref:** SS2.1 examples
**Detail:** Python 3.11+ with `from __future__ import annotations` eliminates string quotes for forward refs. Works with Pydantic v2 but cross-module forward refs require `model_rebuild()`.
**Severity:** Nice-to-have
**Fix:** Add convention: "All modules use `from __future__ import annotations`. Call `model_rebuild()` for Pydantic models with cross-module forward references."

---

## 4. Build Order Reviewer

### 4.1 No per-document error handling in Phase 1 [BLOCKER]

**Spec ref:** SS12 Phase 1 (1.10), SS12 Phase 3 (3.9)
**Detail:** Error handling is deferred to Phase 3 (3.9). Without basic try/except per document in Phase 1, a single malformed PDF crashes the entire `rag index` run. For a first-run with 500 documents, this is unacceptable.
**Severity:** Blocker
**Fix:** Add to 1.10 (pipeline runner): wrap each document in try/except, log error, set `process_status='error'`, continue. Print summary: "Indexed 487/500 documents. 13 errors." Retry logic and poison quarantine remain in Phase 3.

### 4.2 Phase 1 is oversized (16 items in 2 weeks)

**Spec ref:** SS12 Phase 1
**Detail:** Realistic estimates for a single developer:

| Items | Estimated Effort |
|---|---|
| 1.1-1.3 Scaffolding, SQLite, Qdrant | 1.5 days |
| 1.4 Filesystem scanner | 0.5 days |
| 1.5 Docling parser | 2-3 days (highest risk) |
| 1.6 Normalizer | 1 day |
| 1.7 Chunker | 1-2 days |
| 1.8 Embedder | 0.5 days |
| 1.9 Indexer | 1 day |
| 1.10 CLI | 0.5 days |
| 1.11-1.14 Retrieval pipeline | 2-3 days |
| 1.15-1.16 MCP + Claude Desktop | 1-2 days |

Total: 10-14 working days (2-3 weeks).

**Severity:** Should-fix
**Fix:** Split into Phase 1a (indexing pipeline, weeks 1-2) and Phase 1b (retrieval + MCP, week 3). Adjust Phase 2 to week 4, Phase 3 to week 5.

### 4.3 Dedup should be in Phase 1

**Spec ref:** SS12 Phase 3 (3.1), SS6.5
**Detail:** Without dedup, the same content at two file paths (e.g., `~/Downloads/report.pdf` and `~/Dropbox/Work/report.pdf`) produces duplicate vectors and duplicate search results from day one. The atomic per-doc swap only handles re-indexing the same path.
**Severity:** Should-fix
**Fix:** Move exact-hash dedup (SHA-256 on raw bytes) to Phase 1 between normalizer (1.6) and chunker (1.7). It's simple: check `document_hashes` for `raw_hash`, if match, link to canonical, skip. Normalized-hash dedup stays in Phase 3.

### 4.4 Scanned PDFs produce garbage without OCR in Phase 1

**Spec ref:** SS12 Phase 1 (1.5), Phase 3 (3.11), SS6.2, SS6.3
**Detail:** Docling parsing (1.5) is Phase 1 but OCR routing (3.11) is Phase 3. Without OCR, scanned PDFs yield near-zero text, producing empty/garbage chunks that pollute the index.
**Severity:** Should-fix
**Fix:** In Phase 1's Docling parser (1.5), enable OCR as a fallback. Use Docling's `OcrAutoOptions` (see Dependency Audit 1.6) to auto-detect and apply OCR when text extraction yields little content.

### 4.5 Missing Phase 1 items

**Spec ref:** SS9, SS10.2, SS11, SS5.2

| Missing item | Why it's needed in Phase 1 |
|---|---|
| Text parser for TXT/MD | SS9 shows `text_parser.py` but no phase builds it. TXT/MD files crash or are silently skipped. |
| Model download script | SS10.2 step 3 references it. Embedder (1.8) and reranker (1.14) need model weights. |
| `rag search` CLI command | SS11 shows it. Essential for debugging retrieval without Claude Desktop. |
| Qdrant payload indices + text index | SS5.2 specifies them. Required for keyword search (1.12) and filtered retrieval. 1.3 only says "collection creation." |

**Severity:** Should-fix
**Fix:** Add to Phase 1: (a) 1.5b text parser, (b) model download in 1.1/1.3, (c) `rag search` after 1.14, (d) clarify 1.3 includes all Qdrant indices.

### 4.6 No testing in any phase

**Spec ref:** SS9, SS12
**Detail:** The spec lists test files but no phase mentions building them. No test infrastructure (fixtures, test Qdrant, CI config) is planned.
**Severity:** Should-fix
**Fix:** Include testing as continuous work: (a) pytest + CI setup in 1.1, (b) unit tests alongside each component, (c) integration test after 1.14 (index a fixture document, search for it), (d) test fixtures (1 PDF, 1 DOCX, 1 TXT, 1 scanned PDF) in 1.1.

### 4.7 Items that should move earlier

| Item | Current phase | Move to | Rationale |
|---|---|---|---|
| Folder filtering (3.2) | Phase 3 | Phase 1 | Core usability — trivial Qdrant filter condition |
| Startup re-scan (3.12) | Phase 3 | Phase 2 | Needed alongside watch mode (2.5) |
| Query analysis (3.6) | Phase 3 | Phase 2 | Prerequisite for effective pyramid retrieval (2.4) |

**Severity:** Should-fix

### 4.8 Phase 2 bundles three independent workstreams

**Spec ref:** SS12 Phase 2
**Detail:** Summarization (2.1-2.3, 2.9), watch mode (2.5-2.6), and MCP tools (2.7-2.8) have no dependencies on each other but are bundled in one phase.
**Severity:** Nice-to-have
**Fix:** Split into parallel workstreams: 2A (watch mode), 2B (summarization + pyramid retrieval), 2C (additional MCP tools). Each can ship independently.

### 4.9 Retrieval stages 1.11-1.14 are a sequential chain

**Spec ref:** SS12 Phase 1
**Detail:** Items appear independent but have linear dependencies: RRF (1.13) needs outputs from dense (1.11) and keyword (1.12). Reranker (1.14) takes RRF output.
**Severity:** Nice-to-have
**Fix:** Acknowledge as a sequential sub-chain. Estimate as a 2-3 day unit, not four independent items. Add integration test after 1.14.

### 4.10 Highest integration risk items

| Item | Risk Level | Reason | Mitigation |
|---|---|---|---|
| 1.5 Docling parser | Highest | API learning curve, memory management, OCR config | Spike first (day 1-2): standalone script parsing a sample PDF |
| 1.15 MCP server | High | SDK is new/evolving, stdio requires strict stdout discipline | Build minimal "hello" MCP server early, test with Claude Desktop |
| 1.11-1.12 Qdrant Prefetch | Medium | Multi-prefetch filter bug (see Architecture 2.4) | Test text search independently with manual points |
| 1.14 ONNX reranker | Medium | Model download/conversion, Apple Silicon builds | Verify loading and inference on target hardware early |

---

## Consolidated Summary

### Blockers (6 issues — must resolve before implementation)

| # | Source | Issue |
|---|---|---|
| 1 | Dependency | qdrant-client `search()` removed; must use `query_points()` |
| 2 | Architecture | Delete+upsert is not atomic in Qdrant; use deterministic ID overwrites |
| 3 | Typing | Missing Protocol definitions (Summarizer, MetadataDB, VectorStore, Reranker, Parser) |
| 4 | Typing | Result type pattern allows inconsistent states; use discriminated unions |
| 5 | Typing | Missing inter-stage type definitions (FileEvent, ClassificationResult, Chunk, SearchHit, etc.) |
| 6 | Typing | Async/sync strategy unaddressed; MCP SDK requires async |

### Should-Fix (21 issues)

| # | Source | Issue |
|---|---|---|
| 7 | Dependency | Qdrant Docker tag outdated (v1.12 -> v1.17) |
| 8 | Dependency | ruff `TCH` renamed to `TC` |
| 9 | Dependency | mypy missing Pydantic plugin |
| 10 | Dependency | MCP HTTP transport: use `streamable-http` |
| 11 | Dependency | Tesseract -> Docling OcrAutoOptions/OcrMac |
| 12 | Dependency | bge-base-en-v1.5 512-token limit; evaluate BGE-M3 |
| 13 | Architecture | Memory pressure: run Docling in subprocess |
| 14 | Architecture | SQLite missing WAL mode + busy_timeout |
| 15 | Architecture | Qdrant multi-prefetch filter bug |
| 16 | Architecture | Watchdog reliability on network mounts + bulk ops |
| 17 | Architecture | LLM CLI subprocess: no JSON guarantee, stdin size bug |
| 18 | Architecture | Missing error paths (Qdrant down, corrupt files, etc.) |
| 19 | Architecture | UUID5 namespace and separator unspecified |
| 20 | Architecture | Startup re-scan blocks MCP server |
| 21 | Typing | TypedDict vs Pydantic boundary unclear |
| 22 | Typing | Literal vs StrEnum guidance missing |
| 23 | Typing | Config Pydantic model not defined |
| 24 | Typing | Frozen dataclass + Pydantic interop rules needed |
| 25 | Typing | Raw dict prohibition vs external library reality |
| 26 | Build Order | No per-document error handling in Phase 1 |
| 27 | Build Order | Phase 1 oversized; split into 1a/1b |
| 28 | Build Order | Dedup, folder filtering, text parser, model download, `rag search`, test infra missing from phases |

### Nice-to-Have (6 issues)

| # | Source | Issue |
|---|---|---|
| 29 | Dependency | Reranker CPU latency estimate optimistic (~100ms -> ~200-350ms) |
| 30 | Dependency | Evaluate Qdrant native BM25 |
| 31 | Architecture | Config file precedence undefined |
| 32 | Typing | `from __future__ import annotations` convention |
| 33 | Build Order | Phase 2 bundles independent workstreams |
| 34 | Build Order | Retrieval stages should be acknowledged as sequential chain |
