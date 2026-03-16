# Semantic Chunking — Build Specification

## 1. Document Purpose

This spec adds an optional **semantic chunking** strategy to the local-rag indexing pipeline. The current fixed-size chunker (512 tokens, 64-token overlap, sentence-boundary aware) becomes the default `"fixed"` strategy. A new `"semantic"` strategy uses sentence embeddings and the Max-Min similarity algorithm to place chunk boundaries where topic shifts occur, producing chunks that are more topically coherent.

Semantic chunking is **opt-in via config**. The fixed-size chunker remains the default. Both strategies share the same output type (`list[Chunk]`), the same deterministic UUID scheme, and the same downstream pipeline (embed, index, summarize). No changes to retrieval, MCP tools, or Qdrant schema.

This spec covers: algorithm, sentence segmentation, code block handling, configuration, type changes, pipeline integration, guardrails, performance, testing, and migration.

---

## 2. Design Decisions

| Decision | Resolution | Rationale |
|---|---|---|
| Algorithm | **Simplified Max-Min** (Kiss et al. 2025) | Statistically outperforms LlamaIndex SemanticSplitter (AMI 0.85-0.90). Low computational cost because embeddings are already computed for the pipeline. Simpler than percentile-based breakpoint methods. |
| Sentence segmenter | **spaCy `sentencizer`** (rule-based) | F1 92.9, zero model download, fast, already a transitive dependency via Docling. wtpsplit is more accurate (F1 96.5) but adds ~200MB ONNX model and a new dependency for marginal gain. |
| Default strategy | **`"fixed"`** (existing chunker) | Proven, deterministic, no extra compute. Semantic chunking is opt-in. Per NAACL 2025: for most documents with gradual topic transitions, fixed-size matches semantic chunking quality. |
| Sentence embeddings | **Reuse the project's BGE-M3 `Embedder`** | Already loaded for chunk embedding. Sentence embeddings are a batch `embed_batch()` call on the same model. No new model, no new memory. |
| Embedding reuse | **Do not reuse sentence embeddings for chunk embeddings** | Chunk embeddings must represent the full chunk text, not an average of sentence vectors. The quality difference matters for retrieval. Sentence embeddings are used only for boundary detection, then discarded. |
| Code block handling | **Extract-placeholder-restore** | Fenced code blocks break sentence segmentation. Extract them before segmenting, replace with numbered placeholders, restore after chunking. |
| Overlap strategy | **No index-time overlap for semantic chunks** | Semantic boundaries are meaningful topic shifts. Overlapping across a topic boundary adds noise. Query-time context expansion (citation expansion, `get_document_context` with `window`) already provides overlap when needed. |
| Min/max guardrails | **Min 64 tokens, max 768 tokens** | Min prevents degenerate single-sentence chunks. Max prevents runaway chunks in monotopically dense text. Both enforced by merge/split after boundary detection. |
| Section boundaries | **Hard stops** (same as fixed) | Each `ParsedSection` starts a new chunk boundary. Semantic chunking operates within a section. This preserves structural hierarchy from Docling parsing. |
| Chunker version | **`"semantic-v1"`** when semantic strategy active | Already tracked in `documents.chunker_version`. Changing strategy for a document triggers re-index on next scan. Fixed chunker version unchanged. |

---

## 3. Algorithm

### 3.1 Overview

The semantic chunker operates **within each section** of a `NormalizedDocument`, following the same section-first pattern as the fixed chunker. For each section:

1. Extract code blocks, replace with placeholders
2. Segment text into sentences
3. If section has fewer than 3 sentences, emit as a single chunk (no boundary detection needed)
4. Embed all sentences in a single batch call
5. Run simplified Max-Min boundary detection
6. Apply guardrails (merge small chunks, split oversized chunks)
7. Restore code blocks into their containing chunks
8. Emit `Chunk` objects with deterministic UUIDs

### 3.2 Simplified Max-Min Boundary Detection

The full Max-Min algorithm (Kiss et al. 2025) uses three parameters: `hard_thr`, `c`, and `window_size`. We simplify to **two parameters** with fixed defaults that work for 90% of cases.

**Parameters:**

| Parameter | Default | Range | Purpose |
|---|---|---|---|
| `similarity_threshold` | 0.35 | 0.0-1.0 | Base similarity threshold below which a boundary is placed. Lower = fewer boundaries = larger chunks. |
| `max_chunk_sentences` | 15 | 5-30 | Hard cap on sentences per chunk. Triggers a forced boundary via the sigmoid pressure term. |

**Algorithm (pseudocode):**

```python
def detect_boundaries(
    sentence_embeddings: list[list[float]],
    similarity_threshold: float,
    max_chunk_sentences: int,
) -> list[int]:
    """Return list of sentence indices where a new chunk should start.

    Always includes index 0 (first sentence starts first chunk).
    """
    boundaries: list[int] = [0]
    current_chunk_start = 0

    for i in range(1, len(sentence_embeddings)):
        chunk_size = i - current_chunk_start

        # Sigmoid growth pressure: increases as chunk approaches max_chunk_sentences
        # At chunk_size = max_chunk_sentences, sigmoid ~ 1.0
        # At chunk_size = max_chunk_sentences / 2, sigmoid ~ 0.12
        growth_pressure = 1.0 / (1.0 + math.exp(-(chunk_size - max_chunk_sentences) / 2))

        # Dynamic threshold rises with growth pressure
        dynamic_threshold = similarity_threshold + (1.0 - similarity_threshold) * growth_pressure

        # Max similarity: best cosine match to any sentence in current chunk
        max_sim = max(
            cosine_similarity(sentence_embeddings[i], sentence_embeddings[j])
            for j in range(current_chunk_start, i)
        )

        # Place boundary if similarity drops below dynamic threshold
        if max_sim < dynamic_threshold:
            boundaries.append(i)
            current_chunk_start = i

    return boundaries
```

**Key properties:**
- When the chunk is small (1-3 sentences), `growth_pressure` is near 0 and only `similarity_threshold` matters. This prevents premature splits.
- As the chunk grows toward `max_chunk_sentences`, the threshold rises toward 1.0, making it progressively easier to trigger a split. This prevents runaway chunks.
- The `max_similarity` metric (best match to any sentence in current chunk) is more robust than sequential similarity because a new sentence might relate to an earlier sentence in the chunk, not just the previous one.

### 3.3 Cosine Similarity

```python
def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
```

Since BGE-M3 vectors are L2-normalized at embedding time, `cosine_similarity` simplifies to a dot product. Use `numpy.dot` if numpy is available (it is, via sentence-transformers), falling back to pure Python only for tests.

---

## 4. Code Block Preservation

Fenced code blocks (triple-backtick) must be extracted before sentence segmentation and restored after chunking. Inline code (single backtick) is left in place.

### 4.1 Extract Phase

```python
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_PLACEHOLDER_FMT = "\x00CODE_BLOCK_{}\x00"

def extract_code_blocks(text: str) -> tuple[str, list[str]]:
    """Replace fenced code blocks with null-delimited placeholders.

    Returns (text_with_placeholders, list_of_extracted_blocks).
    """
    blocks: list[str] = []
    def replacer(match: re.Match[str]) -> str:
        blocks.append(match.group(0))
        return _PLACEHOLDER_FMT.format(len(blocks) - 1)
    cleaned = _CODE_BLOCK_RE.sub(replacer, text)
    return cleaned, blocks
```

### 4.2 Restore Phase

After chunking, scan each chunk's text for placeholders and replace with the original code block text.

```python
_PLACEHOLDER_RE = re.compile(r"\x00CODE_BLOCK_(\d+)\x00")

def restore_code_blocks(chunk_text: str, blocks: list[str]) -> str:
    def replacer(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        return blocks[idx] if idx < len(blocks) else match.group(0)
    return _PLACEHOLDER_RE.sub(replacer, chunk_text)
```

### 4.3 Oversized Code Blocks

If a code block placeholder is the only content in a chunk and the restored code block exceeds `max_chunk_tokens` (768), the code block becomes its own chunk without further splitting. Code should not be split mid-block.

---

## 5. Sentence Segmentation

### 5.1 Implementation

Use spaCy's rule-based `sentencizer` component. This is a pipeline component that segments on punctuation patterns without requiring a language model.

```python
import spacy

_nlp = spacy.blank("en")
_nlp.add_pipe("sentencizer")

def segment_sentences(text: str) -> list[str]:
    """Split text into sentences using spaCy sentencizer."""
    doc = _nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
```

### 5.2 Why Not the Current Regex

The existing `r"(?<=[.!?])\s+"` regex splits on any period followed by whitespace. This breaks on abbreviations ("Dr. Smith"), decimal numbers ("3.5 million"), URLs, and ellipses. spaCy's sentencizer handles these correctly with minimal overhead (~2ms per section).

The fixed chunker's existing regex splitter is left unchanged. Only the semantic chunker uses spaCy's sentencizer. This avoids an unnecessary behavior change and re-index for users who stay on the fixed strategy.

---

## 6. Configuration

### 6.1 New Config Section

Add a `[chunking]` section to `config.toml`:

```toml
[chunking]
# Chunking strategy: "fixed" (default) or "semantic"
strategy = "fixed"
# Similarity threshold for boundary detection (semantic strategy only)
# Lower = fewer boundaries = larger chunks. Range: 0.0-1.0
similarity_threshold = 0.35
# Max chunk size in tokens (semantic strategy guardrail)
max_chunk_tokens = 768
```

All other chunking parameters (`target_tokens=512`, `overlap_tokens=64`, `min_chunk_tokens=64`, `max_chunk_sentences=15`) are hardcoded defaults. The fixed chunker's existing parameters are unchanged. Only 3 knobs are exposed because most users should never need to tune them.

### 6.2 Pydantic Config Model

```python
ChunkingStrategy = Literal["fixed", "semantic"]

class ChunkingConfig(BaseModel):
    strategy: ChunkingStrategy = "fixed"
    similarity_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    max_chunk_tokens: int = Field(default=768, ge=128, le=2048)
```

Add to `AppConfig`:

```python
class AppConfig(BaseModel):
    folders: FoldersConfig
    database: DatabaseConfig = DatabaseConfig()
    qdrant: QdrantConfig = QdrantConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    chunking: ChunkingConfig = ChunkingConfig()  # NEW
    reranker: RerankerConfig = RerankerConfig()
    summarization: SummarizationConfig = SummarizationConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    mcp: MCPConfig = MCPConfig()
    watcher: WatcherConfig = WatcherConfig()
```

### 6.3 Backward Compatibility

If no `[chunking]` section exists in `config.toml`, the default `ChunkingConfig()` produces identical behavior to the current chunker: `strategy="fixed"`, `target_tokens=512`, `overlap_tokens=64`. Zero config migration required.

---

## 7. Type Changes

### 7.1 New Types in `types.py`

```python
ChunkingStrategy = Literal["fixed", "semantic"]
```

No new Pydantic models are needed. The `Chunk` model is unchanged. The chunker's output contract (`list[Chunk]`) is identical for both strategies.

### 7.2 Internal Data

No new dataclass needed. Sentences with their embeddings are represented as `list[tuple[str, list[float]]]` (text, embedding pairs) within `chunker_semantic.py`. Token counts are computed inline via tiktoken as needed. This keeps the internal representation simple and avoids a dataclass for a single-use, module-internal concern.

---

## 8. Pipeline Integration

### 8.1 Module Structure

```
src/rag/pipeline/
    chunker.py           # Existing — add dispatch at top, fixed logic stays in place
    chunker_semantic.py  # New: semantic chunking + sentence segmentation + code block handling
```

No new `chunker_fixed.py` or `sentence.py`. The fixed chunker logic stays in `chunker.py` untouched. All semantic-specific code (sentence segmentation, code block extraction, boundary detection, guardrails) lives in `chunker_semantic.py`.

### 8.2 Dispatch

`chunker.py` becomes a thin dispatcher:

```python
def chunk_document(
    doc: NormalizedDocument,
    config: ChunkingConfig,
    embedder: Embedder | None = None,
) -> list[Chunk]:
    """Chunk a document using the configured strategy.

    Args:
        doc: Normalized document with sections.
        config: Chunking configuration (strategy, parameters).
        embedder: Required for semantic strategy. Ignored for fixed.
    """
    if config.strategy == "semantic":
        if embedder is None:
            msg = "Embedder required for semantic chunking strategy"
            raise ValueError(msg)
        return chunk_document_semantic(doc, config, embedder)
    return chunk_document_fixed(doc, config)
```

### 8.3 Pipeline Call Site Change

The pipeline currently calls `chunk_document(doc)`. This changes to `chunk_document(doc, config.chunking, embedder)`. The embedder is already instantiated earlier in the pipeline for chunk embedding (step 8). It is passed into the chunker when semantic strategy is active.

### 8.4 Chunker Version

The `chunker_version` stored in the `documents` table must encode the strategy:

- Fixed: unchanged (existing version string)
- Semantic: `"semantic-v1"`

When a document's stored `chunker_version` differs from the current version, the pipeline re-chunks that document on the next index run. This is already supported by the existing version-check logic. The threshold is not encoded in the version string — if a user changes `similarity_threshold`, they should run `rag index --reindex` explicitly.

---

## 9. Guardrails

### 9.1 Minimum Chunk Size (Merge)

After boundary detection, any chunk with fewer than 64 tokens is merged with the previous chunk. If it is the first chunk, merge with the next chunk. No similarity comparison needed — the previous chunk is almost always the right merge target, and the simplicity outweighs any marginal quality gain from similarity-based selection.

### 9.2 Maximum Chunk Size (Split)

After boundary detection, any chunk exceeding `max_chunk_tokens` (configurable, default 768) tokens is split at the sentence boundary closest to the midpoint. This is a simple bisection, not recursive -- if a half still exceeds the max (only possible with very long sentences), it is emitted as-is.

### 9.3 Minimum Sentences for Semantic Mode

Sections with fewer than 3 sentences skip boundary detection and are emitted as a single chunk. The embedding cost of 1-2 sentences is not justified, and there are no meaningful boundaries to detect.

### 9.4 Single Long Sentences

A single sentence exceeding `max_chunk_tokens` is emitted as its own chunk without splitting. Splitting mid-sentence degrades both embedding and retrieval quality. This matches the existing fixed chunker behavior.

---

## 10. Performance

### 10.1 Embedding Cost

The primary added cost of semantic chunking is embedding every sentence in a section. Analysis:

| Metric | Estimate |
|---|---|
| Avg sentences per document | ~100-200 |
| BGE-M3 encode speed (CPU, batch 32) | ~15-30 sentences/sec |
| Time per document (sentence embedding) | ~4-12 seconds |
| Time for 500-doc corpus | ~30-90 minutes added |
| Memory overhead | Negligible (embeddings are 1024 floats per sentence, discarded after chunking) |

Note: CPU embedding speed varies significantly by hardware. The estimates above are conservative for typical developer laptops (M-series Mac, modern x86). The sentence embeddings are shorter texts than full chunks, so per-sentence encoding is faster than per-chunk, but there are more of them.

This is meaningful but acceptable for an opt-in feature. The sentence embedding step happens once per document at index time, not at query time.

### 10.2 Batch Strategy

All sentences within a section are embedded in a single `embed_batch()` call. Sections are processed sequentially (they are already sequential in the pipeline). Cross-section batching is not needed because sections are independent chunking units.

For very long sections (500+ sentences), batch into groups of 128 sentences to avoid memory spikes. This is the only batching consideration.

### 10.3 Cosine Similarity Optimization

The boundary detection loop computes `max_similarity` between the new sentence and all sentences in the current chunk. For a chunk of size `n`, this is `O(n)` dot products. With `max_chunk_sentences=15`, this is at most 15 dot products per sentence -- trivial compared to embedding cost.

Use numpy for the dot product: `float(np.dot(a, b))` where `a` and `b` are numpy arrays. BGE-M3 embeddings are L2-normalized, so dot product equals cosine similarity.

---

## 11. CLI Integration

### 11.1 Progress Display

During `rag index`, the semantic chunker adds a progress phase between parsing and embedding. The existing progress display (`cli.py`) should show:

```
[3/6] Chunking (semantic)... 12/45 sections  [sentence embedding]
```

When using the fixed strategy, display remains unchanged. The phase label changes from "Chunking" to "Chunking (semantic)" only when semantic strategy is active.

### 11.2 `rag status` Dashboard

The `rag status` dashboard should display the active chunking strategy:

```
Chunking:     semantic (threshold=0.35)
```

or:

```
Chunking:     fixed (512 tokens, 64 overlap)
```

This helps users confirm which strategy is active without checking `config.toml`.

### 11.3 `rag doctor` Health Check

Add a check for spaCy availability when semantic strategy is configured:

- Verify `spacy.blank("en")` loads successfully
- Verify `sentencizer` pipe can be added
- Report: `✓ spaCy sentencizer available` or `✗ spaCy not installed (required for semantic chunking)`

No check needed when strategy is `"fixed"`.

### 11.4 Interaction with Local Tools (Claude Code, kiro-cli)

Semantic chunking is invisible to MCP clients by design — Claude Code and kiro-cli call the same 5 MCP tools with the same parameters and get back the same response schema regardless of chunking strategy. The quality improvement is in the chunks themselves, not the API surface.

**What changes for the calling LLM:**
- Search results may have different chunk boundaries (more topically coherent, variable size). The `chunk_id`, `section`, `page_number`, and `score` fields are unchanged.
- Chunk text may be shorter or longer than the fixed 512-token chunks. Callers should not assume a fixed chunk size.
- `get_document_context` with `window` still works — adjacent chunks are still ordered by `chunk_order` within a section.

**What does NOT change:**
- Tool names, parameters, and response schemas — identical.
- Server instructions and MCP prompts (from `mcp-tool-guidance-spec.md`) — no chunking-strategy-specific guidance needed because the tools abstract away the chunking layer.
- `get_sync_status` and `list_recent_documents` — unaffected.

**`get_sync_status` addition:**
Include the active chunking strategy in the sync status response so the calling LLM can see it:

```json
{
  "chunking_strategy": "semantic",
  ...
}
```

This is a single new field in the `get_sync_status` payload. It helps users (and the calling LLM) confirm which strategy is active without running `rag status` in a separate terminal.

---

## 12. Testing

### 12.1 Unit Tests (`tests/test_chunker_semantic.py`)

| Test | What it verifies |
|---|---|
| `test_single_topic_no_split` | A coherent paragraph about one topic produces a single chunk |
| `test_topic_shift_detected` | Two paragraphs about different topics produce two chunks |
| `test_min_chunk_merge` | A 2-sentence mini-chunk is merged with its neighbor |
| `test_max_chunk_split` | A very long monotopic section is split at midpoint when exceeding max |
| `test_few_sentences_skip` | Section with 1-2 sentences skips semantic analysis, returns one chunk |
| `test_code_block_preserved` | Fenced code blocks survive extract/restore roundtrip intact |
| `test_code_block_not_split` | A code block is never split across chunks |
| `test_deterministic_ids` | Same input always produces same chunk UUIDs |
| `test_section_boundary_respected` | Chunks never span across sections |
| `test_empty_section` | Empty section produces no chunks |
| `test_strategy_dispatch` | `chunk_document()` dispatches to correct implementation based on config |
| `test_fixed_strategy_unchanged` | Fixed strategy produces identical output to current chunker (regression) |

### 12.2 Unit Tests (code block handling, in same file)

| Test | What it verifies |
|---|---|
| `test_code_block_extract_restore` | Roundtrip preserves code blocks exactly |
| `test_nested_code_blocks` | Nested triple-backticks handled correctly |
| `test_placeholder_in_original` | Text containing null bytes does not collide with placeholders |

### 12.3 Integration Test

Add a fixture document with two clearly distinct topics (e.g., a document about "machine learning" in the first half and "cooking recipes" in the second half). Assert that semantic chunking places a boundary between the topics while fixed chunking may not.

### 12.4 E2E Test Update

Existing e2e tests use the default `"fixed"` strategy and must continue to pass unchanged. Add one e2e test that sets `strategy = "semantic"` in a test config and verifies end-to-end indexing and retrieval works.

---

## 13. Migration

### 13.1 Existing Indexes

Switching `strategy` from `"fixed"` to `"semantic"` (or vice versa) changes the `chunker_version`. On the next `rag index` run, documents with a mismatched `chunker_version` are re-chunked and re-indexed automatically. No manual migration step is needed.

### 13.2 No Schema Changes

The SQLite schema (`chunks` table) and Qdrant collection schema are unchanged. Semantic chunks produce the same `Chunk` model, the same `VectorPoint` payload, and the same UUID5 point IDs (the UUID inputs are `doc_id`, `section_order`, `chunk_idx` -- identical structure).

### 13.3 Full Re-index

Users switching strategies should expect a full re-index of all documents. This is the same cost as `rag index --reindex`. The `rag status` dashboard already shows re-indexing progress.

### 13.4 Mixed Strategies

A single index can contain documents chunked with different strategies (some fixed, some semantic). This is a valid state during incremental re-indexing. There is no need to enforce uniformity -- retrieval does not depend on chunking strategy.

---

## 14. Implementation Checklist

Ordered by dependency:

1. **`src/rag/types.py`** — Add `ChunkingStrategy` literal
2. **`src/rag/config.py`** — Add `ChunkingConfig` model (3 fields), add `chunking` field to `AppConfig`
3. **`src/rag/pipeline/chunker_semantic.py`** — All semantic code: `segment_sentences()`, `extract_code_blocks()`, `restore_code_blocks()`, `detect_boundaries()`, `chunk_document_semantic()`, guardrail merge/split
4. **`src/rag/pipeline/chunker.py`** — Add dispatch at top of existing `chunk_document()`, accept `ChunkingConfig` + optional `Embedder`. Fixed chunker logic stays in place unchanged.
5. **Pipeline call site** — Pass `config.chunking` and `embedder` to `chunk_document()`
6. **Chunker version logic** — Return `"semantic-v1"` when semantic strategy is active
7. **CLI integration** — Progress display, `rag status`, `rag doctor` (section 11)
8. **Tests** — All tests from section 12
9. **Documentation** — Update `config.toml` example in main spec

Steps 1-2 have no dependencies and can be done in parallel. Step 3 depends on 1-2. Steps 4-6 depend on 3. Steps 7-9 depend on 4-6.
