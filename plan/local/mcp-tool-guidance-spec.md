# MCP Tool Guidance — Build Specification

## 1. Document Purpose

This spec adds three things to improve how calling LLMs interact with local-rag's MCP tools: (a) richer tool and parameter descriptions that teach the model what each tool does, when to use it, and how to formulate good queries; (b) server instructions that describe cross-cutting workflows and conventions; (c) MCP prompts that expose reusable multi-step workflows as slash commands in Claude Code and Kiro CLI.

The goal is to eliminate the most common failure modes observed with the current terse descriptions: the calling LLM picks `search_documents` when `quick_search` would suffice, formulates vague single-word queries, ignores the `detail` parameter, and never uses `get_document_context` for drill-down. These are free wins — the fixes are string changes, not architecture changes.

Parent spec: `plan/local/local-rag-spec.md` §8.

---

## 2. Design Decisions

| Decision | Resolution | Rationale |
|---|---|---|
| Where to put guidance | **Tool descriptions + server instructions** | Tool descriptions are universally supported by all MCP clients. Server instructions are supported by Claude Desktop and Claude Code. Together they cover 100% of our target clients. |
| Description length | **3-5 sentences per tool**, under 500 chars each | Anthropic recommends 3-4 sentences minimum. Our 5 tools are well below any client's tool-count threshold, so descriptions are always fully visible — no risk of truncation. |
| Parameter descriptions | **Enrich `query`, `detail`, `format`, `debug`, `folder_filter`** | These are the parameters where the model most often makes poor choices. Other parameters (top_k, limit, window) are self-explanatory. |
| Server instructions | **Single static string, ~600 words** | Set via `server_instructions` kwarg on MCP `Server()` constructor. Describes the recommended tool workflow (scout → search → drill-down), folder structure context, and query formulation tips. |
| MCP prompts | **3 prompts** (research, discover, catch-up) | Prompts are user-initiated slash commands. They return multi-message templates that guide the LLM through a workflow. Low implementation cost — just register prompt handlers. |
| Folder descriptions in config | **No** — plain folder paths only | Nobody has asked for this. Folder paths in server instructions are sufficient context. Can be added later as a trivial config addition if users request it. |
| Configurability of descriptions | **No** — descriptions are hardcoded | Only 5 tools, all internal. No reason to support env-var overrides like GitHub MCP Server (which has 30+ tools across many use cases). Folder descriptions are the one exception. |
| Backward compatibility | **Fully backward compatible** | All changes are additive string content. No tool renames, no schema changes, no new required parameters. |

---

## 3. Enriched Tool Descriptions

These are the exact description strings to use in `_TOOLS` in `src/rag/mcp/tools.py`. Each replaces the current `description` field.

### 3.1 search_documents

```
Search indexed documents using hybrid dense+keyword retrieval with cross-encoder reranking. Returns cited evidence passages grouped by source document, with section locations and page numbers. Use natural-language queries with specific terms from the domain — e.g., "quarterly revenue growth methodology" rather than single words like "revenue". Use folder_filter to narrow results when you know which folder contains the relevant documents. For a broad overview of which documents match a topic, use quick_search first, then search_documents to extract specific evidence.
```

### 3.2 quick_search

```
Quick document-level scan — returns document titles, summaries, and topics matching a query, without individual passage extraction. Use this as a first step to discover which documents are relevant before drilling into specific ones with search_documents. Runs the same retrieval pipeline but returns document-level results instead of chunk-level evidence. Good for questions like "what documents do we have about X?" or "which reports cover Y?". Not suitable when you need specific quotes, data points, or cited passages — use search_documents for that.
```

### 3.3 get_document_context

```
Retrieve detailed context for a specific document or chunk. Provide doc_id to get a document overview with its summary and all section summaries — useful after quick_search identifies a relevant document. Provide chunk_id to get a specific passage with surrounding chunks for context — useful after search_documents returns a passage you want to read more around. The window parameter controls how many adjacent chunks to include (default 1, meaning 1 before + 1 after). Always requires either doc_id or chunk_id — do not call without one.
```

### 3.4 list_recent_documents

```
List recently indexed documents sorted by modification date, with titles and summaries. Use this to answer "what's new?" or "what changed recently?" questions, or to browse the document collection without a specific search query. Use folder_filter to scope to a specific folder. The detail parameter controls summary length — use "8w" for a quick list, "32w" or "64w" when the user wants to understand what each document covers. This tool does not perform any search — it simply lists documents by recency.
```

### 3.5 get_sync_status

```
Get the current indexing status: total files tracked, how many are indexed, pending, or errored, with a per-folder breakdown. Use this to check whether the index is up to date before searching, or to diagnose why expected documents are not appearing in search results. Returns counts only, not document content — use list_recent_documents or search_documents to see actual documents.
```

---

## 4. Enriched Parameter Descriptions

These replace specific parameter `description` fields in the `inputSchema` dicts.

### 4.1 query (search_documents, quick_search)

```
Natural-language search query. Use specific multi-word phrases rather than single keywords — "employee onboarding process timeline" works better than "onboarding". Include domain-specific terms that would appear in the target documents. The query is used for both semantic (meaning-based) and keyword matching, so exact terminology helps.
```

### 4.2 detail (all tools that support it)

No change to the shared `_DETAIL_SCHEMA` description — the current text is clear:

```
Summary detail level: '8w' (short phrase), '16w' (one sentence), '32w' (1-2 sentences), '64w' (short paragraph), '128w' (detailed paragraph)
```

### 4.3 format (search_documents)

```
Output format: 'text' (default) returns results grouped by document with summaries, topics, and ranked passages — best for answering questions. 'json' returns raw structured data with scores, doc_ids, and chunk_ids — use this when you need IDs for follow-up calls to get_document_context.
```

### 4.4 debug (search_documents)

```
Include retrieval debug info: query classification (broad/specific/navigational), per-lane hit counts, fusion weights, timing breakdown, and reranker scores. Useful for understanding why results are ranked the way they are. Does not change the results themselves.
```

### 4.5 folder_filter (all tools that support it)

```
Restrict results to documents within this folder path. Use an absolute path prefix — e.g., "/Users/you/Documents/Work". Documents in subfolders are included. Omit to search all indexed folders.
```

### 4.6 date_filter (search_documents)

```
ISO 8601 date string (e.g., "2025-01-01"). Only return documents modified on or after this date. Useful for scoping to recent content when the user asks about "recent" or "latest" information.
```

---

## 5. Server Instructions

The following text is set as the `instructions` parameter on the MCP `Server()` constructor. It is returned to the client during the MCP `initialize` handshake and provides cross-cutting guidance that applies across all tools.

The `{folders_block}` placeholder is replaced at startup with a bulleted list of configured folder paths from `config.folders.paths`.

```
local-rag is a local document search system that indexes files (PDF, DOCX, TXT, MD) from configured folders on this machine. It provides hybrid semantic + keyword search with cross-encoder reranking over the indexed collection.

## Indexed Folders

{folders_block}

## Recommended Workflow

1. **Scout** — Start with quick_search or list_recent_documents to understand what documents exist and which are relevant. Use the returned doc_ids for drill-down.
2. **Search** — Use search_documents with specific natural-language queries to extract cited evidence passages. Prefer multi-word queries with domain terminology.
3. **Drill down** — Use get_document_context with a doc_id (for full document overview) or chunk_id (for surrounding passage context) to get deeper information.

## Query Tips

- Use 3-8 word natural-language queries, not single keywords.
- Include domain-specific terms that would appear in the documents.
- If initial results are weak, try rephrasing with different terminology or synonyms.
- Use folder_filter when you know which folder contains the target content.
- Use date_filter on search_documents when the user asks about recent information.

## Important Notes

- This system returns evidence passages, not answers. Synthesize answers from the returned citations.
- Scores range from 0 to 1. Results above 0.5 are typically strong matches; below 0.3 may be tangential.
- The detail parameter on summary tools controls verbosity: "8w" for lists, "32w" for overviews, "128w" for deep reading.
- get_sync_status can verify the index is current before searching.
```

---

## 6. MCP Prompts

Three prompts exposed via `prompts/list` and `prompts/get`. In Claude Code these appear as slash commands (e.g., `/local-rag-research`). Each returns a list of `PromptMessage` objects that guide the calling LLM through a multi-step workflow.

### 6.1 research

| Field | Value |
|---|---|
| Name | `research` |
| Description | `Deep research on a topic across all indexed documents. Scouts for relevant documents, extracts key evidence, and synthesizes findings.` |

**Arguments:**

| Name | Description | Required |
|---|---|---|
| `topic` | The topic or question to research | Yes |
| `folder` | Restrict research to a specific folder path | No |

**Returned messages:**

```python
[
    PromptMessage(
        role="user",
        content=TextContent(
            type="text",
            text=(
                f"Research the following topic across my indexed documents: {topic}\n\n"
                "Follow these steps:\n"
                "1. Use quick_search to find which documents are relevant to this topic."
                f"{folder_clause}\n"
                "2. For the top 3-5 most relevant documents, use get_document_context "
                "with their doc_ids to read their full summaries and section structure.\n"
                "3. Use search_documents with 2-3 specific queries targeting different "
                "aspects of the topic to extract cited evidence passages.\n"
                "4. Synthesize your findings, citing the specific documents and sections "
                "where each piece of information was found."
            ),
        ),
    )
]
```

Where `folder_clause` is `f" Filter to folder: {folder}."` if folder is provided, otherwise `""`.

### 6.2 discover

| Field | Value |
|---|---|
| Name | `discover` |
| Description | `Browse and summarize what documents are available in the indexed collection, optionally filtered by folder.` |

**Arguments:**

| Name | Description | Required |
|---|---|---|
| `folder` | Restrict to a specific folder path | No |

**Returned messages:**

```python
[
    PromptMessage(
        role="user",
        content=TextContent(
            type="text",
            text=(
                "Give me an overview of what documents are in my indexed collection."
                f"{folder_clause}\n\n"
                "Follow these steps:\n"
                "1. Use get_sync_status to see how many documents are indexed "
                "and which folders are tracked.\n"
                "2. Use list_recent_documents with detail '32w' to see what's available."
                f"{folder_clause}\n"
                "3. Summarize the collection: what types of documents are there, "
                "what topics do they cover, how many per folder, and when they were "
                "last updated."
            ),
        ),
    )
]
```

### 6.3 catch-up

| Field | Value |
|---|---|
| Name | `catch-up` |
| Description | `Summarize what documents have been added or changed recently.` |

**Arguments:**

| Name | Description | Required |
|---|---|---|
| `folder` | Restrict to a specific folder path | No |

**Returned messages:**

```python
[
    PromptMessage(
        role="user",
        content=TextContent(
            type="text",
            text=(
                "Summarize what's changed recently in my indexed documents."
                f"{folder_clause}\n\n"
                "Follow these steps:\n"
                "1. Use list_recent_documents with detail '32w' to see recently "
                f"modified documents.{folder_clause}\n"
                "2. Review the modification dates and identify which documents are new "
                "or recently updated.\n"
                "3. For each new or significantly changed document, briefly describe "
                "what it covers and highlight anything that looks important or actionable."
            ),
        ),
    )
]
```

No `days` argument or date arithmetic. `list_recent_documents` already sorts by recency — the LLM can judge what counts as "recent" from the modification dates in the results. If the user wants a specific time window (e.g., "what changed this week?"), they can say so in natural language.

---

## 7. Implementation

All changes are in `src/rag/mcp/` and `src/rag/config.py`. No pipeline, retrieval, or database changes.

### 7.1 Tool Descriptions (src/rag/mcp/tools.py)

Replace the `description` field in each `types.Tool()` in the `_TOOLS` list with the exact text from §3. Replace parameter `description` fields per §4. Pure string replacements — no structural changes.

### 7.2 Server Instructions (src/rag/mcp/server.py)

Modify `create_server()` to build the server instructions string and pass it to the `Server()` constructor:

```python
def create_server(config: AppConfig) -> Server:
    """Create and configure the MCP server with all tools registered."""
    instructions = _build_instructions(config)
    server = Server("local-rag", instructions=instructions)
    register_tools(server, config)
    register_prompts(server, config)
    return server
```

Add `_build_instructions(config: AppConfig) -> str` that:
1. Reads `config.folders.paths`.
2. Builds the `{folders_block}` — a bulleted list of folder paths.
3. Returns the template from §5 with the placeholder replaced.

### 7.3 MCP Prompts (src/rag/mcp/prompts.py — new file)

New file `src/rag/mcp/prompts.py` containing:
- `_PROMPTS`: list of `types.Prompt` objects with names, descriptions, and argument definitions from §6.
- `register_prompts(server: Server, config: AppConfig) -> None`: registers `@server.list_prompts()` and `@server.get_prompt()` handlers.
- `_build_research_messages(topic: str, folder: str | None) -> list[types.PromptMessage]`
- `_build_discover_messages(folder: str | None) -> list[types.PromptMessage]`
- `_build_catchup_messages(folder: str | None) -> list[types.PromptMessage]`

All handlers are `async def` per MCP SDK requirements.

### 7.4 No Config Changes

No changes to `config.py`. The server instructions template uses `config.folders.paths` directly — no new config fields needed. Folder descriptions (optional human-readable labels for each path) could be added later if users request it, but plain paths are sufficient for now.

---

## 8. Type Changes

### 8.1 Prompt Arguments

No new Pydantic models in `types.py`. Prompt arguments are validated inline in `prompts.py` with plain dict access:

```python
topic = args["topic"]  # required — MCP SDK validates presence
folder = args.get("folder")  # optional
```

These arguments don't cross module boundaries and are only used within prompt message builders. Adding Pydantic models for 1-2 field dicts is unnecessary overhead. The MCP SDK already validates required vs optional based on the argument schema in the prompt definition.

---

## 9. Testing

### 9.1 Unit Tests (tests/test_mcp_prompts.py — new file)

- **Prompt listing:** Call `handle_list_prompts()`, assert 3 prompts returned with correct names and argument schemas.
- **Prompt messages:** Call `handle_get_prompt()` for each prompt with sample arguments, assert returned messages contain expected tool names and workflow steps.
- **Folder clause:** Assert folder filtering text is included when `folder` arg is provided, absent when omitted.

### 9.2 Unit Tests (tests/test_mcp_tools.py — existing)

- **Description content:** Assert each tool description in `_TOOLS` contains key phrases (e.g., `search_documents` description contains "natural-language queries", `quick_search` contains "Not suitable").
- **Parameter descriptions:** Assert `query` parameter description contains "multi-word phrases".

### 9.3 Server Instructions (tests/test_mcp_server.py)

- **Instructions present:** Create server via `create_server()`, assert `server.instructions` is not None and contains "Recommended Workflow".
- **Folder paths:** Assert configured folder paths appear in `server.instructions`.

### 9.4 E2E Validation (tests/e2e/)

- **Prompt round-trip:** Connect to MCP server, call `prompts/list`, assert 3 prompts. Call `prompts/get` for "research" with `topic="test"`, assert response contains `PromptMessage` list.
- **Server instructions in handshake:** Connect to MCP server, inspect `InitializeResult`, assert `instructions` field is present and non-empty.

### 9.5 Manual Validation

After deploying enriched descriptions, observe Claude Code behavior on these test scenarios and verify improvement:

1. User asks "what documents do I have?" — model should pick `list_recent_documents` or `quick_search`, not `search_documents`.
2. User asks "find the section about quarterly targets in the ops review" — model should use `search_documents` with a multi-word query, not a single keyword.
3. User asks "is the index up to date?" — model should pick `get_sync_status`.
4. User asks "what changed this week?" — model should pick `list_recent_documents`.

---

## 10. CLI Integration

### 10.1 `rag doctor` Health Check

Add a check that server instructions are buildable:

- Verify `config.folders.paths` is non-empty (instructions need at least one folder)
- Report: `✓ MCP server instructions configured (3 folders)` or `✗ No folders configured — server instructions will be empty`

No changes needed for `rag status` — the enriched descriptions and instructions are static strings baked into the MCP server, not runtime state.

### 10.2 `rag mcp-config --print`

No changes needed. The MCP config JSON snippet already points to the server binary. The enriched descriptions and prompts are served by the MCP server itself — clients discover them via the MCP handshake.

---

## 11. Interaction with Local Tools (Claude Code, kiro-cli)

### 11.1 Client Support Matrix

| Feature | Claude Code | Claude Desktop | kiro-cli |
|---|---|---|---|
| Tool descriptions | Yes | Yes | Yes |
| Parameter descriptions | Yes | Yes | Yes |
| Server instructions | Yes | Yes | Verify at implementation time |
| MCP prompts (slash commands) | Yes (`/prompt-name`) | Limited (via prompt picker) | Verify at implementation time |

Tool descriptions and parameter descriptions are part of the core MCP protocol — all clients support them. Server instructions and prompts are newer MCP features. Claude Code is the primary target and fully supports both.

### 11.2 Graceful Degradation for Prompts

Clients that do not support `prompts/list` simply won't show the slash commands. The prompts are additive — they don't affect tool discovery or usage. Users of unsupported clients still benefit from the enriched tool descriptions and server instructions.

### 11.3 Prompt Naming

MCP prompts appear as slash commands in Claude Code. The prompt names (`research`, `discover`, `catch-up`) should be short and memorable. Claude Code prefixes them with the server name, so they appear as `/local-rag-research`, `/local-rag-discover`, `/local-rag-catch-up`. Verify this naming convention at implementation time.

---

## 12. File Change Summary

| File | Change Type | Description |
|---|---|---|
| `src/rag/mcp/tools.py` | Modify | Replace tool and parameter description strings (§3, §4) |
| `src/rag/mcp/server.py` | Modify | Add `_build_instructions()`, pass `instructions` to `Server()`, call `register_prompts()` (§7.2) |
| `src/rag/mcp/prompts.py` | **New** | Prompt definitions and handlers (§7.3) |
| `tests/test_mcp_prompts.py` | **New** | Prompt unit tests (§9.1) |
| `tests/test_mcp_tools.py` | Modify | Description content assertions (§9.2) |
| `tests/test_mcp_server.py` | Modify | Server instructions assertions (§9.3) |

No changes to `src/rag/config.py` or `src/rag/types.py`.

---

## 13. Build Order

This is a single-phase change. Recommended implementation order:

1. **Tool descriptions** — Replace strings in `_TOOLS` (§3, §4). Run `make test`. Immediate value, zero risk.
2. **Server instructions** — Add `_build_instructions()` to `server.py`, pass `instructions` to `Server()` (§5, §7.2). Run `make test`.
3. **MCP prompts** — Create `prompts.py`, register handlers (§6, §7.3). Run `make test`.
4. **E2E validation** — Run `make test-e2e` to verify MCP handshake and prompt round-trip (§9.4).

Steps 1 and 2 deliver ~90% of the value. Step 3 is additive.
