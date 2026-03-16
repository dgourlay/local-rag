# Agent Team Plan: MCP Tool Guidance

## Overview

Implements `plan/local/mcp-tool-guidance-spec.md` — enriched tool descriptions, server instructions, and MCP prompts.

## Team Composition

Based on agent-organizer advice: scope is small (~200 lines new/changed code), all additive, backward compatible. Minimal team with targeted parallelization.

## Execution Plan

### Phase 1 — Implementation (parallel where safe)

| Agent | Type | Files | Task |
|-------|------|-------|------|
| A: Tool Descriptions | backend-developer | `src/rag/mcp/tools.py` | Replace tool + parameter description strings per spec §3-4 |
| B: MCP Prompts | backend-developer | `src/rag/mcp/prompts.py` (new) | Create prompt definitions + handlers per spec §6-7.3 |

These touch different files with zero overlap — safe to parallelize.

### Phase 2 — Server Integration (sequential, depends on Phase 1)

| Agent | Type | Files | Task |
|-------|------|-------|------|
| Orchestrator (me) | — | `src/rag/mcp/server.py` | Add `_build_instructions()`, pass instructions to Server(), call `register_prompts()` per spec §5, §7.2 |

Depends on prompts.py existing for the import.

### Phase 3 — Tests (parallel)

| Agent | Type | Files | Task |
|-------|------|-------|------|
| C: Prompt Tests | qa-expert | `tests/test_mcp_prompts.py` (new) | Test prompt listing, message content, folder clause per spec §9.1 |
| D: Tool/Server Tests | qa-expert | `tests/test_mcp.py` (modify) | Add description assertions + server instructions test per spec §9.2-9.3 |

### Phase 4 — Validation

Run `make lint` and `make test`. Fix any issues. Do not disband until green.

## Working Branch

Direct on main — all changes are additive, no structural risk.

## Key Constraints

- `from __future__ import annotations` in all files
- mypy strict, ruff linting
- MCP handlers must be `async def`
- Server constructor: `Server("local-rag", instructions=instructions)`
- Config folder paths: `config.folders.paths: list[Path]`
