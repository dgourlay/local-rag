# CLI Model Reference for Summarization

Research conducted 2026-03-12. Documents the `--model` flag behavior and available models for each supported LLM CLI tool used by local-rag's CliSummarizer.

---

## Claude Code (`claude`)

**Flag:** `--model <alias|name>`

### Model Aliases (always resolve to latest version)

| Alias | Resolves To | Notes |
|---|---|---|
| `default` | Depends on account tier (Opus 4.6 for Max/Team Premium, Sonnet 4.6 for Pro/Team Standard) | Always available |
| `sonnet` | Claude Sonnet 4.6 | Daily coding tasks |
| `opus` | Claude Opus 4.6 | Complex reasoning |
| `haiku` | Claude Haiku 4.5 | Fast, efficient, simple tasks |
| `sonnet[1m]` | Sonnet with 1M token context | Long sessions |
| `opusplan` | Opus for planning, Sonnet for execution | Hybrid mode |

Full model names (e.g. `claude-opus-4-6`, `claude-sonnet-4-6`) also accepted.

There is **no `claude model list` command** — [feature request #12612](https://github.com/anthropics/claude-code/issues/12612) is open for this.

**Per-invocation usage:** `claude --model haiku --print < prompt.txt`

**Source:** [Claude Code Model Configuration](https://code.claude.com/docs/en/model-config)

---

## OpenAI Codex (`codex`)

**Flag:** `-m <MODEL>` or `--model <MODEL>` or `-c model="<MODEL>"`

### Available Models

| Model ID | Description |
|---|---|
| `gpt-5.4` | Flagship — coding + reasoning + agentic workflows (recommended by OpenAI) |
| `gpt-5.3-codex` | Industry-leading coding model |
| `gpt-5.3-codex-spark` | Near-instant real-time coding (Pro only) |
| `gpt-5.2-codex` | Previous coding model |
| `gpt-5.2` | Previous general-purpose |
| `gpt-5.1-codex-max` | Long-horizon project-scale work |
| `gpt-5.1-codex` | Long-running agentic coding |
| `gpt-5-codex` | GPT-5 variant for agentic coding |
| `gpt-5-codex-mini` | Smaller, cost-effective |
| `gpt-5` | Base reasoning model |

Also supports `--oss` flag with local providers (Ollama, LM Studio).

**Per-invocation usage:** `codex exec -m gpt-5-codex-mini -o /dev/stdout - < prompt.txt`

**Sources:** [Codex Models](https://developers.openai.com/codex/models) | [Codex CLI Reference](https://developers.openai.com/codex/cli/reference)

---

## Kiro (`kiro-cli`)

**Flag:** None for non-interactive use. Model is set via:
- `/model <name>` in interactive chat
- `kiro-cli settings chat.defaultModel <name>` to persist default
- `/model set-current-as-default` to save current selection

### Available Model IDs

| Model ID | Notes |
|---|---|
| `Auto` | Default — Kiro picks best model automatically |
| `claude-opus4.6` | Experimental, 2.2x credit multiplier |
| `claude-sonnet4.6` | 1.3x credit multiplier |
| `claude-opus4.5` | |
| `claude-sonnet4.5` | |
| `claude-sonnet4.0` | |
| `claude-haiku4.5` | |
| `deepseek-3.2` | Open weight, 0.25x credits |
| `minimax-2.1` | Open weight, 0.15x credits |
| `qwen3-coder-next` | Open weight, 0.05x credits |

**No per-invocation model flag.** Model must be set globally via `kiro-cli settings chat.defaultModel <id>`.

**Sources:** [Kiro CLI Model Selection](https://kiro.dev/docs/cli/chat/model-selection/) | [Kiro Models Changelog](https://kiro.dev/changelog/models/)

---

## Summary: Per-Invocation Model Control

| CLI | Can pass model per-invocation? | Flag syntax | Recommended for summarization |
|---|---|---|---|
| `claude` | Yes | `--model haiku` | `haiku` — fastest, cheapest, sufficient for structured JSON summarization |
| `codex` | Yes | `-m gpt-5-codex-mini` | `gpt-5-codex-mini` — smallest, cheapest |
| `kiro-cli` | **No** | N/A (global default only) | `Auto` or `claude-haiku4.5` via settings |

---

## Design Considerations for `rag init`

### Current behavior
- User picks CLI tool (claude, codex, kiro-cli)
- Tool's default model is used for all summarization
- No model selection step

### Future enhancement (not yet implemented)
- After CLI selection, offer model choice with recommendation
- For claude/codex: store model flag in config, inject into args at runtime
- For kiro-cli: either set global default (invasive — changes user's setting outside local-rag) or accept their current default and document it
- Kiro's lack of per-invocation `--model` makes it the odd one out

### Config shape (proposed)

```toml
[summarization]
enabled = true
command = "claude"
model = "haiku"           # new field — injected as --model arg
args = ["--print"]
input_mode = "stdin"
```

The summarizer would prepend `--model <model>` to the args list at runtime for claude/codex. For kiro-cli, the `model` field would be informational only (or used to call `kiro-cli settings` during init).
