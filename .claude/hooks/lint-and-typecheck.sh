#!/bin/bash
# PostToolUse hook: auto-lint and type-check after Edit/Write
file_path=$(jq -r '.tool_input.file_path // empty' 2>/dev/null)

if [[ -z "$file_path" || ! "$file_path" == *.py ]]; then
  exit 0
fi

ruff check --fix "$file_path" && mypy "$file_path" --ignore-missing-imports
