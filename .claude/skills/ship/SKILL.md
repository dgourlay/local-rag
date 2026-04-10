# /ship - Validate and Push
1. Run `python -m pytest` — all tests must pass
2. Run `ruff check .` — must be clean
3. Run `mypy src/` — no errors
4. Stage all changes: `git add -A`
5. Generate a concise commit message from the diff
6. Commit and push to current branch
