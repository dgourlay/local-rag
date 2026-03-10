.PHONY: lint test test-e2e test-all format install setup qdrant-up qdrant-down download-models

# --- Development ---

lint:
	.venv/bin/ruff check src/ tests/
	.venv/bin/ruff format --check src/ tests/
	.venv/bin/mypy --strict src/

format:
	.venv/bin/ruff format src/ tests/

test:
	.venv/bin/pytest tests/ -k "not e2e" -x -q

test-e2e:
	.venv/bin/pytest tests/e2e/ -v

test-all: lint test test-e2e

# --- Setup ---

install:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

qdrant-up:
	docker compose up -d

qdrant-down:
	docker compose down

download-models:
	bash scripts/download-models.sh

setup: install qdrant-up
	@echo ""
	@echo "Setup complete. Next steps:"
	@echo "  1. source .venv/bin/activate"
	@echo "  2. rag init          # configure folders"
	@echo "  3. rag index         # index documents (downloads models on first run)"
	@echo "  4. rag search \"test\" # verify it works"
