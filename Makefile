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
	.venv/bin/pip install -q --upgrade pip
	.venv/bin/pip install -q -e ".[dev]"

qdrant-up:
	docker compose up -d

qdrant-down:
	docker compose down

download-models:
	bash scripts/download-models.sh

setup: install download-models
	@echo ""
	@if docker info >/dev/null 2>&1; then \
		$(MAKE) qdrant-up; \
		echo ""; \
		echo "Setup complete. Next steps:"; \
		echo "  1. source .venv/bin/activate"; \
		echo "  2. rag init          # configure folders"; \
		echo "  3. rag index         # index documents"; \
		echo "  4. rag search \"test\" # verify it works"; \
	else \
		echo "Python environment ready, but Docker is not running."; \
		echo "Qdrant (via Docker) is required for indexing and search."; \
		echo ""; \
		echo "Next steps:"; \
		echo "  1. Install/start Docker Desktop: https://docs.docker.com/get-docker/"; \
		echo "  2. make qdrant-up    # start Qdrant"; \
		echo "  3. source .venv/bin/activate"; \
		echo "  4. rag init          # configure folders"; \
		echo "  5. rag index         # index documents"; \
	fi
