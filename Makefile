.PHONY: lint test test-e2e test-all

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy --strict src/

test:
	pytest tests/ -k "not e2e" -x -q

test-e2e:
	pytest tests/e2e/ -v

test-all: lint test test-e2e
