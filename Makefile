.PHONY: test lint format check install

install:
	pip install -e ".[dev]"

test:
	pytest --tb=short -q

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

check: lint test
