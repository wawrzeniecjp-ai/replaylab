.PHONY: test test-all lint fix data

# CI set: everything that needs no data, subprocess, network or key.
test:
	uv run pytest -q -m 'not data and not integration and not llm and not slow'

# Everything, including the real-data tests (needs data/processed).
test-all:
	uv run pytest -q

lint:
	uv ruff check . && ruff format --check .

fix:
	uv ruff check . --fix && ruff format .

data:
	uv run python data/build_tables.py --check
	uv run python data/build_clusters.py --check
