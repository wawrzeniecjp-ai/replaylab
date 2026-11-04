.PHONY: test test-all lint fix data audit

# CI set: everything that needs no data, subprocess, network or key.
test:
	uv run pytest -q -m 'not data and not integration and not llm and not slow'

# Everything, including the real-data tests (needs data/processed).
test-all:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

fix:
	uv run ruff check . --fix && uv run ruff format .

data:
	uv run python data/build_tables.py --check
	uv run python data/build_clusters.py --check

# Check the recorded results from outside the package (scripts/audit/README.md).
audit:
	uv run python scripts/audit/rescore.py
	uv run python scripts/audit/leaks.py
	uv run python scripts/audit/controls.py --positive-control
	uv run python scripts/audit/pairing.py
