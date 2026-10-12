# replaylab

A decision engine for protein design selection, validated by replaying real wet-lab
outcomes. An agent buys batches of designs under a budget, results come back (immediately
or after a delay), and the agent decides what to buy next and when to stop. Every strategy
is validated offline by replaying the recorded Proteinbase outcomes; the same code is meant
to drive a live campaign later. Hidden outcomes must never leak to a strategy before it
buys the design.

## Layout

```text
src/replaylab/
  domain/     core types and feature specs — stdlib only (enforced by a test)
  config.py   YAML loading: config/features.yaml, config/games.yaml
  datasets/   Proteinbase tables, 3-mer similarity/clustering, folds, aliases,
              pool loader (the only reader of hidden outcomes), phase-0 analyses
  store/      SQLite campaign store: transactional, forward-only batch states
  backends/   replay backend: serves recorded outcomes only after submit + delay
  engine/     campaign rules (clock, submit/settle/wait), scoring, replay wiring
  advisor/    preprocessing, logistic model, similarity index, rate/stop rules
  policies/   scripted ladder: random, fixed metric, learner, learner_sim (+stop)
  eval/       LLM smoke test
  spec_pin.py OpenAPI spec pin
data/         thin CLIs: build_tables, build_clusters, build_folds, feature_review,
              author_checks
scripts/      pin_spec.py, smoke_llm.py
tests/        one test file per module + import-boundary guard
```

`advisor` and `policies` never import `store`, `backends` or `engine`: they receive a
`CampaignState` snapshot only, so they cannot write and cannot see hidden outcomes
(`tests/test_import_boundaries.py`).

## Quickstart

```bash
uv venv && uv pip install -e ".[dev]"      # or: python -m venv .venv && pip install -e ".[dev]"
make test                                  # CI set: no data, network or key needed
make lint                                  # ruff check + format check
make test-all                              # everything, including real-data checks

# put the Proteinbase CSV at data/raw/proteinbase.csv (never commit it), then:
python data/build_tables.py --csv data/raw/proteinbase.csv --check   # expect 15 x OK
python data/build_clusters.py --check                                # cluster summary check
python data/build_folds.py                                           # writes the frozen split
pytest -m data -q                                                    # real-data checks
```

`--check` prints `OK` or `DIFF` per line; add `--strict` to exit non-zero on any difference.
A `DIFF` is a finding to investigate, not necessarily a bug: the expected numbers came from
exploratory runs. `build_clusters.py` also prints how many pairs sit within 1e-6 of the
distance threshold; if that number is 0, floating-point rounding cannot explain a mismatch.

`build_folds.py` refuses to overwrite `data/splits/folds.csv` or the held-out id file once they
exist (exit code 3); pass `--force` only if you really mean to replace a frozen split.

## Definitions (section 2 of the spec)

* **hit**: any binding record for the (design, target) pair is True.
* **conflict**: the binding records for the pair disagree.
* **strength**: the strongest label among the strength records (Strong, Medium, Weak).
* **kd_molar**: the smallest KD, converted to molar units; a KD with no target is attached
  to the design's only binding target, or dropped if that is ambiguous.
* Metadata (`meta_name`, `meta_author`, `meta_method`) is stored apart from features and is
  never visible to an agent.

## Output (data/processed/, gitignored)

`designs.parquet`, `outcomes.parquet`, `targets.parquet`, `evaluations_long.parquet`,
`build_report.json`.
