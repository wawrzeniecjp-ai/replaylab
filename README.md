# replaylab

A decision engine for protein design selection, validated by replaying real lab
results. Designed to help agents select batches of designs and improve their hit rate under a fixed budget. In simulation mode, results come back immediately
or after a delay, and the agent decides what to buy next and when to stop. Every strategy is validated offline by replaying the recorded Proteinbase outcomes. The forward mode is meant to drive live campaigns.

## Layout

```text
src/replaylab/
  domain/     core types and feature specs
  config.py   YAML configuration loading and validation
  datasets/   Proteinbase tables, 3-mer similarity/clustering, folds, aliases,
              pool loader
  store/      SQLite store: append-only campaign history
  backends/   replay backend: serves recorded outcomes to the agent
  engine/     campaign rules (clock, submit/settle/wait), scoring
  advisor/    preprocessing, logistic model, similarity index, rate/stop rules
  policies/   batch selection strategies: random, fixed metric, learner, learner_sim (+stop)
  eval/       evaluation engine
  interfaces/ the 14 tools, MCP server and CLI (the agent-facing layer)
  spec_pin.py OpenAPI spec pin
data/         data handling and processing scripts
scripts/      convenience scripts
tests/        one test file per module + import-boundary guard
```

## Import boundaries test
`advisor` and `policies` never import `store`, `backends` or `engine`: they receive a
`CampaignState` snapshot only, so they cannot write and cannot see hidden outcomes. This is enforced by `tests/test_import_boundaries.py`, to ensure that future outcome data cannot leak into the agent context.

## Quickstart

```bash
uv sync                                    # project + dev group (pytest, ruff); without uv: pip install -e . pytest ruff
make test                                  # CI set: no data, network or key needed
make lint                                  # ruff check + format check
make test-all                              # everything, including real-data checks

# Download the Proteinbase CSV at data/raw/proteinbase.csv (gitignored), then:
python data/build_tables.py --csv data/raw/proteinbase.csv --check   # expect 15 x OK
python data/build_clusters.py --check                                # cluster summary check
python data/build_folds.py                                           # writes the frozen split
pytest -m data -q                                                    # real-data checks

# Evaluation harness
python scripts/run_ladder.py                                         # reports/ladder.{csv,md}
uv sync --extra eval                                                 # adds the LLM client extras
replaylab-eval conditions                                            # the tool set of every condition
replaylab-eval pilot --run-dir runs/x --model deepseek --games N-dev \
    --conditions A,B1 --trials 20 --max-usd 1                        # first episodes + cost projection
replaylab-eval run --run-dir runs/x --model deepseek --games N-dev \
    --conditions A,B1 --trials 20 --max-usd 1                        # plays only what is missing
replaylab-eval scripted --run-dir runs/x --games N-dev \
    --policies random,learner,learner_sim,learner_sim_purchased --trials 20
replaylab-eval compare --records runs/x/episodes.jsonl --game N-dev \
    --a B1 --b scripted:learner_sim --ci 0.9                         # paired bootstrap over seeds
replaylab-eval view --run-dir runs/x                                 # one HTML page for the run
```

### Note
`build_folds.py` refuses to overwrite `data/splits/folds.csv` or the held-out id file once they
exist (exit code 3); pass `--force` only if you really mean to replace a frozen split.

## Definitions

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
