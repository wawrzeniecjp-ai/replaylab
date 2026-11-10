# Code tour

A reading guide to `src/replaylab`: what the entry points are, which call path each one takes,
and the order in which to read the modules. For a deeper look at the data flow and the leak
rules, see `docs/user_guide/00-architecture.qmd`; for a hands-on REPL session on the real tables,
see `docs/walkthrough.md`.

## 1. The picture in one diagram

```text
                    replaylab-eval run / pilot / scripted            (eval/cli.py)
                                   |
                              eval/runner.py  ---- spend guard, records, final-run permit
                                   |
         +-------------------------+--------------------------+
         | LLM episode                                        | scripted episode
         v                                                    v
   eval/loop.py  <-- eval/llm*.py (model API)            eval/scripted.py
         |                                                    |
   eval/mcp_bridge.py  -- in-memory MCP client                |
         |                                                    |
   interfaces/server.py  (MCP tool schemas)                   v
         |                                             engine/play.py  (the game loop)
   interfaces/tools.py   Toolbox: the 15 tools  ------> policies/ (propose_batch)
         |                                         \--> advisor/  (fit, similarity, rates, stop)
         v                                                    |
   engine/campaign.py   Campaign: submit / wait / settle / stop  <-------+
         |                     |
   store/store.py (SQLite)   backends/replay.py (hidden outcomes)
         ^
   engine/replay.py  wires a game: datasets/pool.py + datasets/aliases.py + config.py
```

Everything above `interfaces/` is about *running agents*; everything below it is the *game*.
`engine/replay.py:build_replay_campaign` is the single place where a game is assembled, and
`interfaces/runtime.py:build_runtime` wraps it with the per-seed advisor objects that the tools
need.

## 2. Entry points

Two console scripts are declared in `pyproject.toml`, plus a few scripts.

| command | defined in | what it does |
| --- | --- | --- |
| `replaylab serve` | `interfaces/cli.py:serve` | one campaign behind a stdio MCP server, for a real MCP client (or you, by hand) |
| `replaylab campaign status / ingest / propose` | `interfaces/cli.py` | inspect or drive a stored campaign from the shell; prints JSON |
| `replaylab run-scripted` | `interfaces/cli.py:run_scripted` | play one scripted policy on one game and seed |
| `replaylab-eval conditions` | `eval/cli.py:show_conditions` | print the tool list of every condition |
| `replaylab-eval scripted` | `eval/cli.py:scripted` | scripted policies on the same seeds as the agents, written as records |
| `replaylab-eval pilot` / `run` | `eval/cli.py:pilot`, `run` | play the games x trials x conditions matrix with an LLM (pilot first: it projects the cost) |
| `replaylab-eval compare` | `eval/cli.py:compare` | paired bootstrap of two conditions over episode seeds |
| `replaylab-eval view` | `eval/cli.py:view` → `eval/replay_view.py` | one HTML page that replays a run (shows hidden outcomes: for people only) |
| `replaylab-eval smoke` | `eval/smoke.py` | check that a provider endpoint works before spending on it |
| `python scripts/run_ladder.py` | → `eval/ladder.py` | every scripted policy, every game, 100 seeds → `reports/ladder.*` |
| `python scripts/export_study.py`, `make_figures.py` | scripts | records → `reports/agent_study.csv` → `reports/figures/*.svg` |
| `python data/build_tables.py`, `build_clusters.py`, `build_folds.py` | `data/` → `datasets/` | build the processed tables and the frozen split from the raw CSV |

`python -m replaylab` and `python -m replaylab.eval` are the same two CLIs (`__main__.py`).
The root `main.py` is an unused stub.

## 3. Three call paths worth tracing

Read these with the files open; each step names the function to jump to.

### 3.1 One LLM episode (`replaylab-eval run`)

1. `eval/cli.py:run` → `_prepare` loads the models, prices, conditions, prompt and games;
   computes the `config_hash`; checks the pilot projection against `--max-usd`
   (`eval/spend.py:refuse_if_over`).
2. `eval/runner.py:run_matrix` expands the matrix (`expand_matrix`), skips episodes already
   in `episodes.jsonl` (`eval/records.py:completed_keys`), and runs the rest with at most
   four in flight.
3. `eval/runner.py:run_llm_episode`, per episode:
   * seed from `eval/seeds.py:episode_seed(game, trial)`: depends on game and trial only, so
     every condition plays the same aliases;
   * `interfaces/runtime.py:build_runtime` → `engine/replay.py:build_replay_campaign` (pool,
     hidden outcomes, aliases, SQLite store, `Campaign`);
   * `interfaces/server.py:build_server(Toolbox(...), condition.tools_arg)`: an MCP server
     exposing only this condition's tools, connected in memory (`mcp.client.Client`);
   * `eval/loop.py:run_episode`: the agent loop below;
   * `campaign.settle()` then `engine/scoring.py:score` → the record.
4. `eval/loop.py:run_episode`: each turn, `guard.check()` (spend cap), one model call
   (`eval/llm.py` for OpenAI-compatible APIs, `eval/llm_anthropic.py` for Claude), then
   every tool call in order through `eval/mcp_bridge.py:McpBridge.call`. Ends on
   `finished`, `agent_stopped`, `agent_ended` or `max_turns`; a truncated reply or a cap
   raises, so the episode is not recorded.
5. `eval/records.py:append_record` writes one JSON line. Failed episodes leave nothing, and
   rerunning the same command plays exactly those.

### 3.2 One tool call (`submit_batch`)

1. `interfaces/server.py`: the `submit_batch` closure validates the arguments against its
   schema and calls `Toolbox.submit_batch`; a `ToolFailure` becomes an MCP error that tells the
   agent how to fix the call.
2. `interfaces/tools.py:Toolbox.submit_batch` → `engine/campaign.py:Campaign.submit`.
3. `Campaign.submit` validates (stopped, rounds, size, duplicates, unknown, already bought,
   budget), records the batch in the store, hands it to `backends/replay.py:ReplayBackend`,
   ingests what is delivered, then ticks the clock. With `delay_rounds > 0` outcomes arrive
   on later ticks (`wait`).
4. The response is a Pydantic model from `interfaces/schemas.py`: it cannot carry a field
   that is not declared, which is part of the leak protection.

The advisor tools follow the same shape: `Toolbox.fit_and_rank` → `advisor/preprocess.py` +
`advisor/model.py:fit_model`; `similar_designs` → 3-mer vectors from `datasets/similarity.py`;
`propose_batch` → `interfaces/policy_factory.py:make_policy` → `policies/`; `estimate_hit_rate`
→ `advisor/rates.py`; `stop_advice` → `advisor/stop.py`; `fit_custom_model` (condition C) →
`advisor/analysis.py`.

### 3.3 One scripted episode

`engine/play.py:play` is the whole game in about fifteen lines: per round, ask the player for
a batch (`choose(state, state.next_round)`), then `campaign.submit(ids)`; an empty batch stops
the campaign; at the end `settle()` and `score()`. No tools, no MCP. It is the best place to
start reading, because it shows the game without the agent machinery around it.

Every scripted run goes through it: `eval/ladder.py:run_game` (the ladder),
`eval/scripted.py:play_scripted` (`replaylab-eval scripted`) and `interfaces/cli.py:run_scripted`
(`replaylab run-scripted`). `policies/core.py:as_choose(policy, seed)` turns a policy into the
player, drawing with the generator of `(seed, round)`.

## 4. The layers, bottom up

The import rules are enforced by `tests/test_import_boundaries.py` (`ALLOWED`). Reading in
this order means you never meet a name you have not seen yet.

| package | read first | role | may import |
| --- | --- | --- | --- |
| `domain/` | `types.py` | `Design`, `Outcome`, `Batch`, `CampaignConfig`, and the immutable `CampaignState` snapshot (delivered outcomes only); `features.py`: `FeatureSpec` with orientation | stdlib only |
| `config.py` | `GameConfig`, `load_games`, `load_feature_config` | YAML in `config/` → typed objects; unknown keys are errors | domain |
| `datasets/` | `pool.py` | `load_pool` (visible features) and `load_outcomes` (the only reader of outcomes, refuses held-out and final-test games); `aliases.py` (`d0001` ids from sha256); `proteinbase.py`, `split_tables.py`, `splits.py`, `similarity.py`: offline table, cluster and fold builders; `anthropic.py`: the final-test pools | domain, config |
| `store/` | `store.py:CampaignStore` | SQLite: designs, batches, outcomes, append-only event log; `snapshot()` returns a `CampaignState` | domain |
| `backends/` | `replay.py:ReplayBackend` | holds the hidden outcomes; `submit` / `poll` / `advance`; `base.py` is the protocol a live backend will implement | domain |
| `engine/` | `campaign.py:Campaign` | the rules of the game (clock, validation, delay); `replay.py` wires a game and resumes a killed one; `scoring.py` scores a settled campaign | domain, store, backends, datasets, config |
| `advisor/` | `preprocess.py`, `model.py` | pure functions over a `CampaignState`: preprocessing (orientation applied here), logistic model with its gate (2 hits, 8 purchases), similarity, Beta hit rates, stop rules, custom analysis | domain, datasets |
| `policies/` | `core.py` | `choose(state, k, rng) -> ids`: `RandomPolicy`, `FixedMetric`, `Learner`, `LearnerSim`, `WithStop`; `purchased.py` has the purchased-scaling variants | domain, advisor |
| `interfaces/` | `tools.py:Toolbox` | the 15 tools (`CAMPAIGN_TOOLS`, `ADVISOR_TOOLS`, `ANALYSIS_TOOLS`); `schemas.py` the responses; `server.py` MCP; `runtime.py` one game + one seed; `cli.py` | everything above; **nothing imports it** |
| `eval/` | `runner.py`, `loop.py` | the experiment harness (next section) | anything; nothing imports it |

## 5. `eval/`, file by file

| file | role |
| --- | --- |
| `cli.py` | the `replaylab-eval` commands; `_prepare` and `_deps` assemble everything a run needs |
| `runner.py` | the matrix: expand, skip completed, run concurrently, record; scripted matrix too |
| `loop.py` | one agent episode: model call, tool calls, stop reasons, closing summary |
| `mcp_bridge.py` | MCP tools ↔ OpenAI-style function specs; compact JSON, `ERROR: ` prefix, 20,000-character cut |
| `llm.py`, `llm_anthropic.py`, `llm_types.py` | provider adapters behind the `LLMClient` protocol; retries; `ResponseTruncated` |
| `models.py`, `spend.py` | `config/models.yaml` and `config/prices.yaml`; `SpendGuard` with the persisted `spend.json`; pilot projection |
| `conditions.py` | `config/conditions.yaml`: a condition is a tool allow-list (A, B1, B2, C, ...) |
| `prompts.py` | loads `prompts/system.md`; the fixed first user message |
| `records.py` | the `episodes.jsonl` format, partial-line repair on read, `config_hash`, git info |
| `seeds.py` | `episode_seed(game, trial)` |
| `scripted.py`, `ladder.py` | scripted play; the ladder report |
| `stats.py`, `scorers.py` | `paired_bootstrap`; per-record metrics (`hits`, argument validity, ...) |
| `final_run.py` | the run-once guard and lock file of the final test |
| `replay_view.py`, `smoke.py` | the HTML replay page; the endpoint probe |

## 6. Configuration and data on disk

| path | holds |
| --- | --- |
| `config/games.yaml` | every game: target, split, rounds, batch size, budget, delay, `final_test` |
| `config/features.yaml` | visible features per target, their orientation, the pre-registered first sort column |
| `config/conditions.yaml` | the tool list of every condition |
| `config/models.yaml`, `config/prices.yaml` | provider endpoints, request parameters, checked prices (`.example` files are templates) |
| `config/frozen_v1.yaml` | frozen model settings the ladder runs under |
| `prompts/system.md` | the one system prompt, shared by all conditions |
| `data/processed/`, `data/splits/` | built tables, clusters and folds (git-ignored except the frozen split) |
| `runs/<dir>/` | `episodes.jsonl`, `spend.json`, `pilot.json`, `errors.jsonl` per run (git-ignored) |
| `tests/snapshots/tools.json` | the tool schemas and descriptions as the agent sees them |

## 7. Suggested first hour

1. `domain/types.py`: the vocabulary (15 minutes).
2. `engine/play.py:play`, then `engine/campaign.py`: the game itself.
3. `policies/core.py:Learner.choose` with `advisor/preprocess.py` and `advisor/model.py`: what
   "a scripted learner" means, including the model gate that explains the E result.
4. `interfaces/tools.py`: skim `query_pool`, `submit_batch`, `fit_and_rank`, `similar_designs`,
   the tools the agents actually use most.
5. `eval/runner.py:run_llm_episode` and `eval/loop.py:run_episode`: how an agent plays.
6. Run something small to watch it move (no API key needed; `run-scripted` needs the built
   tables in `data/processed/`):

```bash
uv run replaylab run-scripted --policy learner_sim --game N-dev --seed 3
```

```bash
uv run replaylab-eval conditions
```

Good tests to read alongside: `tests/test_engine_campaign.py` (the rules), `tests/test_tools_leaks.py`
(what may never appear in a response), `tests/test_eval_loop.py` with `tests/fakes_llm.py` (the
loop against a scripted fake model).
