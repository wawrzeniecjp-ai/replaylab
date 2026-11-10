# Code tour

A reading guide to `src/replaylab`. Read sections 1 and 2 first: six nouns and three call
paths are enough to place every other file. The rest is reference: entry points, the paths step
by step, every guard in one table, the layers, and a suggested first hour.

For the data flow and the leak rules in more depth, see `docs/user_guide/00-architecture.qmd`;
for a hands-on REPL session on the real tables, see `docs/walkthrough.md`.

## 1. Six nouns

Everything in the code is one of these, or serves one of them.

| noun | where | one sentence |
| --- | --- | --- |
| `CampaignState` | `domain/types.py` | an immutable snapshot of a campaign: designs, batches, and *delivered* outcomes only |
| `Campaign` | `engine/campaign.py` | the rules: `submit`, `wait`, `stop`, `settle`; validates each move and advances the clock |
| `ReplayEnvironment` | `engine/replay.py` | one game assembled for one seed: campaign, hidden outcomes, store; `score()` |
| `Policy` | `policies/core.py` | `choose(state, k, rng) -> ids`: a scripted player |
| `Toolbox` | `interfaces/tools/toolbox.py` | the 15 tools an agent can call, over one campaign |
| `run_episode` | `eval/loop.py` | the agent loop: model call, then its tool calls, repeat |

The one fact that explains most of the layout: **only `Campaign` can buy, and only delivered
outcomes ever reach a `CampaignState`.** A policy, a tool and an advisor function all see the
same snapshot, so none of them can read the outcome of a design that was not bought.

## 2. Three call paths

```text
SCRIPTED   build_replay_campaign ──> engine.play(env, as_choose(policy, seed)) ──> Played(score, stopped)
             (the ladder, `replaylab run-scripted` and `replaylab-eval scripted` all call this one function)

AGENT      build_runtime ──> build_server(Toolbox) ──> run_episode(llm, bridge) ──> settle ──> score ──> record

ONE TOOL   server closure ──> Toolbox.<tool> (one line) ──> tools/<group>.py ──> advisor/* or Campaign ──> schema
```

There is **one** game loop, `engine/play.py:play`, and the agent path does not use it, on
purpose: an agent drives the rounds itself, by calling `submit_batch` when it chooses to. So
the agent path ends where the scripted one does (settle, score), but has no loop over rounds.

How the pieces fit together:

```text
                    replaylab-eval run / pilot / scripted            (eval/cli.py, eval/setup.py)
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
   interfaces/tools/     Toolbox: the 15 tools  ------> policies/ (propose_batch)
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
`interfaces/runtime.py:build_runtime` wraps it with the per-seed objects the tools need (the
preprocessor and the similarity index, both built from this campaign's own designs).

## 3. Entry points

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
| `replaylab-eval smoke` | `eval/cli.py:smoke` | three real turns against a provider, no record, before spending on it |
| `python scripts/run_ladder.py` | → `eval/ladder.py` | every scripted policy, every playable game, 100 seeds → `reports/ladder.*` |
| `python scripts/export_study.py`, `make_figures.py` | scripts | records → `reports/agent_study.csv` → `reports/figures/*.svg` |
| `python data/build_tables.py`, `build_clusters.py`, `build_folds.py` | `data/` → `datasets/` | build the processed tables and the frozen split from the raw CSV |

`python -m replaylab` and `python -m replaylab.eval` are the same two CLIs (`__main__.py`).
The root `main.py` is an unused stub.

## 4. The three paths, step by step

Read these with the files open; each step names the function to jump to. Start with the
scripted path: it is the game with nothing around it.

### 4.1 One scripted episode

`engine/play.py:play` is the whole game in about fifteen lines: per round, ask the player for
a batch (`choose(state, state.next_round)`), then `campaign.submit(ids)`; an empty batch stops
the campaign; at the end `settle()` and `score()`. No tools, no MCP.

Every scripted run goes through it: `eval/ladder.py:run_game` (the ladder),
`eval/scripted.py:play_scripted` (`replaylab-eval scripted`) and `interfaces/cli.py:run_scripted`
(`replaylab run-scripted`). `policies/core.py:as_choose(policy, seed)` turns a policy into the
player, drawing with the generator of `(seed, round)`. A policy is built by name in one place,
`policies/factory.py:make_policy`.

### 4.2 One LLM episode (`replaylab-eval run`)

1. `eval/cli.py:run` → `eval/setup.py:prepare` loads the model, its price, the conditions,
   the prompt and the games; computes the `config_hash` (`setup.py:llm_config_hash`); and
   makes every refusal (section 5) before anything is spent. Back in `run`, the pilot's
   projection is checked against `--max-usd` (`eval/spend.py:refuse_if_over`).
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

### 4.3 One tool call (`submit_batch`)

1. `interfaces/server.py`: the `submit_batch` closure validates the arguments against its
   schema and calls `Toolbox.submit_batch`; a `ToolFailure` becomes an MCP error that tells the
   agent how to fix the call.
2. `interfaces/tools/toolbox.py:Toolbox.submit_batch` (one line) →
   `interfaces/tools/campaign.py:submit_batch`, which calls `engine/campaign.py:Campaign.submit`
   and turns each engine error into a message saying what to do instead.
3. `Campaign.submit` validates (stopped, rounds, size, duplicates, unknown, already bought,
   budget), records the batch in the store, hands it to `backends/replay.py:ReplayBackend`,
   ingests what is delivered, then ticks the clock. With `delay_rounds > 0` outcomes arrive
   on later ticks (`wait`).
4. The response is a Pydantic model from `interfaces/schemas.py`: it cannot carry a field
   that is not declared, which is part of the leak protection.

Every tool has the same shape: a one-line `Toolbox` method → a function in the module of its
group (`pool`, `campaign`, `models`, `estimates` under `interfaces/tools/`) → `advisor/`,
`policies/` or `Campaign`. `fit_and_rank` → `advisor/preprocess.py` + `advisor/model.py:fit_model`;
`similar_designs` → `advisor/similarity.py:SimilarityIndex.nearest`; `propose_batch` →
`policies/factory.py:make_policy` → `policies/`; `estimate_hit_rate` → `advisor/rates.py`, and
for a next batch `advisor/calibration.py` (D12); `stop_advice` → `advisor/stop.py`;
`fit_custom_model` (condition C) → `advisor/analysis.py`.

## 5. Guards

Some refusals are written more than once, **on purpose**: each copy guards a different way in
(defence in depth). Not every copy has a test that fails without it: a duplicated held-out
check is one of the mutants deliberately left alive. So if you find the same check in two
places, look it up here before merging them; the tests may not stop you.

### 5.1 Held-out and final-test games

A held-out game (`split: heldout`) is played once, at the end, by the final-run script. A
final-test game (`final_test: true`, the `AB-*` games) is played once, by the runner with
`--final-run`, under a lock.

| where | refuses | who reaches it | why this copy exists |
| --- | --- | --- | --- |
| `config.py:GameConfig` | a game that is both `final_test` and held-out or exploratory | every loader of `games.yaml` | the two rules above must not overlap |
| `datasets/pool.py:load_outcomes` | held-out and final-test outcomes, unless `final_run` | everything that reads hidden outcomes | the last line: the only reader of outcomes, so no path can skip it; it refuses before opening a file |
| `interfaces/runtime.py:build_runtime` | held-out always; final-test unless `final_run` | `replaylab serve`, the campaign CLI, the runner | refuses before a store is created, with a message for a person; `serve` and the CLI never pass `final_run` |
| `eval/runner.py:_check_games` (via `validate_specs`) | held-out; final-test without a permit; a final test mixed with other games; a permit for another game | `replaylab-eval run`, `pilot`, `scripted` | refuses the whole matrix before the first episode, so nothing is spent |
| `eval/runner.py:_check_permit` | an episode outside the permitted trials or conditions, or written to another records file | a final run | a permit covers one leg exactly, not "this game" |
| `eval/setup.py:final_permit` | more than one game; an uncommitted tree (no escape hatch) | `--final-run` | the final run must be reproducible from its commit |
| `eval/final_run.py:acquire` | a leg already run with other settings; a frozen hash or a commit that differs from the lock's | `--final-run` | the run-once rule itself, persisted in a lock file (`--lock-dir`, default `reports/final-test/`) |
| `eval/ladder.py:run_ladder` | held-out and final-test, for every selected game | `scripts/run_ladder.py` | refuses the whole list before the first game is played |
| `eval/ladder.py:run_game` | held-out and final-test | direct callers of one game | `run_game` is public; it must not rely on `run_ladder` having checked |
| `eval/replay_view.py:pool_view` | held-out and final-test | `replaylab-eval view` | the page shows hidden outcomes, so it must never show these |
| `scripts/run_ladder.py:default_games` | (skips, does not refuse) held-out, final-test, rehearsal and exploratory games | the default ladder run | convenience: the default list contains only what the ladder may play |

### 5.2 Money, records and reproducibility

| where | guards | why |
| --- | --- | --- |
| `eval/loop.py:run_episode` → `eval/spend.py:SpendGuard.check` | no model call once the cap is reached | the check comes *before* each call; the total persists in `spend.json`, so a resumed run keeps counting |
| `eval/cli.py:run` → `eval/spend.py:refuse_if_over` | no run without a pilot projection (unless `--skip-pilot`), and none whose projection exceeds `--max-usd` | the cost is known before anything is spent |
| `eval/setup.py:prepare` | bad concurrency, unknown model or price, missing API key, unknown condition | every setup refusal comes before the first model call |
| `eval/setup.py:check_uncommitted` | recording from an uncommitted tree (escape: `--allow-uncommitted`, which marks every record `git_dirty`) | `config_hash` does not cover the code; the recorded commit is the only record of it |
| `eval/loop.py` → `ResponseTruncated` | recording an episode whose reply hit the output limit | a cut reply is a failure, not the agent ending the game |
| `eval/runner.py` (via `loop.py:unwrap`) | a `SpendCapReached` hidden inside an `ExceptionGroup` | the cap must stop the run, not be logged as one failed episode |
| `eval/records.py:append_record` (`_drop_partial_tail`, `fsync`) | a half-written last line gluing onto the next record | an episode is recorded only if it ended normally; rerunning plays exactly the missing ones |
| `tests/test_registered_hashes.py` | any change to the registered config and frozen hashes | the pre-registrations name those hashes |
| `tests/test_import_boundaries.py` | an import across the layer rules (section 6) | keeps `domain` pure and nothing importing `interfaces` or `eval` |

## 6. The layers, bottom up

The import rules are enforced by `tests/test_import_boundaries.py` (`ALLOWED`). Reading in
this order means you never meet a name you have not seen yet.

| package | read first | role | may import |
| --- | --- | --- | --- |
| `domain/` | `types.py` | `Design`, `Outcome`, `Batch`, `CampaignConfig`, and the immutable `CampaignState` snapshot (delivered outcomes only); `features.py`: `FeatureSpec` with orientation | stdlib only |
| `config.py` | `GameConfig`, `load_games`, `load_feature_config` | YAML in `config/` → typed objects; unknown keys are errors | domain |
| `datasets/` | `pool.py` | `load_pool` (visible features) and `load_outcomes` (the only reader of outcomes, refuses held-out and final-test games); `aliases.py` (`d0001` ids from sha256); `proteinbase.py`, `split_tables.py`, `splits.py`, `similarity.py`: offline table, cluster and fold builders; `anthropic.py`: the final-test pools | domain, config |
| `store/` | `store.py:CampaignStore` | SQLite: designs, batches, outcomes, append-only event log; `snapshot()` returns a `CampaignState` | domain |
| `backends/` | `replay.py:ReplayBackend` | holds the hidden outcomes; `submit` / `poll` / `advance`; `base.py` is the protocol a live backend will implement | domain |
| `engine/` | `play.py`, `campaign.py:Campaign` | the game loop and the rules of the game (clock, validation, delay); `replay.py` wires a game and resumes a killed one; `scoring.py` scores a settled campaign | domain, store, backends, datasets, config |
| `advisor/` | `preprocess.py`, `model.py` | pure functions over a `CampaignState`: preprocessing (orientation applied here), logistic model with its gate (2 hits, 8 purchases), similarity, Beta hit rates, calibration of predictions (D12), stop rules, custom analysis | domain, datasets |
| `policies/` | `core.py` | `choose(state, k, rng) -> ids`: `RandomPolicy`, `FixedMetric`, `Learner`, `LearnerSim`, `WithStop`; `purchased.py` has the purchased-scaling variants; `factory.py:make_policy` builds any of them by name (the tools, the CLIs and the ladder all use it) | domain, advisor |
| `interfaces/` | `tools/toolbox.py:Toolbox` | the 15 tools, one line each, grouped into `tools/pool.py`, `campaign.py`, `models.py`, `estimates.py`; `tools/catalog.py` the names, groups and the descriptions the agent reads; `schemas.py` the responses; `server.py` MCP; `runtime.py` one game + one seed; `cli.py` | everything above; **nothing imports it** |
| `eval/` | `runner.py`, `loop.py` | the experiment harness (next section) | anything; nothing imports it |

## 7. `eval/`, file by file

| file | role |
| --- | --- |
| `cli.py` | the `replaylab-eval` commands, one function each |
| `setup.py` | what a command sets up before its first episode: `prepare` and `runner_deps` assemble everything a run needs, `llm_config_hash` is the registered hash, and every refusal (dirty tree, unknown model or price, a second final run) happens here |
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

## 8. Configuration and data on disk

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
| `reports/final-test/` | the final-test lock files, one per game, written by the first `--final-run` (commit them) |
| `tests/snapshots/tools.json` | the tool schemas and descriptions as the agent sees them |

## 9. Suggested first hour

1. `domain/types.py`: the vocabulary (15 minutes).
2. `engine/play.py:play`, then `engine/campaign.py`: the game itself.
3. `policies/core.py:Learner.choose` with `advisor/preprocess.py` and `advisor/model.py`: what
   "a scripted learner" means, including the model gate that explains the E result.
4. `interfaces/tools/toolbox.py`, the table of contents of the 15 tools; then follow
   `query_pool`, `submit_batch`, `fit_and_rank` and `similar_designs`, the tools the agents
   actually use most.
5. `eval/runner.py:run_llm_episode` and `eval/loop.py:run_episode`: how an agent plays.
6. Run something small to watch it move (no API key needed; `run-scripted` needs the built
   tables in `data/processed/`):

```bash
uv run replaylab run-scripted --policy learner_sim --game N-dev --seed 3
```

```bash
uv run replaylab-eval conditions
```

Good tests to read alongside: `tests/test_engine_play.py` (the loop), `tests/test_engine_campaign.py`
(the rules), `tests/test_tools_leaks.py` (what may never appear in a response),
`tests/test_eval_loop.py` with `tests/fakes_llm.py` (the loop against a scripted fake model).
