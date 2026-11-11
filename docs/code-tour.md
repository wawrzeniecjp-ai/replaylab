# Code tour

A reading guide to `src/replaylab`. Sections 1 and 2 are the five-minute version: the six nouns,
then one episode followed from the command line down to a single round. The rest is reference:
the entry points, one tool call step by step, every guard in one table, the layers, and a
suggested first hour.

For the data flow and the leak rules in more depth, see `docs/user_guide/00-architecture.qmd`;
for a hands-on REPL session on the real tables, see `docs/walkthrough.md`.

## 1. Six nouns

Everything in the code is one of these, or serves one of them.

| noun | where | one sentence |
| --- | --- | --- |
| `Campaign` | `engine/campaign.py` | the rules of the game: `submit`, `wait`, `stop`, `settle`; validates each move, advances the clock, records everything in the store |
| `CampaignState` | `domain/types.py` | an immutable snapshot of a campaign: designs, batches, and *delivered* outcomes only |
| `ReplayEnvironment` | `engine/replay.py` | one game assembled for one seed: campaign, store, hidden scoring context |
| `Player` | `engine/episode.py` | who plays: `play_round(campaign)`, then `finish(campaign)`. `ScriptedPlayer` runs a policy; `LLMPlayer` (`eval/llm_player.py`) runs a model with tools |
| `Episode` | `engine/episode.py` | one game and one player; `Episode.play()` is **the** game loop |
| `Toolbox` | `interfaces/tools/toolbox.py` | the 15 tools an LLM player can call, over one campaign |

The one fact that explains most of the layout: **only `Campaign` can buy, and only delivered
outcomes ever reach a `CampaignState`.** A policy, a tool and an advisor function all see the
same snapshot, so none of them can read the outcome of a design that was not bought.

## 2. One episode, top down

Follow `replaylab-eval run` down four levels. Each level does setup, makes one call where the
work happens, and cleans up.

**Level 1, the command** (`eval/cli.py:run`):

1. parse the options;
2. `prepare(...)` (`eval/setup.py`): load the configs, make every refusal (dirty tree, unknown
   model or price, no pilot, projection over the cap), create the model client and the spend
   guard;
3. **`run_matrix(specs, deps)`**: play the episodes;
4. print the summary.

**Level 2, the matrix** (`eval/runner.py:run_matrix`): skip the episodes already recorded, then
for each of the others (at most four at a time) call **`play_episode(spec, deps)`** and append
its record to `episodes.jsonl`. An episode that raises is not recorded, so a rerun plays it
again. `play_episode` is three lines:

```python
async with open_episode(spec, deps) as episode:  # create the resources
    played = await episode.play()  # play the game
return _record(spec, deps, episode, played)  # build the record
```

`open_episode` creates everything one episode needs, in this order, and closes it afterwards:

* the **seed**, from game and trial only (`eval/seeds.py`), so every player of a trial faces
  the same pool under the same aliases;
* the **game** (`interfaces/runtime.py:build_runtime` → `engine/replay.py:build_replay_campaign`):
  pool and hidden outcomes, per-seed aliases, the `ReplayBackend` that holds the answers, the
  SQLite `CampaignStore`, the `Campaign`; plus the preprocessor and similarity index the tools
  use;
* the **player**:
  * `scripted:<policy>` → `ScriptedPlayer(as_choose(policy, seed))`;
  * an LLM condition → an MCP server with only the condition's tools
    (`interfaces/server.py:build_server`), a client connected to it in memory, and an
    `LLMPlayer` talking to the model through it (`eval/mcp_bridge.py`).

**Level 3, the game loop** (`engine/episode.py:Episode.play`), the same for every player:

```python
while not campaign.is_finished():
    if not await player.play_round(campaign):  # False: the player ends the episode
        break
await player.finish(campaign)  # LLM: ask for the closing summary
campaign.settle()  # collect results still outstanding
return Played(score=env.score(), stopped=campaign.stopped)
```

**Level 4, one round**, inside the player:

* `ScriptedPlayer.play_round`: ask the policy for a batch and `campaign.submit` it; an empty
  batch stops the campaign.
* `LLMPlayer.play_round`: run model turns (one model call, then every tool call of the reply,
  in order) until a turn ends with the clock moved (a batch bought or a round waited out), the
  campaign finished, the model replying without tools (`agent_ended`) or the turn limit
  (`max_turns`). The conversation, the turn count and the cost carry over between rounds. The
  agent buys through the `submit_batch` tool, which ends up in the same `Campaign.submit`
  (section 4).

The ladder (`eval/ladder.py:run_game`) and `replaylab run-scripted` enter at level 3 directly,
with a `ScriptedPlayer`. `replaylab-eval scripted` enters at level 2 (`run_scripted_matrix`
calls `play_episode`).

How the pieces fit together:

```text
                replaylab-eval run / pilot / scripted          (eval/cli.py, eval/setup.py)
                                  |
            eval/runner.py   run_matrix -> play_episode -> open_episode
                                  |        (spend guard, records, final-run permit)
                                  v
                 engine/episode.py   Episode.play: the game loop
                     |                                   |
            LLMPlayer (eval/llm_player.py)       ScriptedPlayer (engine/episode.py)
              |   eval/llm*.py: model API                |   policies/: choose a batch
              v                                          |
       eval/mcp_bridge.py -> interfaces/server.py        |
              |                                          |
       interfaces/tools/  Toolbox: the 15 tools  --> policies/, advisor/
              |                                          |
              v                                          v
            engine/campaign.py   Campaign: submit / wait / settle / stop
              |                         |
       store/store.py (SQLite)   backends/replay.py (the hidden outcomes)
```

Everything above `engine/episode.py` is about *running experiments*; everything below
`Campaign` is the *game*.

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
| `python scripts/replay_tool_calls.py`, `replay_agent_loop.py`, `compare_scripted.py` | scripts | refactor checks: replay recorded tool calls, or whole agent episodes with their recorded replies, and compare two runs |
| `python data/build_tables.py`, `build_clusters.py`, `build_folds.py` | `data/` → `datasets/` | build the processed tables and the frozen split from the raw CSV |

`python -m replaylab` and `python -m replaylab.eval` are the same two CLIs (`__main__.py`).

## 4. One tool call (`submit_batch`)

1. `eval/llm_player.py:_execute_call` hands the call to `eval/mcp_bridge.py:McpBridge.call`,
   unless its arguments are not JSON or the condition does not offer the tool (both answered
   with an `ERROR: ` message, without reaching the server).
2. `interfaces/server.py`: the `submit_batch` closure validates the arguments against its
   schema and calls `Toolbox.submit_batch`; a `ToolFailure` becomes an MCP error that tells the
   agent how to fix the call.
3. `interfaces/tools/toolbox.py:Toolbox.submit_batch` (one line) →
   `interfaces/tools/campaign.py:submit_batch`, which calls `engine/campaign.py:Campaign.submit`
   and turns each engine error into a message saying what to do instead.
4. `Campaign.submit` validates (stopped, rounds, size, duplicates, unknown, already bought,
   budget), records the batch in the store, hands it to `backends/replay.py:ReplayBackend`,
   collects what is delivered, then moves the clock one round. With `delay_rounds > 0`
   outcomes arrive on later rounds (`wait`).
5. The response is a Pydantic model from `interfaces/schemas.py`: it cannot carry a field
   that is not declared, which is part of the leak protection.

Every tool has the same shape: a one-line `Toolbox` method → a function in the module of its
group (`pool`, `campaign`, `models`, `estimates` under `interfaces/tools/`) → `advisor/`,
`policies/` or `Campaign`. `fit_and_rank` → `advisor/preprocess.py` + `advisor/model.py:fit_model`;
`similar_designs` → `advisor/similarity.py:SimilarityIndex.nearest`; `propose_batch` →
`policies/factory.py:make_policy` → `policies/`; `estimate_hit_rate` → `advisor/rates.py`, and
for a next batch `advisor/calibration.py`; `stop_advice` → `advisor/stop.py`;
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
| `eval/llm_player.py:LLMPlayer` → `eval/spend.py:SpendGuard.check` | no model call once the cap is reached | the check comes *before* each call; the total persists in `spend.json`, so a resumed run keeps counting |
| `eval/cli.py:run` → `eval/spend.py:refuse_if_over` | no run without a pilot projection (unless `--skip-pilot`), and none whose projection exceeds `--max-usd` | the cost is known before anything is spent |
| `eval/setup.py:prepare` | bad concurrency, unknown model or price, missing API key, unknown condition | every setup refusal comes before the first model call |
| `eval/setup.py:check_uncommitted` | recording from an uncommitted tree (escape: `--allow-uncommitted`, which marks every record `git_dirty`) | `config_hash` does not cover the code; the recorded commit is the only record of it |
| `eval/llm_player.py` → `ResponseTruncated` | recording an episode whose reply hit the output limit | a cut reply is a failure, not the agent ending the game |
| `eval/runner.py:open_episode` (via `llm_player.py:unwrap`) | a `SpendCapReached` hidden inside an `ExceptionGroup` | the cap must stop the run, not be logged as one failed episode |
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
| `engine/` | `episode.py`, `campaign.py:Campaign` | the game loop (`Episode`, `Player`, `ScriptedPlayer`) and the rules of the game (clock, validation, delay); `replay.py` wires a game and resumes a killed one; `scoring.py` scores a settled campaign | domain, store, backends, datasets, config |
| `advisor/` | `preprocess.py`, `model.py` | pure functions over a `CampaignState`: preprocessing (orientation applied here), logistic model with its gate (2 hits, 8 purchases), similarity, Beta hit rates, calibration of predictions, stop rules, custom analysis | domain, datasets |
| `policies/` | `core.py` | `choose(state, k, rng) -> ids`: `RandomPolicy`, `FixedMetric`, `Learner`, `LearnerSim`, `WithStop`; `purchased.py` has the purchased-scaling variants; `factory.py:make_policy` builds any of them by name (the tools, the CLIs and the ladder all use it) | domain, advisor |
| `interfaces/` | `tools/toolbox.py:Toolbox` | the 15 tools, one line each, grouped into `tools/pool.py`, `campaign.py`, `models.py`, `estimates.py`; `tools/catalog.py` the names, groups and the descriptions the agent reads; `schemas.py` the responses; `server.py` MCP; `runtime.py` one game + one seed; `cli.py` | everything above; **nothing imports it** |
| `eval/` | `runner.py`, `llm_player.py` | the experiment harness (next section) | anything; nothing imports it |

## 7. `eval/`, file by file

| file | role |
| --- | --- |
| `cli.py` | the `replaylab-eval` commands, one function each |
| `setup.py` | what a command sets up before its first episode: `prepare` and `runner_deps` assemble everything a run needs, `llm_config_hash` is the registered hash, and every refusal (dirty tree, unknown model or price, a second final run) happens here |
| `runner.py` | one episode (`open_episode`: the resources; `play_episode`: play and record) and the matrix of them: expand, skip completed, run concurrently, record; scripted matrix too |
| `llm_player.py` | the LLM player: model turns and tool calls round by round, stop reasons, closing summary |
| `mcp_bridge.py` | MCP tools ↔ OpenAI-style function specs; compact JSON, `ERROR: ` prefix, 20,000-character cut |
| `llm.py`, `llm_anthropic.py`, `llm_types.py` | provider adapters behind the `LLMClient` protocol; retries; `ResponseTruncated` |
| `models.py`, `spend.py` | `config/models.yaml` and `config/prices.yaml`; `SpendGuard` with the persisted `spend.json`; pilot projection |
| `conditions.py` | `config/conditions.yaml`: a condition is a tool allow-list (A, B1, B2, C, ...) |
| `prompts.py` | loads `prompts/system.md`; the fixed first user message |
| `records.py` | the `episodes.jsonl` format, `scores_dict`, partial-line repair on read, `config_hash`, git info |
| `seeds.py` | `episode_seed(game, trial)` |
| `ladder.py` | the ladder report |
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
2. `engine/episode.py` (the game loop and the scripted player), then `engine/campaign.py`:
   the game itself.
3. `policies/core.py:Learner.choose` with `advisor/preprocess.py` and `advisor/model.py`: what
   "a scripted learner" means, including the model gate that explains the E result.
4. `interfaces/tools/toolbox.py`, the table of contents of the 15 tools; then follow
   `query_pool`, `submit_batch`, `fit_and_rank` and `similar_designs`, the tools the agents
   actually use most.
5. `eval/runner.py:open_episode` and `eval/llm_player.py:LLMPlayer`: how an agent plays.
6. Run something small to watch it move (no API key needed; `run-scripted` needs the built
   tables in `data/processed/`):

```bash
uv run replaylab run-scripted --policy learner_sim --game N-dev --seed 3
```

```bash
uv run replaylab-eval conditions
```

Good tests to read alongside: `tests/test_engine_episode.py` (the loop), `tests/test_engine_campaign.py`
(the rules), `tests/test_tools_leaks.py` (what may never appear in a response),
`tests/test_eval_loop.py` with `tests/fakes_llm.py` (the LLM player against a scripted fake model).
