# Pre-registration: how much did the orientation defect cost?

Written 2026-10-30, **before any episode of this experiment exists** (checked: no `E` records
outside trials 100–119). Descriptive, like design-notes §16: an interval, no pass/fail
threshold. Nothing in §15, §16 or `docs/prereg-anthropic.md` is touched.

## 1. The question

`docs/caveats.md` entry 4 records a defect in the toolkit, not the model: the player saw raw
feature values while the fitted model reported weights in oriented space, and nothing said
which columns were flipped. In the played runs, 113 of 607 `query_pool` calls that sorted on
`proteinmpnn_score` asked for the **worst** designs with no bounding filter, concentrated in
E (62 of 160 episodes), and 86 batches were submitted directly after one.

Because buying the worst-scoring designs *hurts* the agent, E's EXCEEDS verdict was obtained
despite the defect, and the caveat claims **unmeasured headroom**. This experiment measures
it. The result is interesting either way: a gain says tool-surface design is worth real hits;
a null says the agent routed around the defect and the surface mattered less than it looks.

## 2. The arms

| arm | condition | code |
| --- | --- | --- |
| treatment | `B1` | the fixed working tree |
| control | `B1ctl` | a git worktree at **`aaeb1af`**, the commit before the fix |

`B1` and `B1ctl` carry **identical tool lists** (`config/conditions-orient.yaml`); they differ
only in which code serves them. Two labels are needed because the runner keys an episode by
`(game, condition, trial)`, so one label cannot hold both arms in one records file.

**The control reproduces the published E configuration exactly.** Verified rather than
assumed: `git diff ba56753 aaeb1af -- prompts/ config/conditions.yaml
src/replaylab/interfaces/` is empty, so the control's agent-facing surface is the one that
produced §15's results.

The treatment differs from the control in exactly the fix: `describe_campaign` and
`column_summary` report each column's `higher_is_better`; `fit_and_rank` and
`fit_custom_model` report weights in the raw-value direction; `query_pool`'s description
states the hazard without naming a column; the system prompt says the scores do not all point
the same way. Values shown to the player are still raw (D7).

## 3. The block, the game and the models

* **Game E**, the pool where the defect bit hardest (62 of 160 episodes against N-dev's 12 of
  190): two features, one of them the inverted one, so it carries half the signal.
* **Trials 140–159**, a fresh block. Trials 100–119 are used by §15/§16 and 120–139 were
  deliberately left unplayed, so a new idea starts at 140.
* **Both models**, `deepseek` and `mimo`, each in its own run directory, since the runner keys
  episodes without the model.
* **n = 20 per arm per model**, fixed here and not extended after looking. Any extension needs
  its rule written here first.

## 4. What is measured

**Primary (descriptive).** The paired difference in hits, treatment minus control, on the same
seeds: `B1 − B1ctl`, 90% bootstrap interval, 10,000 resamples, seed 0, trial range 140–159 —
the same settings as every other comparison in this project, via the same `compare` command.
Reported per model, and the two pooled. An interval containing 0 means no detectable headroom;
no threshold is registered.

**The manipulation check, and why it matters more than usual.** Before any hit count is read,
count in each arm the `query_pool` calls that sort on `proteinmpnn_score` descending with no
bounding filter on that column — the exact pattern caveat 4 counted. The fix is only *about*
that behaviour, so:

* if the rate does **not** fall in the treatment arm, a null result means *the fix did not
  change behaviour*, not *the surface does not matter*, and must be reported that way;
* if the rate falls and hits do not move, that is a real and more interesting null: the agent
  was already routing around the defect.

Registering this in advance stops a null from being read as whichever of the two is more
convenient afterwards. `scripts/count_orientation_calls.py` computes it from a records file.

**Secondary, descriptive.** Both arms against `scripted:learner_sim` on the same seeds, so
this block can be placed beside §15's numbers; and the per-round curves, to see whether any
gain appears where the neighbour search compounds (rounds 3–4).

## 4b. Registered before the block: the hashes, the baseline and the scripted rows

All of this is in hand before a single agent episode of the block exists.

**Configuration hashes**, from a dry run of each arm (deepseek; the model key is part of the
hash, so mimo's pair differs and is recorded when its dry run is made):

| arm | `config_hash` | tools |
| --- | --- | --- |
| treatment `B1` | `aa5e128a6755` | 14 |
| control `B1ctl` | `e2ef0758ac42` | 14 |

They must differ: the fix changes the prompt and three tool descriptions, which is the
manipulation. The control was verified to load the pre-fix code, not merely to be labelled
that way (`query_pool`'s description still reads "a lower ProteinMPNN score is better", and
`ColumnInfo` has no `higher_is_better` field).

**The pre-fix baseline for the manipulation check**, counted on the played §15 block (trials
100-119, condition B1) with `scripts/count_orientation_calls.py`:

| model | worst-first calls | episodes touched |
| --- | --- | --- |
| deepseek | 7 | 6 of 20 (30%) |
| mimo | 15 | 8 of 20 (40%) |

The control arm on trials 140-159 should land near these; the treatment arm is expected to
fall. Both numbers are read before any hit count of the new block.

**The scripted rows on trials 140-159** (played, free, identical in both run directories):
`random` 9.80, `learner` 7.55, `learner_sim` **10.30**, `learner_sim_purchased` 4.70. These
sit within 0.3 hits of the same policies on trials 100-119 (9.90 / 7.55 / 10.55 / 4.65), so
the fresh block is not an easier or harder draw, and `learner_sim` is again the best scripted
player.

## 5. Validity gates, checked before any hit count is read

The same gates as §15: every episode `finished` (no truncation, no error-stop); each arm's
`config_hash` constant within the arm and different between arms (they must differ — that is
the manipulation); the scripted rows on these seeds identical in both run directories; and no
uncommitted change to `src`, `prompts` or `config` in either tree while the block is played.
A failed gate voids the block, which is then replayed on a fresh one.

## 6. Cost

From the recorded pilot means on E (deepseek 0.0172, mimo 0.0109 USD per episode) and the
usual 1.25 margin: 40 deepseek episodes ≈ 0.86 USD, 40 mimo ≈ 0.55 USD. The caps below are set
well above that and are hard limits, checked before every model call.

## 7. The commands

Scripted baselines on the block (free, no API key, already played):

```bash
replaylab-eval scripted --run-dir runs/orient-ds --games E \
    --policies random,learner,learner_sim,learner_sim_purchased --trials 20 --first-trial 140
```

The treatment arm, from this working tree:

```bash
replaylab-eval run --run-dir runs/orient-ds --model deepseek --games E \
    --conditions B1 --trials 20 --first-trial 140 --max-usd 2 \
    --conditions-config config/conditions-orient.yaml --skip-pilot
```

The control arm, from the worktree at `aaeb1af`, pointing at **this** tree's conditions file
so both arms read one definition:

```bash
PYTHONPATH=../adaptyv-orient-control/src replaylab-eval run --run-dir runs/orient-ds \
    --model deepseek --games E --conditions B1ctl --trials 20 --first-trial 140 --max-usd 2 \
    --conditions-config config/conditions-orient.yaml --skip-pilot
```

Then the same three with `--run-dir runs/orient-mimo --model mimo`, and:

```bash
replaylab-eval compare --records runs/orient-ds/episodes.jsonl --game E \
    --a B1 --b B1ctl --ci 0.9 --resamples 10000 --seed 0 --trial-range 140-159
```


---

# Result (2026-10-30): no headroom measured, because the manipulation failed

Played as registered: E trials 140-159, 20 trials per arm per model, both models.

## Gates: all pass

80 of 80 agent episodes `finished`; 20 distinct trials per arm; `config_hash` constant within
each arm and different between arms (deepseek `aa5e128a6755` / `e2ef0758ac42` exactly as
registered; mimo's pair, `d4651dd4bbff` / `3429038d2521`, was not registered in advance
because the model key enters the hash, and was verified against fresh dry runs of both arms);
the scripted rows identical across the two run directories (80 of 80); both working trees
clean in `src`, `prompts` and `config`. Non-zero flags over all 80 episodes: 5 tool errors and
2 bad-JSON replies, no truncation, no fabricated calls.

## The manipulation check: FAILED

The worst-first rate did not fall. Paired per episode, fixed minus control:

| model | worst-first calls/episode | 90% CI | episodes touched, fixed vs control |
| --- | --- | --- | --- |
| deepseek | **+0.25** | [-0.25, +0.80] | 9/20 vs 8/20 |
| mimo | **+0.25** | [-0.25, +0.85] | 8/20 vs 6/20 |

Both intervals straddle zero and if anything point the wrong way. **The fix did not change
the behaviour it targets**, so section 4's rule applies: whatever the hits do, this block
cannot say whether the orientation surface matters.

## The primary outcome: null, and uninterpretable as headroom

| cell | fixed - control | 90% CI | W/L/T |
| --- | --- | --- | --- |
| deepseek | +2.60 | [-2.00, +7.05] | 12/8/0 |
| mimo | -2.25 | [-5.50, +0.90] | 6/13/1 |
| **pooled, 40 pairs** | **+0.17** | **[-2.70, +3.00]** | 18/21/1 |

Dead null, with the two models pointing opposite ways. Per the registered rule this is **not**
evidence that the tool surface does not matter. The headroom caveat 4 claims remains
**unmeasured**.

## Why the manipulation failed, and why that is my error

The pre-fix description already read: *"Metric values are raw, so a lower ProteinMPNN score is
better."* E's one inverted column **is** `proteinmpnn_score`. So the control arm already
carried a correct, specific warning naming exactly the column in play, and the fix added
generality (true for any pool and any inverted column) plus structured `higher_is_better`
fields -- neither of which adds *information* on this particular pool.

The experiment therefore compared "specific warning" against "general warning plus structured
fields" on the one pool where the specific warning applies. That is a badly targeted test, and
it was foreseeable: caveat 4 quotes that very sentence. A test of what caveat 4 worries about
-- a pool whose inverted column no prose warning names -- needs a pool with a *different*
inverted column, which none of the played games has, or an arm with the warning removed
entirely. Either needs its own pre-registration and a fresh block; trials 140-159 are now used.

## Two things worth keeping

**A finding about tool design, not about this fix.** Roughly 20% of sorts on the inverted
column ask for the worst designs *in both arms* -- that is, with an explicit, correctly named
warning sitting in the tool description. A prose warning in a description does not reliably
change what an agent does. Recorded as `docs/caveats.md` entry 5.

**An independent replication of section 15.** On a fresh seed block, both arms and both models
clear the bar again by a wide margin: deepseek +11.75 [+8.70, +14.65] and +9.15 [+6.95,
+11.35], mimo +13.50 [+11.35, +15.50] and +15.75 [+13.60, +17.80] against
`scripted:learner_sim`. E's EXCEEDS was not a property of trials 100-119.
