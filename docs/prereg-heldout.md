# Pre-registration: the held-out Nipah-G game (`N-held`)

Written 2026-11-11, before any episode of `N-held` was played. No record in any run directory
names `N-held`, no lock file exists for it, and its outcomes have never been read by any code
path (the runner, `serve`, the ladder and the replay page all refused it until this file was
written).

The decisions below were made by the author: the trial count and the secondary bar on
2026-11-11, the conditions, models and analysis by carrying over the Anthropic test's
amendment of 2026-11-01 unchanged. Development is over: no knowledge layer or other tool
change will come before this run.

## 1. What `N-held` is

Fold 2 of the Nipah glycoprotein G designs: **343 designs**, the third of the target that the
frozen split (`data/splits/folds.csv`, `data/splits/nipah_heldout_ids.txt`) set aside on
2026-10-12. Every development game on Nipah-G (`N-dev`, `N-dev-delay2`, the tuning and
confirmatory blocks) played folds 0 and 1 only. The prompt, the tool descriptions, the
conditions and the scripted policies were all developed on those folds and on E, so `N-held`
is the one Nipah-G pool on which nothing was tuned.

**The game**, as committed in `config/games.yaml` on 2026-10-12, before any result of any
kind existed: **4 rounds of 8, budget 32**. That is 9.3% of the pool, against `N-dev`'s 60 of
687 (8.7%): the same share of the pool and the same horizon, not the same batch. The only
change made now is `final_test: true`, which puts the game under the run-once lock (section 5);
it does not change what the game is.

## 2. Disclosure: what is known about the pool

* **Its hit count is known by construction.** The fold builder balances the three folds to
  within 2 designs and 2 hits of each other, and the split summary prints the counts:
  343 designs and 34 hits in fold 2 (`docs/walkthrough.md`). So the base rate (about 9.9%) is
  known, as it is for every fold. No per-design outcome of fold 2 has been read.
* **The pool's ids are public**, in `nipah_heldout_ids.txt`. Players never see them: as in
  every game, designs reach a player only as per-seed aliases.
* **N-dev's results are known**, including its verdicts and per-round behaviour. That is the
  point of a held-out test: the question is whether what was found on folds 0 and 1 holds on
  fold 2.

## 3. The configuration it runs on

The same frozen configuration as the Anthropic final test (`docs/prereg-anthropic.md`
section 7): the orientation fix is in, so this configuration is **not** the one N-dev's
confirmatory blocks were played on. The registered hashes:

| | hash |
| --- | --- |
| `config_hash`, `deepseek` | `0fecb4a7e74a` (as registered for the Anthropic test) |
| `config_hash`, `mimo` | `c2c8dac7e136` (as registered for the Anthropic test) |
| `frozen_hash`, `N-held` | `c1149b131e11` |

The game does not enter `config_hash`, so the two model hashes are shared with the `AB-`
games; `frozen_hash` covers the game and its feature configuration and is pinned in
`tests/test_registered_hashes.py`.

## 4. The run

| | decision |
| --- | --- |
| game | `N-held`, on its own lock |
| conditions | **A, B1, B2** |
| models | `deepseek` and `mimo`, each in its own run directory |
| trials | **40** per condition per model: block **0-39** (never played) |
| scripted, same seeds | `random`, `learner`, `learner_sim`, `learner_purchased`, `learner_sim_purchased` |
| caps | 6 USD per run directory |

**Why 40 trials.** `N-held` is one game, so nothing is pooled. The per-episode paired sd of
`B1 - A` measured on N-dev (9.51 deepseek, 11.78 mimo, on 60 purchases) scales to about 5 to
6.3 hits on 32 purchases, by the purchase-rate factor the Anthropic pre-registration uses. At
20 trials the 90% half-width would be about 1.9 to 2.3 hits; at 40, about 1.3 to 1.6. N-dev's
pooled `B1 - A` of +4.62 hits scales to about +2.5, so 40 trials resolve the primary if the
effect carries over, and 20 likely would not.

**Cost.** About 0.029 USD per agent episode on N-dev (`runs/confirm1`, `runs/mimo2`), on a pool
twice this size and a budget almost twice as large. 120 episodes per model project to about
3.5 USD, an upper bound; the caps leave headroom and are checked before every model call.

## 5. The analysis, fixed in advance

All comparisons are paired by trial, use `replaylab-eval compare` (one paired bootstrap,
`replaylab.eval.stats.paired_bootstrap`) at **90%, 10,000 resamples, resampling seed 0**, on
trials 0-39, and are reported **per model**. The two models are a replication of each other:
if they disagree, both verdicts are reported and the disagreement is the finding.

1. **Primary -- do the advisor tools help on unseen designs?** `B1 - A`. Decision rule: the
   tools **help** if the lower bound of the interval is above zero; **hurt** if the upper
   bound is below zero; otherwise **inconclusive**. No effect-size margin, for the reason the
   Anthropic amendment gives: this is an ablation of our own toolkit, not a contest.
2. **Secondary -- does N-dev's verdict against the best scripted player hold?**
   `B1 - scripted:learner_sim_purchased`, the bar N-dev was judged against
   (`docs/design-notes.md` section 9). Section 9's rule, with its 2-of-60 margin rescaled by
   purchase rate to **1.07 hits** on 32 purchases, as in `docs/prereg-anthropic.md` section 5:
   * **MATCHES:** lower bound > -1.07;
   * **EXCEEDS:** additionally, lower bound > 0;
   * **FAILS:** upper bound < -1.07;
   * **INCONCLUSIVE:** otherwise.
3. **Descriptive, no threshold attached.** `B1 - B2` (what `similar_designs` adds),
   `A - scripted:learner_sim_purchased`, `B1 - scripted:learner_sim`, `B1 - scripted:learner`,
   and the full scripted ladder on the same seeds, so any policy beating the agent appears in
   the same table as the verdicts.

**The hierarchy is a hierarchy.** An inconclusive primary is not rescued by the secondary, and
the secondary may not be promoted to the headline afterwards. Every comparison above is
reported whatever its sign.

**No extension.** If an interval straddles its threshold, the result is inconclusive. Adding
trials after seeing it would be optional stopping, and trials 40 and up of `N-held` stay
unplayed.

## 6. The guard

`N-held` is `split: heldout` and `final_test: true`; the configuration now refuses any
held-out game that is not a final test. So:

* `load_outcomes` reads fold 2 only with `final_run=True`;
* `serve`, the campaign CLI, the ladder and the replay page refuse it outright;
* only `replaylab-eval scripted|pilot|run --final-run` play it, from a committed `src`,
  `prompts` and `config`, under the lock `reports/final-test/N-held.json`, which pins
  `frozen_hash`, the commit, and each leg's trial block and conditions. An identical rerun
  resumes; anything else is refused as a second draw.

## 7. The commands, as registered

From a clean checkout of one commit on `main`, the same commit for every leg.

The scripted ladder first (free, no key):

```bash
replaylab-eval scripted --run-dir runs/held-ds --games N-held \
    --policies random,learner,learner_sim,learner_purchased,learner_sim_purchased \
    --trials 40 --final-run
```

Then the agent: the pilot plays the first 12 episodes of the matrix (trials 0-3, every
condition) and measures their cost; the run plays the rest. The pilot is the start of the
final run, not a separate draw: it takes the lock, and its episodes are kept.

```bash
replaylab-eval pilot --run-dir runs/held-ds --model deepseek --games N-held \
    --conditions A,B1,B2 --trials 40 --max-usd 6 --n 12 --final-run
replaylab-eval run --run-dir runs/held-ds --model deepseek --games N-held \
    --conditions A,B1,B2 --trials 40 --max-usd 6 --final-run
```

Then all three again with `--run-dir runs/held-mimo` and `--model mimo`. Finally, per model:

```bash
replaylab-eval compare --records runs/held-ds/episodes.jsonl --game N-held \
    --a B1 --b A --ci 0.9 --resamples 10000 --seed 0 --trial-range 0-39
replaylab-eval compare --records runs/held-ds/episodes.jsonl --game N-held \
    --a B1 --b scripted:learner_sim_purchased --ci 0.9 --resamples 10000 --seed 0 \
    --trial-range 0-39
```

**Gates, checked before any hit count is read:** every agent episode `finished` or
`agent_stopped` (no `max_turns`, no `agent_ended`); one `config_hash` per model, equal to the
registered one; the lock's `frozen_hash` equal to `c1149b131e11`; the scripted rows present
for every trial; and no uncommitted change in `src`, `prompts` or `config` in the tree that
played them.
