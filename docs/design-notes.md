# Design notes

The numbers behind every choice that the task cards ask to be recorded, plus the
places where the T20 card was ambiguous and what was decided instead.
Date: 2026-10-13.

## 1. Phase 0 analyses (`notes/week0/`, gitignored)

### Feature review (T02)

`notes/week0/feature_review.md`, rules: keep a metric if coverage >= 0.9, drop it
if the hit rates with and without a value differ by more than 0.05 (both groups
>= 30 designs).

* **All seven features in `config/features.yaml` pass** (coverage 0.992-0.998, no
  missingness leak): `boltz2_iptm`, `boltz2_ipsae`, `boltz2_min_ipsae`,
  `boltz2_plddt`, `shape_complimentarity_boltz2_binder_ss`, `esmfold_plddt`,
  `proteinmpnn_score`. The pre-registered list is therefore unchanged.
* Additional metrics that also pass but were **not** added: `boltz2_pdockq`,
  `boltz2_pdockq2`, `boltz2_lis`, `boltz2_ptm`, `boltz2_complex_*`,
  `ted_confidence`, `isoelectric_point`, `molecular_weight`, `seqidentity`,
  `proteinmpnn_seq_recovery`, `redesigned_proteinmpnn_score`. Adding them now
  would be another post-hoc change to the feature set; they are candidates for a
  later, separately documented experiment.
* Excluded: the AFDB/CATH/PDB structure-comparison metrics
  (`aligned-length*`, `evalue*`, `rmsd*`, `seqidentity*`, `tm-score*`) at ~0.20
  coverage, and `evalue*`/`rmsd*`/`seqidentity*`/`tm-score*` additionally because
  their missingness predicts the label (AUC ~0.54: present vs absent hit rates
  0.016 vs 0.174).

`first_column` is unchanged: `boltz2_ipsae` for nipah (chosen **after** seeing
outcomes in the exploratory analysis — a known pre-registration violation, kept
as-is rather than silently "fixed") and `esmfold_plddt` for egfr (reasoning only:
the boltz2 columns have 0 coverage on egfr).

### Author checks (T03)

`notes/week0/author_checks.json`, nipah pool (1030 designs, 206 authors):

* nearest-neighbour pairs are the **same author 44.7% of the time** (random would
  be a few percent), and among hit pairs 51.6% (33/64) are same-author.
* Consequence: sequence similarity partly encodes *authorship* (shared design
  pipelines), not only binding-relevant similarity. `learner_sim` is therefore
  evaluated on its end-to-end hit count, not assumed to be a pure signal; the
  ladder (and later the eval campaigns) is what justifies or rejects it.

### LLM smoke test (T05)

`notes/week0/llm_smoke.txt` against the local endpoint `127.0.0.1:8080`
(llama.cpp, Qwen3-1.7B Q4_K_M): tool-call round trip OK, reasoning lives in
`reasoning_content`, follow-ups with and without reasoning OK, **no reasoning
pass-back needed**, temperature/seed/`tool_choice` probes OK. This is the model
the phase-7 agent loop can be developed against offline.

## 2. Clusters: 631, not 632

Recorded in a comment on `EXPECTED_CLUSTER_SUMMARY` in
`src/replaylab/datasets/split_tables.py`; summary: the pair
(`green-lynx-leaf`, `strong-lion-marble`) has cosine similarity exactly
138/230 = 0.6, i.e. distance exactly on the 0.4 threshold. "At most 0.4" connects
it; strict float rounding in the exploratory run dropped the edge and produced
one extra component (632). Exact integer arithmetic over all 1030x1030 pairs
agrees with the code: **631 clusters**, largest 125, 570 singletons.
`data/splits/*` did not change (the rebuild is byte-identical).

## 3. T20 decisions where the card was ambiguous

1. **The `SimilarityIndex` is built per seed, not once per game.** The card says
   "built once per game ... not per seed", but aliases depend on the seed
   (decision D4, `sha256(seed|id)`), so an index keyed by one seed's aliases
   points at the *wrong sequences* for every other seed and `learner_sim` would
   silently learn from scrambled similarity. The `Preprocessor` (feature values
   only, order-invariant) is built once per game as the card asks.
   *Confirmed in practice:* the first `reproduce_legacy.py` run reused one policy
   object across all 100 seeds (so one cached index), and `legacy learner_sim`
   scored 20.0 instead of ~39 while the index-free `legacy learner` matched 25.93
   vs 25.9 exactly. With a policy factory (fresh per seed) the bug disappears.
   *Re-verified 2026-10-14 as a pre-phase-6 gate:* seed 0 -> seeds 1..5 -> seed 0
   again on one shared `LadderGame` gives byte-identical score cards, so
   `ladder.py` (which builds the index inside `policies(seed)`) never shared one
   and the 26-minute re-run is not needed. Two permanent tests keep it honest:
   `test_a_seed_result_does_not_depend_on_run_order` and
   `test_the_similarity_index_is_rebuilt_per_seed`.
2. **`config/features.yaml` gained entries for the eight stop-game targets.**
   The pack's `features.yaml` only covered nipah and egfr, but `games.yaml` plays
   ten more targets, so the ladder could not run at all. Entries were chosen by
   column availability alone (outcomes not consulted): `first_column =
   esmfold_plddt` everywhere (full coverage, the egfr reasoning-only rule);
   `human-serum-albumin` gets the full nipah-style list (boltz2 coverage 0.98),
   the other seven only `esmfold_plddt` + `proteinmpnn_score` (boltz2 coverage 0,
   exactly the egfr situation).
3. **Pools without cluster rows are scored without `distinct_families`.**
   `data/splits/clusters.csv` only covers targets with >= 300 designs (nipah,
   egfr); `score()` raises for any hit it cannot place, so `run_game` passes
   `context=None` when the pool has no clusters. The ladder CSV then has an empty
   `distinct_families` for stop games.
4. **Held-out refusal** is enforced in `run_game` and again in `run_ladder`
   (ValueError before anything loads). `load_outcomes` already refuses held-out
   reads without `final_run=True`, so the split has three guards.
5. **The legacy script's random stream** is `make_rng(seed, round)`, the same
   stream the library uses; the card does not say. Tie-breaking in the legacy
   top-k is `(-score, design id)` like the library.

## 4. Frozen configuration (T20 step 3)

Comparisons on **N-dev only**, 100 seeds, mean hits (written into
`config/frozen_v1.yaml`):

| variant | learner | learner_sim |
| --- | --- | --- |
| lbfgs, class_weight None | **18.18** (sd 0.99) | 14.52 (sd 3.28) |
| lbfgs, balanced | 18.02 (sd 1.01) | 15.21 (sd 4.04) |
| liblinear, None | 15.67 (sd 1.89) | 15.32 (sd 4.00) |
| liblinear, balanced | 17.66 (sd 1.08) | 15.26 (sd 3.52) |

* **solver**: lbfgs beats liblinear by 2.51 hits on `learner` (> 2, so the rule
  keeps the winner even though lbfgs is also the simpler default); on
  `learner_sim` the gap is 0.8 (< 2) and the simpler lbfgs is kept. → **lbfgs**.
* **class_weight**: None vs balanced differs by 0.16 on `learner` and 0.69 on
  `learner_sim` (both < 2) → prefer the simpler → **None**.
* Both winners are the library defaults, so `frozen_v1` currently equals the
  defaults. The point of freezing is that later phases may not silently change it.
* Observation kept for the eval phase: on N-dev `learner_sim` (14.5) trails
  `learner` (18.2) by ~3.7 hits at w = 0.5. The full ladder shows whether that
  holds across games; the legacy reproduction (section 5) is the other data point.

Stop-rule parameters in `frozen_v1.yaml` equal the library defaults
(`low_yield`: 20 purchases, floor 0.03; `decay`: window 15, drop 0.05);
`load_frozen` refuses a file that disagrees, because the advisor's stop rules are
not configurable yet.

## 5. Legacy reproduction (T20 step 2)

`python scripts/reproduce_legacy.py`, game `legacy` (whole nipah pool, split
`all`, 4 rounds x 25), 100 seeds. The legacy learner exists only in that script:
7 metrics + length + one-hot class, pool medians, no orientation, scaler on the
purchased rows, `LogisticRegression(C=0.1, max_iter=1000, lbfgs)`, ipSAE
fallback below 2 hits, no purchase floor, `floor(0.8*k)` by score then random.

| policy | expected | tolerance | got (mean, 100 seeds) | sd | verdict |
| --- | --- | --- | --- | --- | --- |
| random | 10.2 | +/- 1.5 | 9.86 | 2.57 | OK |
| fixed `boltz2_iptm` | 10.0 | +/- 1 | 10.00 | 0.00 | OK |
| fixed `boltz2_ipsae` | 16.0 | +/- 1 | 16.00 | 0.00 | OK |
| fixed `boltz2_min_ipsae` | 18.0 | +/- 1 | 18.00 | 0.00 | OK |
| legacy learner | 25.9 | +/- 1.5 | 25.93 | 3.04 | OK |
| legacy learner_sim | 38.9 | +/- 1.5 | **38.49** | 2.90 | OK |

**All six reproduce.** The interesting path to get there: the first run scored
`legacy learner_sim` 20.03 (DIFF) because one policy instance (and its cached
`SimilarityIndex`) was reused across all 100 seeds while aliases change per seed
(D4) -- the similarity signal was scrambled for seeds 1..99. A side-by-side
10-seed check made it concrete: cached index = 22.2 mean, per-seed index = 37.5.
The fix (a policy factory, fresh per seed) initially appeared to "not work"
because an editor buffer saved late and the re-run read the old file off disk;
a run after the save gave 38.49. Lesson recorded: `grep`-verify a file on disk
*before* launching a long run that depends on a fresh edit.

Scaling comparison (same legacy script, both scalers, 100 seeds):

| policy | purchased-row (legacy) | pool-wide (library default) | delta |
| --- | --- | --- | --- |
| legacy learner | 25.93 | **27.00** | pool better by 1.07 (< 2) |
| legacy learner_sim | **38.49** | 32.22 | purchased better by **6.27 (> 2)** |

* For the plain learner the difference is under the card's 2-hit threshold and
  the library keeps its pool-wide default (already the library behaviour).
* For `learner_sim`, purchased-row scaling wins by **6.27 hits -- more than 2**,
  so per the card the library must **not** be changed without asking first.
  Asked and decided 2026-10-13 (section 7): keep the library pool-wide and
  track both variants in the ladder, so the `learner_sim` rows in section 6
  (pool-wide) and the `learner_sim_purchased` rows are compared side by side
  until the gap is explained.

## 6. The ladder (T20 step 1)

`python scripts/run_ladder.py` (frozen config loaded): **11,400 rows** = 114
(game, policy) pairs x 100 seeds, 11 games (all but `N-held`), ~26 minutes.
Full tables: `reports/ladder.csv` and `reports/ladder.md`. Highlights from
ladder v2 -- after the two 2026-10-13 decisions (v1 numbers in parentheses):

| game | best policy (mean hits) | worst policy (mean hits) |
| --- | --- | --- |
| N-dev | learner_sim_purchased (26.4) | fixed:proteinmpnn_score (4.0) |
| N-dev-delay2 | learner_sim_purchased (25.1) | fixed:proteinmpnn_score (4.0) |
| E | learner_sim (10.4, ties with its +stop) | learner(_sim)_purchased (4.5) |
| stop games (S-*) | ties (budget = pool size, all buy everything) | +stop can only tie or stop early |

**Stop rules after dropping `decay` (decision 2):** every `+stop` variant now
scores *exactly* its unwrapped twin on every game -- on E, `learner_sim+stop`
went from **1.0 (v1) to 10.4 (v2)**. `low_yield` fires only in the four zero-hit
stop games, stopping at 20 of 30 purchases with zero hits lost: a pure saving.
Nothing is given away anymore.

**Scaling tracker (decision 1, both variants kept)** -- mean hits, pool-wide ->
purchased-row:

| game | learner | learner_sim |
| --- | --- | --- |
| N-dev | 18.2 -> 15.8 (-2.4) | 14.5 -> **26.4 (+11.9)** |
| N-dev-delay2 | 17.3 -> 13.7 (-3.6) | 16.4 -> **25.1 (+8.7)** |
| E | 7.4 -> 4.5 (-2.9) | 10.4 -> 4.5 (-5.9) |
| legacy game (section 5) | 27.0 -> 25.9 (-1.1) | 32.2 -> **38.5 (+6.3)** |
| stop games (S-*) | 0.0 (ties) | 0.0 (ties) |

* The purchased-row boost **replicates on both nipah ladder games** (+11.9,
  +8.7) and on the legacy game (+6.3): it is no longer a single-script artifact.
* But it **reverses on E** (-5.9) and plain `learner` is mildly worse with
  purchased scaling everywhere, so the picture is an interaction between the
  scaler, the similarity fusion and nipah's 7-feature space -- precisely why
  both variants stay tracked until that is explained.
* Notably, purchased+sim (26.4) is the best policy on N-dev, beating plain
  `learner` (18.2); the pooled scaler was what made `learner_sim` look weak
  there in v1 (14.5).

Observations:

* `learner` and `learner_sim` beat every fixed metric on the two nipah games;
  the model + similarity machinery earns its keep there.
* The stop games have `budget == pool size`, so every non-stopping policy buys
  the whole pool and ties at the pool's total hits; only `+stop` variants can
  stop early (now always without losing hits). The games therefore measure
  stopping behaviour, not ranking.

Stop-rule false-stop rates (second table in `reports/ladder.md`; N-dev, 100 seeds,
rules evaluated at every round boundary regardless of the wrapper):

| policy | rule | runs fired | rate |
| --- | --- | --- | --- |
| learner_sim | low_yield | 0/100 | 0% |
| learner_sim | decay | 98/100 | **98%** |
| random | low_yield | 5/100 | 5% |
| random | decay | 77/100 | **77%** |

* `low_yield` is well behaved (needs 20 purchases below a 3% rate).
* `decay` (window 15, drop 0.05) fires in the large majority of runs -- one hit
  of decline inside the window is 1/15 = 0.067 > 0.05, exactly the caveat written
  in `advisor/stop.py`. This is the phase-5 measurement that caveat asked for.
  In ladder v1 it gave away a round almost every game (E: `learner_sim+stop` 1.0
  vs 10.4). **Decided 2026-10-13 (section 7): dropped from `WithStop`'s
  default** -- ladder v2 shows every `+stop` variant tying its base policy.
  The parameters stay frozen, the rule stays measurable in this table and
  explicitly selectable; cross-val retuning is the documented alternative if it
  is ever wanted back.

## 7. Decisions taken (2026-10-13) and what is still open

**Decided after reading sections 5 and 6:**

1. **Keep both scaling variants.** The purchased-row boost may be an artifact,
   so nothing is switched: the library keeps pool-wide scaling as its default,
   and the ladder now runs `learner_purchased` and `learner_sim_purchased`
   next to `learner` and `learner_sim` (same pipeline, only the standardizing
   means/stds come from the delivered rows -- see `purchased_scale` in
   `policies/purchased.py`). Both variants stay in every ladder run until the 6.27-hit
   gap is explained. First tracked run (section 6): +11.9 / +8.7 on the two
   nipah games, +6.3 on legacy, but -5.9 on E -- replicated, not universal,
   so tracking continues.
2. **Drop `decay` from the stop rules' default.** It false-stops 77-98% of
   runs, so `WithStop` now defaults to `rules=("low_yield",)`. `decay` is still
   computed by `stop_advice` (its false-stop rate stays in the report), can be
   requested explicitly, and its parameters remain frozen -- retuning it by
   cross-validation is the documented alternative if it is ever wanted back.
   The spec's section 3.5 should be updated to match.

**Still open:**

* ~~Replace the table in section 2.4 of the spec~~ — paste-ready replacement
  written: `docs/spec-updates.md` (sections 2.4 and 3.5; the project document
  itself lives outside this repo and still needs the paste).
* ~~Send the pack author the T20 outputs~~ — the bundle is assembled in
  `docs/phase6-handoff.md` (make test, ladder tables, legacy table,
  features.yaml, llm_smoke.txt); sending it triggers the phase-6 pack
  (MCP server + CLI).
* ~~Pin `spec/openapi.prod.json`~~ — done 2026-10-13: fetched from
  `https://devs.adaptyvbio.com/api/v1/openapi.json` (openapi 3.1.0,
  info.version 0.0.2), pinned in `spec/openapi.sha256`, the T04 test
  un-skipped (484 passed, 0 skipped); phase 10 unblocked.
* Whether to add the extra reviewed features (section 1) as a second experiment.

## 8. Pre-registration: does a longer horizon change what E measures? (2026-10-14)

Written **before** the run, at the pack author's request: declare the question
and the horizons first, report every horizon, never the best one.

**Question.** E fits a model in exactly one of its four rounds, so every
difference between the learner variants (7.44 / 10.39 / 4.51 / 4.45) is decided
by a single round. The pre-registered questions are: *at what purchase count
does the learner fit regularly, and at what purchase count does it cross
`random`?*

**Facts the horizon is derived from** (measured before the run):

* E's pool (`egfr`, split `all`) holds **826 designs** with **134 hits**
  (16.2%). `random` scores 9.59 hits in 60 purchases = 0.16 per purchase, the
  pool rate exactly.
* `fixed:esmfold_plddt` -- which is byte-for-byte what the learner's fallback
  buys -- scores 5.00 in 60 purchases and **4.00 after the first 45**, so
  the cold-start deficit against `random` (7.33 at 45 purchases) is **3.3 hits**:
  the pack author's 3-4 estimate checks out. *(Correction written after the run:
  this bullet originally claimed "1 hit after 45, deficit ~6". That came from
  misreading a three-seed trace -- 1 hit is the value at 30 purchases, 4 hits at
  45. The 100-seed curve in `reports/e_long.csv` is the authority.)*
* Bounds for the horizon: below ~1/3 of the pool (275) and below the hit
  supply (134).

**Horizon.** `E-long` = same target, split and batch size as E, **8 rounds x
15 = 120 purchases** (14.5% of the pool, under both bounds). Rounds 1-4 are
required to reproduce E exactly, so one run contains every shorter horizon: the
report gives cumulative hits at *every* round, i.e. at every horizon, instead of
picking one.

**Rules.**

1. Only the horizon changes. No policy, model or `frozen_v1` change rides
   along; adding exploration to the fallback would be a design change, a
   separate experiment justified on N-dev and the other games -- not on E.
2. `E-long` carries `exploratory: true`: excluded from the default ladder, from
   `frozen_v1` and from every decision rule. The headline E numbers in sections
   6 and 7 stay as they are.
3. The report (`reports/e_long.md`) gives mean cumulative hits against
   purchases for **every** ladder policy with bootstrap intervals over seeds,
   `random` and `fixed:esmfold_plddt` as references, and the first round each
   learner variant passes the gate.
4. The intervals are over seeds and measure policy randomness only: every seed
   replays the same pool, so they say nothing about variation across targets.
5. E is the only transfer target with a pool this size, so this cannot answer
   whether the learner transfers *across targets*. That stays a separate
   question.

**Gate:** the first four rounds of `E-long` reproduce E's score cards exactly
(`test_e_longs_first_four_rounds_reproduce_e`, marker `data`, needs
`data/processed/`).

**Result (run 2026-10-14: 9 policies x 100 seeds, 378 s,
`reports/e_long.{csv,md}`):**

* **Gate passed.** At 60 purchases this run reproduces every committed E mean
  exactly (nine for nine), so the longer horizon touched nothing upstream.
* **The cold start is exactly three rounds.** The gate opens in round 4 for
  **100% of seeds** and stays open for rounds 5-8. Rounds 1-3 are identical for
  all six learner variants -- 1.00 / 1.00 / 4.00 hits after 15 / 30 / 45
  purchases, always below `random` (2.38 / 4.77 / 7.33).
* **Crossing `random`:** `learner_sim` crosses in round 4 itself (10.39 vs 9.59
  at 60); `learner` and `learner_sim_purchased` cross at ~90 purchases (round 6,
  14.37 and 16.59 vs 14.29); `learner_purchased` at ~105 (round 7, 17.02 vs
  16.97). At 120 purchases: `learner_sim` 34.26, `learner_sim_purchased` 31.70,
  `learner` 26.23, `learner_purchased` 24.07, `random` 19.53,
  `fixed:esmfold_plddt` 11.00.
* **The scaler flip is real and horizon-independent.** Purchased-row scaling
  costs `learner_sim` 5.9 hits at 60 (4.45 vs 10.39) and still 2.6 at 120
  (31.70 vs 34.26), while on N-dev the same variant wins by 11.9. A longer
  horizon does not explain it away -- it is the scaler, not the horizon.
* No policy approaches the 134-hit supply at 120 purchases, so there is no
  convergence artefact, and `+stop` never fires on E (rows identical to their
  base policies).

## 9. Pre-registration: the agent-vs-scripted criterion (2026-10-28)

Written **before trials 5-9 of either model exist** (deepseek `runs/ds2` and
mimo `runs/mimo2` hold only the pilots' trials 0-4), at git `4649e7b`.
Disclosure: the pilot trials 0-4 are known to us (deepseek: A ahead of B1,
paired gap +3.2); the confirmatory weight sits on the full trials 0-9, and
nothing below may change once they are played. Same discipline as section 8.
*Amended the same day, still before any trial 5-9 exists (the commit that adds
`compare --ci` and `--first-trial`): the interval rule is made unambiguous, the
end of the extension is fixed, and validity gates and seed rules are added.*

**Scope.** N-dev, trials 0-9, conditions A and B1, models `deepseek` and
`mimo`, each judged against the four scripted policies played on the same
trials in the same records file. Comparisons are **paired per trial**:

    replaylab-eval compare --records runs/<dir>/episodes.jsonl --game N-dev \
        --a B1 --b scripted:learner_sim_purchased --resamples 10000 --seed 0 --ci 0.9

(`--seed 0` fixes the bootstrap; `--ci 0.9` is a two-sided 90% interval, i.e.
each bound is a one-sided 5% test -- the usual level for a non-inferiority
margin; the margin below is in hits.)

**Primary pass/fail -- the headline question: does the agent match the best
scripted player?**  Bar: `scripted:learner_sim_purchased`, the pre-registered
best scripted variant on N-dev (26.43 mean, section 6).

The verdict is read from the 90% interval [low, high] of the paired difference
(B1 - scripted), never from the mean alone (at n=10 the paired SD of ~5 hits
puts the standard error near 1.6, so a mean near -2 decides nothing):

* **PASS (matches):** low > **-2 hits**. The margin is this repo's own 2-hit
  threshold for deciding that one policy beats another (section 5), not a
  number chosen to fit the pilot.
* **Exceeds:** additionally, low > 0.
* **FAIL:** high < -2.
* **Inconclusive** (the interval contains -2) **pre-commits an action:** both
  models extend to trials 0-19 (same commands, same caps, criterion
  unchanged); that extension is the only permitted escalation. Still
  inconclusive at n=20: reported as inconclusive, nothing further is run.
* **Headline:** the `deepseek` verdict. `mimo` is the replication; if the two
  disagree, both verdicts are reported and the disagreement is the finding.

**Secondary (reported, never pass/fail):**

* `B1` vs `scripted:learner` (bar 18.18: does the agent add anything over a
  plain scripted learner?);
* `B1` vs `scripted:random` (6.03: does it use the features at all?);
* `A` vs the same scripted set (environment-only is a legitimate condition);
* `B1` vs `A` (do the advisor tools help? the pilot says A ahead, declared).

**Rules.** No criterion, prompt, tool or policy edits until both models have
completed trials 0-9 under this note. `mimo` is judged by the **same** bars --
a replication, not a second chance -- and reported separately; model-vs-model
differences are descriptive only. Any relaxation requires a new section below,
written before the data that would benefit from it is seen.

**Setup before judging.** Each records file must hold the four scripted
policies on the same trials (`compare` reads one file). `runs/ds2` has them
for trials 0-9; `runs/mimo2` has none yet:

    replaylab-eval scripted --run-dir runs/mimo2 --games N-dev --trials 10 \
        --policies random,learner,learner_sim,learner_sim_purchased

(`--trials 20` if the extension is triggered, in both files.)

**Validity gates** -- checked per model before any hit count is read; a run
that fails one is fixed and replayed, never analysed:

1. Failed episodes (no record: `LLMError`, `ResponseTruncated`, ...) are at
   most 10% of that model's episodes. More suggests failures that depend on
   the episode, and replaying them would bias the sample.
2. Every agent episode ends `finished` or `agent_stopped`; an `agent_ended`
   with rounds still left counts as a harness problem.
3. A few transcripts are read in the viewer (`replaylab-eval view`) and the
   `flags` totals are reported next to the verdict.

**Seeds and reuse.**

1. *Pairing.* `seed = episode_seed(game, trial)` depends only on game and
   trial (D4), never on condition or model. Every comparison pairs episodes by
   trial within one records file.
2. *One draw per cell.* Each (model, condition, game, trial) is played once.
   A failed episode leaves no record and is replayed (rule 4 of the handoff);
   a recorded episode is never deleted or replayed to get another draw.
3. *One configuration.* Every agent episode in a confirmatory comparison
   carries the same `config_hash`. The hash covers model, price, prompts,
   conditions, tool descriptions and limits, but not tool input schemas or
   policy code, so the `git_sha` may differ only by commits that change
   neither. Check: `git diff --stat <pilot sha>..<run sha> -- src prompts
   config`. From `d4f7e88` (trials 0-4) to this amendment it lists only the
   replay viewer and two CLI options whose defaults reproduce the old
   behaviour (`compare --ci`, `--first-trial`; tested). Runs before
   `d4f7e88` (`runs/ds`, `runs/mimo`) are excluded.
4. *Seen seeds.* Trials 0-4 were inspected (pilots, viewer) before this note,
   and tool descriptions changed after seeing them. A sensitivity line reports
   the primary comparison on trials 5-9 (5-19 if extended) alone; if its
   verdict differs from the full set, both are reported and the note says so.
5. *A fresh block after any change.* Any prompt, tool, condition or policy
   edit after this note makes a new configuration, whose confirmatory claims
   use a block of trials never played under any configuration, chosen with
   `--first-trial`. The next block is **trials 100-119**; trials 20-99 stay
   free for development.
6. *Scripted players are deterministic per seed* under `frozen_v1`, so their
   records may be reused across runs (the viewer's `--ghosts` relies on it).
   Agent episodes are not: a repeated seed is a new sample of the model, and
   rule 2 still applies.
7. *Sampling.* `config/models.yaml` sets no temperature or sampling seed, so
   the providers' defaults apply and a replayed episode is not reproducible
   token for token; the records file, not a rerun, is the result.
