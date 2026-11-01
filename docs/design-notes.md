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

**Results (evaluated 2026-10-28: `--ci 0.9`, 10000 resamples, seed 0).**
Trials 10-19 did not exist when this block was first written. Reproduce with
`compare ... --trial-range 0-9` (sensitivity: `--trial-range 5-9`); the range
is needed once the extension's trials sit in the same file.
*Deviation, disclosed:* gates 1 and 2 were checked before any hit count was
read, but gate 3 (the transcript pass) was not -- it is still pending below.

| model | primary (B1 - `learner_sim_purchased`) | 90% CI | W/L/T | sensitivity, trials 5-9 | gates |
| --- | --- | --- | --- | --- | --- |
| deepseek (`runs/ds2`) | **-9.2** | [-12.4, -5.6] | 2/8/0 | -9.4 [-14.6, -4.0] | 0 failures, 20x `finished`, 5 tool-errors, 1 config hash |
| mimo (`runs/mimo2`) | **-5.2** | [-10.1, -0.4] | 4/6/0 | -10.4 [-16.4, -4.0] | 1/20 failures (replayed), 20x `finished`, 2 tool-errors, 1 config hash |

* **deepseek: FAIL** -- the interval lies wholly below -2 and the sensitivity
  agrees (same verdict on the unseen half).
* **mimo: inconclusive** -- the interval [-10.1, -0.4] contains -2 (FAIL
  needs high < -2), so the pre-committed extension to trials 0-19 fires.
* **Headline: NO-GO.** The headline is the deepseek verdict, and it is final
  at n=10. The rule extends *both* models, so deepseek's trials 10-19 are
  played for symmetry and reported descriptively; they cannot change its
  verdict. The extension decides mimo's verdict only (the replication).
* **The extension runs the same commands and configuration:** only the spend
  cap (`--max-usd`) is raised, to pay for the extra episodes; the turn limit,
  `max_tokens` and the config hash are unchanged.
* **Seed-reuse rule 4 fires for mimo: the verdicts differ.** All of trials
  0-9: inconclusive. The unseen trials 5-9 alone: FAIL (-10.4 [-16.4, -4.0]).
  On the seen trials 0-4 mimo's mean difference is 0.0, on 5-9 -10.4;
  deepseek is steady (-9.0 on 0-4, -9.4 on 5-9). With halves of n=5 this may
  be chance, but it is the pattern rule 4 exists for: the tool descriptions
  were edited after watching trials 0-4. It also means "mimo did better than
  deepseek" holds on the seen half only.
* The mimo, headline, extension and rule-4 items above were written after the
  extension had started (`runs/ds2` held trials >= 10 for A and B1), without
  looking at their outcomes. They resolve what the rule already says and add
  no new criterion.
* **Secondary at n=10:** both agents beat `random` decisively (deepseek +11.9
  [+8.5, +15.8], mimo +15.9 [+11.3, +20.6]); neither is distinguishable from
  plain `learner` (deepseek -0.6 [-3.8, +3.0], losing 7 of 10 trials; mimo
  +3.4 [-1.2, +7.9]); condition A fails the bar on both models (-9.1, -9.7);
  B1 - A shows no detectable difference on either model (deepseek -0.1
  [-5.1, +5.5]; mimo +4.5 [-1.3, +10.6]) -- descriptive only, per scope.
* The pilot's "A ahead on deepseek" did not survive trials 5-9 (full-set
  B1 - A = -0.1): rule 4's sensitivity line earns its keep.
* Extension results: resolved below (2026-10-29). The viewer transcript pass
  (gate 3) is still pending.

**Extension results (n=20, evaluated 2026-10-29: gates first, `--ci 0.9`,
10000 resamples, seed 0).** Gates: `runs/ds2` 40/40 episodes, 0 failures, 40x
`finished`, 1 config hash, $1.153 spent (cap 1.5); `runs/mimo2` 40/40, 1/40
failures (the replayed connection error), 40x `finished`, 1 config hash,
$1.197 spent (cap 1.9); the git range since the pilot still lists only the
replay viewer and the CLI options.

| model | primary (B1 - `learner_sim_purchased`), n=20 | 90% CI | W/L/T | sensitivity 5-19 | verdict |
| --- | --- | --- | --- | --- | --- |
| deepseek | **-6.2** | [-9.0, -3.35] | 6/14/0 | -5.27 [-8.67, -1.8] | **FAIL** -- interval wholly below -2; the extension agrees |
| mimo | **-2.45** | [-5.95, +1.0] | 10/10/0 | -3.27 [-7.6, +1.0] | **FAIL on the point rule; the straddle persists** -- see below |

* **The straddle fired again for mimo at the planned maximum.** The
  registration tied the straddle clause to "the only permitted escalation",
  which is now exhausted -- no trials beyond 19 exist under this plan.
  Reported honestly: the point estimate misses the margin (-2.45 < -2), while
  the interval still includes non-inferior values up to +1.0, so the miss is
  thin and is neither strengthened nor retracted. deepseek's miss is
  unambiguous under either clause.

**Secondary at n=20 (report only):**

* vs `random`: decisive for both agents -- deepseek +14.25 [11.35, 17.3] and
  mimo +18.0 [14.95, 21.05], each 20/20 (condition A: +11.8 and +11.2, also
  20/20).
* vs plain `learner`: mimo **+6.05 [2.8, 9.2]** (13/20 wins, decisive);
  deepseek +2.3 [-0.4, +5.1] (parity).
* B1 - A: mimo **+6.8 [2.6, 11.0]** (14/20) -- the advisor tools help mimo
  decisively; deepseek +2.45 [-0.95, +5.85] -- suggestive only.
* A vs the bar: -8.65 [-10.45, -6.8] and -9.25 [-11.25, -7.15] -- the
  environment-only condition fails the bar on both models, decisively.

**Headline.** Neither model matches the pre-registered best scripted player --
decisively for deepseek, thinly (point rule only, interval straddling) for
mimo. Both crush `random`. mimo clears plain `learner` and its advisor tools
demonstrably help it; deepseek is at parity on both counts. The advisor effect
is model-dependent, which is exactly why this note judged the models
separately under one criterion.

## 10. Capability probe with a frontier model (2026-10-29)

**Question.** Is section 9's NO-GO a limit of the models (capable, not
frontier)? Descriptive only: no new criterion, and section 9's verdicts stand.

**Plumbing.** `eval/llm_anthropic.py` (commit `87d99a0`) adds the Anthropic
Messages API: streamed calls, assistant turns sent back exactly as returned
(signed thinking blocks), cache writes billed at their own price, and no
fallback model (a model switch mid-episode would spoil the episode).
`replaylab-eval smoke` (commit `e81c149`) plays a few real turns of trial 999
with no record and checks thinking pass-back, cache reads and cost.

**Opus 5.5 could not be tested:** the provider declined the requests at the
smoke stage. Decided before trying Sonnet: one more attempt only, Sonnet 5.5;
it plays whole episodes or the hypothesis is reported as untested.

**Sonnet 5.5** (`sonnet`, config hash `167e3086e8cd`): adaptive thinking,
summarized, effort high (the model's default; at medium it did not think in
the smoke run). Prices from the API reference, to be confirmed.

**Results** (`runs/sonnet1`, N-dev B1, trials 0-9, `--ci 0.9`, 10000
resamples, seed 0). Not confirmatory: trials 0-3 were inspected before the
decision to extend. Gates: 10/10 recorded, 0 failures, 10x `finished`, 0 tool
errors, one config hash; $1.24 in total.

| Sonnet B1 vs | mean diff | 90% CI | W/T/L |
| --- | --- | --- | --- |
| `learner_sim_purchased` | **-10.4** | [-13.2, -7.0] | 1/0/9 |
| `learner` | -1.8 | [-4.5, +1.5] | 3/0/7 |
| `learner_sim` | +1.3 | [-0.9, +3.8] | 3/3/4 |
| `random` | +10.7 | [+8.2, +13.4] | 10/0/0 |

Mean hits on the same seeds: Sonnet 16.5, deepseek 17.7, mimo 21.7,
`learner_sim_purchased` 26.9.

**Observations.**
* Trial 1 (31 hits, last batch 15/15) came from buying sequence neighbours of
  a cluster of hits by hand. No other trial went above 19.
* More thinking did not mean more hits (the two trials with the most thinking
  scored 17 and 14).
* Same habits as the flash models: never a purchased-scaling policy, one or
  two `fit_and_rank` calls per episode, `propose_batch` dropped after round 2.

**Reading.** A clearly stronger model did no better, which argues against a
capability gap as the main cause. Not ruled out: that the strongest model
would pass, since it could not be run. Next lever: how the agent understands
and uses the tools (neutral descriptions only, tuned on development trials,
judged on the fresh block 100-119).

## 11. Tuning log (development trials 20-99; never the fresh block)

Each iteration is a new configuration (new config hash), played with deepseek
on development trials and judged against a **paired control**: the previous
configuration on the same seeds, played from a git worktree at its commit.
Without the control, a change of seeds looks like a tuning effect.

### Iteration 1 (2026-10-29): propose_batch interface and status-tool wording

**Changes.**
* `da5c42b`: `policy` became an enum (`learner`, `learner_sim`,
  `learner_purchased`, `learner_sim_purchased`, `random`, `fixed`), `fixed`
  takes a separate `feature` parameter, every parameter has a description,
  and the description lists one neutral bullet per policy plus the fallback
  before a model can be fitted.
* `53861c2`: the default policy is named in the schema only, no longer in
  the text; `describe_campaign`, `get_history`, `pending_batches` and
  `campaign_summary` each say what they return that the others do not.
  Ideas came from asking deepseek to review the descriptions; points without
  support in the transcripts were not taken (renaming `propose_batch`,
  "use this when" hints, which would advise rather than describe, and the
  `wait` wording: `wait` was never called on N-dev).

**Runs** (N-dev, B1, deepseek, `--ci 0.9`, 10000 resamples, seed 0).

| run | configuration | config hash | trials | mean hits |
| --- | --- | --- | --- | --- |
| `runs/control1` | before iteration 1 (`2391d10`, = section 9) | `da8b881a3af2` | 20-29 | 26.0 |
| `runs/tune1` | after both commits (`53861c2`) | `b457e246517b` | 20-29 | 25.2 |
| (not kept) | after `da5c42b` only | not recorded | 20-24 | 19.8 |

Gates: 10/10 episodes recorded per run, all `finished`, one config hash per
run; tune1 cost $0.26. The `da5c42b`-only run left no records on disk; its
hits (36, 27, 11, 13, 12, trial order unknown) are from hand-kept notes
and are reported for completeness only.

| comparison, trials 20-29 | mean diff | 90% CI | W/T/L |
| --- | --- | --- | --- |
| tuned - old | -0.8 | [-5.6, +4.3] | 4/1/5 |
| tuned - `learner_sim_purchased` | -0.9 | [-4.7, +3.0] | 4/1/5 |
| old - `learner_sim_purchased` | -0.1 | [-3.9, +3.7] | 5/1/4 |
| tuned - `learner` | +6.9 | [+3.0, +10.9] | 7/2/1 |

**Result: no detectable effect.** The first 5 tuned trials (25.2 on 20-24)
looked like a gain over section 9 (20.5 on 0-19), but the untuned
configuration scores 26.0 on the same seeds while `learner_sim_purchased`
stays near 26 on both blocks: trials 20-29 suit the agent's hand-picking
better. On trials 20-24 the three configurations scored 23.2 (old), 19.8
(`da5c42b` only) and 25.2 (both): run-to-run sampling on fixed seeds moves a
5-trial mean by several hits, so "the cheap edits pulled their weight" is not
supported.

**Behaviour did not change either:**

| per episode | old | tuned |
| --- | --- | --- |
| `propose_batch` calls | 0.4 | 0.3 |
| batches built by hand (of 4) | 3.9 | 4.0 |
| `query_pool` calls | 12.6 | 12.6 |
| `learner_sim_purchased` chosen (whole run) | 0 | 1 |

Edits to a tool the agent barely calls have little to act on. The agent's
play is hand-picking from `query_pool` and following `similar_designs`
neighbours of hits; the next lever has to change how it picks (the system
prompt's account of the workflow), judged the same way: paired control, at
least 10 seeds, on a development block not used before (30 onwards).

### Iteration 2 (2026-10-29): say what the methods compute -- reverted

**Changes** (`44a6d99`, config `9d476dc1d5d2`). `fit_and_rank` described its
inputs, orientation, pool standardization, L2 regularization (C = 0.1), its
outputs and that a fit is a snapshot; each `propose_batch` bullet said what
the policy computes (the same model as `fit_and_rank`, the exploration split,
the similarity rank blend, standardization on delivered purchases recomputed
every round). `describe_campaign` dropped "after every submit" (agents called
it after 1 of 80 submits; `submit_batch` already returns rounds and results).
All neutral: tests kept ranking words out.

**Runs** (N-dev, B1, deepseek, trials 30-39, `--ci 0.9`, 10000 resamples,
seed 0). `runs/tune2`: iteration 2, $0.28. `runs/control2`: iteration 1
(`53861c2`, `b457e246517b`) from the worktree, $0.26. Gates: 10/10 each, all
`finished`, one config hash per run.

| comparison, trials 30-39 | mean diff | 90% CI | W/T/L |
| --- | --- | --- | --- |
| iteration 2 - iteration 1 | **-5.5** | [-10.2, -1.1] | 2/2/6 |
| iteration 2 - `learner` | +1.1 | [-1.2, +3.7] | 4/2/4 |
| iteration 1 - `learner` | +6.6 | [+3.3, +10.1] | 7/2/1 |
| iteration 2 - `learner_sim_purchased` | -7.8 | [-10.4, -4.9] | 2/0/8 |
| iteration 1 - `learner_sim_purchased` | -2.3 | [-5.8, +1.4] | 4/0/6 |

Mean hits: iteration 2 19.2, iteration 1 24.7, `learner_sim_purchased` 27.0,
`learner` 18.1. Iteration 2 lost the high-scoring episodes (iteration 1's 37,
31 and 32 became 18, 17 and 14) and landed on `learner`.

**Behaviour.** Still 40/40 batches built by hand. `propose_batch` 0.7 calls
per episode (was 0.3), policies `learner_sim` x6 and `random` x1, no
purchased variant. In rounds 3-4 more designs came straight from
`fit_and_rank` rankings (43, was 25; 30% hits); neighbours from
`similar_designs` hit less often (12/27, was 23/35). Hits per round fell in
round 2 (8.6 to 6.4) and round 4 (6.3 to 3.6).

**Reading (plausible, not proven).** Explaining the model made
`fit_and_rank`, which offers pool scaling only, look authoritative; the agent
leaned on its ranking and played like `learner`, where under iteration 1 it
trusted its own hit-cluster follow-ups more. One paired comparison at 90%
leaves about a 1-in-10 chance of an interval this far below zero by noise.

**Decision: reverted in full**, the `describe_campaign` line included, so the
base is iteration 1 again with its config hash `b457e246517b` (checked by a
dry run) and its existing runs (tune1, control2) stay valid references.
Trials 30-39 are used; the next iteration starts at 40.

## 12. Pre-registration: confirmatory run of the frozen configuration (2026-10-29)

Written at git `940520b`, **before any trial >= 40 has been played under any
configuration** (checked over every records file in `runs/`). Nothing below
may change once trial 100 is played. Same discipline as sections 8 and 9.

**Disclosure.** The configuration was chosen after looking at development
data (section 11). Known before writing: iteration 1 on trials 20-39 (tune1
and control2) scored -1.6 [-4.2, +1.0] against `learner_sim_purchased` and
+6.75 [+4.15, +9.35] against `learner`; the untuned configuration did about as
well on trials 20-29, and section 9's configuration scored -6.2 on trials
0-19. Agent results vary a lot between seed blocks; that is why this run uses
20 fresh trials from the start and pre-commits one extension.

**Frozen configuration.** Tuning iteration 1: commit `53861c2`, whose `src`,
`prompts` and `config` are identical to `940520b`'s; model `deepseek`
(`deepseek-flash`, `config/models.yaml` as committed); config hash
**`b457e246517b`**.

**Scope.** N-dev, condition **B1**, model `deepseek`, trials **100-119**,
judged against the four scripted policies played on the same trials in the
same records file, paired per trial:

    replaylab-eval compare --records runs/confirm1/episodes.jsonl --game N-dev \
        --a B1 --b scripted:learner_sim_purchased --resamples 10000 --seed 0 --ci 0.9

Condition A, mimo and Sonnet are not part of this test (A and mimo were not
tuned; any later run of them on these trials is descriptive only).

**Primary pass/fail: section 9's rule, unchanged.** Bar:
`scripted:learner_sim_purchased`. From the 90% interval [low, high] of the
paired difference (B1 - bar):

* **PASS (matches):** low > -2 hits;
* **Exceeds:** additionally, low > 0;
* **FAIL:** high < -2;
* **Inconclusive** (the interval contains -2) pre-commits **one** extension:
  trials 100-139 (n = 40), same commands and configuration, only the spend
  cap raised. The verdict is then read on 100-139. Still inconclusive at
  n = 40: reported as inconclusive, nothing further is run.

**Secondary (reported, never pass/fail):** B1 vs `scripted:learner`,
`scripted:learner_sim` and `scripted:random`; the behaviour profile of section
11 (calls per episode, policies chosen, batches built by hand).

**Validity gates**, checked before any hit count is read; a run that fails
one is fixed and replayed, never analysed:

1. Failed episodes (no record) are at most 10% of the episodes; failed ones
   are replayed by rerunning the same command.
2. Every B1 episode ends `finished` or `agent_stopped`.
3. Every B1 record carries config hash `b457e246517b`, and
   `git diff --stat 940520b..<record git_sha> -- src prompts config` is empty.
4. The four scripted policies are present for every trial judged.

**Seeds.** Section 9's seed rules apply. Trials 100-139 are reserved for
this test; trials 40-99 stay free for development (the next one, a new
condition such as an analysis tool, needs its own fresh block from 140 on).

**Commands** (from the main repo; each prints nothing until it ends):

    replaylab-eval pilot --run-dir runs/confirm1 --model deepseek --games N-dev \
        --conditions B1 --trials 20 --first-trial 100 --n 20 --max-usd 1.5
    replaylab-eval scripted --run-dir runs/confirm1 --games N-dev \
        --policies random,learner,learner_sim,learner_sim_purchased \
        --trials 20 --first-trial 100

Extension, only if inconclusive: the same two commands with `--trials 40`
(and `--n 40 --max-usd 3` for the pilot); existing episodes are skipped.

**Results (evaluated 2026-10-29, gates first, then the primary as written).**
`runs/confirm1`. Gates: (1) 20/20 B1 episodes recorded, no errors file;
(2) 20x `finished`; (3) one config hash `b457e246517b`, played at `e9534a4`,
`git diff --stat 940520b e9534a4 -- src prompts config` empty; (4) the four
scripted policies on every trial 100-119. Flags: 4 tool errors. Cost $0.575.

| B1 vs | mean diff | 90% CI | W/T/L |
| --- | --- | --- | --- |
| `learner_sim_purchased` (primary) | **-6.9** | **[-8.8, -4.95]** | 2/1/17 |
| `learner` | +1.2 | [-0.8, +3.35] | 7/2/11 |
| `learner_sim` | +5.05 | [+3.0, +7.0] | 18/0/2 |
| `random` | +13.75 | [+12.0, +15.55] | 20/0/0 |

Mean hits: B1 19.15, `learner_sim_purchased` 26.05, `learner` 17.95,
`learner_sim` 14.10, `random` 5.40.

* **Verdict: FAIL.** The interval lies wholly below -2; final at n = 20, so
  the extension does not fire and trials 120-139 stay unplayed.
* **The development estimate was optimistic.** Iteration 1 scored -1.6 on
  trials 20-39 and -6.9 here, close to section 9's -6.2 on trials 0-19: the
  20-39 block suited the agent, which is exactly what a fresh block guards
  against.
* **Behaviour as in development:** 77/80 batches built by hand,
  `propose_batch` 0.6 calls per episode (`learner_sim` x10,
  `learner_purchased` x1, never `learner_sim_purchased`), `query_pool` 12.9
  calls per episode.
* **Headline.** With the tuned descriptions, deepseek B1 plays at plain
  `learner` level and about 7 hits short of the best scripted player, on
  fresh seeds -- the same picture as section 9. Description tuning did not
  change it, and neither did a stronger model (section 10).

## 13. Condition C: the agent's own model (`fit_custom_model`) (2026-10-29)

**Why.** Sections 9-12: the agent builds nearly every batch by hand and
rarely delegates to `propose_batch`, but it does call `fit_and_rank` before
most batches. Condition C gives that habit real options instead of a fixed
model. It is the last direction tried: if C shows nothing interesting, the
agent study ends with sections 9-12 as the result.

**Condition.** `C = campaign + advisor + analysis - fit_and_rank`
(`config/conditions.yaml`): B1 with `fit_and_rank` *replaced* by
`fit_custom_model`, so C - B1 measures a better analysis tool, not more tools.
B1's tools and descriptions are unchanged; adding the tool changes the config
hash of every condition (`d103491a18c7`), so B1 and C are played together,
from the same code, in one records file.

**The tool** (`advisor/analysis.py`, `Toolbox.fit_custom_model`): a logistic
regression on the delivered designs with options the agent sets --
`features`; `scaling` pool or purchased (the bar's scaling, now in the tool the
agent uses); `penalty` l2 or l1 and `C` in [0.001, 100]; `similarity_feature`
(largest 3-mer similarity to a hit, never comparing a design with itself, and
recomputed inside every validation fold so a hit's label never reaches its own
input); `rank_by` mean or an optimistic bound from the Laplace posterior of the
l2 model (`kappa` in [0, 5]). Every call reports leave-one-out log loss against
a leave-one-out base-rate baseline (with the standard error of the
difference), the leave-one-out AUC with its Hanley-McNeil standard error, every
weight with its posterior sd (l2), and the ranking. Defaults give exactly
`fit_and_rank`'s model (a test checks the ranking). With `scaling=purchased`,
`similarity_feature=true` the agent can approximate the bar; the bar itself
blends model and similarity ranks 50/50 with random exploration instead.

Not offered, and why: other link functions and linear regression rank almost
like the logistic model at 15-60 labelled designs; MLPs and deep trees overfit
at that size. No cap on fits per round (the agent can try many settings; the
reported standard errors are its guard against reading noise).

**Plan.** A development run first (trials 40-49, B1 and C together), judged on
behaviour before hits: how often C calls `fit_custom_model`, which options it
picks, and how many bought designs come from its rankings. A confirmatory
pre-registration (fresh trials from 140) follows only if that looks worth it.

**Go/no-go rule, set before the development run** (stated in the session that
ran it, before any data): go on to a confirmatory run only if (1) at least a
third of the designs C buys in rounds 2-4 come from `fit_custom_model`
rankings, **and** (2) C - B1 has a mean of at least +2 hits; otherwise stop.

**Development run** (`runs/condC1`, N-dev, deepseek, B1 and C on trials
40-49, one records file). Gates: 10/10 episodes per condition, all
`finished`, one config hash (`d103491a18c7`), the four scripted policies on
every trial; $0.51.

*Behaviour.* C called `fit_custom_model` 2.5 times per episode (B1:
`fit_and_rank` 2.5). Of 25 calls: the similarity input in 18, C set to 0.3 or
1 in about 12, a feature subset in 7, purchased scaling once, the optimistic
bound never. Most fits beat the base-rate baseline by 2-6 standard errors
(leave-one-out AUC mostly 0.8-0.96), and several episodes compared settings
and kept the better one. Batches were still mixed by hand (37/40). Of the 450
designs C bought in rounds 2-4, **47%** had appeared in a `fit_custom_model`
ranking since the previous batch (B1: 20% in a `fit_and_rank` ranking);
credited instead to the *last* tool that listed them, only 10% (the agent
usually looks designs up by id before buying). Hit rates, descriptive: designs
bought from inside the model's rankings 36% (B1: 22%), from outside 49% (both).

*Hits* (90% paired interval):

| comparison, trials 40-49 | mean diff | 90% CI | W/T/L |
| --- | --- | --- | --- |
| C - B1 | **+0.1** | [-6.7, +6.8] | 6/0/4 |
| C - `learner_sim_purchased` | -3.0 | [-7.2, +1.1] | 5/0/5 |
| B1 - `learner_sim_purchased` | -3.1 | [-6.8, +0.6] | 4/0/6 |
| C - `learner` | +5.5 | [+1.4, +9.6] | 6/1/3 |

Mean hits: C 23.3, B1 23.2.

**Verdict: no-go.** Criterion (2) fails (+0.1 < +2). Criterion (1) depends on
how "come from" is read: 47% by "appeared in a ranking since the last batch"
(a pass), 10% by "last tool that listed the design" (a fail). *Disclosed:* the
first reading was chosen after seeing the data; it does not change the
verdict. No confirmatory run is played; trials 140 onwards stay unused.

## 14. Closing the agent study (2026-10-29)

The question (CLAUDE.md): does an LLM agent add anything over a ladder of
scripted baselines on this task? Answer, from pre-registered tests on fresh
seeds: **no, not over the best scripted player.**

| test | configuration | trials | agent - `learner_sim_purchased` | verdict |
| --- | --- | --- | --- | --- |
| section 9 | deepseek B1, first drafts | 0-19 | -6.2 [-9.0, -3.35] | FAIL |
| section 9 | mimo B1, first drafts | 0-19 | -2.45 [-5.95, +1.0] | FAIL on the point rule (interval straddles) |
| section 12 | deepseek B1, tuned (iteration 1) | 100-119 | -6.9 [-8.8, -4.95] | FAIL |

What else was learned, development or descriptive:

* **The agent is a competent player, not a strong one.** Every configuration
  beat `random` decisively and matched or beat plain `learner` (deepseek in
  section 12: +1.2 [-0.8, +3.35]; mimo in section 9: +6.05 [+2.8, +9.2]).
* **Capability was not the bottleneck** (section 10): Sonnet 5.5 scored no
  better than the flash models; Opus 5.5 could not be tested (the provider
  declined the requests).
* **Wording did not move behaviour** (section 11): clearer `propose_batch`
  options changed nothing; explaining the methods made the agent lean on the
  pool-scaled model and lowered hits (-5.5 [-10.2, -1.1], reverted).
* **A better tool was used, but did not pay** (section 13): given its own
  configurable model, the agent fitted, validated and compared models
  sensibly, but its hits matched B1's.
* **The agent prefers its own judgment.** In every configuration it built
  nearly every batch by hand and almost never bought a scripted proposal as
  is, even though `propose_batch(policy="learner_sim_purchased")` would have
  given it the bar's batch. Its hand-picking, mostly following the sequence
  neighbours of its hits, produced both its best episodes (35-37 hits) and its
  shortfall on average.
* **Seed blocks matter for agents.** On trials 20-39 the agent was within the
  margin (-1.6 [-4.2, +1.0]); on the fresh block 100-119 it was -6.9. The
  scripted bar stayed near 26 on every block. Development estimates were
  optimistic; only fresh blocks are evidence.

Model spend recorded in `runs/*/spend.json`: **$6.15** in all (deepseek at
its peak tariff, so an upper bound; Sonnet $1.28 with its smoke tests). Not
recorded there: the declined Opus smoke test and the section 11 run that was
not kept. Trials 120-139 (section 12's unused
extension) and 140 onwards remain unplayed.

## 15. Pre-registration: the frozen agent on the transfer set E (EGFR) (2026-10-29)

Written after commit `dbff08e`, **before any agent episode on E** (every agent
episode so far is N-dev, checked over every records file in `runs/`) and
before any episode on E trials >= 100 (the only E episodes are the phase 6
ladder and one load check of the four scripted players on trial 999, whose
hits were not read). Nothing below may change once trial 100 is played on E.
Same discipline as sections 9 and 12.

**Why E.** The agent study (section 14) found no gain over the best scripted
player on N-dev. On E the scripted ladder is weak and ordered differently:
`learner_sim` 10.4, `random` 9.6, `learner` 7.4, both purchased variants 4.5
(100 seeds, section 6). The model gate (2 hits, 8 purchases) holds in only one
of four rounds, and E has two scoring features (`esmfold_plddt`,
`proteinmpnn_score`) plus length and design class. If the agent's own
judgment adds anything, a pool where the scripts are weak is where it should
show. E is the transfer set: the agent is run **frozen**, nothing is tuned on
it.

**Frozen configuration.** The code at `dbff08e` (the N-dev configuration, plus
the `query_pool` fix that reads a quoted number as a number); conditions **B1**
and **C** (`config/conditions.yaml`); models **deepseek** (config hash
`d103491a18c7`) and **mimo** (`1665298211ef`), as committed in
`config/models.yaml`.

**Scope.** Game E, trials **100-119**, both conditions, both models, each
model's records in one file with the four scripted policies on the same trials:
`runs/egfr-ds` and `runs/egfr-mimo`. Paired per trial:

    replaylab-eval compare --records runs/egfr-ds/episodes.jsonl --game E \
        --a B1 --b scripted:learner_sim --resamples 10000 --seed 0 --ci 0.9

**Bar: `scripted:learner_sim`**, the best policy on E in the 100-seed ladder
(not `learner_sim_purchased`, the N-dev bar, which is among the worst on E).

**Primary pass/fail: section 9's rule, unchanged** (90% interval of B1 - bar):
PASS low > -2; **Exceeds** additionally low > 0; FAIL high < -2; otherwise
inconclusive. The 2-hit margin is kept for consistency although it is about
20% of the bar on E; "Exceeds" is the outcome that would say the agent adds
something.

* **Headline:** deepseek B1 (the configuration of sections 9 and 12).
* **Replications, same rule, reported separately:** deepseek C, mimo B1,
  mimo C. Disagreements are reported, not resolved.
* **Inconclusive** in any of the four cells pre-commits **one** extension of
  all four cells to trials 100-139 (same commands, only caps raised); verdicts
  are then read on 100-139 and are final.

**Secondary (reported, never pass/fail):** C - B1 per model (does the custom
model help where the fixed model is weak?); each agent cell against `random`,
`learner` and `learner_sim_purchased`; the behaviour profile of sections 11
and 13 (batches built by hand, `fit_custom_model` options, share of buys found
in a model ranking).

**Validity gates**, per model, checked before any hit count is read; a run
that fails one is fixed and replayed, never analysed:

1. Failed episodes (no record) are at most 10%; they are replayed by rerunning
   the same command.
2. Every agent episode ends `finished` or `agent_stopped`.
3. Every agent record carries its model's config hash above, and
   `git diff --stat <this section's commit>..<record git_sha> -- src prompts config`
   is empty.
4. The four scripted policies are present for every trial judged.

**Seeds.** Section 9's rules. E trials 100-139 are reserved for this test;
E trials 0-99 were played by the scripted ladder only.

**Commands** (main repo; each prints nothing until it ends; about $1.1 per
model):

    replaylab-eval pilot --run-dir runs/egfr-ds --model deepseek --games E \
        --conditions B1,C --trials 20 --first-trial 100 --n 40 --max-usd 3
    replaylab-eval pilot --run-dir runs/egfr-mimo --model mimo --games E \
        --conditions B1,C --trials 20 --first-trial 100 --n 40 --max-usd 3
    replaylab-eval scripted --run-dir runs/egfr-ds --games E \
        --policies random,learner,learner_sim,learner_sim_purchased --trials 20 --first-trial 100
    replaylab-eval scripted --run-dir runs/egfr-mimo --games E \
        --policies random,learner,learner_sim,learner_sim_purchased --trials 20 --first-trial 100

Extension, only if triggered: the same commands with `--trials 40` (and
`--n 80 --max-usd 6` for the pilots); existing episodes are skipped.


**Results (2026-10-30: gates before hit counts).** All four gates pass per
model: 0/40 failures, 40/40 `finished`, config hashes `d103491a18c7` (ds) and
`1665298211ef` (mimo) exactly as registered, `git diff dbff08e..4672c20 --
src prompts config` empty, four scripted policies complete on all 20 trials.

| cell | agent - `learner_sim` (bar) | 90% CI | W/L/T | §15 verdict |
| --- | --- | --- | --- | --- |
| deepseek B1 (headline) | **+12.25** | [8.85, 15.55] | 17/3/0 | **EXCEEDS** (low > 0) |
| deepseek C | +12.35 | [9.2, 15.25] | 17/2/1 | **EXCEEDS** |
| mimo B1 | **+13.7** | [12.05, 15.35] | 20/0/0 | **EXCEEDS** |
| mimo C | +11.85 | [9.1, 14.5] | 18/1/1 | **EXCEEDS** |

**All four cells exceed the bar outright** -- nothing is inconclusive, so the
extension never triggers and the verdicts are **final on trials 100-119**.
This is the outcome §15 named as the one that would say the agent adds
something: a frozen configuration, a transfer set, a fresh block.

* Secondary (registered): vs `random` +12.5 to +14.35, vs `learner` +14.85 to
  +16.7, vs `learner_sim_purchased` +17.75 to +19.6 -- every interval above
  zero. C - B1: +0.1 [-4.85, 4.75] (ds), -1.85 [-5.65, 1.7] (mimo) -- the
  custom model neither helps nor hurts, consistent with section 13.
* Behaviour profile (registered secondary): 24-28 tool calls per episode,
  dominated by `query_pool` (214-228 calls per 20 episodes); `submit_batch`
  always full; batches are hand-built (`propose_batch` only 12-29 calls);
  errors rare (1-3 per 20 episodes). `fit_custom_model` chose
  `similarity_feature: true` in 24-29 of 30-32 calls per model pair.
* **Mechanism (round profile vs the bar):** mean deltas by round -- ds B1
  [0.2, +5.8, +9.2, +12.2], ds C [-0.1, +4.0, +9.8, +12.3], mimo B1
  [-0.2, +4.5, +11.1, +13.7], mimo C [-0.2, +3.9, +9.7, +11.8]. Round 1 is a
  wash (~1 hit each: both start blind). The lead is **built in round 2, when
  no model can fit for anyone** -- the agent's hand-rolled `query_pool` /
  `column_summary` filtering finds 4-6 hits where the scripted fallback finds
  ~0, and those hits open the agent's own model gate a full round early
  (section 8: the scripted gate only opens in round 4). Rounds 3-4 compound.
  Mean hits 22.4-24.2 against a bar at 10.4: a ~38-40% hit rate on a
  16%-rate pool with two features, length and class.

## 16. Amendment: condition A on the transfer set (2026-10-30)

Written **before any condition-A episode on E exists** (checked: the E records
hold only B1 and C, trials 100-119). §15's cells are judged and final; this
adds a **descriptive ablation** -- no pass/fail claim attaches to it, and
nothing in §15 is re-judged.

* **Question:** is E's gain produced by campaign-side judgement (filters,
  class/threshold queries) or by the advisor tools? §15's mechanism says the
  lead opens in round 2, before any model can fit; `A` (campaign tools only)
  tests that directly. Expected shapes: A ~= B1 (judgement carries it) or
  B1 > A (advisor compounds from round 3).
* **Setup:** condition `A`, game E, trials 100-119, both models, the same
  frozen `dbff08e` tree. The A cell carries its own config hash (conditions
  are hashed) -- expected, and confined to that cell.
* **Comparisons, same settings** (`--ci 0.9 --resamples 10000 --seed 0
  --trial-range 100-119`): A vs `scripted:learner_sim`, then B1 - A and
  C - A per model. Intervals reported; no threshold registered -- this is
  description, not a claim.

    replaylab-eval pilot --run-dir runs/egfr-ds --model deepseek --games E \
        --conditions A --trials 20 --first-trial 100 --n 20 --max-usd 1
    replaylab-eval pilot --run-dir runs/egfr-mimo --model mimo --games E \
        --conditions A --trials 20 --first-trial 100 --n 20 --max-usd 1
    # the four scripted rows for trials 100-119 already exist in both files


**§16 results (2026-10-30: descriptive; both models complete, n=20, settings
identical to §15).** deepseek's A needed one resume after three provider 500s
(`LLMError: the response has no choices`, trials 104/109/110 -- replayed per
the failure rule; the final set holds 0 unrecovered failures). mimo's A ran
clean. Config hashes unchanged per model (`d103491a18c7` / `1665298211ef`);
all A episodes end 20/20 `finished`.

| comparison | deepseek | mimo |
| --- | --- | --- |
| A - `learner_sim` (bar) | **+5.55** [2.2, 8.75] (13/7/0) | **+9.2** [6.45, 11.75] (18/2/0) |
| B1 - A | **+6.7** [1.25, 12.2] (14/6/0) | **+4.5** [1.5, 7.6] (13/6/1) |
| C - A | +6.8 [2.5, 11.25] (14/6/0) | +2.65 [-0.65, 5.65] (14/6/0) |

Readings (intervals reported; §16 registered no threshold):

* **Campaign-side judgment alone clears E's bar on both models** -- both
  intervals above zero. A means: 16.1 (ds) and 19.8 (mimo) against the same
  bar at 10.55 on these seeds. The round-2 cold-start escape needs no advisor
  tools at all.
* **Advisor tools add on top on both models** (B1 - A intervals above zero:
  +6.7 ds, +4.5 mimo), splitting §15's gain roughly in half. The deepseek round
  profile makes it concrete. Lead over the bar at the four round boundaries,
  cumulative in both rows: A [-0.1, +3.8, +5.4, +5.6] against B1
  [+0.2, +5.9, +9.2, +12.2]. A takes its whole lead in round 2 and holds it
  (the lead grows by +3.8, then +1.6, then +0.2); B1 keeps compounding once it
  can fit (+5.6, then +3.4, then +3.1). Note that A's *hits* rise throughout
  (0.95, 4.75, 9.35, 16.1) -- it is the lead that stops growing, not the
  buying.

  (Correction, 2026-10-30: this bullet previously read "A vs bar
  [-0.1, +3.8, +1.5, +0.2] -- lifts in round 2, then flatlines" against B1's
  [0.2, +5.8, +9.2, +12.2]. The two brackets were different quantities -- A's
  were increments of the lead, B1's the cumulative lead -- so the contrast read
  starker than a like-for-like comparison supports, and A's third increment was
  +1.6, printed as +1.5. Both rows are now cumulative. The qualitative reading
  is unchanged and still holds.)
* C - A is decisive for ds (+6.8) and straddles zero for mimo (+2.65),
  matching C - B1 in §15: *having* an analysis tool matters more than which
  one is exposed.
* Model contrast: mimo's bare judgment is the stronger (A alone 19.8 vs 16.1);
  deepseek leans more on the advisor (+6.7 vs +4.5).
* **Which advisor tool.** Attributing B1's rounds 2-4 purchases by whether the
  design had appeared in a `similar_designs` reply: ds 24.1 such buys per
  episode at a 0.67 hit rate against 17.8 others at 0.26; mimo 29.7 at 0.67
  against 15.3 at 0.24. So the round-2 escape is judgment (A reproduces it
  with no advisor tool at all), while the rounds 3-4 compounding runs through
  the neighbour tool: buy a feature sort, keep the hits, ask for their
  neighbours, buy those. `fit_and_rank` / `propose_batch` rankings account for
  about 3 buys per episode. **Correlational:** a design in a neighbour list
  may have been chosen for another reason, and E's hit-rich families are
  small, so neighbours of a hit are hits nearly by construction -- which is
  real pool structure being exploited, not an artifact, but it is not a
  controlled attribution. `B2` (`exclude: [similar_designs]`, already in
  `config/conditions.yaml`) would test it directly, on a fresh block from
  trial 140 and with its own pre-registration.

**Follow-up, now fixed (outside §15/§16).** `prompts/system.md` promises a
final plain-text summary that the loop never requests on the `finished` /
`agent_stopped` paths: `loop.py` returns immediately after the last tool call,
and every agent episode on record ends on a tool result (60 ds + 60 mimo);
only a voluntary `agent_ended` could show the summary, and no model ever took
that path.

Fixed after §15 and §16 were judged, so no episode here is affected: on
`finished` and `agent_stopped` the loop now makes one further call **without
tools** and appends the reply as the last assistant message. It is best effort
-- a cap hit, a model failure (`ResponseTruncated` included) or an empty reply
only raises a new `closing_summary_failed` flag, because a finished campaign
must not be discarded over its epilogue. The usage row carries `closing: True`,
so "one usage row per turn" still holds for the turns.

**This does not change the config hash**, which covers the model and price
entries, both prompts, the conditions file, the tool descriptions and `Limits`
-- not `loop.py`. So the hash alone cannot separate episodes played before and
after it; what does is §15 gate 3's second half, the `git diff ... -- src`
check against the record's `git_sha`. Any later confirmatory claim still needs
a fresh block (§9 rule 5), and the epilogue adds one call per episode at the
episode's largest context (mostly cache reads, and outside the pilot's
projection), so recorded per-episode cost shifts slightly upwards.

## 17. Amendment: condition B2 on the transfer set (2026-10-30)

Written **before any condition-B2 episode on E exists** (checked: the E records
hold A, B1 and C only). Descriptive, like §16: no threshold, no pass/fail;
§15 and §16 are untouched.

* **Question:** how much of the advisor gain (§16: B1 - A = +6.7 ds, +4.5
  mimo) comes from `similar_designs` specifically? B2 is B1 minus that one
  tool, so **B1 - B2** isolates it. The N-dev forensics after §9 implicated the
  neighbour crawl in the agent's worst runs; E is where that gets a clean test
  in a cell where the advisor demonstrably helps.
* **Setup:** condition `B2` (`config/conditions.yaml`: campaign + advisor -
  `similar_designs`), game E, trials 100-119, both models, working tree
  `ba56753` (episodes therefore include the closing-summary call;
  `config_hash` is unchanged by it -- prompts, conditions, tool descriptions
  and limits are what the hash covers). B2's own cell carries a different
  config hash (its conditions text differs) -- expected and confined to it.
* **Comparisons** (same settings as §15/§16: `--ci 0.9 --resamples 10000
  --seed 0 --trial-range 100-119`): **B1 - B2** per model (the isolating
  pair), **B2 - `scripted:learner_sim`** (does the reduced agent still clear
  the bar?), C - B2 as context. Intervals reported, no threshold registered.
* **Commands** (the four scripted rows for trials 100-119 already exist in
  both files; caps cover existing spend plus the projection -- spend is
  cumulative per run-dir, and pilot means are ds 0.0172 / mimo 0.0109 per
  episode):

      replaylab-eval pilot --run-dir runs/egfr-ds --model deepseek --games E \
          --conditions B2 --trials 20 --first-trial 100 --n 20 --max-usd 1.5
      replaylab-eval pilot --run-dir runs/egfr-mimo --model mimo --games E \
          --conditions B2 --trials 20 --first-trial 100 --n 20 --max-usd 1.4


**§17 results (2026-10-30: gates before hit counts).** Both cells complete:
20/20 episodes, all `finished`, 0 B2 failures, 80 scripted rows present,
spend $1.280 (cap 1.5) and $0.934 (cap 1.4); flags: 5 and 3 tool-errors
(+1 validation). **Correction to §17 above:** B2 does *not* carry its own
config hash -- the hash covers the whole `conditions` file, so all four
conditions in a run-dir share it (the registered `d103491a18c7` /
`1665298211ef`). The expectation recorded above was wrong; the useful
consequence is that A, B1, B2 and C are provably one configuration file.

| comparison (n=20) | deepseek | mimo |
| --- | --- | --- |
| B1 - B2 | **+7.15** [1.65, 12.55] (15/5/0) | **+3.85** [0.55, 7.3] (11/7/2) |
| B2 - `learner_sim` (bar) | **+5.1** [2.2, 8.0] (14/6/0) | **+9.85** [7.35, 12.2] (18/2/0) |
| C - B2 | +7.25 [3.0, 11.3] (14/5/1) | +2.0 [-0.9, 5.1] (11/9/0) |

Readings (descriptive; §17 registered no threshold):

* **`similar_designs` is where the advisor's value comes from on E.**
  B1 - B2 clears zero on both models, and B2 drops back to campaign-tools
  level: B2 - bar (+5.1 ds, +9.85 mimo) sits at A - bar (+5.55, +9.2, section
  16). The rest of the advisor kit -- fit, estimate, stop advice -- adds
  little beyond the pool filters on this pool.
* **B2 alone still beats the bar** on both models (intervals above zero):
  the round-2 cold-start escape of section 15 needs no advisor tools at all.
* C - B2: decisive on ds (+7.25; C keeps the similarity tool), straddles
  zero on mimo (+2.0) -- consistent with C - B1 in section 15 being
  null-to-negative for mimo.
* The N-dev forensics (neighbour crawling in the worst runs) and this
  ablation point opposite ways per pool: on E, with sparse, clustered hits,
  following up on hits through the similarity index is the single most
  valuable advisor behaviour; on N-dev it correlated with failure. The
  ablation (randomised tool presence) outranks the observational forensics
  -- recorded as such.