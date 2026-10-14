# E-long: does a longer horizon change what E measures?

Pre-registered in `docs/design-notes.md` section 8 (2026-10-14): the question,
the facts and the horizon were written down before this run, and every horizon
is reported -- not the best one. This game is **exploratory**: excluded from the
default ladder, from `frozen_v1` and from every decision rule. The headline E
numbers stand as they are.

- game: `E-long` (target `egfr`, split `all`, 8 rounds x 15 = 120 purchases)
- pool: 826 designs, 134 hits (16.2%); the horizon stays below a third of the pool and below the hit supply
- runs: 9 policies x 100 seeds, 900 campaigns
- intervals: 95% percentile bootstrap over seeds (B=2000, rng 20261014).
  Every seed replays the **same pool**, so the intervals measure policy
  randomness only -- they say nothing about variation across targets.

## Score at every horizon

Mean cumulative hits after each batch of purchases, 95% bootstrap interval
over seeds.

| policy | 15 purchases | 30 purchases | 45 purchases | 60 purchases | 75 purchases | 90 purchases | 105 purchases | 120 purchases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed:esmfold_plddt | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 4.00 [4.00, 4.00] | 5.00 [5.00, 5.00] | 5.00 [5.00, 5.00] | 6.00 [6.00, 6.00] | 10.00 [10.00, 10.00] | 11.00 [11.00, 11.00] |
| fixed:proteinmpnn_score | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 3.00 [3.00, 3.00] | 5.00 [5.00, 5.00] | 5.00 [5.00, 5.00] | 6.00 [6.00, 6.00] | 6.00 [6.00, 6.00] | 6.00 [6.00, 6.00] |
| learner | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 4.00 [4.00, 4.00] | 7.44 [7.32, 7.57] | 8.05 [7.85, 8.28] | 14.37 [14.00, 14.73] | 18.54 [17.99, 19.06] | 26.23 [25.77, 26.69] |
| learner+stop | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 4.00 [4.00, 4.00] | 7.44 [7.32, 7.56] | 8.05 [7.84, 8.26] | 14.37 [14.00, 14.74] | 18.54 [17.98, 19.08] | 26.23 [25.75, 26.68] |
| learner_purchased | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 4.00 [4.00, 4.00] | 4.51 [4.38, 4.66] | 7.13 [6.69, 7.59] | 12.62 [12.08, 13.16] | 17.02 [16.24, 17.78] | 24.07 [23.06, 25.08] |
| learner_sim | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 4.00 [4.00, 4.00] | 10.39 [10.29, 10.50] | 13.24 [12.89, 13.60] | 19.78 [19.25, 20.32] | 26.86 [26.24, 27.43] | 34.26 [33.62, 34.89] |
| learner_sim+stop | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 4.00 [4.00, 4.00] | 10.39 [10.29, 10.50] | 13.24 [12.90, 13.60] | 19.78 [19.23, 20.31] | 26.86 [26.30, 27.45] | 34.26 [33.62, 34.89] |
| learner_sim_purchased | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 4.00 [4.00, 4.00] | 4.45 [4.34, 4.57] | 8.33 [7.94, 8.75] | 16.59 [16.16, 17.02] | 24.03 [23.52, 24.56] | 31.70 [31.28, 32.15] |
| random | 2.38 [2.12, 2.65] | 4.77 [4.41, 5.12] | 7.33 [6.91, 7.75] | 9.59 [9.05, 10.13] | 11.80 [11.22, 12.42] | 14.29 [13.63, 14.93] | 16.97 [16.25, 17.71] | 19.53 [18.81, 20.27] |

## The reference curve

The two references are `random` (the bar to cross) and `fixed:esmfold_plddt`
(exactly what the fallback buys). Rounds 1-4 are E's 60 purchases.

| purchases | random | fixed:esmfold_plddt | learner | learner_sim | learner_purchased | learner_sim_purchased |
| --- | --- | --- | --- | --- | --- | --- |
| 15 | 2.38 [2.10, 2.64] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| 30 | 4.77 [4.42, 5.14] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| 45 | 7.33 [6.90, 7.76] | 4.00 [4.00, 4.00] | 4.00 [4.00, 4.00] | 4.00 [4.00, 4.00] | 4.00 [4.00, 4.00] | 4.00 [4.00, 4.00] |
| 60 | 9.59 [9.05, 10.13] | 5.00 [5.00, 5.00] | 7.44 [7.32, 7.57] | 10.39 [10.29, 10.50] | 4.51 [4.38, 4.65] | 4.45 [4.34, 4.57] |
| 75 | 11.80 [11.15, 12.40] | 5.00 [5.00, 5.00] | 8.05 [7.85, 8.27] | 13.24 [12.89, 13.62] | 7.13 [6.68, 7.62] | 8.33 [7.92, 8.76] |
| 90 | 14.29 [13.65, 14.93] | 6.00 [6.00, 6.00] | 14.37 [14.02, 14.73] | 19.78 [19.25, 20.35] | 12.62 [12.07, 13.16] | 16.59 [16.15, 17.04] |
| 105 | 16.97 [16.23, 17.71] | 10.00 [10.00, 10.00] | 18.54 [18.00, 19.07] | 26.86 [26.30, 27.46] | 17.02 [16.23, 17.77] | 24.03 [23.53, 24.55] |
| 120 | 19.53 [18.76, 20.28] | 11.00 [11.00, 11.00] | 26.23 [25.73, 26.72] | 34.26 [33.62, 34.89] | 24.07 [23.02, 25.03] | 31.70 [31.30, 32.14] |

## When does the model start fitting?

Share of seeds in which the gate (>= 2 hits, >= 8 purchases, both classes)
passed in that round -- the fallback runs everywhere the cell says 0%.

| policy | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| learner | 0% | 0% | 0% | 100% | 100% | 100% | 100% | 100% |
| learner+stop | 0% | 0% | 0% | 100% | 100% | 100% | 100% | 100% |
| learner_purchased | 0% | 0% | 0% | 100% | 100% | 100% | 100% | 100% |
| learner_sim | 0% | 0% | 0% | 100% | 100% | 100% | 100% | 100% |
| learner_sim+stop | 0% | 0% | 0% | 100% | 100% | 100% | 100% | 100% |
| learner_sim_purchased | 0% | 0% | 0% | 100% | 100% | 100% | 100% | 100% |

## Parity with the committed ladder

At 60 purchases (E's horizon) this run must reproduce
`reports/ladder.csv` exactly -- the longer horizon may not touch the early rounds:

- `fixed:esmfold_plddt`: E ladder 5.00 vs this run 5.00 -> OK
- `fixed:proteinmpnn_score`: E ladder 5.00 vs this run 5.00 -> OK
- `learner`: E ladder 7.44 vs this run 7.44 -> OK
- `learner+stop`: E ladder 7.44 vs this run 7.44 -> OK
- `learner_purchased`: E ladder 4.51 vs this run 4.51 -> OK
- `learner_sim`: E ladder 10.39 vs this run 10.39 -> OK
- `learner_sim+stop`: E ladder 10.39 vs this run 10.39 -> OK
- `learner_sim_purchased`: E ladder 4.45 vs this run 4.45 -> OK
- `random`: E ladder 9.59 vs this run 9.59 -> OK

## What this does not say

- It is one target. E is the only transfer target with a pool this size, so
  nothing here answers whether the learner transfers **across targets**.
- No policy changed: adding exploration to the fallback is a design change and
  a separate experiment, justified on N-dev and the other games -- not on E.
