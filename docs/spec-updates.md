# Spec updates (paste-ready)

Prepared 2026-10-13 for the project document, which lives outside this repo.
Source of every number: `reports/ladder.csv` (11,400 rows = 11 games x 114
policies x 100 seeds) and `scripts/reproduce_legacy.py`; rationale in
`docs/design-notes.md`.

## Section 2.4 — replace the ladder table

### 2.4.a Reproduction of the exploratory ladder (game `legacy`: whole Nipah-G
pool, split `all`, 4 rounds x 25, 100 seeds)

| policy | expected (exploratory) | tolerance | reproduced (mean) | sd | verdict |
| --- | --- | --- | --- | --- | --- |
| random | 10.2 | +/- 1.5 | 9.86 | 2.57 | OK |
| fixed `boltz2_iptm` | 10.0 | +/- 1 | 10.00 | 0.00 | OK |
| fixed `boltz2_ipsae` | 16.0 | +/- 1 | 16.00 | 0.00 | OK |
| fixed `boltz2_min_ipsae` | 18.0 | +/- 1 | 18.00 | 0.00 | OK |
| legacy learner | 25.9 | +/- 1.5 | 25.93 | 3.04 | OK |
| legacy learner + similarity, w = 0.5 | 38.9 | +/- 1.5 | 38.49 | 2.90 | OK |

All six reproduce. (The legacy learner exists only in
`scripts/reproduce_legacy.py`; it is not part of the library.)

### 2.4.b Current ladder — mean hits over 100 seeds, frozen config v1

| game | random | learner | learner_sim | learner_purchased | learner_sim_purchased | best fixed |
| --- | --- | --- | --- | --- | --- | --- |
| N-dev | 6.0 | 18.2 | 14.5 | 15.8 | 26.4 | 14.0 |
| N-dev-delay2 | 6.3 | 17.3 | 16.4 | 13.7 | 25.1 | 14.0 |
| E | 9.6 | 7.4 | 10.4 | 4.5 | 4.5 | 5.0 |
| S-hnmt | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 |
| S-human-ambp | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| S-human-gm2a | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| S-human-idi2 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| S-human-orm2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| S-human-rfk | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 |
| S-human-serum-albumin | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| S-human-tnfa | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Notes for the table:

* The stop games (S-*) have `budget == pool size`, so every policy ties at the
  pool's total hits; they measure stopping, not ranking.
* `+stop` variants are omitted because they tie their base policy exactly on
  every game (see 3.5).
* `*_purchased` are the tracked scaling variants (decision, 2026-10-13): the
  purchased-row scaler wins big on Nipah (+11.9, +8.7, and +6.3 on the legacy
  game) but loses on E (-5.9); both variants stay in the ladder until the
  difference is explained. Full per-policy tables: `reports/ladder.md`.

## Section 3.5 — replace the stop-rule text

> Stop rules are advice, evaluated on delivered purchases at round boundaries.
> **`low_yield`** (default): fires from 20 delivered purchases when the
> cumulative hit rate is below 0.03. In the ladder it fires only in the four
> zero-hit stop games, stopping at 20 of 30 purchases with no hits lost — a
> pure saving; every `+stop` variant ties its unwrapped policy exactly.
> **`decay`** (window 15, drop 0.05) is **no longer part of the default**: one
> hit of decline inside the window is 1/15 = 0.067 > 0.05, and measurement
> showed it false-stopping 77% of random runs and 98% of `learner_sim` runs on
> N-dev — in the previous ladder it cost game E `learner_sim+stop` 10.4 hits
> (1.0 vs 10.4). The rule remains implemented, measured and explicitly
> selectable; its parameters stay frozen. Retuning it (e.g. by cross-validation)
> would be a spec change.
