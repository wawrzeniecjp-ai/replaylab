# Audits: checking the results from outside the code

These scripts check that the recorded results mean what the README says they mean. Audits 1,
2 and 4 import nothing from `replaylab`: they rebuild pools, aliases, outcomes and statistics
from `data/raw/proteinbase.csv`, the split files, `config/games.yaml` and the recorded
`runs/*/episodes.jsonl`, by the rules the README's Methods describe; audit 2 also reads the
Anthropic release's own tables (`design_summary`, `cofold_predictions`) by the rules of
`docs/prereg-anthropic.md`. A bug in the package
would therefore not be shared by its audit. Audit 3 runs the package's own scripted policies,
because they are what it tests, but on tables it writes to a temporary folder.

```bash
make audit                                   # all four, about 3 minutes
python scripts/audit/rescore.py              # 1
python scripts/audit/leaks.py                # 2
python scripts/audit/controls.py             # 3 (add --positive-control for part D)
python scripts/audit/pairing.py              # 4
```

Each script takes episodes files as arguments (default: every `runs/*/episodes.jsonl`),
prints what it checked, and exits with 1 on any failure.

| audit | claim it tests | how |
| --- | --- | --- |
| 1 `rescore.py` | the replay served the real lab results, and each recorded score is what the purchases earned | rebuild the pool and the seed's aliases, read the accepted `submit_batch` calls, re-score from the raw CSV; check every outcome shown to the agent against the raw value |
| 2 `leaks.py` | no sequence, no real id, no outcome before delivery; feature values are raw | scan every message; replay the clock to know what was delivered when; check `nearest_hit`, the aggregate counts and every feature value shown |
| 3 `controls.py` | scoring is unbiased and no policy gets outcome information except through the features | random against n·K/N; one result per scripted (game, policy, trial) in every run; the ladder on labels shuffled afresh per trial must fall to chance; with `--positive-control`, a planted leak must be found |
| 4 `pairing.py` | conditions are paired by seed and the reported numbers follow from the records | seeds are sha256(game\|trial); one config hash per arm, the registered hashes, commits that exist (old ones through the history-rewrite table); every comparison the README reports recomputed: mean, win/loss/tie, 90% bootstrap interval |

Each audit was checked against tampered input (a score off by one, a flipped outcome, the
wrong seed, an outcome planted on a pool row, a real id, a sign-flipped value, an inflated
count), and it failed on every one.

## Audit 2 on the Anthropic pools (2026-11-19)

Audit 2 now covers the 480 Anthropic agent episodes as well. It rebuilds their pools from the
release's tables, without the package: a target's pool is every design with a binder call;
the features are the seed-best co-fold row (highest `ipsae_min` at a design-complex
stoichiometry on the design target, ties to the lowest seed); real ids include the release's
UUIDs. It passes on all 1,310 agent episodes (830 Proteinbase, 480 Anthropic): 70,062 outcome
rows, 9,578 `nearest_hit` hints and 1,783,328 feature values checked, every Anthropic value
equal to the release's. Planted into an Anthropic episode, a UUID, a 12-residue sequence
fragment, an outcome on an undelivered pool row and a feature value off by one part in 10^9
each made it fail. Audit 1 and part A of audit 3 still skip these episodes.

## What the audits found (2026-11-13, after the final tests)

All four pass on the 21 recorded runs. Audits 1 and 2 re-scored and scanned 830 agent
episodes (N-dev 350, E 240, N-held 240): 51,460 outcomes shown to agents and 1,383,948 feature
values checked. Audit 4 checked the seeds of 3,610 records and reproduced 44 reported
comparisons, including every final-test number in the README (pooled and per game, the
held-out test and the round-by-round table). Two limits to know:

* **The Anthropic pools are only partly audited.** Their designs are not in the Proteinbase
  export these audits rebuild from, so audits 1 and 2 and part A of audit 3 skip their 480
  agent episodes (each says so). Audit 4 covers them: seeds, registered hashes, commits and
  the statistics. (Audit 2 has covered them since 2026-11-19; see above.)
* **Records name commits from before the history rewrite.** Audit 4 maps the 16 such commits
  through the table in `docs/history-rewrite.md`; all 16 map to commits in this repository.

## What the audits found (2026-11-04)

All four pass on the 17 recorded runs: 590 agent episodes re-scored, 42,012 outcomes shown to
agents and 990,999 feature values checked, 18 reported comparisons reproduced. Three things
came up that are not failures but should be known:

* **A shown pLDDT is the mean of two records.** The export holds two `esmfold_plddt` values
  for 4,934 designs (median difference 0.35, largest 30.6), and the package shows their mean.
  This is a reasonable choice but the README does not state it; 136,039 of the values shown
  to agents were such means.
* **The orientation control arm names the wrong commit.** In `orient-ds` and `orient-mimo`
  the control arm (B1ctl) was played from a worktree at `74acb41` but its records say
  `be91889f` (a pre-rewrite hash, now `a162359a`; see `docs/history-rewrite.md`), because
  `git_sha` was then read from the shell's folder, not the code's (fixed in `6f01231`). The
  arms still differ in config hash, so the comparison stands; only the recorded commit is
  wrong.
* **Some arms span several commits** under one config hash (part of `ds2`, `mimo2`,
  `egfr-ds`). Allowed by design (the hash pins what the model saw), and listed by audit 4.

Not covered: scripted episodes record only their scores, so audit 1 re-scores agent episodes
only (audit 3 checks the scripted baselines in aggregate). No agent run used a delayed game,
so audit 2's delivery-timing replay has run only on games where results arrive at once.
