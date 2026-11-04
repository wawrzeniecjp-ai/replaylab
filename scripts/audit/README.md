# Audits: checking the results from outside the code

These scripts check that the recorded results mean what the README says they mean. Audits 1,
2 and 4 import nothing from `replaylab`: they rebuild pools, aliases, outcomes and statistics
from `data/raw/proteinbase.csv`, the split files, `config/games.yaml` and the recorded
`runs/*/episodes.jsonl`, by the rules the README's Methods describe. A bug in the package
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
| 4 `pairing.py` | conditions are paired by seed and the reported numbers follow from the records | seeds are sha256(game\|trial); one config hash per arm, the registered hashes, commits that exist; every comparison the README reports recomputed: mean, win/loss/tie, 90% bootstrap interval |

Each audit was checked against tampered input (a score off by one, a flipped outcome, the
wrong seed, an outcome planted on a pool row, a real id, a sign-flipped value, an inflated
count), and it failed on every one.

## What the audits found (2026-11-04)

All four pass on the 17 recorded runs: 590 agent episodes re-scored, 42,012 outcomes shown to
agents and 990,999 feature values checked, 18 reported comparisons reproduced. Three things
came up that are not failures but should be known:

* **A shown pLDDT is the mean of two records.** The export holds two `esmfold_plddt` values
  for 4,934 designs (median difference 0.35, largest 30.6), and the package shows their mean.
  This is a reasonable choice but the README does not state it; 136,039 of the values shown
  to agents were such means.
* **The orientation control arm names the wrong commit.** In `orient-ds` and `orient-mimo`
  the control arm (B1ctl) was played from a worktree at `aaeb1af` but its records say
  `be91889f`, because `git_sha` was then read from the shell's folder, not the code's (fixed
  in `df94134`). The arms still differ in config hash, so the comparison stands; only the
  recorded commit is wrong.
* **Some arms span several commits** under one config hash (part of `ds2`, `mimo2`,
  `egfr-ds`). Allowed by design (the hash pins what the model saw), and listed by audit 4.

Not covered: scripted episodes record only their scores, so audit 1 re-scores agent episodes
only (audit 3 checks the scripted baselines in aggregate). No agent run used a delayed game,
so audit 2's delivery-timing replay has run only on games where results arrive at once.
