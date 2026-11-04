"""Audit 4: seeds, pairing, provenance and the reported statistics.

    python scripts/audit/pairing.py

Independently of ``replaylab``:

1. **Seeds.** Every record's seed is the documented sha256("<game>|<trial>") value, so
   the same (game, trial) means the same pool order and aliases in every condition.
2. **Provenance.** Within a run, all agent records of one model and condition share one
   config hash (a run may hold two arms played from different trees on purpose, as the
   orientation study's B1 and B1ctl are),
   and the hashes registered in the docs are the ones recorded; every recorded git
   commit exists in this repository, and no record is marked ``git_dirty``.
3. **Pairing and statistics.** For each comparison the README reports, both sides have
   exactly one record per trial of the block, and the paired mean difference, the
   win/loss/tie ledger and a 90% percentile bootstrap interval (10,000 resamples of the
   per-trial differences, own implementation) are recomputed. Means and ledgers must
   match exactly; interval endpoints within ``CI_TOLERANCE`` (the draws differ).
"""

from __future__ import annotations

import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass

import _common as c
import numpy as np

CI_TOLERANCE = 0.35
RESAMPLES = 10_000


@dataclass(frozen=True)
class Claim:
    """One comparison as the README (or the design notes it cites) reports it."""

    run: str
    game: str
    a: str
    b: str
    trials: tuple[int, int]
    mean: float
    ci: tuple[float, float] | None = None
    ledger: tuple[int, int, int] | None = None  # wins, losses, ties of a over b


LSP, LS = "scripted:learner_sim_purchased", "scripted:learner_sim"
CLAIMS = [
    # Nipah-G: the agent against the strongest baseline (Figure 2).
    Claim("ds2", "N-dev", "B1", LSP, (0, 19), -6.20),
    Claim("confirm1", "N-dev", "B1", LSP, (100, 119), -6.90, (-8.8, -4.95)),
    Claim("confirm1", "N-dev", "B1", "scripted:learner", (100, 119), 1.20, (-0.8, 3.35)),
    Claim("mimo2", "N-dev", "B1", LSP, (0, 19), -2.45, (-5.95, 1.0)),
    # EGFR: the agent against the strongest baseline there (Figure 2).
    Claim("egfr-ds", "E", "B1", LS, (100, 119), 12.25, (8.85, 15.55), (17, 3, 0)),
    Claim("egfr-mimo", "E", "B1", LS, (100, 119), 13.70, (12.05, 15.35), (20, 0, 0)),
    Claim("egfr-ds", "E", "C", LS, (100, 119), 12.35, (9.2, 15.25), (17, 2, 1)),
    Claim("egfr-mimo", "E", "C", LS, (100, 119), 11.85, (9.1, 14.5), (18, 1, 1)),
    # The advisor effect, B1 - A (Figure 1).
    Claim("ds2", "N-dev", "B1", "A", (0, 19), 2.45),
    Claim("mimo2", "N-dev", "B1", "A", (0, 19), 6.80),
    Claim("egfr-ds", "E", "B1", "A", (100, 119), 6.70),
    Claim("egfr-mimo", "E", "B1", "A", (100, 119), 4.50),
    # The EGFR ablation (Figure 4).
    Claim("egfr-ds", "E", "A", LS, (100, 119), 5.55),
    Claim("egfr-mimo", "E", "A", LS, (100, 119), 9.20),
    Claim("egfr-ds", "E", "B1", "B2", (100, 119), 7.15),
    Claim("egfr-mimo", "E", "B1", "B2", (100, 119), 3.85),
    # The similarity tool on Nipah-G (Figure 5).
    Claim("auth-ds", "N-dev", "B1", "B2", (140, 179), 5.30, (3.15, 7.40)),
    Claim("auth-mimo", "N-dev", "B1", "B2", (140, 179), 4.88),
]
# Config hashes registered in docs/design-notes.md for the confirmatory runs.
REGISTERED = {"confirm1": "b457e246517b", "egfr-ds": "d103491a18c7", "egfr-mimo": "1665298211ef"}


def percentile_bootstrap(diffs: np.ndarray, level: float = 0.9) -> tuple[float, float]:
    rng = np.random.default_rng(12345)
    means = diffs[rng.integers(0, len(diffs), size=(RESAMPLES, len(diffs)))].mean(axis=1)
    tail = (1 - level) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def commit_exists(sha: str) -> bool:
    done = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=c.ROOT, capture_output=True
    )
    return done.returncode == 0


def main(arguments: list[str]) -> int:
    report = c.Report("audit 4: seeds, pairing, provenance and statistics")
    runs: dict[str, list[dict]] = defaultdict(list)
    for path, record in c.read_records(c.record_paths(arguments)):
        runs[path.parent.name].append(record)

    # 1. Seeds.
    n_records = sum(len(v) for v in runs.values())
    bad = [
        (name, r["game"], r["condition"], r["trial"])
        for name, records in runs.items()
        for r in records
        if r["seed"] != c.episode_seed(r["game"], r["trial"])
    ]
    report.note(f"1. seeds checked on {n_records} records in {len(runs)} runs; wrong: {len(bad)}")
    for item in bad[:10]:
        report.fail(f"1. seed of {item} is not sha256(game|trial)")

    # 2. Provenance.
    shas: set[str] = set()
    for name, records in sorted(runs.items()):
        hashes = defaultdict(set)
        for r in records:
            if c.is_agent(r):
                hashes[(r["model"], r["condition"])].add(r["config_hash"])
            if r.get("git_sha"):
                shas.add(r["git_sha"])
            report.check(not r.get("git_dirty"), f"2. {name}: a record is marked git_dirty")
        for (model, condition), found in hashes.items():
            report.check(
                len(found) == 1,
                f"2. {name}: {model} {condition} played under several hashes {sorted(found)}",
            )
        if name in REGISTERED:
            found = {h for v in hashes.values() for h in v}
            report.check(
                found == {REGISTERED[name]},
                f"2. {name}: hash {sorted(found)} is not the registered one",
            )
        # Informational: an arm played over several commits under one config hash is
        # allowed (the hash pins what the model saw), but the code may have changed.
        commits = defaultdict(set)
        for r in records:
            if c.is_agent(r):
                commits[(r["model"], r["condition"])].add((r.get("git_sha") or "?")[:8])
        spread = {f"{m} {cond}": sorted(v) for (m, cond), v in commits.items() if len(v) > 1}
        if spread:
            report.note(f"2. {name}: arms recorded over several commits: {spread}")
    missing = sorted(s for s in shas if not commit_exists(s))
    report.note(f"2. recorded commits: {len(shas)}; missing from this repository: {len(missing)}")
    for sha in missing:
        report.fail(f"2. recorded commit {sha} is not in this repository")

    # 3. The reported comparisons.
    for claim in CLAIMS:
        label = (
            f"{claim.run} {claim.game} {claim.a} - {claim.b} ({claim.trials[0]}-{claim.trials[1]})"
        )
        block = range(claim.trials[0], claim.trials[1] + 1)
        sides = {}
        for side in (claim.a, claim.b):
            rows = [
                r
                for r in runs.get(claim.run, [])
                if r["game"] == claim.game and r["condition"] == side
            ]
            per_trial = defaultdict(list)
            for r in rows:
                if r["trial"] in block:
                    per_trial[r["trial"]].append(r["scores"]["hits"])
            if sorted(per_trial) != list(block) or any(len(v) != 1 for v in per_trial.values()):
                report.fail(f"3. {label}: {side} does not have exactly one record per trial")
                sides = {}
                break
            sides[side] = per_trial
        if not sides:
            continue
        diffs = np.array([sides[claim.a][t][0] - sides[claim.b][t][0] for t in block], dtype=float)
        mean = float(diffs.mean())
        ledger = (int((diffs > 0).sum()), int((diffs < 0).sum()), int((diffs == 0).sum()))
        low, high = percentile_bootstrap(diffs)
        wins, losses, ties = ledger
        line = f"3. {label}: {mean:+.2f} [{low:+.2f}, {high:+.2f}] W/L/T {wins}/{losses}/{ties}"
        report.check(
            abs(mean - claim.mean) < 0.005 + 1e-9,
            f"{label}: mean {mean:+.3f}, reported {claim.mean:+.2f}",
        )
        if claim.ci is not None:
            gap = max(abs(low - claim.ci[0]), abs(high - claim.ci[1]))
            line += (
                f"  (reported [{claim.ci[0]:+.2f}, {claim.ci[1]:+.2f}], endpoints within {gap:.2f})"
            )
            report.check(
                gap <= CI_TOLERANCE, f"{label}: interval differs from the reported one by {gap:.2f}"
            )
        if claim.ledger is not None:
            report.check(
                ledger == claim.ledger, f"{label}: ledger {ledger}, reported {claim.ledger}"
            )
        report.note(line)
    return report.finish()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
