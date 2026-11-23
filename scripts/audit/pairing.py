"""Audit 4: seeds, pairing, provenance and the reported statistics.

    python scripts/audit/pairing.py

Independently of ``replaylab``:

1. **Seeds.** Every record's seed is the documented sha256("<game>|<trial>") value, so
   the same (game, trial) means the same pool order and aliases in every condition.
2. **Provenance.** Within a run, all agent records of one model and condition share one
   config hash (a run may hold two arms played from different trees on purpose, as the
   orientation study's B1 and B1ctl are),
   and the hashes registered in the docs are the ones recorded; every recorded git
   commit exists in this repository (records played before the history rewrite name the old
   commit, which ``docs/history-rewrite.md`` maps to its new hash), and no record is marked
   ``git_dirty``.
3. **Pairing and statistics.** For each comparison the README reports, both sides have
   exactly one record per trial of the block (per game and trial, for a comparison pooled
   over several games), and the paired mean difference, the win/loss/tie ledger and a 90%
   percentile bootstrap interval (10,000 resamples of the paired differences, own
   implementation) are recomputed. A claim may be about the hits at a round boundary rather
   than at the end (the README's round-by-round table). Means and ledgers must
   match exactly; interval endpoints within ``CI_TOLERANCE`` (the draws differ).
"""

from __future__ import annotations

import re
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
    game: str  # one game, or a label for the games in ``pooled``
    a: str
    b: str
    trials: tuple[int, int]
    mean: float
    ci: tuple[float, float] | None = None
    ledger: tuple[int, int, int] | None = None  # wins, losses, ties of a over b
    pooled: tuple[str, ...] = ()  # the games of a pooled comparison, paired on (game, trial)
    boundary: int | None = None  # compare the hits after this round instead of at the end

    @property
    def games(self) -> tuple[str, ...]:
        return self.pooled or (self.game,)


LSP, LS = "scripted:learner_sim_purchased", "scripted:learner_sim"
AB = ("AB-nipah", "AB-egfr", "AB-bbf14", "AB-bhrf1")
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
    # The Anthropic final test, pooled over the four games (Figure 6 and the text).
    Claim("ab-ds", "AB pooled", "B1", "A", (0, 19), 0.94, (0.59, 1.29), (41, 14, 25), AB),
    Claim("ab-mimo", "AB pooled", "B1", "A", (0, 19), 0.69, (0.36, 1.04), (34, 15, 31), AB),
    Claim("ab-ds", "AB pooled", "B1", LS, (0, 19), 2.24, pooled=AB),
    Claim("ab-mimo", "AB pooled", "B1", LS, (0, 19), 1.95, pooled=AB),
    Claim("ab-ds", "AB pooled", "A", LS, (0, 19), 1.30, pooled=AB),
    Claim("ab-mimo", "AB pooled", "A", LS, (0, 19), 1.26, pooled=AB),
    Claim("ab-ds", "AB pooled", "B1", "B2", (0, 19), 0.55, (0.14, 0.97), pooled=AB),
    Claim("ab-mimo", "AB pooled", "B1", "B2", (0, 19), 0.25, (-0.06, 0.57), pooled=AB),
    # The Anthropic final test, game by game (the README's per-game table).
    Claim("ab-ds", "AB-nipah", "B1", "A", (0, 19), 1.60, (1.00, 2.20)),
    Claim("ab-ds", "AB-egfr", "B1", "A", (0, 19), 0.00, (-0.60, 0.65)),
    Claim("ab-ds", "AB-bbf14", "B1", "A", (0, 19), 0.20, (0.05, 0.35)),
    Claim("ab-ds", "AB-bhrf1", "B1", "A", (0, 19), 1.95, (1.05, 2.85)),
    Claim("ab-mimo", "AB-nipah", "B1", "A", (0, 19), -0.15, (-0.75, 0.45)),
    Claim("ab-mimo", "AB-egfr", "B1", "A", (0, 19), 0.25, (-0.25, 0.80)),
    Claim("ab-mimo", "AB-bbf14", "B1", "A", (0, 19), 0.10, (0.00, 0.20)),
    Claim("ab-mimo", "AB-bhrf1", "B1", "A", (0, 19), 2.55, (1.85, 3.30)),
    # The held-out Nipah-G test (Figure 6 and the text).
    Claim("held-ds", "N-held", "B1", "A", (0, 39), 0.53, (-0.80, 1.80)),
    Claim("held-mimo", "N-held", "B1", "A", (0, 39), -0.38, (-2.33, 1.52)),
    Claim("held-ds", "N-held", "B1", LSP, (0, 39), 4.15),
    Claim("held-mimo", "N-held", "B1", LSP, (0, 39), 5.28),
    # The round-by-round table: B1 - A after round 2 in the main study, and the pooled
    # Anthropic gap after round 3.
    Claim("ds2", "N-dev", "B1", "A", (0, 19), 0.45, boundary=2),
    Claim("mimo2", "N-dev", "B1", "A", (0, 19), 2.55, boundary=2),
    Claim("egfr-ds", "E", "B1", "A", (100, 119), 2.10, boundary=2),
    Claim("egfr-mimo", "E", "B1", "A", (100, 119), -0.80, boundary=2),
    Claim("ab-ds", "AB pooled", "B1", "A", (0, 19), 0.96, pooled=AB, boundary=3),
    Claim("ab-mimo", "AB pooled", "B1", "A", (0, 19), 0.85, pooled=AB, boundary=3),
]
# Config hashes registered in docs/design-notes.md for the confirmatory runs, and in
# docs/prereg-anthropic.md and docs/prereg-heldout.md for the final tests.
FINAL_DS, FINAL_MIMO = "0fecb4a7e74a", "c2c8dac7e136"
REGISTERED = {
    "confirm1": "b457e246517b",
    "egfr-ds": "d103491a18c7",
    "egfr-mimo": "1665298211ef",
    "ab-ds": FINAL_DS,
    "held-ds": FINAL_DS,
    "ab-mimo": FINAL_MIMO,
    "held-mimo": FINAL_MIMO,
}
REWRITE_TABLE = c.ROOT / "docs/history-rewrite.md"


def rewritten_commits() -> dict[str, str]:
    """Old commit -> new commit, from the table in ``docs/history-rewrite.md``."""
    rows = re.findall(r"^\| `([0-9a-f]{40})` \| `([0-9a-f]{40})` \|$", REWRITE_TABLE.read_text(),
                      re.MULTILINE)  # fmt: skip
    return dict(rows)


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
    rewritten = rewritten_commits()
    missing, mapped = [], 0
    for sha in sorted(shas):
        if commit_exists(sha):
            continue
        if sha in rewritten and commit_exists(rewritten[sha]):
            mapped += 1
        else:
            missing.append(sha)
    report.note(
        f"2. recorded commits: {len(shas)}; found through the history-rewrite table: {mapped};"
        f" missing from this repository: {len(missing)}"
    )
    for sha in missing:
        report.fail(f"2. recorded commit {sha} is not in this repository")

    # 3. The reported comparisons.
    for claim in CLAIMS:
        label = (
            f"{claim.run} {claim.game} {claim.a} - {claim.b} ({claim.trials[0]}-{claim.trials[1]})"
        )
        if claim.boundary is not None:
            label += f" after round {claim.boundary}"
        block = range(claim.trials[0], claim.trials[1] + 1)
        keys = [(game, t) for game in claim.games for t in block]
        sides = {}
        for side in (claim.a, claim.b):
            rows = [
                r
                for r in runs.get(claim.run, [])
                if r["game"] in claim.games and r["condition"] == side
            ]
            per_key = defaultdict(list)
            for r in rows:
                if r["trial"] in block:
                    hits = r["scores"]["hits"]
                    if claim.boundary is not None:
                        hits = r["scores"]["hits_at_boundaries"][claim.boundary - 1]
                    per_key[(r["game"], r["trial"])].append(hits)
            if sorted(per_key) != sorted(keys) or any(len(v) != 1 for v in per_key.values()):
                report.fail(f"3. {label}: {side} does not have exactly one record per trial")
                sides = {}
                break
            sides[side] = per_key
        if not sides:
            continue
        diffs = np.array([sides[claim.a][k][0] - sides[claim.b][k][0] for k in keys], dtype=float)
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
