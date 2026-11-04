"""Audit 1: re-score every agent episode from the raw Proteinbase export.

    python scripts/audit/rescore.py [runs/x/episodes.jsonl ...]   # default: every run

For each recorded agent episode, independently of ``replaylab``:

1. rebuild the game's pool from the raw CSV and the episode's aliases from its seed;
2. read the designs it bought (every accepted ``submit_batch``, in order);
3. look up their lab results in the raw CSV and recompute the score card (hits, curve,
   hits per round, purchases, rounds used, best KD, distinct families);
4. compare with the score recorded in the episode, field by field, exactly;
5. check every outcome the agent was shown (submit_batch, wait, get_history) against
   the raw lab result, exactly (KD values are never rounded).

A pass means the replay served the real outcomes under the right aliases and the
recorded scores are what those purchases earned. Scripted episodes record only their
scores, not their purchases, so they cannot be re-scored here (audit 3 checks the
scripted random baseline against its analytical expectation instead).
"""

from __future__ import annotations

import sys
from collections import Counter

import _common as c

OUTCOME_TOOLS = ("submit_batch", "wait", "get_history")


def main(arguments: list[str]) -> int:
    report = Report("audit 1: re-score agent episodes from the raw export")
    raw, games = c.load_raw(), c.load_games()
    folds, clusters = c.read_csv(c.FOLDS), c.read_csv(c.CLUSTERS)
    cluster_of = {(r["design_id"], r["target"]): int(r["cluster_id"]) for r in clusters}
    pools: dict[str, list[str]] = {}
    checked: Counter[str] = Counter()
    shown = skipped = 0

    for path, record in c.read_records(c.record_paths(arguments)):
        if not c.is_agent(record):
            continue
        where = f"{path.parent.name} {record['game']}/{record['condition']}/{record['trial']}"
        game = games.get(record["game"])
        if game is None or game["target"] not in raw.pool_by_target:
            skipped += 1
            continue
        target = game["target"]
        if record["game"] not in pools:
            pools[record["game"]] = c.pool_ids(raw, game, folds)
        alias_to_real = c.aliases(
            pools[record["game"]], c.episode_seed(record["game"], record["trial"])
        )

        # The purchases, batch by batch, as the agent made them.
        batches = [
            call["arguments"]["design_ids"]
            for call in record["tool_calls"]
            if call["name"] == "submit_batch" and call["ok"]
        ]
        bought = [a for batch in batches for a in batch]
        unknown = [a for a in bought if a not in alias_to_real]
        if unknown:
            report.fail(f"{where}: bought aliases not in the pool: {unknown[:3]}")
            continue
        report.check(len(set(bought)) == len(bought), f"{where}: a design was bought twice")
        report.check(len(bought) <= game["budget"], f"{where}: {len(bought)} purchases over budget")
        report.check(len(batches) <= game["rounds"], f"{where}: {len(batches)} batches over rounds")

        # The score card those purchases earn, from the raw lab results.
        curve, boundaries, hits = [], [], []
        total = 0
        for batch in batches:
            for alias in batch:
                if raw.truth[(alias_to_real[alias], target)].hit:
                    total += 1
                    hits.append(alias_to_real[alias])
                curve.append(total)
            boundaries.append(total)
        kds = [
            raw.truth[(d, target)].kd_molar
            for d in hits
            if raw.truth[(d, target)].kd_molar is not None
        ]
        families = (
            len({cluster_of[(d, target)] for d in hits})
            if all((d, target) in cluster_of for d in hits)
            else None
        )
        expected = {
            "hits": total,
            "purchases": len(curve),
            "curve": curve,
            "hits_at_boundaries": boundaries,
            "rounds_used": len(batches),
            "best_kd": min(kds) if kds else None,
            "distinct_families": families,
        }
        for key, value in expected.items():
            if record["scores"].get(key) != value:
                report.fail(
                    f"{where}: {key} recorded {record['scores'].get(key)!r}, raw gives {value!r}"
                )
        checked[record["game"]] += 1

        # Every outcome the agent was shown must be the raw lab result, exactly.
        for call, text in c.calls_with_results(record):
            if call["name"] not in OUTCOME_TOOLS or not call["ok"]:
                continue
            data = c.parse(text) or {}
            for row in data.get("results", []) + data.get("designs", []):
                truth = raw.truth[(alias_to_real[row["design_id"]], target)]
                shown += 1
                if (row["hit"], row["strength"], row["kd_molar"]) != (
                    truth.hit,
                    truth.strength,
                    truth.kd_molar,
                ):
                    report.fail(
                        f"{where}: {call['name']} showed {row['design_id']} as "
                        f"{(row['hit'], row['strength'], row['kd_molar'])}, raw is "
                        f"{(truth.hit, truth.strength, truth.kd_molar)}"
                    )

    report.note(f"agent episodes re-scored: {dict(checked)} ({sum(checked.values())} in all)")
    report.note(f"outcomes shown to agents and checked against raw: {shown}")
    if skipped:
        report.note(f"skipped {skipped} episode(s) on games whose target is not in the raw export")
    report.check(sum(checked.values()) > 0, "no agent episode was checked")
    return report.finish()


Report = c.Report

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
