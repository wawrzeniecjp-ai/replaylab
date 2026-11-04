"""Audit 2: scan every agent transcript for information the agent must not have.

    python scripts/audit/leaks.py [runs/x/episodes.jsonl ...]   # default: every run

The rule (README Methods): nothing the agent sees contains a sequence, a real design id,
or an outcome for a design that was not bought *and delivered*. Checked on what the
agent actually received, message by message, independently of ``replaylab``:

1. **Real ids and sequences.** No real Proteinbase id appears in any message, and no
   amino-acid run of 10+ letters that occurs in a real sequence.
2. **Per-design outcomes.** Every row that carries ``hit``/``strength``/``kd_molar``
   names a design delivered by then. Delivery is simulated from the documented clock
   rule: a batch submitted at clock c is due once the clock reaches c + delay; each
   submit polls, then ticks the clock once and polls again; each wait ticks once.
3. **Indirect hints.** A ``nearest_hit`` must be a delivered hit; the aggregate counts
   (``hits_found``, ``campaign_summary.hits``, ``best_kd``, the cumulative curve) must
   equal what the delivered designs give, so no aggregate counts an undelivered result.
4. **Raw values.** Every feature value shown (query_pool rows, history rows) equals the
   raw export's value for that design: never re-oriented or rounded. Where the export
   holds several records of one metric for a design (two esmfold_plddt values for most
   designs), the package shows their mean; that rule is checked, and how often it
   applied is reported, because the README does not mention it.
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from typing import Any

import _common as c

ID_PATTERN = re.compile(r"\b[a-z]+(?:-[a-z]+){2}\b")  # real ids look like "amber-bat-vine"
AA_RUN = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{10,}")
OUTCOME_KEYS = ("hit", "strength", "kd_molar")
NOT_FEATURES = {"design_id", "length", "design_class", "batch_id", "round", "features"} | set(
    OUTCOME_KEYS
)


def rows_with_outcomes(value: Any) -> list[dict[str, Any]]:
    """Every dict anywhere in a tool result that names a design and carries an outcome."""
    found = []
    if isinstance(value, dict):
        if "design_id" in value and any(k in value for k in OUTCOME_KEYS):
            found.append(value)
        for child in value.values():
            found += rows_with_outcomes(child)
    elif isinstance(value, list):
        for child in value:
            found += rows_with_outcomes(child)
    return found


def nearest_hits(value: Any) -> list[str]:
    found = []
    if isinstance(value, dict):
        if isinstance(value.get("nearest_hit"), str):
            found.append(value["nearest_hit"])
        for child in value.values():
            found += nearest_hits(child)
    elif isinstance(value, list):
        for child in value:
            found += nearest_hits(child)
    return found


def due(submitted: list[tuple[list[str], int]], clock: int, delay: int) -> set[str]:
    """Designs whose batch is due at ``clock``: submitted at c, due once clock >= c + delay."""
    return {d for ids, at in submitted if clock >= at + delay for d in ids}


def main(arguments: list[str]) -> int:
    report = c.Report("audit 2: leak scan of agent transcripts")
    raw, games = c.load_raw(), c.load_games()
    folds = c.read_csv(c.FOLDS)
    real_ids = set(raw.sequences)
    sequences = "\n".join(raw.sequences.values())
    pools: dict[str, list[str]] = {}
    counts: Counter[str] = Counter()

    for path, record in c.read_records(c.record_paths(arguments)):
        if not c.is_agent(record):
            continue
        where = f"{path.parent.name} {record['game']}/{record['condition']}/{record['trial']}"
        game = games.get(record["game"])
        if game is None or game["target"] not in raw.pool_by_target:
            continue
        target, delay = game["target"], int(game.get("delay_rounds", 0))
        if record["game"] not in pools:
            pools[record["game"]] = c.pool_ids(raw, game, folds)
        alias_to_real = c.aliases(
            pools[record["game"]], c.episode_seed(record["game"], record["trial"])
        )
        counts["episodes"] += 1

        # 1. Real ids and sequences, anywhere in what the agent was sent.
        for message in record["messages"]:
            if message.get("role") == "assistant":
                continue
            text = message.get("content") or ""
            for token in ID_PATTERN.findall(text):
                if token in real_ids:
                    report.fail(f"{where}: real design id {token!r} in a {message['role']} message")
            for run in AA_RUN.findall(text):
                if run in sequences:
                    report.fail(
                        f"{where}: sequence fragment {run[:20]!r} in a {message['role']} message"
                    )

        # 2-4. Replay the calls in order, tracking the clock and what has been delivered.
        clock = 0
        submitted: list[tuple[list[str], int]] = []
        delivered: set[str] = set()

        for call, text in c.calls_with_results(record):
            if call["name"] == "submit_batch" and call["ok"]:
                submitted.append((call["arguments"]["design_ids"], clock))
                delivered |= due(submitted, clock, delay)
                clock += 1
                delivered |= due(submitted, clock, delay)
            elif call["name"] == "wait" and call["ok"]:
                clock += 1
                delivered |= due(submitted, clock, delay)
            data = c.parse(text)
            if data is None:
                continue
            counts["tool results"] += 1
            truth = {a: raw.truth[(alias_to_real[a], target)] for a in delivered}
            delivered_hits = [a for a in delivered if truth[a].hit]

            for row in rows_with_outcomes(data):
                counts["outcome rows"] += 1
                if row["design_id"] not in delivered:
                    report.fail(
                        f"{where}: {call['name']} gave an outcome for undelivered "
                        f"{row['design_id']}"
                    )
            for alias in nearest_hits(data):
                counts["nearest_hit hints"] += 1
                if alias not in delivered_hits:
                    report.fail(
                        f"{where}: {call['name']} named {alias} as a hit "
                        "before it was a delivered hit"
                    )

            if call["name"] == "describe_campaign":
                report.check(
                    data["hits_found"] == len(delivered_hits),
                    f"{where}: hits_found {data['hits_found']} "
                    f"but {len(delivered_hits)} delivered hits",
                )
            if call["name"] == "campaign_summary":
                kds = [truth[a].kd_molar for a in delivered_hits if truth[a].kd_molar is not None]
                report.check(
                    data["hits"] == len(delivered_hits), f"{where}: summary hits {data['hits']}"
                )
                report.check(
                    data["best_kd"] == (min(kds) if kds else None), f"{where}: summary best_kd"
                )
                curve = data["cumulative_curve"]
                report.check(
                    not curve or curve[-1] == len(delivered_hits),
                    f"{where}: summary curve ends at {curve[-1:]}",
                )

            # Raw values: features of query_pool rows and history rows.
            rows = data.get("rows", []) if call["name"] == "query_pool" else []
            rows += [
                {**r["features"], "design_id": r["design_id"]}
                for r in data.get("designs", [])
                if "features" in r
            ]
            for row in rows:
                real = alias_to_real[row["design_id"]]
                values = raw.metrics[(real, target)]
                for column, shown in row.items():
                    if column in NOT_FEATURES or shown is None or column.startswith("class="):
                        continue
                    counts["feature values"] += 1
                    published = values.get(column, [])
                    if len(published) > 1:
                        counts["feature values that are a mean of several records"] += 1
                    expected = sum(published) / len(published) if published else None
                    if expected is None or not math.isclose(
                        shown, expected, rel_tol=1e-12, abs_tol=0
                    ):
                        report.fail(
                            f"{where}: {column} of {row['design_id']} shown {shown}, "
                            f"raw {published}"
                        )
                if "length" in row:
                    counts["feature values"] += 1
                    report.check(
                        row["length"] == len(raw.sequences[real]),
                        f"{where}: length of {row['design_id']}",
                    )
                if "design_class" in row:
                    counts["feature values"] += 1
                    report.check(
                        row["design_class"] == raw.design_class[real],
                        f"{where}: design_class of {row['design_id']}",
                    )

    report.note("checked: " + ", ".join(f"{n} {k}" for k, n in counts.items()))
    report.check(counts["episodes"] > 0, "no agent episode was checked")
    return report.finish()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
