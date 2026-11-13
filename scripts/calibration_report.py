"""How well the agents' hit-rate forecasts matched what their batches delivered.

    python scripts/calibration_report.py --runs runs

Reads the recorded transcripts of the study's blocks (``scripts/export_study.py``: the main
study and the final tests) and writes two committed CSVs, with no design ids:

* ``reports/calibration_forecasts.csv``: one row per ``estimate_hit_rate(next_batch=...)``
  result the agent saw: its round, batch size, predicted rate, 90% interval and ``calibrated``
  flag, and, when the agent then bought exactly that batch, the realized rate and whether the
  interval covered it.
* ``reports/calibration_proposals.csv``: one row per ``propose_batch`` result: its round,
  batch size, basis (``model`` or ``posterior_mean``) and predicted rate, whether the agent
  bought exactly that batch, and if so its realized rate.

A forecast or a proposal is scored only against a batch with **exactly** the same designs,
bought in the same round, because a prediction is about that batch and no other. Rows the
agent did not buy keep an empty ``realized``. Then it prints the summary that design notes
section 21 reports.

Descriptive only: every episode here is from trials already played.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from export_study import BLOCKS

FORECAST_FIELDS = ["run", "game", "model", "condition", "trial", "round", "k", "predicted",
                   "lower", "upper", "calibrated", "realized", "covered"]  # fmt: skip
PROPOSAL_FIELDS = ["run", "game", "model", "condition", "trial", "round", "k", "basis",
                   "predicted", "bought", "realized"]  # fmt: skip


def _load(text):
    """A tool call's arguments or a tool result as a dict or list; a model sometimes sends
    them as a JSON string, and a failed call's result is plain text."""
    if isinstance(text, str):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    return text


def _ids(value) -> frozenset[str] | None:
    value = _load(value)
    return frozenset(value) if isinstance(value, list) else None


def episode_predictions(record: dict) -> tuple[list[dict], list[dict]]:
    """The forecasts and proposals of one agent episode, each matched to its round's purchase.

    Returns:
        (forecasts, proposals): dicts with the fields of ``FORECAST_FIELDS`` and
        ``PROPOSAL_FIELDS`` past the episode key. A prediction is matched to the batch bought
        at the end of the round it was made in, and only if that batch has exactly its designs.
    """
    replies = {m.get("tool_call_id"): m.get("content") for m in record["messages"]
               if m.get("role") == "tool"}  # fmt: skip
    forecasts: list[dict] = []
    proposals: list[dict] = []
    open_forecasts: list[tuple[frozenset[str], dict]] = []
    open_proposals: list[tuple[frozenset[str], dict]] = []
    round_no = 1
    for message in record["messages"]:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function", call)
            name, arguments = function.get("name"), _load(function.get("arguments")) or {}
            result = _load(replies.get(call.get("id")))
            if not isinstance(arguments, dict) or not isinstance(result, dict):
                continue  # a failed call: nothing was predicted or bought
            if name == "estimate_hit_rate" and result.get("next_batch"):
                ids = _ids(arguments.get("next_batch"))
                nb = result["next_batch"]
                row = {
                    "round": round_no,
                    "k": len(ids or ()),
                    "predicted": nb["predicted_rate"],
                    "lower": nb["interval90"][0],
                    "upper": nb["interval90"][1],
                    "calibrated": nb["calibrated"],
                    "realized": None,
                    "covered": None,
                }
                forecasts.append(row)
                if ids:
                    open_forecasts.append((ids, row))
            elif name == "propose_batch" and "predicted_hit_rate" in result:
                ids = frozenset(result.get("design_ids") or ())
                row = {"round": round_no, "k": len(ids),
                       "basis": (result.get("calibration") or {}).get("basis"),
                       "predicted": result["predicted_hit_rate"], "bought": False,
                       "realized": None}  # fmt: skip
                proposals.append(row)
                open_proposals.append((ids, row))
            elif name == "submit_batch" and result.get("results"):
                bought = _ids(arguments.get("design_ids"))
                hits = [bool(r.get("hit")) for r in result["results"]]
                realized = (
                    sum(hits) / len(hits) if hits and bought and len(hits) == len(bought) else None
                )
                if realized is not None:
                    # The last forecast of exactly this batch is the one the agent acted on.
                    for ids, row in reversed(open_forecasts):
                        if ids == bought:
                            row["realized"] = realized
                            row["covered"] = row["lower"] <= realized <= row["upper"]
                            break
                    for ids, row in open_proposals:
                        if ids == bought:
                            row["bought"], row["realized"] = True, realized
                open_forecasts, open_proposals = [], []
                round_no += 1
    return forecasts, proposals


def collect(runs_dir: Path):
    forecasts, proposals = [], []
    for run, blocks in sorted(BLOCKS.items()):
        wanted = dict(blocks)
        path = runs_dir / run / "episodes.jsonl"
        if not path.exists():
            print(f"  skipped {run}: no {path}")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["trial"] not in wanted.get(record["game"], ()):
                continue
            if record["condition"].startswith("scripted:"):
                continue
            key = {"run": run, "game": record["game"], "model": record["model"],
                   "condition": record["condition"], "trial": record["trial"]}  # fmt: skip
            f, p = episode_predictions(record)
            forecasts += [{**key, **row} for row in f]
            proposals += [{**key, **row} for row in p]
    return forecasts, proposals


def summary(forecasts: list[dict], proposals: list[dict]) -> None:
    print(f"\nnext-batch forecasts: {len(forecasts)}; calibrated: "
          f"{sum(bool(f['calibrated']) for f in forecasts)}")  # fmt: skip
    scored = [f for f in forecasts if f["realized"] is not None]
    if scored:
        bias = statistics.fmean(f["realized"] - f["predicted"] for f in scored)
        mae = statistics.fmean(abs(f["realized"] - f["predicted"]) for f in scored)
        width = statistics.fmean(f["upper"] - f["lower"] for f in scored)
        covered = sum(f["covered"] for f in scored)
        print(f"  of the batch then bought: {len(scored)}; covered {covered}"
              f" ({covered / len(scored):.1%}); bias {bias:+.3f}; MAE {mae:.3f};"
              f" mean width {width:.3f}")  # fmt: skip
        for label, key in (("game", "game"), ("model", "model"), ("round", "round")):
            groups = defaultdict(list)
            for f in scored:
                groups[f[key]].append(f)
            parts = [f"{g} {sum(x['covered'] for x in v)}/{len(v)}"
                     for g, v in sorted(groups.items())]  # fmt: skip
            print(f"  coverage by {label}: " + ", ".join(parts))
    print(f"\nproposals: {len(proposals)}; bought exactly as proposed: "
          f"{sum(p['bought'] for p in proposals)}")  # fmt: skip
    for basis in ("posterior_mean", "model"):
        rows = [p for p in proposals if p["basis"] == basis]
        bought = [p for p in rows if p["bought"]]
        line = f"  {basis}: {len(rows)}, bought {len(bought)}"
        if bought:
            bias = statistics.fmean(p["realized"] - p["predicted"] for p in bought)
            line += f"; bias of the bought ones {bias:+.3f}"
        print(line + "; by round " + str(dict(sorted(Counter(p["round"] for p in rows).items()))))
    # Residuals the calibration could ever have used: model-based proposals bought exactly.
    per_episode = Counter(
        (p["run"], p["game"], p["condition"], p["trial"])
        for p in proposals
        if p["basis"] == "model" and p["bought"]
    )
    print("  realized model-based residuals per episode that had any: "
          + str(dict(sorted(Counter(per_episode.values()).items()))))  # fmt: skip


def write(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()
    forecasts, proposals = collect(args.runs)
    write(args.out / "calibration_forecasts.csv", FORECAST_FIELDS, forecasts)
    write(args.out / "calibration_proposals.csv", PROPOSAL_FIELDS, proposals)
    summary(forecasts, proposals)


if __name__ == "__main__":
    main()
