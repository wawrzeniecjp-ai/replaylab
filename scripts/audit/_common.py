"""Shared helpers for the audit scripts, written without importing ``replaylab``.

Everything here is rebuilt from the published inputs only:

* ``data/raw/proteinbase.csv``: the Proteinbase export (sequences, computational
  scores, lab results), parsed directly;
* ``data/raw/anthropic/.../tables``: the Anthropic release's ``design_summary`` and
  ``cofold_predictions`` (the final test's pools), read with the rules of
  ``docs/prereg-anthropic.md``;
* ``data/splits/folds.csv`` and ``data/splits/clusters.csv``: the fixed split and
  cluster files, read as plain tables;
* ``config/games.yaml``: which target, split and budget each game uses;
* ``runs/*/episodes.jsonl``: the recorded episodes.

The point is independence: if the package had a bug in how it builds pools, aliases,
outcomes or scores, these helpers would not share it, because they implement the
rules again from their written description (README Methods), not from the code.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/proteinbase.csv"
ANTHROPIC = ROOT / "data/raw/anthropic/anthropic_binders_core/data/tables"
FOLDS = ROOT / "data/splits/folds.csv"
CLUSTERS = ROOT / "data/splits/clusters.csv"
GAMES = ROOT / "config/games.yaml"
HELDOUT_FOLD = 2
STRENGTH_RANK = {"Weak": 1, "Medium": 2, "Strong": 3}


@dataclass(frozen=True)
class Truth:
    """What the lab says about one (design, target) pair."""

    hit: bool  # at least one binding record says True
    kd_molar: float | None  # the smallest KD on record (all KDs in the export are molar)
    strength: str | None  # the strongest strength label on record


@dataclass
class Raw:
    """The Proteinbase export, indexed for the audits."""

    sequences: dict[str, str]  # design id -> sequence
    truth: dict[tuple[str, str], Truth]  # (design id, target) -> lab result
    # (design id, target) -> numeric computational metric -> every published value.
    # Metrics without a target apply to every target of the design. Most designs carry
    # two esmfold_plddt records; the package shows their mean (see leaks.py).
    metrics: dict[tuple[str, str], dict[str, list[float]]]
    design_class: dict[str, str | None]
    pool_by_target: dict[str, list[str]] = field(default_factory=dict)


def load_raw(path: Path = RAW) -> Raw:
    """Parse the raw CSV: one row per design, its records in a JSON ``evaluations`` list."""
    csv.field_size_limit(sys.maxsize)
    sequences: dict[str, str] = {}
    binding: dict[tuple[str, str], list[bool]] = {}
    kds: dict[tuple[str, str], list[float]] = {}
    strengths: dict[tuple[str, str], list[str]] = {}
    shared: dict[str, dict[str, list[float]]] = {}
    targeted: dict[tuple[str, str], dict[str, list[float]]] = {}
    design_class: dict[str, str | None] = {}

    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            design = row["id"]
            sequences[design] = row["sequence"]
            design_class[design] = None
            for record in json.loads(row["evaluations"] or "[]"):
                kind, metric, value = record.get("type"), record.get("metric"), record.get("value")
                target = record.get("target")
                if kind == "experimental" and metric == "binding" and target:
                    if isinstance(value, bool):
                        binding.setdefault((design, target), []).append(value)
                elif kind == "experimental" and metric == "kd" and target:
                    if record.get("unit") != "M":
                        raise ValueError(f"{design}: KD unit {record.get('unit')!r} is not molar")
                    kds.setdefault((design, target), []).append(float(value))
                elif kind == "experimental" and metric == "binding_strength" and target:
                    if isinstance(value, str) and value.strip().title() in STRENGTH_RANK:
                        strengths.setdefault((design, target), []).append(value.strip().title())
                elif kind == "computational" and metric == "design_class":
                    design_class[design] = value
                elif kind == "computational" and isinstance(value, int | float):
                    if isinstance(value, bool):
                        continue
                    bucket = (
                        targeted.setdefault((design, target), {})
                        if target
                        else shared.setdefault(design, {})
                    )
                    bucket.setdefault(metric, []).append(float(value))

    truth: dict[tuple[str, str], Truth] = {}
    pool_by_target: dict[str, list[str]] = {}
    metrics: dict[tuple[str, str], dict[str, list[float]]] = {}
    for (design, target), flags in binding.items():
        best = max(strengths.get((design, target), []), key=STRENGTH_RANK.get, default=None)
        truth[(design, target)] = Truth(
            hit=any(flags),
            kd_molar=min(kds[(design, target)]) if (design, target) in kds else None,
            strength=best,
        )
        pool_by_target.setdefault(target, []).append(design)
        metrics[(design, target)] = {**shared.get(design, {}), **targeted.get((design, target), {})}
    for target in pool_by_target:
        pool_by_target[target].sort()
    return Raw(sequences, truth, metrics, design_class, pool_by_target)


# The Anthropic release (docs/prereg-anthropic.md): replaylab feature -> (predictor, column).
ANTHROPIC_FEATURES = {
    "boltz2_min_ipsae": ("boltz2", "ipsae_min"),
    "boltz2_iptm": ("boltz2", "iptm_pae"),
    "boltz2_plddt": ("boltz2", "plddt_binder"),
    "esmfold_plddt": ("ef2full", "plddt_binder"),
}
ANTHROPIC_DESIGN_STOICH = {"1to1", "1to2", "1to3", "rnp"}


def anthropic_target(code: str) -> str:
    """The target id of a release target code: "ab-" and the code in lower-case words."""
    return "ab-" + "-".join(re.findall(r"[a-z0-9]+", code.lower()))


def load_anthropic(tables: Path = ANTHROPIC) -> Raw:
    """The Anthropic release, read with the pre-registered rules, as a :class:`Raw`.

    * A target's pool is every design with a binder call (``binder_final`` not null).
    * A hit is ``binder_final``; the KD is ``kd_nM_final`` in molar; no strength label.
    * Features are the co-fold row the release bundles per design and predictor: on the
      design target (empty ``target_form``) at a design-complex stoichiometry, the highest
      ``ipsae_min``, ties to the lowest seed number. Each is one published value.
    * The design class is the ``generator`` column.
    """
    import pandas as pd  # parquet; the Proteinbase reader above needs only the csv module

    summary = pd.read_parquet(tables / "design_summary.parquet")
    cofold = pd.read_parquet(tables / "insilico/cofold_predictions.parquet")
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in cofold.to_dict("records"):
        if row["target_form"] != "" or row["stoichiometry"] not in ANTHROPIC_DESIGN_STOICH:
            continue
        key = (row["uuid"], row["cofolding_model"])
        rank = (-float(row["ipsae_min"]), int(row["seed"]))
        if key not in best or rank < best[key]["_rank"]:
            best[key] = {**row, "_rank": rank}

    raw = Raw({}, {}, {}, {})
    for row in summary.to_dict("records"):
        design = row["uuid"]
        raw.sequences[design] = row["sequence"]
        raw.design_class[design] = row["generator"]
        if row["binder_final"] is None or row["binder_final"] != row["binder_final"]:
            continue  # no binder call: not in any pool
        target = anthropic_target(row["target"])
        kd = row["kd_nM_final"]
        raw.truth[(design, target)] = Truth(
            hit=bool(row["binder_final"]),
            kd_molar=None if kd is None or kd != kd else float(kd) * 1e-9,
            strength=None,
        )
        raw.metrics[(design, target)] = {
            name: [float(best[(design, predictor)][column])]
            for name, (predictor, column) in ANTHROPIC_FEATURES.items()
            if (design, predictor) in best
            and best[(design, predictor)][column] == best[(design, predictor)][column]
        }
        raw.pool_by_target.setdefault(target, []).append(design)
    for ids in raw.pool_by_target.values():
        ids.sort()
    return raw


def load_games(path: Path = GAMES) -> dict[str, dict[str, Any]]:
    return yaml.safe_load(path.read_text())["games"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def pool_ids(
    raw: Raw, game: dict[str, Any], folds: list[dict[str, str]] | None = None
) -> list[str]:
    """The designs of a game: every design with a binding record on its target, cut by split."""
    ids = raw.pool_by_target.get(game["target"], [])
    split = game.get("split", "all")
    if split == "all":
        return list(ids)
    folds = folds if folds is not None else read_csv(FOLDS)
    fold = {r["design_id"]: int(r["fold"]) for r in folds if r["target"] == game["target"]}
    if split == "heldout":
        return [d for d in ids if fold[d] == HELDOUT_FOLD]
    return [d for d in ids if fold[d] != HELDOUT_FOLD]


def episode_seed(game: str, trial: int) -> int:
    """The documented seed rule: first 8 bytes of sha256("<game>|<trial>"), big-endian."""
    return int.from_bytes(hashlib.sha256(f"{game}|{trial}".encode()).digest()[:8], "big")


def aliases(real_ids: Iterable[str], seed: int) -> dict[str, str]:
    """alias -> real id: ids sorted on sha256("<seed>|<id>"), named d0001, d0002, ..."""
    ordered = sorted(
        real_ids, key=lambda d: (hashlib.sha256(f"{seed}|{d}".encode()).hexdigest(), d)
    )
    width = max(4, len(str(len(ordered))))
    return {f"d{i:0{width}d}": real for i, real in enumerate(ordered, 1)}


def read_records(paths: Iterable[Path]) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Every complete line of the given episodes.jsonl files, with the file it came from."""
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        yield path, json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a half-written last line is not a record


def record_paths(arguments: list[str]) -> list[Path]:
    """The episodes files to audit: the arguments, or every runs/*/episodes.jsonl."""
    if arguments:
        return [Path(a) for a in arguments]
    return sorted((ROOT / "runs").glob("*/episodes.jsonl"))


def is_agent(record: dict[str, Any]) -> bool:
    return not record["condition"].startswith("scripted:")


def calls_with_results(record: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    """Each tool call paired with the text the agent got back, in the order they ran.

    The loop appends one tool message per call, in call order, so the k-th tool
    message answers the k-th call; the count is checked rather than assumed.
    """
    replies = [m.get("content") or "" for m in record["messages"] if m.get("role") == "tool"]
    calls = record["tool_calls"]
    if len(replies) != len(calls):
        raise ValueError(f"{len(calls)} tool calls but {len(replies)} tool messages")
    return list(zip(calls, replies, strict=True))


def parse(text: str) -> Any:
    if text.startswith("ERROR"):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


class Report:
    """Collects pass/fail lines and prints a verdict; exit code 1 on any failure."""

    def __init__(self, title: str) -> None:
        self.title = title
        self.failures: list[str] = []
        self.notes: list[str] = []
        print(f"== {title}")

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)
        print(f"   {message}")

    def check(self, condition: bool, message: str) -> None:
        if not condition:
            self.fail(message)

    def finish(self) -> int:
        for line in self.failures[:50]:
            print(f"   FAIL {line}")
        if len(self.failures) > 50:
            print(f"   ... and {len(self.failures) - 50} more failures")
        verdict = "PASS" if not self.failures else f"FAIL ({len(self.failures)})"
        print(f"== {self.title}: {verdict}\n")
        return 0 if not self.failures else 1
