"""Episode records: one JSON line per episode.

Lines are appended and flushed, so a crash can only damage the last line; readers ignore a
half-written last line. Scripted policies write the same format with the condition
``scripted:<name>``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from replaylab.engine import ScoreCard

SCHEMA_VERSION = 2  # 2 (2026-10-28): adds `reasoning` (one entry per assistant
# message, None when the endpoint returned none); lines written before that
# simply lack the key and readers must use .get().


def scores_dict(card: ScoreCard) -> dict[str, Any]:
    """Return a score card as the JSON-ready dict a record's ``scores`` field holds.

    Args:
        card (ScoreCard): the score of a settled campaign.

    Returns:
        dict[str, Any]: hits, purchases, rounds used, early stop, distinct families,
        best Kd, the hit curve and the hits at each round boundary.
    """
    return {
        "hits": card.hits,
        "purchases": card.purchases,
        "rounds_used": card.rounds_used,
        "stopped_early": card.stopped_early,
        "distinct_families": card.distinct_families,
        "best_kd": card.best_kd,
        "curve": list(card.curve),
        "hits_at_boundaries": list(card.hits_at_boundaries),
    }


class CorruptRecords(Exception):
    """A line that is not the last one cannot be parsed."""


@dataclass
class EpisodeRecord:
    """Everything one finished episode leaves behind: one JSON line in ``episodes.jsonl``.

    ``scores`` is the campaign's outcome, ``messages``/``tool_calls`` the full
    play-by-play, ``config_hash`` + ``git_sha`` pin what produced it, and
    ``reasoning`` (schema 2) holds one entry per assistant message.

    ``git_dirty`` (optional, no schema bump) says the run was recorded while
    ``src``/``prompts``/``config`` had uncommitted changes, so ``git_sha`` is not
    the code that played it. Lines written before the field existed lack the key
    and mean "not checked"; the runner refuses such a run unless it is asked to.
    """

    game: str
    condition: str
    trial: int
    seed: int
    model: str | None
    config_hash: str
    git_sha: str | None
    index_version: str | None
    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    usage: list[dict[str, Any]]
    cost_usd: float
    stop_reason: str
    turns: int
    scores: dict[str, Any]
    git_dirty: bool = False  # declared after the non-default fields; see the docstring
    flags: dict[str, int] = field(default_factory=dict)
    reasoning: list[str | None] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    @property
    def key(self) -> tuple[str, str, int]:
        """The identity used to skip completed episodes: (game, condition, trial)."""
        return (self.game, self.condition, self.trial)

    def to_dict(self) -> dict[str, Any]:
        """The JSON-ready mapping written to disk (``sort_keys`` at the door)."""
        return asdict(self)


def _drop_partial_tail(path: Path) -> None:
    """A run killed mid-write leaves a last line without its newline: cut it off.

    Without this the next record would be glued to the fragment and both would be lost.
    """
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as handle:
        data = handle.read()
        if data.endswith(b"\n"):
            return
        handle.truncate(data.rfind(b"\n") + 1)


def append_record(path: Path | str, record: EpisodeRecord | Mapping[str, Any]) -> None:
    """Append one record as a single flushed, fsynced line (see the module docstring)."""
    data = record.to_dict() if isinstance(record, EpisodeRecord) else dict(record)
    line = json.dumps(data, sort_keys=True, separators=(",", ":"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _drop_partial_tail(path)
    with path.open("a", encoding="utf-8") as handle:
        # One write per record, then flush Python's buffer and ask the OS to put it on
        # disk, so a record that was reported as saved survives a crash.
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_records(path: Path | str) -> list[dict[str, Any]]:
    """All parseable records. A half-written last line is ignored; any other bad line raises."""
    path = Path(path)
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = []
    for number, line in enumerate(lines, start=1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if number == len(lines):
                break
            raise CorruptRecords(f"{path}: line {number} is not valid JSON") from None
    return records


def completed_keys(records: Iterable[Mapping[str, Any]]) -> set[tuple[str, str, int]]:
    """The (game, condition, trial) identities already in ``records`` -- what the runner skips."""
    return {(r["game"], r["condition"], int(r["trial"])) for r in records}


def config_hash(parts: Mapping[str, Any]) -> str:
    """First 12 hex characters of the sha256 of the canonical JSON of the run configuration."""
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def git_info(repo_dir: Path | str = ".") -> tuple[str | None, bool]:
    """(HEAD sha or None, whether the working tree has uncommitted changes)."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None, False
    return sha, dirty
