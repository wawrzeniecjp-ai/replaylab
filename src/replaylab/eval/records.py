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
    the code that played it; the runner records such a run only with
    ``--allow-uncommitted``. A line without the key means "not checked".
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
    """Append one record as a single flushed, fsynced line.

    A half-written last line left by a crash is cut off first, so the new record cannot be
    glued onto it.

    Args:
        path (Path | str): the episodes file (created with its folder if missing).
        record (EpisodeRecord | Mapping[str, Any]): the record.
    """
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
    """Read every record of an episodes file.

    Args:
        path (Path | str): the episodes file.

    Returns:
        list[dict[str, Any]]: the records in file order; empty if the file does not exist.
        A half-written last line is ignored.

    Raises:
        CorruptRecords: a line other than the last is not valid JSON.
    """
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
    """Collect the (game, condition, trial) identities of ``records``: what the runner skips.

    Args:
        records (Iterable[Mapping[str, Any]]): episode records.

    Returns:
        set[tuple[str, str, int]]: their keys.
    """
    return {(r["game"], r["condition"], int(r["trial"])) for r in records}


def config_hash(parts: Mapping[str, Any]) -> str:
    """Hash a run configuration.

    Args:
        parts (Mapping[str, Any]): the configuration; serialized as canonical JSON (sorted
            keys, ``str`` for anything JSON cannot hold).

    Returns:
        str: the first 12 hex characters of the sha256.
    """
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def head_sha(repo_dir: Path | str = ".") -> str | None:
    """Return the commit checked out in ``repo_dir``.

    Whether the tree is clean is checked separately, by
    :func:`replaylab.eval.final_run.uncommitted_frozen_paths`.

    Args:
        repo_dir (Path | str): a directory inside the repository.

    Returns:
        str | None: the full sha of ``HEAD``, or None when ``repo_dir`` is not in a git
        repository or git cannot be run.
    """
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
