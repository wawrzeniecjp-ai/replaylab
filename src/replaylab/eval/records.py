"""Episode records: one JSON line per episode (spec 8.2).

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

SCHEMA_VERSION = 1


class CorruptRecords(Exception):
    """A line that is not the last one cannot be parsed."""


@dataclass
class EpisodeRecord:
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
    flags: dict[str, int] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.game, self.condition, self.trial)

    def to_dict(self) -> dict[str, Any]:
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
    data = record.to_dict() if isinstance(record, EpisodeRecord) else dict(record)
    line = json.dumps(data, sort_keys=True, separators=(",", ":"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _drop_partial_tail(path)
    with path.open("a", encoding="utf-8") as handle:
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
