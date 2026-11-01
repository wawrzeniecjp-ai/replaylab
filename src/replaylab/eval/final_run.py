"""The run-once guard of the final test (CLAUDE.md, planned direction A).

A game marked ``final_test`` in ``config/games.yaml`` is played only by the runner, and only
under a :class:`FinalRunPermit`; :func:`acquire` is the one way to get one. Everything else
(``serve``, the campaign CLI, the ladder, the replay page) refuses the game outright.

The lock file ``<lock_dir>/<game>.json`` is meant to be committed with the results. It
records:

* ``frozen_hash``: prompts, conditions, tool descriptions, limits, and the game's own
  ``games.yaml`` and ``features.yaml`` entries (:func:`frozen_hash`). Every leg of the final
  run must share it: a change after the first leg started is refused.
* ``git_sha``: the commit every leg is played from (a dirty ``src``, ``prompts`` or
  ``config`` is refused before that, by :func:`dirty_paths` in the CLI).
* one entry per leg (a model key, or ``scripted``): its config hash, records file, trial
  block and conditions (the policies, for ``scripted``).

Starting a leg that is already in the lock is allowed only with exactly the recorded
settings: that is the resume after an interruption, and the runner then plays only the
missing episodes (an episode is recorded only if it ends normally). Any other setting --
another trial block, another records file, another configuration -- would be a second draw
of the test, and is refused with :class:`FinalRunRefused`.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from replaylab.config import GameConfig
from replaylab.domain.features import FeatureConfig
from replaylab.eval.records import config_hash
from replaylab.interfaces.runtime import SetupError

SCRIPTED_LEG = "scripted"
FROZEN_PATHS = ("src", "prompts", "config")


class FinalRunRefused(SetupError):
    """The final test cannot be played with these settings (it would be a second run)."""


@dataclass(frozen=True)
class FinalLeg:
    """One leg of the final run: one model (or the scripted ladder) on one trial block."""

    game: str
    leg: str
    config_hash: str
    records: str  # resolved path of the episodes file
    first_trial: int
    trials: int
    conditions: tuple[str, ...]

    @classmethod
    def make(
        cls,
        *,
        game: str,
        leg: str,
        config_hash: str,
        records: Path | str,
        first_trial: int,
        trials: int,
        conditions: Sequence[str],
    ) -> FinalLeg:
        return cls(
            game=game,
            leg=leg,
            config_hash=config_hash,
            records=str(Path(records).resolve()),
            first_trial=first_trial,
            trials=trials,
            conditions=tuple(sorted(conditions)),
        )

    @property
    def trial_range(self) -> range:
        return range(self.first_trial, self.first_trial + self.trials)

    def settings(self) -> dict[str, Any]:
        """What a resume must repeat exactly."""
        values = asdict(self)
        values["conditions"] = list(self.conditions)
        del values["game"], values["leg"]
        return values


@dataclass(frozen=True)
class FinalRunPermit:
    """Proof that :func:`acquire` accepted this leg. Only ``acquire`` should build one."""

    leg: FinalLeg
    lock_path: Path
    written: bool  # False for a check-only permit (a dry run): the lock was not touched

    @property
    def game(self) -> str:
        return self.leg.game


def frozen_hash(
    *,
    system_prompt: str,
    user_prompt: str,
    conditions_text: str,
    tool_descriptions: Mapping[str, str],
    limits: Mapping[str, Any],
    game: GameConfig,
    feature_config: FeatureConfig,
) -> str:
    """The hash every leg of one final run must share (the agent-facing configuration)."""
    return config_hash(
        {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "conditions": conditions_text,
            "tool_descriptions": dict(tool_descriptions),
            "limits": dict(limits),
            "game": asdict(game),
            "features": {
                "target": feature_config.target,
                "first_column": feature_config.first_column,
                "features": [[s.name, s.higher_is_better] for s in feature_config.specs],
            },
        }
    )


def dirty_paths(repo_dir: Path | str = ".", paths: Sequence[str] = FROZEN_PATHS) -> list[str]:
    """Uncommitted changes (including untracked files) under ``paths``, as git reports them.

    Raises:
        FinalRunRefused: git is unavailable, so the frozen state cannot be checked.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--", *paths],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as error:
        raise FinalRunRefused(f"cannot check the working tree with git: {error}") from None
    return [line for line in out.splitlines() if line.strip()]


def lock_path(lock_dir: Path | str, game: str) -> Path:
    return Path(lock_dir) / f"{game}.json"


def read_lock(path: Path | str) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_atomically(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def acquire(
    lock_dir: Path | str,
    leg: FinalLeg,
    *,
    frozen: str,
    git_sha: str,
    check_only: bool = False,
    now: datetime | None = None,
) -> FinalRunPermit:
    """Accept ``leg`` into the final run of its game, or refuse it.

    Args:
        lock_dir: folder of the lock files (one per final-test game).
        leg: the leg about to be played.
        frozen: :func:`frozen_hash` of the current configuration.
        git_sha: the commit being played.
        check_only: validate against the lock without writing it (a dry run).
        now: timestamp for a new leg (tests); defaults to the current UTC time.

    Raises:
        FinalRunRefused: the frozen configuration or the commit differs from the lock's, or
            the leg is already in the lock with other settings.
    """
    if not git_sha:
        raise FinalRunRefused("the final run needs a git commit to record; none was found")
    path = lock_path(lock_dir, leg.game)
    lock = read_lock(path)
    if lock is None:
        lock = {"game": leg.game, "frozen_hash": frozen, "git_sha": git_sha, "legs": {}}
    where = f"final test {leg.game!r} ({path})"
    if lock["frozen_hash"] != frozen:
        raise FinalRunRefused(
            f"{where}: the frozen configuration changed since the final run started "
            f"(lock {lock['frozen_hash']}, now {frozen}); the final test is played once"
        )
    if lock["git_sha"] != git_sha:
        raise FinalRunRefused(
            f"{where}: the final run started at commit {lock['git_sha']}, this is {git_sha}; "
            "every leg is played from the same commit"
        )
    recorded = lock["legs"].get(leg.leg)
    if recorded is not None:
        wanted = leg.settings()
        changed = sorted(k for k in wanted if recorded.get(k) != wanted[k])
        if changed:
            raise FinalRunRefused(
                f"{where}: leg {leg.leg!r} already ran with other settings ({', '.join(changed)}); "
                "only an identical resume is allowed, which plays just the missing episodes"
            )
        return FinalRunPermit(leg, path, written=False)
    if check_only:
        return FinalRunPermit(leg, path, written=False)
    stamp = (now or datetime.now(UTC)).isoformat(timespec="seconds")
    lock["legs"][leg.leg] = {**leg.settings(), "started_at": stamp}
    _write_atomically(path, lock)
    return FinalRunPermit(leg, path, written=True)
