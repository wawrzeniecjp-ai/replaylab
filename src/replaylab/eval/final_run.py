"""The run-once guard of the final tests (``docs/prereg-anthropic.md``,
``docs/prereg-heldout.md``).

A game marked ``final_test`` in ``config/games.yaml`` is played only by the runner, and only
under a :class:`FinalRunPermit`; :func:`acquire` is the one way to get one. Everything else
(``serve``, the campaign CLI, the ladder, the replay page) refuses the game outright.

The lock file ``<lock_dir>/<game>.json`` is meant to be committed with the results. It
records:

* ``frozen_hash``: prompts, conditions, tool descriptions, limits, and the game's own
  ``games.yaml`` and ``features.yaml`` entries (:func:`frozen_hash`). Every leg of the final
  run must share it: a change after the first leg started is refused.
* ``git_sha``: the commit every leg is played from (a dirty ``src``, ``prompts`` or
  ``config`` is refused before that, by :func:`uncommitted_frozen_paths`, which the CLI
  calls through ``eval/setup.py``).
* one entry per leg (a model key, or ``scripted``): its config hash, records file, trial
  block and conditions (the policies, for ``scripted``).

The scripted leg may be recorded into more than one records file (one per model's run
directory, so each file can be compared on its own), but every copy must have the same trial
block, policies and configuration: scripted play is deterministic, so a copy reveals nothing
new, while another block or another policy would.

Starting a leg that is already in the lock is allowed only with exactly the recorded
settings: that is the resume after an interruption, and the runner then plays only the
missing episodes (an episode is recorded only if it ends normally). Any other setting --
another trial block, another records file, another configuration -- would be a second draw
of the test, and is refused with :class:`FinalRunRefused`.

The lock does not record which legs the pre-registration names. A leg under a new model key,
from the same commit and with the same frozen configuration, is accepted as a new leg; the
pre-registration, not the lock, is what rules out adding a model after results are seen.
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


class UncommittedTree(SetupError):
    """A recording run was asked for while ``src``/``prompts``/``config`` were uncommitted.

    ``config_hash`` covers what the model *sees* (prompts, conditions, tool
    descriptions, model parameters), not the code that runs the tools, the policies or
    the scorer. ``git_sha`` is the only pin on those, so a dirty tree would leave
    records claiming a commit that did not produce them.
    """


def code_root() -> Path:
    """The repository root of the *imported* package -- not of the working directory.

    A control run is played from a git worktree with ``PYTHONPATH=<worktree>/src``
    while the shell stays in the main checkout; resolving ``"."`` would then check the
    wrong tree and stamp the wrong sha.
    """
    import replaylab

    return Path(replaylab.__file__).resolve().parents[2]


def uncommitted_frozen_paths(repo_dir: Path | str | None = None) -> list[str]:
    """Uncommitted changes under :data:`FROZEN_PATHS`, for the repo that holds the code.

    Raises:
        UncommittedTree: the repository cannot be located, so cleanliness cannot be
            checked; treating that as clean is the trap this guard exists to close.
    """
    root = code_root() if repo_dir is None else Path(repo_dir)
    if not (root / ".git").exists():
        raise UncommittedTree(
            f"{root} is not a git repository, so it cannot be shown that src, prompts and "
            "config are committed"
        )
    try:
        return dirty_paths(root)
    except FinalRunRefused as error:
        raise UncommittedTree(str(error)) from None


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
        """Build a leg with its records path resolved and its conditions sorted.

        Args:
            game (str): the final-test game.
            leg (str): the model key, or ``scripted``.
            config_hash (str): the leg's config hash.
            records (Path | str): the episodes file the leg writes.
            first_trial (int): the first trial of the block.
            trials (int): the number of trials.
            conditions (Sequence[str]): the conditions (or ``scripted:<policy>`` names) played.

        Returns:
            FinalLeg: the leg.
        """
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
        """The trials of the block, first to last."""
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
        """The game the permit is for."""
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
    """Hash the agent-facing configuration that every leg of one final run must share.

    Args:
        system_prompt (str): the system prompt.
        user_prompt (str): the user prompt.
        conditions_text (str): the conditions file, as text.
        tool_descriptions (Mapping[str, str]): the description of every tool.
        limits (Mapping[str, Any]): the episode limits.
        game (GameConfig): the game's ``games.yaml`` entry.
        feature_config (FeatureConfig): the game's ``features.yaml`` entry.

    Returns:
        str: the first 12 hex characters of the sha256 (:func:`config_hash`).
    """
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
    """List the uncommitted changes (including untracked files) under ``paths``.

    Args:
        repo_dir (Path | str): a directory inside the repository.
        paths (Sequence[str]): the paths to check, relative to ``repo_dir``.

    Returns:
        list[str]: one ``git status --porcelain`` line per changed path; empty when clean.

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
    """Return where the lock file of ``game`` lives.

    Args:
        lock_dir (Path | str): the folder of the lock files.
        game (str): the final-test game.

    Returns:
        Path: ``<lock_dir>/<game>.json``.
    """
    return Path(lock_dir) / f"{game}.json"


def read_lock(path: Path | str) -> dict[str, Any] | None:
    """Read a lock file.

    Args:
        path (Path | str): the lock file.

    Returns:
        dict[str, Any] | None: its content, or None if the file does not exist.

    Raises:
        json.JSONDecodeError: the file is not valid JSON.
    """
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
    key = leg.leg
    if leg.leg == SCRIPTED_LEG:
        key, recorded = _scripted_entry(lock, leg, where)
    else:
        recorded = lock["legs"].get(key)
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
    lock["legs"][key] = {**leg.settings(), "started_at": stamp}
    _write_atomically(path, lock)
    return FinalRunPermit(leg, path, written=True)


def _scripted_entry(
    lock: Mapping[str, Any], leg: FinalLeg, where: str
) -> tuple[str, dict[str, Any] | None]:
    """Find the lock key of a scripted leg, and its recorded entry if it has one.

    The first scripted leg is keyed ``scripted``; a copy into another records file is keyed
    ``scripted@<records>``.

    Args:
        lock (Mapping[str, Any]): the lock being checked.
        leg (FinalLeg): the scripted leg about to be played.
        where (str): the game and lock path, for the error message.

    Returns:
        tuple[str, dict[str, Any] | None]: the key, and the entry recorded under it (None
        for a new copy or the first scripted leg).

    Raises:
        FinalRunRefused: a new copy differs from the scripted leg already recorded in
            anything but its records file.
    """
    scripted = {
        key: entry
        for key, entry in lock["legs"].items()
        if key == SCRIPTED_LEG or key.startswith(f"{SCRIPTED_LEG}@")
    }
    for key, entry in scripted.items():
        if entry.get("records") == leg.records:
            return key, entry
    if not scripted:
        return SCRIPTED_LEG, None
    wanted = {k: v for k, v in leg.settings().items() if k != "records"}
    first = scripted.get(SCRIPTED_LEG) or next(iter(scripted.values()))
    changed = sorted(k for k in wanted if first.get(k) != wanted[k])
    if changed:
        raise FinalRunRefused(
            f"{where}: a copy of the scripted leg must repeat its settings, but "
            f"{', '.join(changed)} differ; another trial block or policy would be a second run"
        )
    return f"{SCRIPTED_LEG}@{leg.records}", None
