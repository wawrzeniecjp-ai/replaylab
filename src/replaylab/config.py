"""Loading and validating the YAML configuration files.

* ``config/features.yaml``: visible features per target and the pre-registered first column.
* ``config/games.yaml``: the games (pool, split, rounds, batch size, budget, delay).

All validation errors raise :class:`ConfigError` with a message that names the place
(target or game) and the problem. Unknown or missing keys are errors: there are no
silent defaults beyond the ones documented here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from replaylab.domain.features import FeatureConfig, FeatureSpec
from replaylab.domain.types import CampaignConfig

SPLITS = ("dev", "heldout", "all")
KINDS = ("rank", "stop")
SUPPORTED_VERSION = 1


class ConfigError(ValueError):
    """A configuration file is invalid."""


def _check_keys(obj: Any, required: set[str], optional: set[str], where: str) -> None:
    if not isinstance(obj, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {type(obj).__name__}")
    unknown = set(obj) - required - optional
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown, key=str)}")
    missing = required - set(obj)
    if missing:
        raise ConfigError(f"{where}: missing key(s) {sorted(missing)}")


def _check_version(data: Any, where: str) -> None:
    if data.get("version") != SUPPORTED_VERSION or isinstance(data.get("version"), bool):
        raise ConfigError(f"{where}: unsupported version {data.get('version')!r}; expected 1")


def _read_yaml(path: Path | str) -> Any:
    path = Path(path)
    try:
        return yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise ConfigError(f"{path}: invalid YAML ({error})") from error


def _require_int(value: Any, where: str, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}: {field_name} must be an integer, got {value!r}")
    return value


# --- features ---------------------------------------------------------------


def parse_feature_config(data: Any) -> dict[str, FeatureConfig]:
    """Validate parsed YAML and return one :class:`FeatureConfig` per target."""
    _check_keys(data, {"version", "targets"}, set(), "features config")
    _check_version(data, "features config")
    targets = data["targets"]
    if not isinstance(targets, Mapping) or not targets:
        raise ConfigError("features config: targets must be a non-empty mapping")

    result: dict[str, FeatureConfig] = {}
    for target, body in targets.items():
        where = f"target {target}"
        _check_keys(body, {"first_column", "features"}, set(), where)
        items = body["features"]
        if not isinstance(items, list) or not items:
            raise ConfigError(f"{where}: features must be a non-empty list")
        specs = []
        for index, item in enumerate(items):
            item_where = f"{where} feature #{index + 1}"
            _check_keys(item, {"name"}, {"higher_is_better"}, item_where)
            name = item["name"]
            if not isinstance(name, str) or not name:
                raise ConfigError(f"{item_where}: name must be a non-empty string")
            higher = item.get("higher_is_better", True)
            if not isinstance(higher, bool):
                raise ConfigError(f"{item_where}: higher_is_better must be true or false")
            specs.append(FeatureSpec(name, higher))
        try:
            result[str(target)] = FeatureConfig(str(target), body["first_column"], tuple(specs))
        except ValueError as error:
            raise ConfigError(str(error)) from error
    return result


def load_feature_config(path: Path | str) -> dict[str, FeatureConfig]:
    return parse_feature_config(_read_yaml(path))


# --- games ------------------------------------------------------------------


@dataclass(frozen=True)
class GameConfig:
    """One game: which pool, how many rounds, how big a batch, and any feedback delay."""

    id: str
    target: str
    split: str
    rounds: int
    batch_size: int
    budget: int
    delay_rounds: int = 0
    kind: str = "rank"

    def __post_init__(self) -> None:
        if not self.id:
            raise ConfigError("game id must be non-empty")
        where = f"game {self.id}"
        if self.split not in SPLITS:
            raise ConfigError(f"{where}: unknown split {self.split!r}; expected one of {SPLITS}")
        if self.kind not in KINDS:
            raise ConfigError(f"{where}: unknown kind {self.kind!r}; expected one of {KINDS}")
        if self.rounds < 1:
            raise ConfigError(f"{where}: rounds must be at least 1")
        if self.batch_size < 1:
            raise ConfigError(f"{where}: batch_size must be at least 1")
        if self.delay_rounds < 0:
            raise ConfigError(f"{where}: delay_rounds cannot be negative")
        product = self.rounds * self.batch_size
        if self.budget != product:
            raise ConfigError(
                f"{where}: budget must equal rounds * batch_size "
                f"({self.rounds} * {self.batch_size} = {product}), got {self.budget}"
            )

    def check_pool_size(self, n_designs: int) -> None:
        """Raise if the budget exceeds the number of designs in the pool."""
        if self.budget > n_designs:
            raise ConfigError(f"game {self.id}: budget {self.budget} exceeds pool size {n_designs}")

    def to_campaign_config(
        self, features: Iterable[str] = (), seed: int = 0, mode: str = "replay"
    ) -> CampaignConfig:
        return CampaignConfig(
            name=self.id,
            target=self.target,
            mode=mode,  # type: ignore[arg-type]
            budget=self.budget,
            batch_size=self.batch_size,
            rounds=self.rounds,
            delay_rounds=self.delay_rounds,
            seed=seed,
            features=tuple(features),
        )


def parse_games(data: Any) -> dict[str, GameConfig]:
    _check_keys(data, {"version", "games"}, set(), "games config")
    _check_version(data, "games config")
    games = data["games"]
    if not isinstance(games, Mapping) or not games:
        raise ConfigError("games config: games must be a non-empty mapping")
    result: dict[str, GameConfig] = {}
    for game_id, body in games.items():
        where = f"game {game_id}"
        _check_keys(
            body,
            {"target", "split", "rounds", "batch_size", "budget"},
            {"delay_rounds", "kind"},
            where,
        )
        result[str(game_id)] = GameConfig(
            id=str(game_id),
            target=str(body["target"]),
            split=body["split"],
            rounds=_require_int(body["rounds"], where, "rounds"),
            batch_size=_require_int(body["batch_size"], where, "batch_size"),
            budget=_require_int(body["budget"], where, "budget"),
            delay_rounds=_require_int(body.get("delay_rounds", 0), where, "delay_rounds"),
            kind=body.get("kind", "rank"),
        )
    return result


def load_games(path: Path | str) -> dict[str, GameConfig]:
    return parse_games(_read_yaml(path))
