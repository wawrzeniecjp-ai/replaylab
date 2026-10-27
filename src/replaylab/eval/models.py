"""Model configuration (``config/models.yaml``): endpoint, id, key variable and request parameters.

Everything here comes from the file; the code supplies no defaults for provider details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

RESERVED_PARAMS = ("model", "messages", "tools")


class ModelConfigError(ValueError):
    """The models file is invalid."""


@dataclass(frozen=True)
class ModelConfig:
    key: str
    base_url: str
    model: str
    api_key_env: str
    price_key: str
    reasoning_passback: bool = False
    reasoning_field: str = "reasoning_content"
    params: dict[str, Any] = field(default_factory=dict)


def load_models(path: Path | str) -> dict[str, ModelConfig]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or set(data) != {"version", "models"} or data["version"] != 1:
        raise ModelConfigError("models file needs exactly the keys version (1) and models")
    required = {"base_url", "model", "api_key_env", "price_key"}
    optional = {"reasoning_passback", "reasoning_field", "params"}
    models: dict[str, ModelConfig] = {}
    for key, body in (data["models"] or {}).items():
        where = f"model {key}"
        if not isinstance(body, dict):
            raise ModelConfigError(f"{where}: expected a mapping")
        unknown = set(body) - required - optional
        if unknown:
            raise ModelConfigError(f"{where}: unknown key(s) {sorted(unknown)}")
        missing = sorted(required - set(body))
        if missing:
            raise ModelConfigError(f"{where}: missing key(s) {missing}")
        if not isinstance(body.get("reasoning_passback", False), bool):
            raise ModelConfigError(f"{where}: reasoning_passback must be true or false")
        params = body.get("params") or {}
        if not isinstance(params, dict):
            raise ModelConfigError(f"{where}: params must be a mapping")
        reserved = [p for p in RESERVED_PARAMS if p in params]
        if reserved:
            raise ModelConfigError(
                f"{where}: params must not contain {reserved}; the adapter sets them"
            )
        models[str(key)] = ModelConfig(
            key=str(key),
            base_url=str(body["base_url"]),
            model=str(body["model"]),
            api_key_env=str(body["api_key_env"]),
            price_key=str(body["price_key"]),
            reasoning_passback=bool(body.get("reasoning_passback", False)),
            reasoning_field=str(body.get("reasoning_field", "reasoning_content")),
            params=dict(params),
        )
    if not models:
        raise ModelConfigError("no models defined")
    return models
