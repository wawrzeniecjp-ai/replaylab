"""Episode-local aliases for design ids (decision D4).

Agents see ``d0001``, ``d0002``, ... instead of the real design ids. The mapping is a
bijection that depends only on the set of ids and an integer seed. It is built by
sorting ids on ``sha256(f'{seed}|{id}')``, so it is identical on every machine and
Python/NumPy version (no random number generator is involved).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


class UnknownIdError(KeyError):
    """An id or alias is not part of the mapping."""


def _digest(seed: int, design_id: str) -> str:
    return hashlib.sha256(f"{seed}|{design_id}".encode()).hexdigest()


class Aliaser:
    """Bijective mapping between real design ids and aliases."""

    def __init__(self, real_to_alias: dict[str, str]) -> None:
        self._real_to_alias = dict(real_to_alias)
        self._alias_to_real = {alias: real for real, alias in self._real_to_alias.items()}
        if len(self._alias_to_real) != len(self._real_to_alias):
            raise ValueError("aliases must be unique")

    @classmethod
    def from_seed(cls, real_ids: Iterable[str], seed: int) -> Aliaser:
        ids = list(real_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("real_ids must be unique")
        ordered = sorted(ids, key=lambda design_id: (_digest(seed, design_id), design_id))
        width = max(4, len(str(len(ordered))))
        return cls({real: f"d{position:0{width}d}" for position, real in enumerate(ordered, 1)})

    def alias(self, real_id: str) -> str:
        try:
            return self._real_to_alias[real_id]
        except KeyError:
            raise UnknownIdError(f"unknown design id {real_id!r}") from None

    def real(self, alias: str) -> str:
        try:
            return self._alias_to_real[alias]
        except KeyError:
            raise UnknownIdError(f"unknown alias {alias!r}") from None

    def aliases(self) -> list[str]:
        """All aliases in alias order (d0001, d0002, ...)."""
        return sorted(self._alias_to_real)

    def __len__(self) -> int:
        return len(self._real_to_alias)

    def __contains__(self, real_id: object) -> bool:
        return real_id in self._real_to_alias
