"""Episode seeds (decision D4).

The seed depends on the game and the trial only, never on the condition, so every condition
(and every scripted policy) plays the same pool shuffle and the same design aliases: results
can be paired by seed.
"""

from __future__ import annotations

import hashlib


def episode_seed(game: str, trial: int) -> int:
    """A 64-bit seed: the first 8 bytes of sha256 of ``"<game>|<trial>"``, read big-endian."""
    digest = hashlib.sha256(f"{game}|{trial}".encode()).digest()
    return int.from_bytes(digest[:8], "big")
