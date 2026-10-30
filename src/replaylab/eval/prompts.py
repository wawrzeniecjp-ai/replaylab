"""The system prompt (one for every condition) and the first user message."""

from __future__ import annotations

import hashlib
from pathlib import Path

USER_PROMPT = (
    "Start the campaign. Work until your budget or rounds are used up or you decide to stop, "
    "then reply with a short summary of what you did and why."
)


def load_system_prompt(path: Path | str) -> str:
    """Read the shared system prompt, stripped of surrounding whitespace."""
    return Path(path).read_text().strip()


def text_hash(text: str) -> str:
    """First 12 hex characters of the sha256 of a text (used in the config hash)."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]
