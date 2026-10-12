"""Pin the published Adaptyv OpenAPI spec by its sha256.

The pin file uses the ``sha256sum`` format: ``<64 hex characters>  <file name>``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def openapi_info(path: Path | str) -> tuple[str, str]:
    """Return (openapi version, info.version) from a spec file."""
    data = json.loads(Path(path).read_text())
    try:
        return str(data["openapi"]), str(data["info"]["version"])
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path}: not an OpenAPI document (missing {error})") from error


def write_pin(spec_path: Path | str, pin_path: Path | str) -> str:
    """Write the pin file for ``spec_path`` and return the hash."""
    digest = sha256_file(spec_path)
    Path(pin_path).write_text(f"{digest}  {Path(spec_path).name}\n")
    return digest


def read_pin(pin_path: Path | str) -> str:
    """Return the hash stored in a pin file."""
    first = Path(pin_path).read_text().split()
    if not first or len(first[0]) != 64 or any(c not in "0123456789abcdef" for c in first[0]):
        raise ValueError(f"{pin_path}: not a valid sha256 pin file")
    return first[0]


def verify_pin(spec_path: Path | str, pin_path: Path | str) -> bool:
    return sha256_file(spec_path) == read_pin(pin_path)
