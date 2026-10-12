#!/usr/bin/env python
"""Run the LLM smoke test against an OpenAI-compatible endpoint.

Required environment variables (nothing is read from files in the repository):

    LLM_API_KEY    the API key
    LLM_MODEL      the model id to test (list the endpoint's models first if unsure)
Optional:
    LLM_BASE_URL   the endpoint's base URL (omit for the OpenAI default)

    python scripts/smoke_llm.py

Writes notes/week0/llm_smoke.txt (gitignored) and prints the report.
Exit code 0: the tool-call round trip worked. 1: it did not. 2: bad setup.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    import replaylab  # noqa: F401
except ModuleNotFoundError:  # allow running without an editable install
    sys.path.insert(0, str(REPO_ROOT / "src"))

from replaylab.eval.smoke import format_report, run_smoke  # noqa: E402


def _openai_client(api_key: str, base_url: str | None):
    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        print("the openai package is not installed: pip install openai", file=sys.stderr)
        raise SystemExit(2) from None
    return OpenAI(api_key=api_key, base_url=base_url)


def main(
    argv: list[str] | None = None,
    env: Mapping[str, str] | None = None,
    client_factory: Callable[[str, str | None], object] | None = None,
    out: Path | None = None,
) -> int:
    env = os.environ if env is None else env
    for name in ("LLM_API_KEY", "LLM_MODEL"):
        if not env.get(name):
            print(f"environment variable {name} is not set", file=sys.stderr)
            return 2
    factory = client_factory or _openai_client
    client = factory(env["LLM_API_KEY"], env.get("LLM_BASE_URL") or None)
    report = run_smoke(client, env["LLM_MODEL"])
    text = format_report(report)
    out = out or REPO_ROOT / "notes/week0/llm_smoke.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print(f"wrote {out}")
    return 0 if report.tool_call_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
