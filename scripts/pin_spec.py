#!/usr/bin/env python
"""Pin or verify the published Adaptyv OpenAPI spec.

    python scripts/pin_spec.py write     # after saving the spec to spec/openapi.prod.json
    python scripts/pin_spec.py verify    # exit 0 if the file matches the pin, 1 if not

Exit code 2 means a file is missing or not valid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    import replaylab  # noqa: F401
except ModuleNotFoundError:  # allow running without an editable install
    sys.path.insert(0, str(REPO_ROOT / "src"))

from replaylab.spec_pin import openapi_info, read_pin, sha256_file, verify_pin, write_pin  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["write", "verify"])
    parser.add_argument("--spec", type=Path, default=REPO_ROOT / "spec/openapi.prod.json")
    parser.add_argument("--pin", type=Path, default=REPO_ROOT / "spec/openapi.sha256")
    args = parser.parse_args(argv)

    if not args.spec.exists():
        print(f"missing {args.spec}", file=sys.stderr)
        return 2
    try:
        openapi, version = openapi_info(args.spec)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    if args.command == "write":
        digest = write_pin(args.spec, args.pin)
        print(f"openapi {openapi}, info.version {version}")
        print(f"sha256 {digest}")
        print(f"wrote {args.pin}")
        return 0

    if not args.pin.exists():
        print(f"missing {args.pin}", file=sys.stderr)
        return 2
    try:
        matches = verify_pin(args.spec, args.pin)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    if matches:
        print(f"OK {args.spec.name} matches the pin (openapi {openapi}, info.version {version})")
        return 0
    print(f"MISMATCH: {sha256_file(args.spec)} != {read_pin(args.pin)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
