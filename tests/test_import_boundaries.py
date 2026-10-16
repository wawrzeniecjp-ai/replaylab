"""Architecture guard: which package may import which (spec section 4.3).

advisor and policies must never see the store or a backend, so they cannot write and
cannot peek at hidden outcomes. This test fails the build if that rule is broken.
"""

from __future__ import annotations

import ast
from pathlib import Path

import replaylab

SRC = Path(replaylab.__file__).resolve().parent

# package -> the replaylab sub-packages it may import (besides itself).
ALLOWED: dict[str, set[str]] = {
    "domain": set(),
    "config": {"domain"},
    "datasets": {"domain", "config"},
    "store": {"domain"},
    "backends": {"domain"},
    "advisor": {"domain", "datasets"},
    "policies": {"domain", "advisor"},
    "engine": {"domain", "store", "backends", "datasets", "config"},
    # the agent-facing layer: the one package allowed to see everything,
    # which is why nothing may import it in return
    "interfaces": {
        "domain",
        "config",
        "datasets",
        "store",
        "backends",
        "engine",
        "advisor",
        "policies",
    },
}


def imported_subpackages(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = [node.module]
        for module in modules:
            parts = module.split(".")
            if parts[0] == "replaylab" and len(parts) > 1:
                found.add(parts[1])
    return found


def test_packages_only_import_what_they_are_allowed_to():
    violations = []
    for package, allowed in ALLOWED.items():
        package_path = SRC / (package + ".py" if package == "config" else package)
        files = (
            [package_path] if package_path.suffix == ".py" else sorted(package_path.rglob("*.py"))
        )
        for path in files:
            if not path.exists():
                continue
            illegal = imported_subpackages(path) - allowed - {package, "spec_pin"}
            if illegal:
                violations.append(f"{path.relative_to(SRC)} imports {sorted(illegal)}")
    assert not violations, "forbidden imports: " + "; ".join(violations)


def test_the_guard_covers_every_existing_package():
    existing = {p.name for p in SRC.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    unguarded = existing - set(ALLOWED) - {"eval"}  # eval is allowed to import anything
    assert not unguarded, f"add these packages to ALLOWED: {sorted(unguarded)}"
