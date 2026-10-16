"""Pytest options shared by the tests."""

from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption(
        "--update-tool-snapshot",
        action="store_true",
        default=False,
        help="rewrite tests/snapshots/tools.json from the current tool list",
    )
