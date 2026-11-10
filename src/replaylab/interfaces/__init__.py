"""User-facing interfaces: the tool layer (``tools/``), the MCP server and the CLI.

May import every layer below it; nothing imports it except ``eval``, which may import
anything. ``runtime.py:build_runtime`` assembles one game and one seed for the tools.
"""
