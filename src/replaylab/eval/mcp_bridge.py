"""Connect an MCP client to an OpenAI-style function-calling loop (plan 7.1).

* ``tool_specs`` turns each MCP tool into ``{"type": "function", "function": {name, description,
  parameters}}`` with ``parameters`` equal to the tool's input schema.
* ``call`` runs one tool and returns text for the model. Success text is the compact JSON of the
  structured result (about 30% shorter than the SDK's indented text); errors are prefixed with
  ``ERROR: `` and lose pydantic's documentation-link line; text longer than ``max_chars`` is cut
  and ends with a visible marker, with ``truncated`` set.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

FUNCTION_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
DEFAULT_MAX_CHARS = 20_000


class BridgeError(Exception):
    """A tool cannot be offered to the model."""


@dataclass(frozen=True)
class ToolResult:
    text: str
    ok: bool
    truncated: bool
    chars: int  # length of the text before truncation
    error_kind: str | None = None  # validation, unknown_tool or tool_error


def truncation_marker(max_chars: int) -> str:
    return f"\n[TRUNCATED: the output was longer than {max_chars} characters and was cut here]"


def classify_error(text: str) -> str:
    lowered = text.lower()
    if "validation error" in lowered:
        return "validation"
    if lowered.startswith("unknown tool"):
        return "unknown_tool"
    return "tool_error"


def _clean_error(text: str) -> str:
    kept = [
        line
        for line in text.splitlines()
        if not line.strip().startswith("For further information visit")
    ]
    return "\n".join(kept).strip()


class McpBridge:
    def __init__(self, client: Any, *, max_chars: int = DEFAULT_MAX_CHARS) -> None:
        self._client = client
        self._max_chars = max_chars
        self._specs: list[dict[str, Any]] | None = None

    async def tool_specs(self) -> list[dict[str, Any]]:
        if self._specs is None:
            listing = await self._client.list_tools()
            specs = []
            for tool in listing.tools:
                if not FUNCTION_NAME.fullmatch(tool.name):
                    raise BridgeError(f"tool name {tool.name!r} is not a valid function name")
                specs.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.input_schema,
                        },
                    }
                )
            self._specs = specs
        return self._specs

    async def tool_names(self) -> list[str]:
        return [spec["function"]["name"] for spec in await self.tool_specs()]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        result = await self._client.call_tool(name, arguments)
        texts = [block.text for block in result.content if getattr(block, "type", None) == "text"]
        if result.is_error:
            body = _clean_error("\n".join(texts))
            text, kind = "ERROR: " + body, classify_error(body)
        else:
            structured = result.structured_content
            text = (
                json.dumps(structured, separators=(",", ":"))
                if structured is not None
                else "\n".join(texts)
            )
            kind = None
        chars = len(text)
        truncated = chars > self._max_chars
        if truncated:
            text = text[: self._max_chars] + truncation_marker(self._max_chars)
        return ToolResult(
            text=text, ok=not result.is_error, truncated=truncated, chars=chars, error_kind=kind
        )
