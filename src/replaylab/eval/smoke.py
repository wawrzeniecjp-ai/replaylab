"""Smoke test for an OpenAI-compatible chat API with tool calling.

It answers the following questions:

1. Which models does the endpoint list?
2. Does one round trip with a tool call work (right tool, right arguments)?
3. Does the response carry a separate reasoning field, and must that field be passed
   back in the next turn of a tool loop?
4. Are ``temperature``, ``seed`` and ``tool_choice='required'`` accepted?
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two integers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        },
    }
]
PROMPT = "Use the add tool to add 2 and 3."
REASONING_FIELDS = ("reasoning_content", "reasoning")


@dataclass(frozen=True)
class ProbeResult:
    """One capability probe's outcome: pass/fail plus a detail line."""

    ok: bool
    detail: str = ""


@dataclass
class SmokeReport:
    """Everything the smoke test observed about an OpenAI-compatible
    endpoint: the model listing, the tool-call round trip, where reasoning
    text came back, token use, and the capability probes (temperature,
    seed, tool_choice)."""

    model: str
    available_models: list[str] = field(default_factory=list)
    tool_call_ok: bool = False
    tool_call_detail: str = ""
    reasoning_field: str | None = None
    message_fields: list[str] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    followup_plain: ProbeResult | None = None
    followup_with_reasoning: ProbeResult | None = None
    param_probes: dict[str, ProbeResult] = field(default_factory=dict)


def _error(exc: Exception) -> ProbeResult:
    return ProbeResult(False, f"{type(exc).__name__}: {str(exc)[:200]}")


def _message_fields(message: Any) -> list[str]:
    """The field names the endpoint's message actually carries.

    Pydantic extras count (the OpenAI SDK parses unknown fields like
    ``reasoning_content`` as extras): this list is the evidence behind the
    "reasoning field:" verdict, so a "none" can be told apart from a missed
    field at a different location.
    """
    names = set(getattr(message, "model_fields", None) or ())
    names |= set(getattr(message, "model_extra", None) or ())
    if not names and hasattr(message, "__dict__"):
        names = {k for k in vars(message) if not k.startswith("_")}
    return sorted(names)


def _try(client: Any, **kwargs: Any) -> tuple[Any, ProbeResult]:
    try:
        return client.chat.completions.create(**kwargs), ProbeResult(True, "ok")
    except Exception as exc:  # noqa: BLE001 - any API failure is data for the report
        return None, _error(exc)


def run_smoke(client: Any, model: str) -> SmokeReport:
    """Run the probe sequence against ``client`` and collect the report.

    The sequence: list available models, do a tool-call round trip (without and with
    reasoning content), then probe temperature, seed and tool_choice support. Each step
    records itself in the report, so one failure does not hide the rest.

    Args:
        client (Any): an OpenAI-compatible client.
        model (str): the model id to probe.

    Returns:
        SmokeReport: what each probe observed.
    """
    report = SmokeReport(model=model)

    # 1. Which models the endpoint lists (informational; a failure is not fatal).
    try:
        listing = client.models.list()
        report.available_models = sorted(m.id for m in getattr(listing, "data", listing))
    except Exception:  # noqa: BLE001
        report.available_models = []

    # 2. One call that should come back as a tool call add(a=2, b=3).
    messages = [{"role": "user", "content": PROMPT}]
    response, probe = _try(client, model=model, messages=messages, tools=TOOLS)
    if response is None:
        report.tool_call_detail = probe.detail
        return report  # nothing else is worth trying if the basic call fails

    usage = getattr(response, "usage", None)
    report.prompt_tokens = getattr(usage, "prompt_tokens", None)
    report.completion_tokens = getattr(usage, "completion_tokens", None)
    message = response.choices[0].message
    calls = list(getattr(message, "tool_calls", None) or [])
    for name in REASONING_FIELDS:
        value = getattr(message, name, None)
        if isinstance(value, str) and value:
            report.reasoning_field = name
            break
    report.message_fields = _message_fields(message)

    if not calls:
        report.tool_call_detail = "no tool call in the response"
    else:
        call = calls[0]
        try:
            arguments = json.loads(call.function.arguments)
        except (TypeError, ValueError):
            arguments = None
        if call.function.name != "add":
            report.tool_call_detail = f"unexpected tool name {call.function.name!r}"
        elif arguments != {"a": 2, "b": 3}:
            report.tool_call_detail = f"unexpected arguments {call.function.arguments!r}"
        else:
            report.tool_call_ok = True
            report.tool_call_detail = "ok"

    # 3. Send the tool result back, once without and once with the reasoning text, to
    #    learn whether the endpoint accepts (or needs) reasoning passed back.
    if report.tool_call_ok:
        assistant = {
            "role": "assistant",
            "content": getattr(message, "content", None),
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.function.name, "arguments": c.function.arguments},
                }
                for c in calls
            ],
        }
        tool_result = {"role": "tool", "tool_call_id": calls[0].id, "content": "5"}
        _, report.followup_plain = _try(
            client, model=model, messages=[*messages, assistant, tool_result], tools=TOOLS
        )
        if report.reasoning_field is not None:
            with_reasoning = {
                **assistant,
                report.reasoning_field: getattr(message, report.reasoning_field),
            }
            _, report.followup_with_reasoning = _try(
                client, model=model, messages=[*messages, with_reasoning, tool_result], tools=TOOLS
            )

    # 4. Which optional request parameters the endpoint accepts.
    for name, extra in (
        ("temperature", {"temperature": 0}),
        ("seed", {"seed": 0}),
        ("tool_choice_required", {"tool_choice": "required"}),
    ):
        _, report.param_probes[name] = _try(
            client, model=model, messages=messages, tools=TOOLS, **extra
        )
    return report


def passback_needed(report: SmokeReport) -> bool | None:
    """True if the reasoning field must be passed back; None if it cannot be told."""
    plain, with_reasoning = report.followup_plain, report.followup_with_reasoning
    if plain is None:
        return None
    if plain.ok:
        return False
    if with_reasoning is not None and with_reasoning.ok:
        return True
    return None


def _status(probe: ProbeResult | None) -> str:
    if probe is None:
        return "not applicable"
    return "OK" if probe.ok else f"FAILED ({probe.detail})"


def format_report(report: SmokeReport) -> str:
    """Render a SmokeReport as the fixed-width text written to
    ``notes/week0/llm_smoke.txt``."""
    needed = passback_needed(report)
    shown = ", ".join(report.available_models[:20])
    lines = [
        f"model: {report.model}",
        f"available models: {len(report.available_models)} ({shown})",
        "tool call: " + ("OK" if report.tool_call_ok else f"FAILED ({report.tool_call_detail})"),
        f"reasoning field: {report.reasoning_field or 'none'}",
        f"message fields: {', '.join(report.message_fields) or '(none)'}",
        f"tokens: prompt={report.prompt_tokens} completion={report.completion_tokens}",
        f"follow-up without reasoning: {_status(report.followup_plain)}",
        f"follow-up with reasoning: {_status(report.followup_with_reasoning)}",
        "reasoning pass-back needed: "
        + ("unknown" if needed is None else "yes" if needed else "no"),
    ]
    for name, probe in report.param_probes.items():
        lines.append(f"probe {name}: {_status(probe)}")
    return "\n".join(lines) + "\n"
