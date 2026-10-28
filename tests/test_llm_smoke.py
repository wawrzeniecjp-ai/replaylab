"""T05: LLM smoke test, using a fake client (no network, no key)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from replaylab.eval.smoke import (
    ProbeResult,
    SmokeReport,
    format_report,
    passback_needed,
    run_smoke,
)

ROOT = Path(__file__).resolve().parents[1]


def response(
    arguments='{"a": 2, "b": 3}', name="add", reasoning=None, usage=(11, 7), no_call=False
):
    calls = (
        None
        if no_call
        else [
            SimpleNamespace(id="call_1", function=SimpleNamespace(name=name, arguments=arguments))
        ]
    )
    message = SimpleNamespace(content=None, tool_calls=calls)
    if reasoning is not None:
        message.reasoning_content = reasoning
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]),
    )


class FakeClient:
    """Calls ``handler(kwargs)`` for every chat completion; records all calls."""

    def __init__(self, handler, model_ids=("m-b", "m-a"), list_fails=False):
        self.calls = []
        self._handler = handler
        outer = self

        class Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                return outer._handler(kwargs)

        class Models:
            def list(self):
                if list_fails:
                    raise RuntimeError("models endpoint unavailable")
                return SimpleNamespace(data=[SimpleNamespace(id=i) for i in model_ids])

        self.chat = SimpleNamespace(completions=Completions())
        self.models = Models()


def is_followup(kwargs) -> bool:
    return any(m["role"] == "tool" for m in kwargs["messages"])


def assistant_message(kwargs) -> dict:
    return next(m for m in kwargs["messages"] if m["role"] == "assistant")


def test_happy_path_without_a_reasoning_field():
    client = FakeClient(lambda kwargs: response())
    report = run_smoke(client, "m-a")
    assert report.model == "m-a" and report.available_models == ["m-a", "m-b"]  # sorted
    assert report.tool_call_ok and report.tool_call_detail == "ok"
    assert report.reasoning_field is None
    assert (report.prompt_tokens, report.completion_tokens) == (11, 7)
    assert report.followup_plain == ProbeResult(True, "ok")
    assert report.followup_with_reasoning is None
    assert passback_needed(report) is False
    assert set(report.param_probes) == {"temperature", "seed", "tool_choice_required"}
    assert all(p.ok for p in report.param_probes.values())
    assert len(client.calls) == 5  # tool call, follow-up, three probes


def test_the_first_call_offers_the_add_tool_and_the_documented_prompt():
    client = FakeClient(lambda kwargs: response())
    run_smoke(client, "m-a")
    first = client.calls[0]
    assert first["model"] == "m-a"
    assert first["messages"] == [{"role": "user", "content": "Use the add tool to add 2 and 3."}]
    assert first["tools"][0]["function"]["name"] == "add"
    probes = {tuple(sorted(set(c) - {"model", "messages", "tools"})) for c in client.calls[2:]}
    assert probes == {("temperature",), ("seed",), ("tool_choice",)}
    assert any(c.get("tool_choice") == "required" for c in client.calls)


def test_the_followup_sends_the_assistant_tool_call_and_the_tool_result():
    client = FakeClient(lambda kwargs: response())
    run_smoke(client, "m-a")
    followup = client.calls[1]
    assistant = assistant_message(followup)
    assert assistant["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
        }
    ]
    assert followup["messages"][-1] == {"role": "tool", "tool_call_id": "call_1", "content": "5"}


def test_reasoning_must_be_passed_back_when_the_plain_followup_is_rejected():
    def handler(kwargs):
        if is_followup(kwargs) and "reasoning_content" not in assistant_message(kwargs):
            raise RuntimeError("reasoning_content must be passed back")
        return response(reasoning="thinking...")

    client = FakeClient(handler)
    report = run_smoke(client, "m")
    assert report.reasoning_field == "reasoning_content"
    assert not report.followup_plain.ok and "must be passed back" in report.followup_plain.detail
    assert report.followup_with_reasoning.ok
    assert passback_needed(report) is True
    sent = [assistant_message(c) for c in client.calls if is_followup(c)]
    assert sent[1]["reasoning_content"] == "thinking..."  # echoed unchanged


def test_reasoning_not_needed_when_the_plain_followup_works():
    client = FakeClient(lambda kwargs: response(reasoning="thinking..."))
    report = run_smoke(client, "m")
    assert report.reasoning_field == "reasoning_content"
    assert report.followup_plain.ok and report.followup_with_reasoning.ok
    assert passback_needed(report) is False


def test_the_other_reasoning_field_name_is_detected():
    message = response().choices[0].message
    message.reasoning = "because"
    client = FakeClient(
        lambda kwargs: SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)
    )
    report = run_smoke(client, "m")
    assert report.reasoning_field == "reasoning"
    assert (report.prompt_tokens, report.completion_tokens) == (None, None)  # usage missing


@pytest.mark.parametrize(
    ("kwargs", "detail"),
    [
        ({"arguments": '{"a": "2", "b": 3}'}, "unexpected arguments"),
        ({"arguments": "not json"}, "unexpected arguments"),
        ({"name": "subtract"}, "unexpected tool name"),
        ({"no_call": True}, "no tool call"),
    ],
)
def test_a_wrong_tool_call_fails_without_followups_but_still_probes(kwargs, detail):
    client = FakeClient(lambda c: response(**kwargs))
    report = run_smoke(client, "m")
    assert not report.tool_call_ok and detail in report.tool_call_detail
    assert report.followup_plain is None and report.followup_with_reasoning is None
    assert passback_needed(report) is None
    assert len(client.calls) == 4  # the call and the three parameter probes


def test_if_the_first_call_raises_nothing_else_is_tried():
    def handler(kwargs):
        raise PermissionError("invalid api key " + "x" * 500)

    client = FakeClient(handler)
    report = run_smoke(client, "m")
    assert not report.tool_call_ok
    assert report.tool_call_detail.startswith("PermissionError: invalid api key")
    assert len(report.tool_call_detail) <= len("PermissionError: ") + 200
    assert len(client.calls) == 1 and report.param_probes == {}


def test_a_failing_models_endpoint_is_tolerated():
    report = run_smoke(FakeClient(lambda k: response(), list_fails=True), "m")
    assert report.available_models == [] and report.tool_call_ok


def test_a_rejected_parameter_is_recorded_not_raised():
    def handler(kwargs):
        if "seed" in kwargs:
            raise ValueError("seed is not supported")
        return response()

    report = run_smoke(FakeClient(handler), "m")
    assert report.param_probes["temperature"].ok
    assert not report.param_probes["seed"].ok
    assert "seed is not supported" in report.param_probes["seed"].detail
    assert report.param_probes["tool_choice_required"].ok


def test_passback_is_unknown_when_both_followups_fail():
    report = SmokeReport(
        "m",
        tool_call_ok=True,
        followup_plain=ProbeResult(False, "boom"),
        followup_with_reasoning=ProbeResult(False, "boom"),
    )
    assert passback_needed(report) is None


def test_format_report_lines():
    def handler(kwargs):
        if is_followup(kwargs) and "reasoning_content" not in assistant_message(kwargs):
            raise RuntimeError("must pass back")
        if "seed" in kwargs:
            raise ValueError("no seed")
        return response(reasoning="hm")

    text = format_report(run_smoke(FakeClient(handler), "deepseek-x"))
    lines = text.splitlines()
    assert lines[0] == "model: deepseek-x"
    assert lines[1] == "available models: 2 (m-a, m-b)"
    assert "tool call: OK" in lines
    assert "reasoning field: reasoning_content" in lines
    assert "tokens: prompt=11 completion=7" in lines
    assert "follow-up without reasoning: FAILED (RuntimeError: must pass back)" in lines
    assert "follow-up with reasoning: OK" in lines
    assert "reasoning pass-back needed: yes" in lines
    assert "probe temperature: OK" in lines
    assert "probe seed: FAILED (ValueError: no seed)" in lines
    assert text.endswith("\n")


def test_format_report_for_a_report_without_reasoning():
    text = format_report(run_smoke(FakeClient(lambda k: response()), "m"))
    assert "reasoning field: none" in text
    assert "follow-up with reasoning: not applicable" in text
    assert "reasoning pass-back needed: no" in text


# --- the script -------------------------------------------------------------------------


def load_script():
    spec = importlib.util.spec_from_file_location("smoke_script", ROOT / "scripts" / "smoke_llm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_requires_the_key_and_the_model(tmp_path, capsys):
    script = load_script()
    assert script.main([], env={"LLM_MODEL": "m"}) == 2
    assert "LLM_API_KEY" in capsys.readouterr().err
    assert script.main([], env={"LLM_API_KEY": "k"}) == 2
    assert "LLM_MODEL" in capsys.readouterr().err
    assert script.main([], env={"LLM_API_KEY": "", "LLM_MODEL": "m"}) == 2  # empty counts as unset


def test_script_runs_with_an_injected_client_and_writes_the_report(tmp_path, capsys):
    script = load_script()
    seen = {}

    def factory(api_key, base_url):
        seen["args"] = (api_key, base_url)
        return FakeClient(lambda kwargs: response())

    out = tmp_path / "notes" / "smoke.txt"
    env = {"LLM_API_KEY": "secret-key", "LLM_MODEL": "m", "LLM_BASE_URL": "https://example.test/v1"}
    assert script.main([], env=env, client_factory=factory, out=out) == 0
    assert seen["args"] == ("secret-key", "https://example.test/v1")
    assert out.read_text().startswith("model: m\n")
    printed = capsys.readouterr().out
    assert "secret-key" not in printed and "secret-key" not in out.read_text()  # never echoed


def test_script_exit_code_is_1_when_the_tool_call_fails(tmp_path):
    script = load_script()
    factory = lambda key, url: FakeClient(lambda kwargs: response(no_call=True))  # noqa: E731
    env = {"LLM_API_KEY": "k", "LLM_MODEL": "m"}
    assert script.main([], env=env, client_factory=factory, out=tmp_path / "o.txt") == 1


def test_base_url_is_none_when_not_set(tmp_path):
    script = load_script()
    seen = {}

    def factory(api_key, base_url):
        seen["base_url"] = base_url
        return FakeClient(lambda kwargs: response())

    script.main(
        [], env={"LLM_API_KEY": "k", "LLM_MODEL": "m"}, client_factory=factory, out=tmp_path / "o"
    )
    assert seen["base_url"] is None


def test_the_report_lists_the_message_fields():
    """Which fields the message carries is the evidence behind the reasoning verdict."""
    with_reasoning = format_report(run_smoke(FakeClient(lambda k: response(reasoning="hm")), "m"))
    field_lines = [ln for ln in with_reasoning.splitlines() if ln.startswith("message fields:")]
    assert len(field_lines) == 1
    assert "tool_calls" in field_lines[0] and "reasoning_content" in field_lines[0]
    plain = format_report(run_smoke(FakeClient(lambda k: response()), "m"))
    plain_line = [ln for ln in plain.splitlines() if ln.startswith("message fields:")]
    assert plain_line and "content" in plain_line[0]
    assert "reasoning_content" not in plain_line[0]
