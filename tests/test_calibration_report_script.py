"""scripts/calibration_report.py: a prediction is scored only against the batch it describes."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    sys.path.insert(0, str(ROOT / "scripts"))  # the script imports export_study beside it
    try:
        spec = importlib.util.spec_from_file_location(
            "calibration_report", ROOT / "scripts" / "calibration_report.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    return module


def call(call_id: str, name: str, arguments: dict) -> dict:
    return {"id": call_id, "function": {"name": name, "arguments": json.dumps(arguments)}}


def reply(call_id: str, content) -> dict:
    text = content if isinstance(content, str) else json.dumps(content)
    return {"role": "tool", "tool_call_id": call_id, "content": text}


def forecast(rate: float, low: float, high: float) -> dict:
    nb = {"predicted_rate": rate, "interval90": [low, high], "calibrated": False}
    return {"n": 15, "hits": 2, "next_batch": nb}


def bought(*hits: bool) -> dict:
    results = [{"design_id": f"d{i}", "hit": h} for i, h in enumerate(hits)]
    return {"batch_id": "b001", "status": "complete", "results": results}


def episode(*turns) -> dict:
    messages = []
    for calls, replies in turns:
        messages.append({"role": "assistant", "content": None, "tool_calls": calls})
        messages.extend(replies)
    return {"messages": messages}


def test_a_forecast_is_scored_only_against_exactly_its_batch():
    script = load_script()
    record = episode(
        (
            [
                call("1", "estimate_hit_rate", {"next_batch": ["a", "b"]}),
                call("2", "estimate_hit_rate", {"next_batch": ["a", "c"]}),
                call("3", "estimate_hit_rate", {"next_batch": ["c", "a"]}),
            ],
            [
                reply("1", forecast(0.5, 0.1, 0.9)),
                reply("2", forecast(0.2, 0.0, 0.4)),
                reply("3", forecast(0.6, 0.55, 0.65)),
            ],
        ),
        (
            [call("4", "submit_batch", {"design_ids": ["a", "c"]})],
            [reply("4", bought(True, False))],
        ),
        # Round 2: a forecast of a batch never bought stays unscored.
        (
            [call("5", "estimate_hit_rate", {"next_batch": ["x"]})],
            [reply("5", forecast(0.3, 0, 1))],
        ),
        ([call("6", "submit_batch", {"design_ids": ["y"]})], [reply("6", bought(True))]),
    )
    forecasts, _ = script.episode_predictions(record)
    assert [f["round"] for f in forecasts] == [1, 1, 1, 2]
    first, second, last, other = forecasts
    assert first["realized"] is None  # a different batch
    assert second["realized"] is None  # superseded by a later forecast of the same set
    assert (last["realized"], last["covered"]) == (0.5, False)  # 0.5 lies outside [0.55, 0.65]
    assert other["realized"] is None


def test_a_proposal_counts_as_bought_only_when_bought_whole():
    script = load_script()
    proposal = {
        "batch_id": "b001",
        "design_ids": ["a", "b", "c"],
        "predicted_hit_rate": 0.4,
        "calibration": {"basis": "model"},
    }
    record = episode(
        ([call("1", "propose_batch", {"k": 3})], [reply("1", proposal)]),
        # Two of the three proposed designs: a subset, not the proposal.
        ([call("2", "submit_batch", {"design_ids": ["a", "b"]})], [reply("2", bought(True, True))]),
        ([call("3", "propose_batch", {"k": 3})], [reply("3", proposal)]),
        (
            [call("4", "submit_batch", {"design_ids": ["c", "b", "a"]})],
            [reply("4", bought(True, False, False))],
        ),  # fmt: skip
    )
    _, proposals = script.episode_predictions(record)
    assert [(p["round"], p["bought"], p["realized"]) for p in proposals] == [
        (1, False, None),
        (2, True, 1 / 3),
    ]


def test_a_failed_call_is_skipped():
    script = load_script()
    record = episode(
        (
            [call("1", "estimate_hit_rate", {"next_batch": ["a"]})],
            [reply("1", "ERROR: insufficient_data: need at least 2 hits")],
        ),
        ([call("2", "submit_batch", {"design_ids": ["a"]})], [reply("2", bought(False))]),
    )
    assert script.episode_predictions(record) == ([], [])
