"""T04: pinning the OpenAPI spec."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from replaylab.spec_pin import openapi_info, read_pin, sha256_file, verify_pin, write_pin

ROOT = Path(__file__).resolve().parents[1]
SPEC = {"openapi": "3.1.0", "info": {"title": "x", "version": "0.0.2"}, "paths": {}}


@pytest.fixture()
def spec(tmp_path) -> Path:
    path = tmp_path / "openapi.prod.json"
    path.write_text(json.dumps(SPEC))
    return path


def load_script():
    spec = importlib.util.spec_from_file_location("pin_script", ROOT / "scripts" / "pin_spec.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sha256_matches_hashlib(spec):
    assert sha256_file(spec) == hashlib.sha256(spec.read_bytes()).hexdigest()


def test_pin_file_uses_sha256sum_format(spec, tmp_path):
    pin = tmp_path / "openapi.sha256"
    digest = write_pin(spec, pin)
    assert pin.read_text() == f"{digest}  openapi.prod.json\n"
    assert read_pin(pin) == digest


def test_verify_detects_a_changed_file(spec, tmp_path):
    pin = tmp_path / "pin"
    write_pin(spec, pin)
    assert verify_pin(spec, pin) is True
    spec.write_text(json.dumps({**SPEC, "paths": {"/x": {}}}))
    assert verify_pin(spec, pin) is False


@pytest.mark.parametrize("content", ["", "nothex  file\n", "abc  file\n", ("Z" * 64) + "  f\n"])
def test_malformed_pin_files_raise(tmp_path, content):
    pin = tmp_path / "pin"
    pin.write_text(content)
    with pytest.raises(ValueError, match="not a valid sha256 pin file"):
        read_pin(pin)


def test_openapi_info(spec, tmp_path):
    assert openapi_info(spec) == ("3.1.0", "0.0.2")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"hello": 1}))
    with pytest.raises(ValueError, match="not an OpenAPI document"):
        openapi_info(bad)


def test_script_write_then_verify(spec, tmp_path, capsys):
    script = load_script()
    pin = tmp_path / "pin"
    assert script.main(["write", "--spec", str(spec), "--pin", str(pin)]) == 0
    printed = capsys.readouterr().out
    assert "openapi 3.1.0, info.version 0.0.2" in printed and "sha256" in printed
    assert script.main(["verify", "--spec", str(spec), "--pin", str(pin)]) == 0
    assert "OK openapi.prod.json matches the pin" in capsys.readouterr().out
    spec.write_text(json.dumps({**SPEC, "paths": {"/x": {}}}))
    assert script.main(["verify", "--spec", str(spec), "--pin", str(pin)]) == 1
    assert "MISMATCH" in capsys.readouterr().err


def test_script_exit_code_2_for_missing_or_invalid_files(spec, tmp_path):
    script = load_script()
    missing = tmp_path / "missing.json"
    assert script.main(["write", "--spec", str(missing), "--pin", str(tmp_path / "p")]) == 2
    assert script.main(["verify", "--spec", str(spec), "--pin", str(tmp_path / "nopin")]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert script.main(["write", "--spec", str(bad), "--pin", str(tmp_path / "p")]) == 2


@pytest.mark.skipif(
    not (ROOT / "spec" / "openapi.prod.json").exists()
    or not (ROOT / "spec" / "openapi.sha256").exists(),
    reason="spec/openapi.prod.json is not pinned yet (run scripts/pin_spec.py write)",
)
def test_committed_pin_matches_the_committed_spec():
    assert verify_pin(ROOT / "spec" / "openapi.prod.json", ROOT / "spec" / "openapi.sha256")
    assert openapi_info(ROOT / "spec" / "openapi.prod.json") == ("3.1.0", "0.0.2")
