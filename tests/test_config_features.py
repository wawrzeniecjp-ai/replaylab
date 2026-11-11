"""Feature configuration (FeatureConfig in domain, loader in replaylab.config)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from replaylab.config import ConfigError, load_feature_config, parse_feature_config
from replaylab.domain.features import FeatureConfig, FeatureSpec

ROOT = Path(__file__).resolve().parents[1]

VALID = {
    "version": 1,
    "targets": {
        "t1": {
            "first_column": "a",
            "features": [{"name": "a"}, {"name": "b", "higher_is_better": False}],
        }
    },
}


def mutated(mutator) -> dict:
    data = deepcopy(VALID)
    mutator(data)
    return data


# --- FeatureConfig (domain) -------------------------------------------------------------


def test_feature_config_names_and_orientation():
    config = FeatureConfig("t", "a", (FeatureSpec("a"), FeatureSpec("b", higher_is_better=False)))
    assert config.names == ("a", "b")
    assert config.orientation == {"a": 1.0, "b": -1.0}


@pytest.mark.parametrize(
    ("specs", "first", "message"),
    [
        ((), "a", "at least one feature"),
        ((FeatureSpec("a"), FeatureSpec("a")), "a", "duplicate feature"),
        ((FeatureSpec("a"),), "zzz", "first_column"),
    ],
)
def test_feature_config_rejects_bad_input(specs, first, message):
    with pytest.raises(ValueError, match=message):
        FeatureConfig("t", first, specs)


# --- parsing ----------------------------------------------------------------------------


def test_valid_config_parses_and_defaults_to_higher_is_better():
    result = parse_feature_config(VALID)
    config = result["t1"]
    assert isinstance(config, FeatureConfig)
    assert config.first_column == "a"
    assert config.names == ("a", "b")
    assert config.orientation == {"a": 1.0, "b": -1.0}


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda d: d.update(extra=1), "unknown key"),
        (lambda d: d.pop("targets"), "missing key"),
        (lambda d: d.update(version=2), "unsupported version"),
        (lambda d: d.update(version=True), "unsupported version"),
        (lambda d: d.update(targets={}), "non-empty mapping"),
        (lambda d: d["targets"]["t1"].update(extra=1), "unknown key"),
        (lambda d: d["targets"]["t1"].pop("first_column"), "missing key"),
        (lambda d: d["targets"]["t1"].update(features=[]), "non-empty list"),
        (lambda d: d["targets"]["t1"]["features"].append({"name": "a"}), "duplicate feature"),
        (lambda d: d["targets"]["t1"].update(first_column="zzz"), "first_column"),
        (lambda d: d["targets"]["t1"]["features"][0].update(color="red"), "unknown key"),
        (lambda d: d["targets"]["t1"]["features"][0].update(name=""), "non-empty string"),
        (
            lambda d: d["targets"]["t1"]["features"][0].update(higher_is_better="yes"),
            "true or false",
        ),
    ],
)
def test_invalid_configs_raise_config_error(mutator, message):
    with pytest.raises(ConfigError, match=message):
        parse_feature_config(mutated(mutator))


def test_non_mapping_input_is_rejected():
    with pytest.raises(ConfigError, match="expected a mapping"):
        parse_feature_config(["not", "a", "mapping"])


def test_config_error_is_a_value_error():
    assert issubclass(ConfigError, ValueError)


# --- loading from disk ------------------------------------------------------------------


def test_load_from_file(tmp_path):
    path = tmp_path / "features.yaml"
    path.write_text(
        "version: 1\ntargets:\n  t1:\n    first_column: a\n    features:\n"
        "      - {name: a}\n      - {name: b, higher_is_better: false}\n"
    )
    assert load_feature_config(path)["t1"].names == ("a", "b")


def test_invalid_yaml_raises_config_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("version: 1\ntargets: [unclosed")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_feature_config(path)


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_feature_config(tmp_path / "nope.yaml")


def test_committed_features_yaml_is_valid():
    config = load_feature_config(ROOT / "config" / "features.yaml")
    nipah = config["nipah-glycoprotein-g"]
    assert nipah.first_column == "boltz2_ipsae"
    assert nipah.orientation["proteinmpnn_score"] == -1.0
    assert nipah.orientation["boltz2_iptm"] == 1.0
    assert len(nipah.names) == 7
    egfr = config["egfr"]
    assert not any(name.startswith("boltz2_") for name in egfr.names)  # EGFR has no Boltz-2
