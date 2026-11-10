"""scripts/run_ladder.py: which games the ladder plays when no --game is given."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from replaylab.config import load_games

ROOT = Path(__file__).resolve().parents[1]

# The games of the committed reports/ladder.csv. The default selection must reproduce it:
# a final-test game or the rehearsal copy slipping in would make the default run fail
# (the ladder refuses the first; the second's target is not in data/processed).
COMMITTED_LADDER = {
    "N-dev",
    "N-dev-delay2",
    "E",
    "S-human-serum-albumin",
    "S-human-tnfa",
    "S-human-gm2a",
    "S-human-orm2",
    "S-human-rfk",
    "S-hnmt",
    "S-human-ambp",
    "S-human-idi2",
}


def load_script():
    spec = importlib.util.spec_from_file_location("run_ladder", ROOT / "scripts" / "run_ladder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_selection_is_the_committed_ladder():
    games = load_games(ROOT / "config/games.yaml")
    selected, notes = load_script().default_games(games)
    assert set(selected) == COMMITTED_LADDER
    assert len(selected) == len(COMMITTED_LADDER)
    assert not any(games[g].final_test or games[g].split == "heldout" for g in selected)
    text = "\n".join(notes)
    assert "skipping final-test game(s) ['AB-nipah', 'AB-egfr', 'AB-bbf14', 'AB-bhrf1']" in text
    assert "skipping rehearsal game(s) ['AB-rehearsal']" in text
    assert "skipping held-out game(s) ['N-held']" in text
    assert "skipping exploratory game(s) ['E-long']" in text


def test_include_exploratory_adds_only_the_exploratory_games():
    games = load_games(ROOT / "config/games.yaml")
    selected, notes = load_script().default_games(games, include_exploratory=True)
    assert set(selected) == COMMITTED_LADDER | {"E-long"}
    assert not any("exploratory" in note for note in notes)
    assert any("final-test" in note for note in notes)


def test_main_never_reaches_a_final_test_game(monkeypatch, capsys):
    script = load_script()
    seen = {}

    def fake_run_ladder(games, features, *, game_ids, **kwargs):
        seen["game_ids"] = list(game_ids)
        raise ValueError("stop here")  # nothing is played or written

    monkeypatch.setattr(script, "run_ladder", fake_run_ladder)
    assert script.main(["--seeds", "1"]) == 1
    assert set(seen["game_ids"]) == COMMITTED_LADDER
    assert "skipping final-test game(s)" in capsys.readouterr().out
