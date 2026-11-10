"""The hashes that recorded results and the final test's pre-registration depend on.

``config_hash`` is stamped into every LLM record; two records compare only if it matches.
``frozen_hash`` is what the final test's lock pins for each ``AB-`` game. The deepseek and mimo
values below are the ones registered in ``docs/prereg-anthropic.md`` section 6; the others were
recorded on 2026-11-10 (commit ``aaee2ce``) as the baseline for the legibility refactor
(``docs/refactor-proposal.md``).

A failure means something the model sees (prompt, tool descriptions, conditions, limits, a
model's request parameters or price) or a final-test game's frozen configuration has changed.
A refactor must never cause that. Update a value only on purpose, together with a new
pre-registration or a fresh trial block, never to make this test pass.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from replaylab.config import load_feature_config, load_games
from replaylab.eval import cli
from replaylab.eval.final_run import frozen_hash
from replaylab.eval.loop import Limits
from replaylab.eval.models import load_models
from replaylab.eval.prompts import USER_PROMPT, load_system_prompt
from replaylab.eval.spend import load_prices
from replaylab.interfaces.tools import TOOL_DESCRIPTIONS

ROOT = Path(__file__).resolve().parents[1]
FILES = cli.Files(
    **{
        name: ROOT / getattr(cli.DEFAULT, name)
        for name in cli.Files.__dataclass_fields__  # every default path, from the repo root
    }
)

CONFIG_HASHES = {
    "deepseek": "0fecb4a7e74a",  # registered, prereg-anthropic section 6
    "mimo": "c2c8dac7e136",  # registered, prereg-anthropic section 6
    "opus": "c660bcb365e4",
    "sonnet": "dfc8c3e686b3",
    "local-qwen": "a9d264ab36b3",
}

FROZEN_HASHES = {
    "AB-bbf14": "3bb06367dc15",
    "AB-bhrf1": "4fda9854f46d",
    "AB-egfr": "25e93623ed1a",
    "AB-nipah": "de44e444dfcd",
}


@pytest.mark.parametrize("model", sorted(CONFIG_HASHES))
def test_config_hash_is_unchanged(model):
    cfg = load_models(FILES.models_config)[model]
    price = load_prices(FILES.prices_config)[cfg.price_key]
    assert cli._llm_config_hash(cfg, price, FILES) == CONFIG_HASHES[model]


@pytest.mark.parametrize("game_id", sorted(FROZEN_HASHES))
def test_final_test_frozen_hash_is_unchanged(game_id):
    game = load_games(FILES.games_config)[game_id]
    frozen = frozen_hash(
        system_prompt=load_system_prompt(FILES.system_prompt),
        user_prompt=USER_PROMPT,
        conditions_text=FILES.conditions_config.read_text(),
        tool_descriptions=TOOL_DESCRIPTIONS,
        limits=asdict(Limits()),
        game=game,
        feature_config=load_feature_config(FILES.features_config)[game.target],
    )
    assert frozen == FROZEN_HASHES[game_id]


def test_every_final_test_game_is_pinned():
    games = load_games(FILES.games_config)
    assert {g for g, game in games.items() if game.final_test} == set(FROZEN_HASHES)
