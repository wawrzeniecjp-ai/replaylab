"""Aliases for design ids."""

from __future__ import annotations

import hashlib
import re

import pytest

from replaylab.datasets.aliases import Aliaser, UnknownIdError

IDS = [f"real-{i:04d}" for i in range(25)]


def test_aliases_are_d_followed_by_four_digits_starting_at_one():
    aliaser = Aliaser.from_seed(IDS, seed=0)
    assert aliaser.aliases() == [f"d{i:04d}" for i in range(1, 26)]
    assert len(aliaser) == 25


def test_mapping_is_a_bijection_and_round_trips():
    aliaser = Aliaser.from_seed(IDS, seed=3)
    aliases = [aliaser.alias(real) for real in IDS]
    assert len(set(aliases)) == len(IDS)
    assert [aliaser.real(alias) for alias in aliases] == IDS
    assert all(real in aliaser for real in IDS)
    assert "real-9999" not in aliaser


def test_same_seed_same_mapping_regardless_of_input_order():
    first = Aliaser.from_seed(IDS, seed=5)
    second = Aliaser.from_seed(list(reversed(IDS)), seed=5)
    assert all(first.alias(real) == second.alias(real) for real in IDS)


def test_different_seeds_give_different_mappings():
    first = Aliaser.from_seed(IDS, seed=1)
    second = Aliaser.from_seed(IDS, seed=2)
    assert any(first.alias(real) != second.alias(real) for real in IDS)


def test_order_is_defined_by_sha256_of_seed_and_id():
    """The mapping must not depend on any RNG: it is the sha256 order, ties by id."""
    seed = 11
    expected = sorted(IDS, key=lambda i: (hashlib.sha256(f"{seed}|{i}".encode()).hexdigest(), i))
    aliaser = Aliaser.from_seed(IDS, seed=seed)
    assert [aliaser.real(f"d{n:04d}") for n in range(1, 26)] == expected


def test_alias_width_grows_beyond_9999_designs():
    ids = [f"x{i}" for i in range(10_000)]
    aliaser = Aliaser.from_seed(ids, seed=0)
    assert set(map(len, aliaser.aliases())) == {6}  # d + 5 digits
    assert aliaser.aliases()[0] == "d00001"


def test_alias_strings_never_contain_a_real_id():
    aliaser = Aliaser.from_seed(IDS, seed=0)
    assert all(re.fullmatch(r"d\d{4}", alias) for alias in aliaser.aliases())
    for alias in aliaser.aliases():
        assert not any(real in alias for real in IDS)


def test_unknown_ids_raise_a_key_error_subclass():
    aliaser = Aliaser.from_seed(IDS, seed=0)
    with pytest.raises(UnknownIdError, match="unknown design id"):
        aliaser.alias("nope")
    with pytest.raises(UnknownIdError, match="unknown alias"):
        aliaser.real("d9999")
    assert issubclass(UnknownIdError, KeyError)


def test_duplicate_real_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        Aliaser.from_seed(["a", "a"], seed=0)


def test_empty_input_gives_an_empty_mapping():
    assert len(Aliaser.from_seed([], seed=0)) == 0
