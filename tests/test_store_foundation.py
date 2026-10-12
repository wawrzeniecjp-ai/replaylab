"""T10: store foundation: schema, migrations, campaigns, designs, events, clock."""

from __future__ import annotations

import sqlite3

import pytest

from replaylab.domain.types import CampaignConfig, Design
from replaylab.store import CampaignStore, StoreError, UnknownCampaign
from replaylab.store.errors import SchemaError

FIXED_TIME = "2026-01-01T00:00:00+00:00"


def fixed_clock() -> str:
    return FIXED_TIME


def make_config(**overrides) -> CampaignConfig:
    base = dict(
        name="test",
        target="nipah-glycoprotein-g",
        mode="replay",
        budget=6,
        batch_size=2,
        rounds=3,
        delay_rounds=2,
        seed=7,
        features=("boltz2_ipsae", "esmfold_plddt"),
    )
    base.update(overrides)
    return CampaignConfig(**base)


def make_designs(n: int = 3) -> list[Design]:
    return [
        Design(
            f"d{i:04d}",
            sequence="ACDEFGHIKL" * (i + 1),
            design_class="Peptide",
            features={"x": i / 10},
        )
        for i in range(1, n + 1)
    ]


@pytest.fixture()
def store():
    with CampaignStore.open(":memory:", clock=fixed_clock) as opened:
        yield opened


# --- opening and migrating --------------------------------------------------------------


def test_in_memory_store_is_migrated_to_version_one(store):
    assert store.schema_version == 1


def test_in_memory_stores_are_independent():
    with CampaignStore.open() as first, CampaignStore.open() as second:
        first.create_campaign(make_config())
        assert first.campaign_ids() == [1]
        assert second.campaign_ids() == []


def test_file_store_uses_wal_and_can_be_reopened(tmp_path):
    path = tmp_path / "campaign.db"
    with CampaignStore.open(path) as first:
        first.create_campaign(make_config())
    with CampaignStore.open(path) as second:
        assert second.schema_version == 1
        assert second.campaign_ids() == [1]
    mode = sqlite3.connect(path).execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"


def test_an_empty_version_zero_database_is_migrated(tmp_path):
    path = tmp_path / "empty.db"
    sqlite3.connect(path).close()
    with CampaignStore.open(path) as opened:
        assert opened.schema_version == 1


def test_a_database_from_a_newer_version_is_refused(tmp_path):
    path = tmp_path / "newer.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    conn.execute("INSERT INTO schema_version VALUES (99)")
    conn.commit()
    conn.close()
    with pytest.raises(SchemaError, match="newer"):
        CampaignStore.open(path)


def test_all_store_errors_share_a_base_class():
    assert issubclass(SchemaError, StoreError)
    assert issubclass(UnknownCampaign, StoreError)


def test_using_a_closed_store_fails():
    opened = CampaignStore.open()
    opened.close()
    with pytest.raises(sqlite3.ProgrammingError):
        opened.campaign_ids()


# --- campaigns --------------------------------------------------------------------------


def test_create_campaign_and_get_config_round_trip(store):
    config = make_config()
    campaign_id = store.create_campaign(config)
    assert campaign_id == 1
    assert store.get_config(campaign_id) == config
    assert store.get_config(campaign_id).features == ("boltz2_ipsae", "esmfold_plddt")  # a tuple
    second = store.create_campaign(make_config(name="other"))
    assert store.campaign_ids() == [campaign_id, second]


def test_unknown_campaigns_raise(store):
    for call in (
        lambda: store.get_config(999),
        lambda: store.get_clock(999),
        lambda: store.add_designs(999, make_designs()),
        lambda: store.append_event(999, "x", {}),
        lambda: store.get_designs(999),
    ):
        with pytest.raises(UnknownCampaign, match="999"):
            call()


# --- designs ----------------------------------------------------------------------------


def test_add_designs_inserts_and_keeps_insertion_order(store):
    campaign_id = store.create_campaign(make_config())
    assert store.add_designs(campaign_id, make_designs(3)) == 3
    assert [d.design_id for d in store.get_designs(campaign_id)] == ["d0001", "d0002", "d0003"]
    store.add_designs(campaign_id, [Design("d0000", sequence="WWW")])
    assert [d.design_id for d in store.get_designs(campaign_id)][-1] == "d0000"


def test_add_designs_is_idempotent_for_identical_rows(store):
    campaign_id = store.create_campaign(make_config())
    store.add_designs(campaign_id, make_designs(3))
    assert store.add_designs(campaign_id, make_designs(3)) == 0
    kinds = [e.kind for e in store.events(campaign_id)]
    assert kinds.count("designs_added") == 1  # no event for a no-op


def test_conflicting_redefinition_raises_and_rolls_back_the_whole_call(store):
    campaign_id = store.create_campaign(make_config())
    store.add_designs(campaign_id, make_designs(1))
    changed = Design("d0001", sequence="DIFFERENT")
    new = Design("d9999", sequence="ACDEF")
    with pytest.raises(StoreError, match="already exists with different values"):
        store.add_designs(campaign_id, [new, changed])
    assert [d.design_id for d in store.get_designs(campaign_id)] == ["d0001"]  # new one not kept


def test_features_round_trip_exactly(store):
    campaign_id = store.create_campaign(make_config())
    features = {"a": 0.1 + 0.2, "b": 1e-12, "c": 123456789.123456789, "d": 3, "e": -0.0}
    store.add_designs(campaign_id, [Design("d0001", sequence="ACD", features=features)])
    loaded = store.get_designs(campaign_id)[0]
    assert dict(loaded.features) == features
    assert loaded.features["a"] == 0.30000000000000004  # exact, not rounded


def test_design_class_none_and_sequence_survive(store):
    campaign_id = store.create_campaign(make_config())
    store.add_designs(campaign_id, [Design("d0001", sequence="ACDEF", design_class=None)])
    loaded = store.get_designs(campaign_id)[0]
    assert loaded.design_class is None and loaded.sequence == "ACDEF"


def test_designs_of_different_campaigns_do_not_mix(store):
    first = store.create_campaign(make_config(name="one"))
    second = store.create_campaign(make_config(name="two"))
    store.add_designs(first, make_designs(3))
    store.add_designs(second, make_designs(1))
    assert len(store.get_designs(first)) == 3 and len(store.get_designs(second)) == 1


# --- events -----------------------------------------------------------------------------


def test_events_are_ordered_and_use_the_injected_clock(store):
    campaign_id = store.create_campaign(make_config())
    store.add_designs(campaign_id, make_designs(2))
    store.append_event(campaign_id, "custom", {"b": 2, "a": 1})
    events = store.events(campaign_id)
    assert [e.kind for e in events] == ["campaign_created", "designs_added", "custom"]
    assert [e.id for e in events] == sorted(e.id for e in events)
    assert all(e.ts == FIXED_TIME and e.campaign_id == campaign_id for e in events)
    assert events[0].payload == {"name": "test", "target": "nipah-glycoprotein-g", "mode": "replay"}
    assert events[1].payload == {"n": 2}
    assert events[2].payload == {"a": 1, "b": 2}


def test_append_event_rejects_an_empty_kind(store):
    campaign_id = store.create_campaign(make_config())
    with pytest.raises(StoreError, match="kind"):
        store.append_event(campaign_id, "", {})


def test_events_are_append_only_at_the_database_level(store):
    """White-box test: the triggers must stop raw UPDATE and DELETE statements."""
    campaign_id = store.create_campaign(make_config())
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        store._conn.execute("UPDATE events SET kind = 'x' WHERE campaign_id = ?", (campaign_id,))
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        store._conn.execute("DELETE FROM events WHERE campaign_id = ?", (campaign_id,))
    assert len(store.events(campaign_id)) == 1


# --- clock ------------------------------------------------------------------------------


def test_clock_starts_at_zero_and_only_moves_forward(store):
    campaign_id = store.create_campaign(make_config())
    assert store.get_clock(campaign_id) == 0
    store.set_clock(campaign_id, 3)
    assert store.get_clock(campaign_id) == 3
    store.set_clock(campaign_id, 3)  # same value: no-op
    with pytest.raises(StoreError, match="cannot decrease"):
        store.set_clock(campaign_id, 2)
    assert store.get_clock(campaign_id) == 3
    clock_events = [e for e in store.events(campaign_id) if e.kind == "clock_advanced"]
    assert [e.payload for e in clock_events] == [{"from": 0, "to": 3}]  # exactly one event


# --- integrity and concurrency ----------------------------------------------------------


def test_foreign_keys_are_enforced(store):
    """White-box test: a row pointing at a missing campaign must be rejected."""
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "INSERT INTO designs (campaign_id, design_id, position, sequence, features_json) "
            "VALUES (999, 'x', 0, 'ACD', '{}')"
        )


def test_a_second_reader_sees_committed_data_while_the_writer_is_open(tmp_path):
    path = tmp_path / "shared.db"
    with CampaignStore.open(path) as writer:
        campaign_id = writer.create_campaign(make_config())
        writer.add_designs(campaign_id, make_designs(2))
        reader = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            assert reader.execute("SELECT COUNT(*) FROM designs").fetchone()[0] == 2
        finally:
            reader.close()
