"""Database schema and migrations.

``MIGRATIONS[n]`` upgrades a database from version ``n`` to ``n + 1``. Never edit an
existing migration once released; add a new one.
"""

from __future__ import annotations

import sqlite3

from replaylab.store.errors import SchemaError

V1 = """
CREATE TABLE schema_version (version INTEGER NOT NULL);

CREATE TABLE campaigns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    target      TEXT    NOT NULL,
    mode        TEXT    NOT NULL CHECK (mode IN ('replay', 'live')),
    config_json TEXT    NOT NULL,
    clock       INTEGER NOT NULL DEFAULT 0 CHECK (clock >= 0),
    created_at  TEXT    NOT NULL
);

CREATE TABLE designs (
    campaign_id   INTEGER NOT NULL REFERENCES campaigns(id),
    design_id     TEXT    NOT NULL,
    position      INTEGER NOT NULL,
    sequence      TEXT    NOT NULL,
    design_class  TEXT,
    features_json TEXT    NOT NULL,
    PRIMARY KEY (campaign_id, design_id)
);

CREATE TABLE batches (
    campaign_id INTEGER NOT NULL REFERENCES campaigns(id),
    id          TEXT    NOT NULL,
    round       INTEGER NOT NULL CHECK (round >= 1),
    status      TEXT    NOT NULL CHECK (status IN ('proposed', 'submitted', 'partial', 'complete')),
    backend_ref TEXT,
    created_at  TEXT    NOT NULL,
    PRIMARY KEY (campaign_id, id)
);

CREATE TABLE batch_items (
    campaign_id INTEGER NOT NULL,
    batch_id    TEXT    NOT NULL,
    design_id   TEXT    NOT NULL,
    position    INTEGER NOT NULL,
    PRIMARY KEY (campaign_id, batch_id, design_id),
    FOREIGN KEY (campaign_id, batch_id)  REFERENCES batches(campaign_id, id),
    FOREIGN KEY (campaign_id, design_id) REFERENCES designs(campaign_id, design_id)
);

CREATE TABLE outcomes (
    campaign_id INTEGER NOT NULL,
    design_id   TEXT    NOT NULL,
    batch_id    TEXT    NOT NULL,
    hit         INTEGER CHECK (hit IN (0, 1)),
    strength    TEXT,
    kd_molar    REAL,
    raw_json    TEXT    NOT NULL,
    received_at TEXT    NOT NULL,
    PRIMARY KEY (campaign_id, design_id),
    FOREIGN KEY (campaign_id, batch_id)  REFERENCES batches(campaign_id, id),
    FOREIGN KEY (campaign_id, design_id) REFERENCES designs(campaign_id, design_id)
);

CREATE TABLE predictions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id    INTEGER NOT NULL REFERENCES campaigns(id),
    round          INTEGER NOT NULL CHECK (round >= 1),
    batch_id       TEXT,
    predicted_rate REAL    NOT NULL CHECK (predicted_rate >= 0 AND predicted_rate <= 1),
    model          TEXT    NOT NULL,
    created_at     TEXT    NOT NULL
);

CREATE TABLE events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id  INTEGER NOT NULL REFERENCES campaigns(id),
    ts           TEXT    NOT NULL,
    kind         TEXT    NOT NULL,
    payload_json TEXT    NOT NULL
);

CREATE TRIGGER events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'events is append-only'); END;

CREATE TRIGGER events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'events is append-only'); END;
"""

MIGRATIONS: tuple[str, ...] = (V1,)
LATEST_VERSION = len(MIGRATIONS)


def current_version(conn: sqlite3.Connection) -> int:
    """The schema version recorded in the database; 0 for a fresh file
    (no ``schema_version`` table yet)."""
    has_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'"
    ).fetchone()
    if has_table is None:
        return 0
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    return int(row[0]) if row else 0


def migrate(conn: sqlite3.Connection) -> int:
    """Bring the database to the latest version and return it."""
    version = current_version(conn)
    if version > LATEST_VERSION:
        raise SchemaError(
            f"database version {version} is newer than the supported version {LATEST_VERSION}"
        )
    for number in range(version, LATEST_VERSION):
        conn.executescript(MIGRATIONS[number])
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (number + 1,))
        conn.commit()
    return LATEST_VERSION
