"""The campaign store: durable, idempotent persistence for one or more campaigns.

Rules enforced here (see the implementation plan, tasks T10 and T11):

* Every mutating method is one transaction: it either fully succeeds or changes nothing.
* A batch moves only forward: proposed -> submitted -> (partial) -> complete.
* A design may be in at most one batch that has left ``proposed``; proposed batches
  may overlap freely.
* Recording an outcome identical to the stored one is a no-op; a different outcome for
  the same design raises :class:`OutcomeConflict` and writes nothing.
* The event log is append-only (enforced by database triggers).
* A snapshot contains delivered outcomes only.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from replaylab.domain.types import (
    Batch,
    BatchStatus,
    CampaignConfig,
    CampaignState,
    Design,
    Outcome,
)
from replaylab.store.errors import (
    InvalidTransition,
    OutcomeConflict,
    SchemaError,
    StoreError,
    UnknownCampaign,
    UnknownDesign,
)
from replaylab.store.schema import current_version, migrate

__all__ = [
    "CampaignStore",
    "Event",
    "InvalidTransition",
    "OutcomeConflict",
    "Prediction",
    "SchemaError",
    "StoreError",
    "UnknownCampaign",
    "UnknownDesign",
]

_ALLOWED_TRANSITIONS = {
    (BatchStatus.PROPOSED, BatchStatus.SUBMITTED),
    (BatchStatus.SUBMITTED, BatchStatus.PARTIAL),
    (BatchStatus.SUBMITTED, BatchStatus.COMPLETE),
    (BatchStatus.PARTIAL, BatchStatus.COMPLETE),
}


@dataclass(frozen=True)
class Event:
    """One row of a campaign's append-only event log: what happened, when
    (UTC), and the payload that the event kind defines. Replaying this log
    is what restores a backend after a crash (kill-and-resume)."""

    id: int
    campaign_id: int
    ts: str
    kind: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class Prediction:
    """A recorded buy prediction: which round and batch, the predicted hit
    rate, which model predicted it, and when -- kept for the later
    calibration analysis."""

    round: int
    batch_id: str | None
    predicted_rate: float
    model: str
    created_at: str


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


class CampaignStore:
    """SQLite-backed store. Use :meth:`open` to create one."""

    def __init__(self, conn: sqlite3.Connection, clock: Callable[[], str]) -> None:
        self._conn = conn
        self._clock = clock

    # --- lifecycle ---------------------------------------------------------

    @classmethod
    def open(
        cls, path: Path | str = ":memory:", *, clock: Callable[[], str] | None = None
    ) -> CampaignStore:
        """Open (or create) the SQLite store at ``path`` (``":memory:"`` for a
        throwaway), creating or migrating the schema as needed. ``clock``
        injects the timestamp source; scripts leave it alone and tests pass
        a fixed one.
        """
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        if str(path) != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        try:
            migrate(conn)
        except Exception:
            conn.close()
            raise
        return cls(conn, clock or _utc_now)

    def close(self) -> None:
        """Close the database connection (prefer the context manager)."""
        self._conn.close()

    def __enter__(self) -> CampaignStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        """The store's schema version (0 for an empty file)."""
        return current_version(self._conn)

    # --- helpers -------------------------------------------------------------

    def _require_campaign(self, campaign_id: int) -> None:
        row = self._conn.execute("SELECT 1 FROM campaigns WHERE id = ?", (campaign_id,)).fetchone()
        if row is None:
            raise UnknownCampaign(f"unknown campaign {campaign_id}")

    def _event(self, campaign_id: int, kind: str, payload: Mapping[str, Any]) -> int:
        cursor = self._conn.execute(
            "INSERT INTO events (campaign_id, ts, kind, payload_json) VALUES (?, ?, ?, ?)",
            (campaign_id, self._clock(), kind, _dumps(dict(payload))),
        )
        return int(cursor.lastrowid)

    def _batch_row(self, campaign_id: int, batch_id: str) -> sqlite3.Row:
        row = self._conn.execute(
            "SELECT * FROM batches WHERE campaign_id = ? AND id = ?", (campaign_id, batch_id)
        ).fetchone()
        if row is None:
            raise StoreError(f"unknown batch {batch_id!r}")
        return row

    def _check_not_purchased(
        self, campaign_id: int, design_ids: Iterable[str], exclude_batch: str | None
    ) -> None:
        for design_id in design_ids:
            row = self._conn.execute(
                "SELECT b.id FROM batch_items bi JOIN batches b "
                "ON b.campaign_id = bi.campaign_id AND b.id = bi.batch_id "
                "WHERE bi.campaign_id = ? AND bi.design_id = ? AND b.status != 'proposed' "
                "AND b.id IS NOT ?",
                (campaign_id, design_id, exclude_batch),
            ).fetchone()
            if row is not None:
                raise StoreError(f"design {design_id!r} already purchased in batch {row['id']!r}")

    # --- campaigns -----------------------------------------------------------

    def create_campaign(self, config: CampaignConfig) -> int:
        """Insert a campaign row (clock 0, empty log) and return its id."""
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO campaigns (name, target, mode, config_json, clock, created_at) "
                "VALUES (?, ?, ?, ?, 0, ?)",
                (config.name, config.target, config.mode, _dumps(asdict(config)), self._clock()),
            )
            campaign_id = int(cursor.lastrowid)
            self._event(
                campaign_id,
                "campaign_created",
                {"name": config.name, "target": config.target, "mode": config.mode},
            )
        return campaign_id

    def campaign_ids(self) -> list[int]:
        """Every campaign id in the store, ascending (build_replay_campaign
        uses the first one to decide create-vs-resume)."""
        return [int(r["id"]) for r in self._conn.execute("SELECT id FROM campaigns ORDER BY id")]

    def get_config(self, campaign_id: int) -> CampaignConfig:
        """The stored CampaignConfig.

        Raises:
            UnknownCampaign: if the id is not in the store.
        """
        row = self._conn.execute(
            "SELECT config_json FROM campaigns WHERE id = ?", (campaign_id,)
        ).fetchone()
        if row is None:
            raise UnknownCampaign(f"unknown campaign {campaign_id}")
        data = json.loads(row["config_json"])
        data["features"] = tuple(data["features"])
        return CampaignConfig(**data)

    def get_clock(self, campaign_id: int) -> int:
        """The campaign's stored clock (elapsed rounds)."""
        row = self._conn.execute(
            "SELECT clock FROM campaigns WHERE id = ?", (campaign_id,)
        ).fetchone()
        if row is None:
            raise UnknownCampaign(f"unknown campaign {campaign_id}")
        return int(row["clock"])

    def set_clock(self, campaign_id: int, clock: int) -> None:
        """Advance the stored clock. It can only stay the same or go up."""
        with self._conn:
            current = self.get_clock(campaign_id)
            if clock < current:
                raise StoreError(f"clock cannot decrease ({current} -> {clock})")
            if clock == current:
                return
            self._conn.execute("UPDATE campaigns SET clock = ? WHERE id = ?", (clock, campaign_id))
            self._event(campaign_id, "clock_advanced", {"from": current, "to": clock})

    # --- designs -------------------------------------------------------------

    def add_designs(self, campaign_id: int, designs: Iterable[Design]) -> int:
        """Insert designs; identical re-adds are skipped. Returns how many were new."""
        designs = list(designs)
        with self._conn:
            self._require_campaign(campaign_id)
            position = self._conn.execute(
                "SELECT COUNT(*) FROM designs WHERE campaign_id = ?", (campaign_id,)
            ).fetchone()[0]
            inserted = 0
            for design in designs:
                features_json = _dumps(dict(design.features))
                row = self._conn.execute(
                    "SELECT sequence, design_class, features_json FROM designs "
                    "WHERE campaign_id = ? AND design_id = ?",
                    (campaign_id, design.design_id),
                ).fetchone()
                if row is not None:
                    same = (row["sequence"], row["design_class"], row["features_json"]) == (
                        design.sequence,
                        design.design_class,
                        features_json,
                    )
                    if not same:
                        raise StoreError(
                            f"design {design.design_id!r} already exists with different values"
                        )
                    continue
                self._conn.execute(
                    "INSERT INTO designs (campaign_id, design_id, position, sequence, "
                    "design_class, features_json) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        campaign_id,
                        design.design_id,
                        position,
                        design.sequence,
                        design.design_class,
                        features_json,
                    ),
                )
                position += 1
                inserted += 1
            if inserted:
                self._event(campaign_id, "designs_added", {"n": inserted})
        return inserted

    def get_designs(self, campaign_id: int) -> list[Design]:
        """All designs of the campaign in insertion order."""
        self._require_campaign(campaign_id)
        return [
            Design(
                design_id=r["design_id"],
                sequence=r["sequence"],
                design_class=r["design_class"],
                features=json.loads(r["features_json"]),
            )
            for r in self._conn.execute(
                "SELECT * FROM designs WHERE campaign_id = ? ORDER BY position", (campaign_id,)
            )
        ]

    # --- batches -------------------------------------------------------------

    def record_batch(self, campaign_id: int, batch: Batch) -> None:
        """Record a newly proposed batch (nothing spent yet); the batch id
        must be free."""
        with self._conn:
            self._require_campaign(campaign_id)
            exists = self._conn.execute(
                "SELECT 1 FROM batches WHERE campaign_id = ? AND id = ?",
                (campaign_id, batch.batch_id),
            ).fetchone()
            if exists:
                raise StoreError(f"batch {batch.batch_id!r} already exists")
            for design_id in batch.design_ids:
                known = self._conn.execute(
                    "SELECT 1 FROM designs WHERE campaign_id = ? AND design_id = ?",
                    (campaign_id, design_id),
                ).fetchone()
                if known is None:
                    raise UnknownDesign(f"unknown design {design_id!r}")
            if batch.status.is_purchased:
                self._check_not_purchased(campaign_id, batch.design_ids, None)
            self._conn.execute(
                "INSERT INTO batches (campaign_id, id, round, status, backend_ref, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    campaign_id,
                    batch.batch_id,
                    batch.round,
                    batch.status.value,
                    batch.backend_ref,
                    self._clock(),
                ),
            )
            for position, design_id in enumerate(batch.design_ids):
                self._conn.execute(
                    "INSERT INTO batch_items (campaign_id, batch_id, design_id, position) "
                    "VALUES (?, ?, ?, ?)",
                    (campaign_id, batch.batch_id, design_id, position),
                )
            self._event(
                campaign_id,
                f"batch_{batch.status.value}",
                {"batch_id": batch.batch_id, "round": batch.round, "n": len(batch.design_ids)},
            )

    def set_batch_status(
        self,
        campaign_id: int,
        batch_id: str,
        status: BatchStatus,
        backend_ref: str | None = None,
    ) -> None:
        """Move a batch forward to ``status`` (proposed -> submitted ->
        partial -> complete). A backward move raises InvalidTransition; the
        same status is a no-op that may still fill in ``backend_ref``.
        """
        with self._conn:
            row = self._batch_row(campaign_id, batch_id)
            current = BatchStatus(row["status"])
            if status is current:
                if backend_ref is not None:
                    self._conn.execute(
                        "UPDATE batches SET backend_ref = ? WHERE campaign_id = ? AND id = ?",
                        (backend_ref, campaign_id, batch_id),
                    )
                return
            if (current, status) not in _ALLOWED_TRANSITIONS:
                raise InvalidTransition(
                    f"batch {batch_id!r}: cannot move from {current.value} to {status.value}"
                )
            if current is BatchStatus.PROPOSED:
                items = self._conn.execute(
                    "SELECT design_id FROM batch_items WHERE campaign_id = ? AND batch_id = ?",
                    (campaign_id, batch_id),
                ).fetchall()
                self._check_not_purchased(campaign_id, [r["design_id"] for r in items], batch_id)
            self._conn.execute(
                "UPDATE batches SET status = ?, backend_ref = COALESCE(?, backend_ref) "
                "WHERE campaign_id = ? AND id = ?",
                (status.value, backend_ref, campaign_id, batch_id),
            )
            self._event(
                campaign_id,
                f"batch_{status.value}",
                {"batch_id": batch_id, "backend_ref": backend_ref},
            )

    def discard_batch(self, campaign_id: int, batch_id: str) -> None:
        """Remove a batch that is still ``proposed`` (a rejected proposal);
        anything that left the process raises InvalidTransition."""
        with self._conn:
            row = self._batch_row(campaign_id, batch_id)
            if BatchStatus(row["status"]) is not BatchStatus.PROPOSED:
                raise InvalidTransition(
                    f"batch {batch_id!r}: only proposed batches can be discarded"
                )
            self._conn.execute(
                "DELETE FROM batch_items WHERE campaign_id = ? AND batch_id = ?",
                (campaign_id, batch_id),
            )
            self._conn.execute(
                "DELETE FROM batches WHERE campaign_id = ? AND id = ?", (campaign_id, batch_id)
            )
            self._event(campaign_id, "batch_discarded", {"batch_id": batch_id})

    # --- outcomes and predictions ----------------------------------------------

    def record_outcomes(self, campaign_id: int, batch_id: str, outcomes: Iterable[Outcome]) -> int:
        """Store delivered outcomes. Returns how many were new; conflicts write nothing."""
        outcomes = list(outcomes)
        with self._conn:
            row = self._batch_row(campaign_id, batch_id)
            if not BatchStatus(row["status"]).is_purchased:
                raise InvalidTransition(f"batch {batch_id!r} has not been submitted")
            members = {
                r["design_id"]
                for r in self._conn.execute(
                    "SELECT design_id FROM batch_items WHERE campaign_id = ? AND batch_id = ?",
                    (campaign_id, batch_id),
                )
            }
            to_insert: dict[str, Outcome] = {}
            for outcome in outcomes:
                if outcome.design_id not in members:
                    raise StoreError(
                        f"outcome for design {outcome.design_id!r} is not in batch {batch_id!r}"
                    )
                key = (
                    None if outcome.hit is None else int(outcome.hit),
                    outcome.strength,
                    outcome.kd_molar,
                )
                stored = self._conn.execute(
                    "SELECT hit, strength, kd_molar FROM outcomes "
                    "WHERE campaign_id = ? AND design_id = ?",
                    (campaign_id, outcome.design_id),
                ).fetchone()
                if stored is not None:
                    if (stored["hit"], stored["strength"], stored["kd_molar"]) != key:
                        raise OutcomeConflict(
                            f"design {outcome.design_id!r}: delivered outcome differs "
                            "from the stored one"
                        )
                    continue
                earlier = to_insert.get(outcome.design_id)
                if earlier is not None:
                    if earlier != outcome:
                        raise OutcomeConflict(
                            f"design {outcome.design_id!r}: two different outcomes in one delivery"
                        )
                    continue
                to_insert[outcome.design_id] = outcome
            for outcome in to_insert.values():
                self._conn.execute(
                    "INSERT INTO outcomes (campaign_id, design_id, batch_id, hit, strength, "
                    "kd_molar, raw_json, received_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        campaign_id,
                        outcome.design_id,
                        batch_id,
                        None if outcome.hit is None else int(outcome.hit),
                        outcome.strength,
                        outcome.kd_molar,
                        _dumps(dict(outcome.raw)),
                        self._clock(),
                    ),
                )
            if to_insert:
                self._event(
                    campaign_id, "outcomes_delivered", {"batch_id": batch_id, "n": len(to_insert)}
                )
        return len(to_insert)

    def record_prediction(
        self,
        campaign_id: int,
        round: int,
        predicted_rate: float,
        model: str,
        batch_id: str | None = None,
    ) -> None:
        """Store a predicted hit rate for a round (rate in [0, 1], round
        1-indexed) for the later calibration analysis."""
        if not 0.0 <= predicted_rate <= 1.0:
            raise StoreError("predicted_rate must be between 0 and 1")
        if round < 1:
            raise StoreError("round is 1-indexed")
        with self._conn:
            self._require_campaign(campaign_id)
            self._conn.execute(
                "INSERT INTO predictions (campaign_id, round, batch_id, predicted_rate, model, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (campaign_id, round, batch_id, predicted_rate, model, self._clock()),
            )
            self._event(
                campaign_id,
                "prediction_recorded",
                {"round": round, "batch_id": batch_id, "predicted_rate": predicted_rate},
            )

    def predictions(self, campaign_id: int) -> list[Prediction]:
        """Every recorded prediction of a campaign, in creation order."""
        rows = self._conn.execute(
            "SELECT * FROM predictions WHERE campaign_id = ? ORDER BY id", (campaign_id,)
        ).fetchall()
        return [
            Prediction(r["round"], r["batch_id"], r["predicted_rate"], r["model"], r["created_at"])
            for r in rows
        ]

    # --- events ----------------------------------------------------------------

    def append_event(self, campaign_id: int, kind: str, payload: Mapping[str, Any]) -> int:
        """Append one event to the campaign's append-only log; returns its id.

        The log is what restore replays, so event kinds are part of the
        store's contract (``submission_clock``, ``campaign_stopped``, ...).
        """
        if not kind:
            raise StoreError("event kind must be non-empty")
        with self._conn:
            self._require_campaign(campaign_id)
            return self._event(campaign_id, kind, payload)

    def events(self, campaign_id: int) -> list[Event]:
        """The campaign's events in insertion order (oldest first)."""
        rows = self._conn.execute(
            "SELECT * FROM events WHERE campaign_id = ? ORDER BY id", (campaign_id,)
        ).fetchall()
        return [
            Event(r["id"], r["campaign_id"], r["ts"], r["kind"], json.loads(r["payload_json"]))
            for r in rows
        ]

    # --- snapshot ----------------------------------------------------------------

    def snapshot(self, campaign_id: int) -> CampaignState:
        """Rebuild the immutable :class:`CampaignState` (delivered outcomes only)."""
        config = self.get_config(campaign_id)
        designs = {d.design_id: d for d in self.get_designs(campaign_id)}
        batches = []
        for r in self._conn.execute(
            "SELECT * FROM batches WHERE campaign_id = ? ORDER BY rowid", (campaign_id,)
        ).fetchall():
            ids = tuple(
                i["design_id"]
                for i in self._conn.execute(
                    "SELECT design_id FROM batch_items WHERE campaign_id = ? AND batch_id = ? "
                    "ORDER BY position",
                    (campaign_id, r["id"]),
                )
            )
            batches.append(
                Batch(r["id"], r["round"], ids, BatchStatus(r["status"]), r["backend_ref"])
            )
        outcomes = {
            r["design_id"]: Outcome(
                design_id=r["design_id"],
                hit=None if r["hit"] is None else bool(r["hit"]),
                strength=r["strength"],
                kd_molar=r["kd_molar"],
                raw=json.loads(r["raw_json"]),
            )
            for r in self._conn.execute(
                "SELECT * FROM outcomes WHERE campaign_id = ? ORDER BY rowid", (campaign_id,)
            )
        }
        return CampaignState(config, designs, tuple(batches), outcomes, self.get_clock(campaign_id))
