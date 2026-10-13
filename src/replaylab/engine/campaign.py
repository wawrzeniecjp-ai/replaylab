"""The Campaign: the rules of the game (decisions D1 and D8, plan task T13).

Time model, verbatim:

* Rounds are batch slots; ``state.clock`` counts elapsed rounds.
* ``submit`` creates (or reuses) a batch, hands it to the backend, ingests (so a delay
  of 0 delivers immediately), then ticks: ``backend.advance()``, clock + 1, ingest again.
* With delay d, a batch submitted at clock t is delivered when the clock reaches t + d.
* ``wait`` ticks without submitting and is allowed only while something is pending.
* ``settle`` calls ``wait`` until nothing is pending.
* ``stop`` ends buying. ``wait`` and ``settle`` still work afterwards.

Validation order for ``submit`` (the first failing check raises): stopped, no rounds
left, empty, too large, duplicates, unknown, already purchased, over budget.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from replaylab.backends.base import BackendError, LabBackend, SubmissionHandle
from replaylab.domain.types import (
    Batch,
    BatchStatus,
    CampaignConfig,
    CampaignState,
    Design,
    Outcome,
)
from replaylab.engine.errors import (
    AlreadyPurchased,
    BatchTooLarge,
    BudgetExceeded,
    CampaignStopped,
    DuplicateDesigns,
    EmptyBatch,
    EngineError,
    NoRoundsLeft,
    NothingPending,
    UnknownDesignError,
)
from replaylab.store import CampaignStore

_PENDING = (BatchStatus.SUBMITTED, BatchStatus.PARTIAL)
_BATCH_NUMBER = re.compile(r"^b(\d+)$")
_SETTLE_LIMIT = 10_000


@dataclass(frozen=True)
class SubmitResult:
    """What one ``submit`` did: the batch's new status, the outcomes that
    arrived during the call (in arrival order), and the rounds left."""

    batch_id: str
    status: BatchStatus
    delivered: tuple[Outcome, ...]  # outcomes that arrived during this call, in arrival order
    rounds_left: int


class Campaign:
    """One campaign: the rules of the game over a store and a backend.

    Nothing here is random and nothing here decides *what* to buy -- it only
    enforces the rules (validation order in the module docstring) and records
    every mutation durably, which is what makes kill-and-resume work.
    """

    def __init__(self, store: CampaignStore, backend: LabBackend, campaign_id: int) -> None:
        self._store = store
        self._backend = backend
        self._cid = campaign_id
        self._handles: dict[str, SubmissionHandle] = {}

    # --- construction ------------------------------------------------------------

    @classmethod
    def create(
        cls,
        config: CampaignConfig,
        designs: Iterable[Design],
        store: CampaignStore,
        backend: LabBackend,
    ) -> Campaign:
        """Create a campaign from scratch (no stored campaign yet).

        Raises:
            EngineError: if the backend's delay disagrees with the config.
        """
        delay = getattr(backend, "delay_rounds", None)
        if delay is not None and delay != config.delay_rounds:
            raise EngineError(
                f"backend delay_rounds {delay} differs from "
                f"config delay_rounds {config.delay_rounds}"
            )
        campaign_id = store.create_campaign(config)
        store.add_designs(campaign_id, designs)
        return cls(store, backend, campaign_id)

    @classmethod
    def resume(cls, store: CampaignStore, backend: LabBackend, campaign_id: int) -> Campaign:
        """Re-attach to a stored campaign; pending batches re-register their
        backend handles. The caller must have restored the backend first
        (see :func:`replaylab.engine.replay.restore_backend`).
        """
        campaign = cls(store, backend, campaign_id)
        for batch in store.snapshot(campaign_id).batches:
            if batch.status in _PENDING:
                if not batch.backend_ref:
                    raise EngineError(f"batch {batch.batch_id} is pending but has no backend_ref")
                campaign._handles[batch.batch_id] = SubmissionHandle(
                    batch.backend_ref, batch.batch_id
                )
        return campaign

    # --- views ---------------------------------------------------------------------

    @property
    def campaign_id(self) -> int:
        """The store's id for this campaign."""
        return self._cid

    @property
    def config(self) -> CampaignConfig:
        """The campaign's static configuration, as stored."""
        return self._store.get_config(self._cid)

    @property
    def stopped(self) -> bool:
        """True once a ``campaign_stopped`` event exists (stop is one-way)."""
        return any(e.kind == "campaign_stopped" for e in self._store.events(self._cid))

    def state(self) -> CampaignState:
        """An immutable snapshot of the campaign right now."""
        return self._store.snapshot(self._cid)

    def is_finished(self) -> bool:
        """True when no further batch can be submitted."""
        state = self.state()
        return self.stopped or state.rounds_left == 0 or state.budget_remaining <= 0

    # --- validation ----------------------------------------------------------------

    def _validate(self, state: CampaignState, ids: Sequence[str], *, submitting: bool) -> None:
        if self.stopped:
            raise CampaignStopped("campaign is stopped")
        if submitting and state.rounds_left == 0:
            raise NoRoundsLeft("no rounds left")
        if not ids:
            raise EmptyBatch("a batch needs at least one design")
        if len(ids) > state.config.batch_size:
            raise BatchTooLarge(
                f"batch has {len(ids)} designs but batch_size is {state.config.batch_size}"
            )
        duplicates = sorted({i for i in ids if list(ids).count(i) > 1})
        if duplicates:
            raise DuplicateDesigns(f"duplicate design ids in the batch: {duplicates[:3]}")
        unknown = [i for i in ids if i not in state.designs]
        if unknown:
            raise UnknownDesignError(f"unknown design(s): {unknown[:3]}")
        bought = [i for i in ids if i in state.purchased_ids]
        if bought:
            raise AlreadyPurchased(f"already purchased: {bought[:3]}")
        if submitting and len(ids) > state.budget_remaining:
            raise BudgetExceeded(
                f"batch of {len(ids)} exceeds the remaining budget of {state.budget_remaining}"
            )

    @staticmethod
    def _next_batch_id(state: CampaignState) -> str:
        numbers = [int(m.group(1)) for b in state.batches if (m := _BATCH_NUMBER.match(b.batch_id))]
        return f"b{max(numbers, default=0) + 1:03d}"

    @staticmethod
    def _slot(state: CampaignState) -> int:
        return sum(1 for b in state.batches if b.status.is_purchased) + 1

    # --- acting ------------------------------------------------------------------------

    def propose(
        self,
        design_ids: Sequence[str],
        *,
        predicted_rate: float | None = None,
        model: str = "manual",
    ) -> Batch:
        """Record a proposed batch. Nothing is spent and nothing leaves the process."""
        state = self.state()
        self._validate(state, design_ids, submitting=False)
        batch = Batch(self._next_batch_id(state), self._slot(state), tuple(design_ids))
        self._store.record_batch(self._cid, batch)
        if predicted_rate is not None:
            self._store.record_prediction(
                self._cid, batch.round, predicted_rate, model, batch_id=batch.batch_id
            )
        return batch

    def submit(self, design_ids: Sequence[str]) -> SubmitResult:
        """Buy a batch: validate, hand it to the backend, ingest what is
        ready, then tick the clock once and ingest again (decision D1).

        Validation runs in a fixed order and the first failure raises:
        stopped, no rounds left, empty, too large, duplicates, unknown,
        already purchased, over budget.
        """
        state = self.state()
        self._validate(state, design_ids, submitting=True)
        ids = tuple(design_ids)
        proposal = next(
            (b for b in state.batches if b.status is BatchStatus.PROPOSED and b.design_ids == ids),
            None,
        )
        created_here = proposal is None
        if created_here:
            proposal = Batch(self._next_batch_id(state), self._slot(state), ids)
            self._store.record_batch(self._cid, proposal)
        try:
            handle = self._backend.submit(proposal)
        except BackendError:
            if created_here:
                self._store.discard_batch(self._cid, proposal.batch_id)
            raise
        self._store.set_batch_status(
            self._cid, proposal.batch_id, BatchStatus.SUBMITTED, backend_ref=handle.backend_ref
        )
        self._handles[proposal.batch_id] = handle
        self._store.append_event(
            self._cid, "submission_clock", {"batch_id": proposal.batch_id, "clock": state.clock}
        )
        delivered = self._ingest() + self._tick()
        after = self.state()
        status = next(b.status for b in after.batches if b.batch_id == proposal.batch_id)
        return SubmitResult(proposal.batch_id, status, tuple(delivered), after.rounds_left)

    def ingest(self) -> tuple[Outcome, ...]:
        """Poll every pending batch and record new outcomes. Does not move the clock."""
        return tuple(self._ingest())

    def wait(self) -> tuple[Outcome, ...]:
        """Advance one round without submitting. Only allowed while something is pending."""
        if not self._pending(self.state()):
            raise NothingPending("nothing is pending")
        return tuple(self._tick())

    def settle(self) -> tuple[Outcome, ...]:
        """Wait until nothing is pending. Returns every outcome that arrived."""
        delivered: list[Outcome] = []
        for _ in range(_SETTLE_LIMIT):
            if not self._pending(self.state()):
                return tuple(delivered)
            delivered.extend(self._tick())
        raise EngineError("settle did not converge")

    def stop(self, reason: str) -> None:
        """End buying for good (recorded as an event, with the reason).

        Waiting and settling still work afterwards, so pending deliveries
        can finish arriving.
        """
        if not self.stopped:
            self._store.append_event(self._cid, "campaign_stopped", {"reason": reason})

    # --- internals ----------------------------------------------------------------------

    @staticmethod
    def _pending(state: CampaignState) -> list[Batch]:
        return [b for b in state.batches if b.status in _PENDING]

    def _tick(self) -> list[Outcome]:
        self._backend.advance()
        self._store.set_clock(self._cid, self.state().clock + 1)
        return self._ingest()

    def _ingest(self) -> list[Outcome]:
        state = self.state()
        known = set(state.outcomes)
        fresh_all: list[Outcome] = []
        for batch in self._pending(state):
            result = self._backend.poll(self._handles[batch.batch_id])
            fresh = [o for o in result.delivered if o.design_id not in known]
            if result.delivered:
                self._store.record_outcomes(self._cid, batch.batch_id, result.delivered)
            known.update(o.design_id for o in fresh)
            fresh_all.extend(fresh)
            if result.status is BatchStatus.COMPLETE:
                self._store.set_batch_status(self._cid, batch.batch_id, BatchStatus.COMPLETE)
                del self._handles[batch.batch_id]
            elif result.status is BatchStatus.PARTIAL and batch.status is not BatchStatus.PARTIAL:
                self._store.set_batch_status(self._cid, batch.batch_id, BatchStatus.PARTIAL)
        return fresh_all
