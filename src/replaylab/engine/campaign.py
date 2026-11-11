"""The Campaign: the rules of the game, over a store and a lab backend.

A campaign is a fixed number of rounds. In each round a player may buy one batch of
designs; the lab measures them and the results arrive some rounds later.

How time moves:

* ``state.clock`` counts the rounds that have passed. Only :meth:`Campaign.submit` and
  :meth:`Campaign.wait` move it, one round per call.
* :meth:`Campaign.submit` buys a batch and then moves the clock one round. Results due by
  then are collected on the way, so with no delay a batch's results arrive within the
  same call.
* With a delay of ``d`` rounds, a batch bought at clock ``t`` arrives when the clock
  reaches ``t + d``.
* :meth:`Campaign.wait` moves the clock without buying; it is allowed only while results
  are outstanding. :meth:`Campaign.settle` waits until nothing is outstanding.
* :meth:`Campaign.stop` ends buying for good; waiting and settling still work afterwards.
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
# A safety net: with a finite delay settle needs at most `delay` ticks, so hitting this
# limit means a backend that never completes a batch.
_SETTLE_LIMIT = 10_000


@dataclass(frozen=True)
class SubmitResult:
    """Report what one :meth:`Campaign.submit` did.

    Attributes:
        batch_id (str): the id of the batch that was bought.
        status (BatchStatus): the batch's status after the call.
        delivered (tuple[Outcome, ...]): the outcomes of any batch that arrived during
            the call, in arrival order.
        rounds_left (int): the rounds still available for buying.
    """

    batch_id: str
    status: BatchStatus
    delivered: tuple[Outcome, ...]
    rounds_left: int


class Campaign:
    """Enforce the rules of one campaign and record every change of its state in the store.

    The campaign decides nothing about *what* to buy and uses no randomness: it checks
    each request against the rules, passes valid batches to the backend, and writes every
    change (batches, outcomes, the clock, events) to the store as it happens. Because the
    store is always up to date, a killed process can resume from it.

    Build one with :meth:`create` (a new campaign) or :meth:`resume` (a stored one).
    """

    def __init__(self, store: CampaignStore, backend: LabBackend, campaign_id: int) -> None:
        self._store = store
        self._backend = backend
        self._cid = campaign_id
        # The backend's receipt (SubmissionHandle) for each batch that is still waiting
        # for results, keyed by batch id; polling the backend needs it. It lives in
        # memory only: resume() rebuilds it from the backend_ref stored with each batch.
        self._handles: dict[str, SubmissionHandle] = {}

    # --- acting ------------------------------------------------------------------------

    def propose(
        self,
        design_ids: Sequence[str],
        *,
        predicted_rate: float | None = None,
        model: str = "manual",
    ) -> Batch:
        """Record a proposed batch without buying it. Nothing is spent and nothing is
        sent to the backend; submitting the same designs later buys this proposal.

        Args:
            design_ids (Sequence[str]): the designs to propose.
            predicted_rate (float | None): the hit rate predicted for the batch, recorded
                so that later predictions can be calibrated against what arrived.
            model (str): the name of whatever produced ``predicted_rate``.

        Returns:
            Batch: the recorded batch, with status ``proposed``.

        Raises:
            EngineError: the batch breaks a rule (see :meth:`_validate`; the round and
                budget checks do not apply to a proposal).
        """
        state = self.state()
        self._validate(state, design_ids, submitting=False)
        batch = Batch(self._next_batch_id(state), state.next_round, tuple(design_ids))
        self._store.record_batch(self._cid, batch)
        if predicted_rate is not None:
            self._store.record_prediction(
                self._cid, batch.round, predicted_rate, model, batch_id=batch.batch_id
            )
        return batch

    def submit(self, design_ids: Sequence[str]) -> SubmitResult:
        """Buy a batch: validate it, hand it to the backend, collect any results that are
        ready, then move the clock one round and collect again.

        Collecting before the clock moves is what lets a game with no delay deliver the
        batch within this call.

        Args:
            design_ids (Sequence[str]): the designs to buy.

        Returns:
            SubmitResult: the batch id, its status, what arrived and the rounds left.

        Raises:
            EngineError: the batch breaks a rule (see :meth:`_validate`).
            BackendError: the backend refused the batch; nothing is recorded.
        """
        state = self.state()
        self._validate(state, design_ids, submitting=True)
        ids = tuple(design_ids)
        # Buying exactly a batch proposed earlier submits that proposal rather than
        # recording a second batch with the same designs.
        proposal = next(
            (b for b in state.batches if b.status is BatchStatus.PROPOSED and b.design_ids == ids),
            None,
        )
        created_here = proposal is None
        if created_here:
            proposal = Batch(self._next_batch_id(state), state.next_round, ids)
            self._store.record_batch(self._cid, proposal)
        try:
            handle = self._backend.submit(proposal)
        except BackendError:
            # The backend refused the batch, so nothing was bought: remove the record we
            # just made, leaving the store as it was before the call.
            if created_here:
                self._store.discard_batch(self._cid, proposal.batch_id)
            raise
        self._store.set_batch_status(
            self._cid, proposal.batch_id, BatchStatus.SUBMITTED, backend_ref=handle.backend_ref
        )
        self._handles[proposal.batch_id] = handle
        # The clock at submission is what restore_backend replays after a restart.
        self._store.append_event(
            self._cid, "submission_clock", {"batch_id": proposal.batch_id, "clock": state.clock}
        )
        delivered = self._ingest() + self._tick()
        after = self.state()
        status = next(b.status for b in after.batches if b.batch_id == proposal.batch_id)
        return SubmitResult(proposal.batch_id, status, tuple(delivered), after.rounds_left)

    def ingest(self) -> tuple[Outcome, ...]:
        """Ask the backend for results of every outstanding batch and record what is new.
        The clock does not move.

        Returns:
            tuple[Outcome, ...]: the outcomes that arrived since the last collection.
        """
        return tuple(self._ingest())

    def wait(self) -> tuple[Outcome, ...]:
        """Let one round pass without buying, and collect the results that become due.

        Returns:
            tuple[Outcome, ...]: the outcomes that arrived during the round.

        Raises:
            NothingPending: no batch is waiting for results.
        """
        if not self._pending(self.state()):
            raise NothingPending("nothing is pending")
        return tuple(self._tick())

    def settle(self) -> tuple[Outcome, ...]:
        """Let rounds pass until no batch is waiting for results.

        Returns:
            tuple[Outcome, ...]: every outcome that arrived meanwhile.

        Raises:
            EngineError: the backend never completed a batch (a safety limit of
                10,000 rounds).
        """
        delivered: list[Outcome] = []
        for _ in range(_SETTLE_LIMIT):
            if not self._pending(self.state()):
                return tuple(delivered)
            delivered.extend(self._tick())
        raise EngineError("settle did not converge")

    def stop(self, reason: str) -> None:
        """End buying for good, recording ``reason`` as an event. Stopping twice records
        nothing more. Waiting and settling still work afterwards, so outstanding results
        can still arrive.

        Args:
            reason (str): why the campaign stops, kept in the event log.
        """
        if not self.stopped:
            self._store.append_event(self._cid, "campaign_stopped", {"reason": reason})

    # --- views ---------------------------------------------------------------------

    @property
    def campaign_id(self) -> int:
        """The store's id for this campaign."""
        return self._cid

    @property
    def config(self) -> CampaignConfig:
        """The campaign's fixed configuration, as stored."""
        return self._store.get_config(self._cid)

    @property
    def stopped(self) -> bool:
        """True once the campaign has been stopped."""
        return self.state().stopped

    def state(self) -> CampaignState:
        """Return an immutable snapshot of the campaign as it is now."""
        return self._store.snapshot(self._cid)

    def is_finished(self) -> bool:
        """Return True when no further batch can be bought."""
        return self.state().finished

    # --- construction ------------------------------------------------------------

    @classmethod
    def create(
        cls,
        config: CampaignConfig,
        designs: Iterable[Design],
        store: CampaignStore,
        backend: LabBackend,
    ) -> Campaign:
        """Start a new campaign: write its configuration and designs to the store.

        Args:
            config (CampaignConfig): rounds, batch size, budget, delay and the rest.
            designs (Iterable[Design]): the pool the player may buy from.
            store (CampaignStore): where the campaign is recorded.
            backend (LabBackend): the lab that measures the batches.

        Returns:
            Campaign: the new campaign.

        Raises:
            EngineError: the backend's delay differs from ``config.delay_rounds``.
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
        """Re-attach to a stored campaign and rebuild the receipts of its outstanding
        batches. A replay backend must be restored first
        (:func:`replaylab.engine.replay.restore_backend`).

        Args:
            store (CampaignStore): the store that holds the campaign.
            backend (LabBackend): the backend, in the state the store describes.
            campaign_id (int): the campaign to resume.

        Returns:
            Campaign: the resumed campaign.

        Raises:
            EngineError: an outstanding batch has no stored backend reference.
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

    # --- validation ----------------------------------------------------------------

    def _validate(self, state: CampaignState, ids: Sequence[str], *, submitting: bool) -> None:
        """Check a batch against the rules, reading only ``state`` (never the store).

        The checks run in this order and the first failure raises: stopped, no rounds
        left, empty, too large, duplicates, unknown, already purchased, over budget.

        Args:
            state (CampaignState): the snapshot to check against.
            ids (Sequence[str]): the batch's design ids.
            submitting (bool): True for a purchase. A proposal (False) skips the two
                checks that only matter when buying: rounds left and budget.

        Raises:
            CampaignStopped: the campaign is stopped.
            NoRoundsLeft: every round is used (purchases only).
            EmptyBatch: ``ids`` is empty.
            BatchTooLarge: ``ids`` is larger than the batch size.
            DuplicateDesigns: a design appears twice in ``ids``.
            UnknownDesignError: a design is not in the pool.
            AlreadyPurchased: a design was bought before.
            BudgetExceeded: the batch costs more than the budget left (purchases only).
        """
        if state.stopped:
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

    # --- internals ----------------------------------------------------------------------

    @staticmethod
    def _next_batch_id(state: CampaignState) -> str:
        """Return the next free batch id: ``b001``, ``b002``..."""
        numbers = [int(m.group(1)) for b in state.batches if (m := _BATCH_NUMBER.match(b.batch_id))]
        return f"b{max(numbers, default=0) + 1:03d}"

    @staticmethod
    def _pending(state: CampaignState) -> list[Batch]:
        """Return the batches still waiting for results."""
        return [b for b in state.batches if b.status in _PENDING]

    def _tick(self) -> list[Outcome]:
        """Move the backend's and the store's clocks one round, then collect what is due."""
        self._backend.advance()
        self._store.set_clock(self._cid, self.state().clock + 1)
        return self._ingest()

    def _ingest(self) -> list[Outcome]:
        """Poll every outstanding batch, record its outcomes and status, and return only
        the outcomes not seen before (a poll repeats earlier outcomes; the store ignores
        the repeats)."""
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
