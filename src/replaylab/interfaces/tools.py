"""The tool layer: every tool as a plain method, independent of any transport.

The MCP server and the CLI both call this class, so what an agent sees through MCP is
exactly what the tests check here. Rules enforced at this boundary:

* sequences, real design ids and unpurchased outcomes never appear in a response;
* responses are deterministic given the campaign state;
* errors an agent can fix raise :class:`ToolFailure` with a message that says how to fix it.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import beta

from replaylab.advisor.analysis import AnalysisSettings, analyse
from replaylab.advisor.model import FittedModel, InsufficientData, fit_model
from replaylab.advisor.preprocess import Preprocessor
from replaylab.advisor.rates import estimate_hit_rate as rates_estimate
from replaylab.advisor.similarity import SimilarityIndex
from replaylab.advisor.stop import stop_advice as advisor_stop_advice
from replaylab.datasets.similarity import kmer_matrix
from replaylab.domain.features import FeatureConfig
from replaylab.domain.types import CampaignState, Design, Outcome
from replaylab.engine import (
    AlreadyPurchased,
    BatchTooLarge,
    BudgetExceeded,
    Campaign,
    CampaignStopped,
    DuplicateDesigns,
    EmptyBatch,
    NoRoundsLeft,
    NothingPending,
    UnknownDesignError,
)
from replaylab.interfaces import schemas as s
from replaylab.interfaces.policy_factory import UnknownPolicy, make_policy
from replaylab.policies import make_rng
from replaylab.policies.purchased import purchased_scale
from replaylab.store import CampaignStore

CAMPAIGN_TOOLS = (
    "describe_campaign",
    "column_summary",
    "query_pool",
    "submit_batch",
    "pending_batches",
    "wait",
    "get_history",
    "stop",
)
ADVISOR_TOOLS = (
    "propose_batch",
    "fit_and_rank",
    "similar_designs",
    "estimate_hit_rate",
    "stop_advice",
    "campaign_summary",
)
# Condition C: the agent's own model, offered in place of fit_and_rank (design-notes
# section 13).
ANALYSIS_TOOLS = ("fit_custom_model",)
# Reserved for a planned knowledge layer; empty, so conditions may already name it.
KNOWLEDGE_TOOLS: tuple[str, ...] = ()
TOOL_GROUPS = {
    "campaign": CAMPAIGN_TOOLS,
    "advisor": ADVISOR_TOOLS,
    "analysis": ANALYSIS_TOOLS,
    "knowledge": KNOWLEDGE_TOOLS,
}

OPERATORS = (">", ">=", "<", "<=", "==", "!=")
DEFAULT_QUERY_LIMIT = 20
MAX_QUERY_LIMIT = 50
MAX_RANK = 50
MAX_SIMILAR = 20
CALIBRATION_MIN_ROUNDS = 3  # realized model predictions needed before calibrating
_FLOAT_GUARD = 1e-9
_Z90 = 1.6448536269514722  # 95th percentile of the standard normal

# What the model reads about each tool. These strings are part of the experiment: they
# are hashed into every run's config_hash, so editing one makes later runs a different
# configuration (and tests/snapshots/tools.json must be refreshed and reviewed).
TOOL_DESCRIPTIONS: dict[str, str] = {
    "describe_campaign": (
        "Show the campaign's state: budget, rounds left, batch size, counts of purchased and pending designs, "
        "hits found, and the columns you can query. Call it first and after every submit. It never shows outcomes "
        "of designs you have not bought."
    ),
    "column_summary": (
        "Summarize each column over the whole pool (min, median, max, share missing and higher_is_better; counts for classes). "
        "higher_is_better says which direction is the good one, since values are raw; use it to pick thresholds "
        "before querying. It says nothing about which designs are hits."
    ),
    "query_pool": (
        "List unpurchased designs with their raw metric values, optionally filtered and sorted (at most 50 rows). "
        "Use it to look at candidates before buying, or pass design_ids to look up specific designs (purchased ones included). "
        "Values are raw and never reoriented, so for some columns a LOWER value is better and sorting "
        "descending returns the worst designs: check the column's higher_is_better, which describe_campaign "
        "and column_summary report."
    ),
    "submit_batch": (
        "Buy up to batch_size designs that are not yet purchased or pending, within budget. "
        "Results come back immediately when there is no delay, otherwise through wait or later submits. "
        "Do not use it to explore: every call spends a round."
    ),
    "pending_batches": (
        "List batches that were submitted but whose results have not arrived; "
        "this happens only in games with a result delay. It does not advance time."
    ),
    "wait": (
        "Advance one round without buying and return any results that arrive. "
        "Use it only when something is pending. It fails if nothing is pending."
    ),
    "get_history": (
        "Return one row per delivered purchase, with its features and outcome. "
        "Use it to study which of your designs hit and why. Pending designs are not listed."
    ),
    "stop": (
        "End the campaign with a reason; no further batches can be submitted. "
        "Use it when more purchases are unlikely to pay off. It cannot be undone, but pending results can still arrive."
    ),
    "propose_batch": (
        "Run a pre-registered scripted policy on your data so far and get the batch it would buy, with reasons. "
        "It spends nothing: to buy the batch, pass its design_ids to submit_batch. "
        "Until a model can be fitted (at least 2 hits and 8 delivered purchases), the model-based policies "
        "rank by a preset feature instead.\n"
        "Policies:\n"
        "- learner: fits a model on your delivered results and ranks the pool by it, plus random exploration.\n"
        "- learner_sim: like learner, and also ranks by sequence similarity to your hits.\n"
        "- learner_purchased: like learner, but standardizes the features on your purchased designs, not the whole pool.\n"
        "- learner_sim_purchased: like learner_sim, but standardizes the features on your purchased designs, not the whole pool.\n"
        "- random: a uniformly random batch of unpurchased designs.\n"
        "- fixed: the top unpurchased designs by one feature, given as feature."
    ),
    "fit_and_rank": (
        "Fit a regularized logistic model on delivered designs and rank unpurchased designs by predicted hit probability. "
        "Needs at least 2 hits and 8 delivered purchases, otherwise it returns insufficient_data. "
        "Use it to see which features matter: each weight is in the raw-value direction, so positive means a "
        "higher raw value of that column goes with binding."
    ),
    "fit_custom_model": (
        "Fit your own logistic regression on the delivered results, with the options below, and rank the unpurchased "
        "designs; each reported weight is in the raw-value direction, so positive means a higher raw value goes with binding. "
        "It needs at least 2 hits and 8 delivered purchases, otherwise it returns insufficient_data; with the default options "
        "it is the same model as fit_and_rank. "
        "validation compares the model's leave-one-out log loss with a base-rate baseline (improvement above 0 means better "
        "than the baseline; compare it with improvement_se) and gives the leave-one-out AUC with its standard error.\n"
        "Options:\n"
        "- features: the predicted-score features to use (default: all); length and design class are always inputs.\n"
        "- scaling: pool standardizes each input with the mean and standard deviation of the whole pool; purchased, "
        "with those of your delivered designs.\n"
        "- penalty and C: l2 shrinks every weight toward zero, l1 can set weights to exactly zero; a smaller C "
        "shrinks more (0.001 to 100).\n"
        "- similarity_feature: adds each design's largest 3-mer sequence similarity to one of your hits as an input "
        "(a design is never compared with itself).\n"
        "- rank_by: mean ranks by predicted hit probability; upper by an optimistic bound, "
        "sigmoid(log-odds + kappa x sd), from the model's uncertainty (l2 only; kappa 0 to 5).\n"
        "- top_k: how many designs to return (at most 50)."
    ),
    "similar_designs": (
        "Find unpurchased designs whose sequences are most similar (3-mer cosine) to the designs you give, typically hits. "
        "Use it to follow up on a hit. Similarity partly reflects shared authorship, so do not over-buy one family."
    ),
    "estimate_hit_rate": (
        "Estimate the hit rate so far with a 90% interval, optionally per design class, or predict the rate of a proposed next batch. "
        "Use it to decide whether more purchases are worthwhile. Next-batch predictions are flagged uncalibrated until 3 rounds of predictions are realized."
    ),
    "stop_advice": (
        "Report the hit-rate trend and whether the low_yield or decay stop rules have fired. "
        "Use it before deciding to stop. The decay rule is noisy and often fires by chance, so treat it as weak evidence."
    ),
    "campaign_summary": (
        "Summarize progress as totals and trends, with no per-design rows: rounds used, hits, best affinity, the cumulative hit curve, "
        "hit rate by design class, and predicted against realized rates. "
        "Use it for a status review; per-design outcomes are in get_history, budget and rounds in describe_campaign."
    ),
}


class ToolFailure(Exception):
    """An error the agent can act on. The message is shown to the agent verbatim."""


def resolve_tool_selection(selection: str | None, registered: Sequence[str]) -> list[str]:
    """Turn ``--tools a,b,campaign`` into tool names (group names expand; order follows ``registered``)."""
    if selection is None or not selection.strip():
        return list(registered)
    wanted: set[str] = set()
    for token in (t.strip() for t in selection.split(",") if t.strip()):
        if token in TOOL_GROUPS:
            wanted.update(TOOL_GROUPS[token])
        elif token in registered:
            wanted.add(token)
        else:
            valid = ", ".join([*TOOL_GROUPS, *registered])
            raise ValueError(f"unknown tool or group {token!r}; valid: {valid}")
    return [name for name in registered if name in wanted]


def _finite_number(text: str) -> float | None:
    """The finite number a text spells, or None ("180" -> 180.0; "nan", "abc" -> None)."""
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _r6(value: float | None) -> float | None:
    # Six decimals keep model-facing numbers short and identical across platforms.
    # Not used for dissociation constants, which are always reported exactly.
    return None if value is None else round(float(value), 6)


def _round_detail(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: _round_detail(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_detail(v) for v in value]
    return value


@dataclass(frozen=True)
class ToolContext:
    """Everything the tools may use. Deliberately has no alias mapping and no outcome source."""

    campaign: Campaign
    store: CampaignStore
    feature_config: FeatureConfig
    preprocessor: Preprocessor
    similarity: SimilarityIndex
    seed: int


class _PoolSimilarity:
    """3-mer vectors of the pool, for similar_designs (built once)."""

    def __init__(self, designs: dict[str, Design]) -> None:
        self._ids = list(designs)
        self._row = {d: i for i, d in enumerate(self._ids)}
        self._matrix = kmer_matrix([designs[d].sequence for d in self._ids])

    def matrix(self, rows: Sequence[str], columns: Sequence[str]) -> np.ndarray:
        """Cosine similarity of every row design to every column design (dense)."""
        if not rows or not columns:
            return np.zeros((len(rows), len(columns)))
        return (
            self._matrix[[self._row[r] for r in rows]]
            @ self._matrix[[self._row[c] for c in columns]].T
        ).toarray()

    def nearest(
        self, given: Sequence[str], candidates: Sequence[str], k: int
    ) -> list[tuple[str, float, str]]:
        """The ``k`` candidates closest to any of ``given``, with the nearest given id."""
        given_sorted = sorted(given)
        if not candidates:
            return []
        sims = (
            self._matrix[[self._row[c] for c in candidates]]
            @ self._matrix[[self._row[g] for g in given_sorted]].T
        ).toarray()
        best = sims.argmax(axis=1)  # first maximum: the smallest given id
        rows = [
            (c, float(sims[i, best[i]]), given_sorted[best[i]]) for i, c in enumerate(candidates)
        ]
        rows.sort(key=lambda r: (-r[1], r[0]))
        return rows[:k]


class Toolbox:
    """The 15 tools over one campaign, independent of any transport.

    The MCP server and the CLI both call these methods, so what an agent sees is
    exactly what the tool tests check here. Every method reads the state fresh.
    """

    def __init__(self, context: ToolContext) -> None:
        self._ctx = context
        self._campaign = context.campaign
        self._store = context.store
        self._fc = context.feature_config
        state = context.campaign.state()
        self._pool_similarity = _PoolSimilarity(dict(state.designs))
        self._columns = ("length", "design_class", *self._fc.names)

    # ------------------------------------------------------------------ helpers

    def _state(self) -> CampaignState:
        return self._campaign.state()

    def _value(self, design: Design, column: str) -> Any:
        if column == "length":
            return len(design.sequence)
        if column == "design_class":
            return design.design_class
        return design.features.get(column)

    def _check_column(self, column: str) -> None:
        if column == "design_id":
            raise ToolFailure(
                "design_id is not a column to filter or sort on; "
                "to look up designs by id, pass design_ids=[...] to query_pool"
            )
        if column not in self._columns:
            raise ToolFailure(f"unknown column {column!r}; valid columns: {list(self._columns)}")

    @staticmethod
    def _result(outcome: Outcome) -> s.Result:
        return s.Result(
            design_id=outcome.design_id,
            hit=outcome.hit,
            strength=outcome.strength,
            kd_molar=outcome.kd_molar,
        )

    def _pending_batches(self, state: CampaignState) -> list:
        return [b for b in state.batches if b.status.value in ("submitted", "partial")]

    # ------------------------------------------------------------------ campaign tools

    def describe_campaign(self) -> s.DescribeCampaign:
        """Budget, progress and queryable columns -- never outcomes of designs not bought."""
        state = self._state()
        n = len(state.designs)
        columns = []
        for name in self._columns:
            present = sum(1 for d in state.designs.values() if self._value(d, name) is not None)
            kind = (
                "integer"
                if name == "length"
                else "category"
                if name == "design_class"
                else "number"
            )
            columns.append(
                s.ColumnInfo(
                    name=name,
                    type=kind,
                    coverage=_r6(present / n if n else 0.0),
                    higher_is_better=self._higher_is_better(name),
                )
            )
        config = state.config
        return s.DescribeCampaign(
            target=config.target,
            mode=config.mode,
            n_designs=n,
            rounds_left=state.rounds_left,
            batch_size=config.batch_size,
            budget=config.budget,
            budget_remaining=state.budget_remaining,
            purchased=state.n_purchased,
            pending=len(state.pending_ids),
            hits_found=state.n_hits,
            delay_rounds=config.delay_rounds,
            stopped=self._campaign.stopped,
            columns=columns,
        )

    def _higher_is_better(self, name: str) -> bool | None:
        """Whether a higher raw value of this column is the better one, or None if undefined.

        ``length`` and ``design_class`` have no pre-registered direction, so they get None
        rather than a guess.
        """
        for spec in self._fc.specs:
            if spec.name == name:
                return spec.higher_is_better
        return None

    def column_summary(self, columns: list[str] | None = None) -> s.ColumnSummary:
        """min/median/max and missing share (or class counts) per column, whole pool."""
        wanted = list(self._columns) if columns is None else list(columns)
        for name in wanted:
            self._check_column(name)
        state = self._state()
        total = len(state.designs)
        summary: dict[str, s.NumericSummary | s.CategorySummary] = {}
        for name in wanted:
            values = [self._value(d, name) for d in state.designs.values()]
            present = [v for v in values if v is not None]
            missing = _r6(1 - len(present) / total) if total else 1.0
            if name == "design_class":
                counts: dict[str, int] = {}
                for v in present:
                    counts[v] = counts.get(v, 0) + 1
                summary[name] = s.CategorySummary(
                    counts=dict(sorted(counts.items())), missing_share=missing
                )
            elif present:
                summary[name] = s.NumericSummary(
                    higher_is_better=self._higher_is_better(name),
                    min=_r6(min(present)),
                    median=_r6(statistics.median(present)),
                    max=_r6(max(present)),
                    missing_share=missing,
                )
            else:
                summary[name] = s.NumericSummary(min=None, median=None, max=None, missing_share=1.0)
        return s.ColumnSummary(columns=summary)

    def query_pool(
        self,
        where: list[s.Condition] | None = None,
        sort_by: str | None = None,
        descending: bool = True,
        limit: int = DEFAULT_QUERY_LIMIT,
        exclude_purchased: bool = True,
        design_ids: list[str] | None = None,
    ) -> s.QueryPoolResult:
        """Raw-valued pool rows, filtered and sorted; purchased designs excluded by default.

        ``design_ids`` restricts the rows to those designs, purchased ones included (a
        lookup of designs another tool named; rows never carry outcomes).
        """
        # Check every argument before touching the pool.
        if limit < 1:
            raise ToolFailure("limit must be at least 1")
        limit = min(limit, MAX_QUERY_LIMIT)
        where = self._checked_conditions(where or [])
        if sort_by is not None:
            self._check_column(sort_by)

        # Which designs are eligible at all: an id lookup, or the unpurchased pool.
        state = self._state()
        wanted: set[str] | None = None
        if design_ids is not None:
            unknown = [d for d in design_ids if d not in state.designs]
            if unknown:
                raise ToolFailure(f"unknown design id(s) {unknown[:3]}; use ids from query_pool")
            wanted = set(design_ids)
        # An explicit id lookup returns those designs even if bought: rows carry no outcome.
        blocked = state.purchased_ids if exclude_purchased and wanted is None else frozenset()

        # Filter.
        matching = []
        for design_id, design in state.designs.items():
            if design_id in blocked or (wanted is not None and design_id not in wanted):
                continue
            if all(self._matches(design, c) for c in where):
                matching.append(design)

        # Sort. Sort by id first so ties in the value sort below come out in a fixed order.
        matching.sort(key=lambda d: d.design_id)
        if sort_by is not None:
            # Values are raw, so descending=True puts the *worst* designs first on a
            # lower-is-better column (proteinmpnn_score). That is documented to the agent
            # through higher_is_better rather than corrected here. Missing values always go
            # last, whichever the direction.
            with_value = [d for d in matching if self._value(d, sort_by) is not None]
            without = [d for d in matching if self._value(d, sort_by) is None]
            with_value.sort(key=lambda d: self._value(d, sort_by), reverse=descending)
            matching = with_value + without  # stable: ties keep ascending design id

        # One row per design, raw values only; total_matching counts beyond the limit.
        rows = []
        for design in matching[:limit]:
            row: dict[str, float | int | str | None] = {"design_id": design.design_id}
            for column in self._columns:
                row[column] = self._value(design, column)
            rows.append(row)
        return s.QueryPoolResult(rows=rows, total_matching=len(matching))

    def _checked_conditions(self, where: list[s.Condition]) -> list[s.Condition]:
        """Validate query_pool's filters; a quoted number on a numeric column becomes one."""
        checked: list[s.Condition] = []
        for condition in where:
            self._check_column(condition.column)
            if condition.column == "design_class":
                if condition.op not in ("==", "!=") or not isinstance(condition.value, str):
                    raise ToolFailure("design_class compares only with == or != and a text value")
            elif isinstance(condition.value, str):
                number = _finite_number(condition.value)
                if number is None:
                    raise ToolFailure(
                        f"column {condition.column!r} is numeric; use a number as the value"
                    )
                condition = condition.model_copy(update={"value": number})
            checked.append(condition)
        return checked

    def _matches(self, design: Design, condition: s.Condition) -> bool:
        value = self._value(design, condition.column)
        if value is None:
            return False
        op, target = condition.op, condition.value
        return {
            ">": lambda: value > target,
            ">=": lambda: value >= target,
            "<": lambda: value < target,
            "<=": lambda: value <= target,
            "==": lambda: value == target,
            "!=": lambda: value != target,
        }[op]()

    def submit_batch(self, design_ids: list[str]) -> s.SubmitBatchResult:
        """Buy a batch; returns every outcome delivered during this call (delay included)."""
        config = self._state().config
        try:
            result = self._campaign.submit(list(design_ids))
        except CampaignStopped as error:
            raise ToolFailure(
                f"{error}. No more batches can be submitted; use get_history or campaign_summary."
            ) from None
        except NoRoundsLeft as error:
            raise ToolFailure(
                f"{error}. Use pending_batches and wait to collect pending results, or stop."
            ) from None
        except EmptyBatch as error:
            raise ToolFailure(
                f"{error}. Pass 1 to {config.batch_size} design ids from query_pool."
            ) from None
        except BatchTooLarge as error:
            raise ToolFailure(
                f"{error}. Submit at most {config.batch_size} designs per batch."
            ) from None
        except DuplicateDesigns as error:
            raise ToolFailure(f"{error}. Remove the repeated ids.") from None
        except UnknownDesignError as error:
            raise ToolFailure(f"{error}. Use ids returned by query_pool.") from None
        except AlreadyPurchased as error:
            raise ToolFailure(
                f"{error}. Those designs are already bought or pending; choose others with query_pool (exclude_purchased is on by default)."
            ) from None
        except BudgetExceeded as error:
            raise ToolFailure(f"{error}. Submit fewer designs.") from None
        return s.SubmitBatchResult(
            batch_id=result.batch_id,
            status=result.status.value,
            rounds_left=result.rounds_left,
            results=[self._result(o) for o in result.delivered],
        )

    def pending_batches(self) -> s.PendingBatches:
        """Batches that were submitted and have not fully delivered yet."""
        state = self._state()
        return s.PendingBatches(
            batches=[
                s.PendingBatch(
                    batch_id=b.batch_id,
                    design_ids=list(b.design_ids),
                    submitted_round=b.round,
                    status=b.status.value,
                )
                for b in self._pending_batches(state)
            ]
        )

    def wait(self) -> s.WaitResult:
        """Advance one round without buying and return the outcomes that arrive."""
        try:
            delivered = self._campaign.wait()
        except NothingPending:
            raise ToolFailure(
                "Nothing is pending, so waiting would do nothing. Submit a batch, or stop if you are finished."
            ) from None
        return s.WaitResult(
            results=[self._result(o) for o in delivered],
            pending_batches=len(self._pending_batches(self._state())),
        )

    def get_history(self) -> s.History:
        """Every delivered purchase with its visible features and its outcome."""
        state = self._state()
        rows = []
        for batch in state.batches:
            if not batch.status.is_purchased:
                continue
            for design_id in batch.design_ids:
                outcome = state.outcomes.get(design_id)
                if outcome is None:
                    continue
                design = state.designs[design_id]
                rows.append(
                    s.HistoryRow(
                        design_id=design_id,
                        batch_id=batch.batch_id,
                        round=batch.round,
                        length=len(design.sequence),
                        design_class=design.design_class,
                        features=dict(design.features),
                        hit=outcome.hit,
                        strength=outcome.strength,
                        kd_molar=outcome.kd_molar,
                    )
                )
        return s.History(designs=rows)

    def stop(self, reason: str) -> s.StopResult:
        """End the campaign with a reason (one-way; pending results can still arrive)."""
        if not reason or not reason.strip():
            raise ToolFailure("Give a short reason for stopping.")
        self._campaign.stop(reason.strip())
        return s.StopResult(stopped=True, pending_batches=len(self._pending_batches(self._state())))

    # ------------------------------------------------------------------ calibration

    def _prediction_pairs(self, state: CampaignState) -> list[tuple[int, float, float | None, str]]:
        """(round, predicted, realized or None, model) for every recorded prediction."""
        batches = {b.batch_id: b for b in state.batches}
        pairs = []
        for p in self._store.predictions(self._campaign.campaign_id):
            batch = batches.get(p.batch_id) if p.batch_id else None
            realized = None
            if (
                batch is not None
                and batch.status.is_purchased
                and all(d in state.outcomes for d in batch.design_ids)
            ):
                realized = sum(state.outcomes[d].is_hit for d in batch.design_ids) / len(
                    batch.design_ids
                )
            pairs.append((p.round, p.predicted_rate, realized, p.model))
        return pairs

    def _residuals(self, state: CampaignState) -> list[float]:
        """realized - predicted for realized, model-based predictions only."""
        return [
            realized - predicted
            for _, predicted, realized, model in self._prediction_pairs(state)
            if realized is not None and model.endswith(":model")
        ]

    def _calibration(self, state: CampaignState, basis: str) -> s.CalibrationInfo:
        # Every model-based proposal records a predicted hit rate; once its batch is
        # delivered, realized - predicted is a residual. With enough residuals, the
        # estimates can be corrected by their mean bias.
        residuals = self._residuals(state)
        return s.CalibrationInfo(
            basis=basis,  # type: ignore[arg-type]
            n_realized=len(residuals),
            mean_residual=_r6(statistics.fmean(residuals)) if residuals else None,
            calibrated=len(residuals) >= CALIBRATION_MIN_ROUNDS,
        )

    # ------------------------------------------------------------------ advisor tools

    def _fitted(
        self, state: CampaignState, preprocessor: Preprocessor | None = None
    ) -> FittedModel:
        fitted = fit_model(state, preprocessor or self._ctx.preprocessor)
        if isinstance(fitted, InsufficientData):
            raise ToolFailure(
                "insufficient_data: need at least 2 hits and 8 delivered purchases; "
                f"have {fitted.n_hits} hits and {fitted.n_purchases} purchases. Buy more, or use query_pool."
            )
        return fitted

    def _raw_direction(self, feature: str) -> float:
        """+1 or -1: the sign that turns an oriented weight into the raw-value direction.

        The model fits on oriented features (lower-is-better columns are flipped so that
        higher always means more likely to bind), but every value the tools show is raw.
        Reporting an oriented weight beside raw values invites reading a lower-is-better
        column backwards, so the reported weight is multiplied by the column's orientation
        and a positive weight then means "a higher **raw** value goes with binding".
        Magnitudes, and which weights l1 set to zero, are unchanged.
        """
        return float(self._ctx.preprocessor.orientation.get(feature, 1.0))

    def _preprocessor_for(self, state: CampaignState, features: list[str] | None) -> Preprocessor:
        """The pool preprocessor, or one refitted on a subset of the features."""
        if features is None:
            return self._ctx.preprocessor
        if not features:
            raise ToolFailure("features must not be empty; omit it to use every feature")
        unknown = [f for f in features if f not in self._fc.names]
        if unknown:
            raise ToolFailure(
                f"unknown feature(s) {unknown}; valid features: {list(self._fc.names)}"
            )
        subset = FeatureConfig(
            self._fc.target,
            features[0],
            tuple(spec for spec in self._fc.specs if spec.name in features),
        )
        return Preprocessor.fit(state.designs, subset)

    def propose_batch(
        self,
        k: int,
        policy: str = "learner_sim",
        exploration: float = 0.2,
        feature: str | None = None,
    ) -> s.ProposeBatchResult:
        """Ask a scripted policy for a batch and record the proposal with reasons; spends nothing.

        Policy ``fixed`` ranks by ``feature``; ``fixed:<feature>`` in one string also works here
        (``run-scripted`` uses it), but the MCP tool offers only the separate form.
        """
        if policy == "fixed":
            valid = ", ".join(self._fc.names)
            if not feature:
                raise ToolFailure(f"policy fixed needs feature, one of: {valid}")
            if feature not in self._fc.names:
                raise ToolFailure(f"unknown feature {feature!r} for policy fixed; one of: {valid}")
            policy = f"fixed:{feature}"
        elif feature is not None:
            raise ToolFailure("feature is used only with policy fixed")

        # Refuse a request that cannot produce a batch, before any policy runs.
        state = self._state()
        if not 1 <= k <= state.config.batch_size:
            raise ToolFailure(f"k must be between 1 and the batch size {state.config.batch_size}")
        if not 0.0 <= exploration <= 1.0:
            raise ToolFailure("exploration must be between 0 and 1")
        if self._campaign.stopped:
            raise ToolFailure("The campaign is stopped; there is nothing to propose.")
        if state.rounds_left == 0:
            raise ToolFailure("No rounds left; there is nothing to propose.")

        # Let the policy choose.
        try:
            scripted = make_policy(
                policy, self._fc, self._ctx.preprocessor, self._ctx.similarity, eps=exploration
            )
        except UnknownPolicy as error:
            raise ToolFailure(str(error)) from None
        # The same generator a scripted run uses for this round, so asking a policy here
        # gives the batch that policy would have bought on this seed.
        ids = scripted.choose(state, k, make_rng(self._ctx.seed, state.next_round))
        if not ids:
            raise ToolFailure("No unpurchased designs are left to propose.")

        # Predict the batch's hit rate: the model's mean score once it can be fitted,
        # the Beta posterior of the hit rate before that.
        fitted = fit_model(state, self._ctx.preprocessor)
        chosen = [state.designs[d] for d in ids]
        if isinstance(fitted, InsufficientData):
            basis = "posterior_mean"
            scores: np.ndarray | None = None
            predicted = rates_estimate(state).overall.mean
        else:
            basis = "model"
            scores = fitted.score(chosen)
            predicted = float(scores.mean())
        # Recording the proposal with its prediction is what later feeds calibration.
        batch = self._campaign.propose(
            ids, predicted_rate=predicted, model=f"{scripted.name}:{basis}"
        )

        # One reason per design: its score, what drives it, and its nearest bought hit.
        nearest = self._ctx.similarity.max_to_hits(state)
        reasons = []
        for index, design in enumerate(chosen):
            top = [] if isinstance(fitted, InsufficientData) else self._top_features(design, fitted)
            similarity, hit_id = nearest[design.design_id]
            reasons.append(
                s.Reason(
                    design_id=design.design_id,
                    score=None if scores is None else _r6(scores[index]),
                    top_features=top,
                    nearest_hit=hit_id if similarity > 0 else None,
                    similarity=_r6(similarity),
                )
            )
        return s.ProposeBatchResult(
            batch_id=batch.batch_id,
            design_ids=list(ids),
            predicted_hit_rate=_r6(predicted),
            calibration=self._calibration(state, basis),
            reasons=reasons,
        )

    def _top_features(self, design: Design, fitted: Any) -> list[s.TopFeature]:
        """The three features that move this design's log-odds the most.

        A feature's contribution is its weight x the design's standardized value; ties
        are broken by name, and zero contributions are left out.
        """
        row = self._ctx.preprocessor.transform([design])[0]
        contributions = sorted(
            zip(fitted.columns, fitted.coef, row, strict=True),
            key=lambda t: (-abs(t[1] * t[2]), t[0]),
        )
        return [
            s.TopFeature(feature=name, contribution=_r6(coef * x))
            for name, coef, x in contributions[:3]
            if coef * x != 0
        ]

    def fit_and_rank(
        self, features: list[str] | None = None, top_k: int = 20, exploration: float = 0.2
    ) -> s.FitAndRankResult:
        """Fit the logistic model, rank the unpurchased designs, and pick the exploration ones."""
        if top_k < 1:
            raise ToolFailure("top_k must be at least 1")
        if not 0.0 <= exploration <= 1.0:
            raise ToolFailure("exploration must be between 0 and 1")
        top_k = min(top_k, MAX_RANK)
        state = self._state()
        preprocessor = self._preprocessor_for(state, features)
        fitted = self._fitted(state, preprocessor)

        available = list(state.unpurchased_ids)
        top_k = min(top_k, len(available))
        scores = fitted.score([state.designs[d] for d in available])
        log_odds = fitted.log_odds([state.designs[d] for d in available])
        order = sorted(range(len(available)), key=lambda i: (-scores[i], available[i]))
        ranked = [
            s.Ranked(design_id=available[i], score=_r6(scores[i]), log_odds=_r6(log_odds[i]))
            for i in order[:top_k]
        ]
        # Same split as the scripted learners: the best share by score, the rest drawn at
        # random from the remaining designs (see policies.core._top_and_explore).
        n_top = min(top_k, int(math.floor((1 - exploration) * top_k + _FLOAT_GUARD)))
        taken = {r.design_id for r in ranked[:n_top]}
        remaining = [d for d in available if d not in taken]
        n_explore = min(top_k - n_top, len(remaining))
        rng = make_rng(self._ctx.seed, state.next_round)
        picks = (
            sorted(rng.choice(len(remaining), size=n_explore, replace=False).tolist())
            if n_explore
            else []
        )
        return s.FitAndRankResult(
            n_train=fitted.n_train,
            n_hits=fitted.n_hits,
            converged=fitted.converged,
            top_weights=[
                s.Weight(feature=f, weight=_r6(w * self._raw_direction(f)))
                for f, w in fitted.top_weights(5)
            ],
            ranked=ranked,
            exploration_ids=[remaining[i] for i in picks],
        )

    def fit_custom_model(
        self,
        features: list[str] | None = None,
        scaling: str = "pool",
        penalty: str = "l2",
        C: float = 0.1,  # noqa: N803 - the name every reference uses for this parameter
        similarity_feature: bool = False,
        rank_by: str = "mean",
        kappa: float = 1.0,
        top_k: int = 20,
    ) -> s.FitCustomModelResult:
        """The agent's own logistic regression: options, leave-one-out validation, ranking."""
        # Check the options first: a bad one costs the agent a message, not a fit.
        settings = AnalysisSettings(
            penalty=penalty,
            C=C,
            similarity=similarity_feature,
            scaling=scaling,
            rank_by=rank_by,
            kappa=kappa,
        )
        try:
            settings.check()
        except ValueError as error:
            raise ToolFailure(str(error)) from None
        if top_k < 1:
            raise ToolFailure("top_k must be at least 1")

        # Inputs: standardized features of the delivered designs (to fit) and of the
        # unpurchased ones (to rank), plus their similarity matrices if asked for.
        state = self._state()
        preprocessor = self._preprocessor_for(state, features)
        fitted = self._fitted(state, preprocessor)  # the same data gate as fit_and_rank
        # The scaler is the only thing "purchased" changes (as in the scripted variants).
        if scaling == "purchased":
            preprocessor = purchased_scale(preprocessor, state)
        delivered = sorted(d for d, o in state.outcomes.items() if o.hit is not None)
        available = sorted(state.unpurchased_ids)
        y = np.array([int(bool(state.outcomes[d].hit)) for d in delivered])
        similarity = {}
        if similarity_feature:
            similarity = {
                "s_train": self._pool_similarity.matrix(delivered, delivered),
                "s_cand": self._pool_similarity.matrix(available, delivered),
            }

        # Fit, validate by leave-one-out, and rank (advisor.analysis does the work).
        try:
            result = analyse(
                preprocessor.transform(state.designs[d] for d in delivered),
                y,
                preprocessor.transform(state.designs[d] for d in available),
                preprocessor.columns,
                settings,
                **similarity,
            )
        except ValueError as error:
            raise ToolFailure(str(error)) from None

        # Report the weights largest first, in the raw-value direction the agent sees.
        sds = result.coef_sd or (None,) * len(result.coef)
        weights = sorted(
            zip(result.columns, result.coef, sds, strict=True), key=lambda w: (-abs(w[1]), w[0])
        )
        return s.FitCustomModelResult(
            penalty=penalty,
            C=C,
            scaling=scaling,
            similarity_feature=similarity_feature,
            rank_by=rank_by,
            kappa=kappa,
            n_train=fitted.n_train,
            n_hits=fitted.n_hits,
            converged=result.converged,
            validation=s.Validation(
                loo_log_loss=_r6(result.loo_log_loss),
                baseline_log_loss=_r6(result.baseline_log_loss),
                improvement=_r6(result.baseline_log_loss - result.loo_log_loss),
                improvement_se=_r6(result.improvement_se),
                loo_auc=_r6(result.loo_auc),
                loo_auc_se=_r6(result.loo_auc_se),
            ),
            weights=[
                s.CustomWeight(feature=f, weight=_r6(w * self._raw_direction(f)), sd=_r6(sd))
                for f, w, sd in weights
            ],
            ranked=[
                s.CustomRanked(
                    design_id=available[i],
                    score=_r6(result.mean[i]),
                    upper=_r6(result.upper[i]) if result.upper is not None else None,
                )
                for i in result.order[: min(top_k, MAX_RANK)]
            ],
        )

    def similar_designs(self, design_ids: list[str], k: int = 10) -> s.SimilarDesignsResult:
        """The nearest unpurchased designs by 3-mer similarity to the given ones."""
        if k < 1:
            raise ToolFailure("k must be at least 1")
        if not design_ids:
            raise ToolFailure("Give at least one design id (typically hits from get_history).")
        state = self._state()
        unknown = [d for d in design_ids if d not in state.designs]
        if unknown:
            raise ToolFailure(
                f"unknown design id(s) {unknown[:3]}; use ids from query_pool or get_history"
            )
        # Only unpurchased designs are returned: the point is to find designs to buy.
        given = set(design_ids)
        candidates = [d for d in state.unpurchased_ids if d not in given]
        rows = self._pool_similarity.nearest(list(given), candidates, min(k, MAX_SIMILAR))
        return s.SimilarDesignsResult(
            neighbors=[
                s.Neighbor(design_id=d, similarity=_r6(sim), nearest_given_id=g)
                for d, sim, g in rows
            ]
        )

    def estimate_hit_rate(
        self,
        by: str | None = None,
        threshold: float | None = None,
        next_batch: list[str] | None = None,
    ) -> s.HitRateResult:
        """Posterior hit rate, optionally per class, plus a prediction for a proposed batch."""
        if by not in (None, "design_class"):
            raise ToolFailure("by must be omitted or 'design_class'")
        if threshold is not None and not 0.0 < threshold < 1.0:
            raise ToolFailure("threshold must be strictly between 0 and 1")
        state = self._state()
        report = rates_estimate(state, by=by, threshold=threshold)
        groups = None
        if by == "design_class":
            groups = {
                name: s.GroupRate(
                    n=g.estimate.n,
                    hits=g.estimate.hits,
                    mean=_r6(g.estimate.mean),
                    interval90=[_r6(g.estimate.lower), _r6(g.estimate.upper)],
                    low_n=g.low_n,
                )
                for name, g in report.groups.items()
            }
        nb = None
        if next_batch is not None:
            nb = self._next_batch_estimate(state, next_batch)
        overall = report.overall
        return s.HitRateResult(
            n=overall.n,
            hits=overall.hits,
            mean=_r6(overall.mean),
            interval90=[_r6(overall.lower), _r6(overall.upper)],
            p_above_threshold=_r6(report.p_above_threshold),
            groups=groups,
            next_batch=nb,
        )

    def _next_batch_estimate(self, state: CampaignState, ids: list[str]) -> s.NextBatchEstimate:
        if not ids:
            raise ToolFailure("next_batch must list at least one design id")
        unknown = [d for d in ids if d not in state.designs]
        if unknown:
            raise ToolFailure(f"unknown design id(s) {unknown[:3]}; use ids from query_pool")
        bought = [d for d in ids if d in state.purchased_ids]
        if bought:
            raise ToolFailure(
                f"already purchased: {bought[:3]}; next_batch must be unpurchased designs"
            )
        fitted = self._fitted(state)
        predicted = float(fitted.score([state.designs[d] for d in ids]).mean())
        residuals = self._residuals(state)
        # Calibrated: shift the model's prediction by its mean past error, and take the
        # 90% interval from the spread of those errors.
        if len(residuals) >= CALIBRATION_MIN_ROUNDS:
            adjusted = min(1.0, max(0.0, predicted + statistics.fmean(residuals)))
            half = _Z90 * statistics.stdev(residuals)
            interval = [max(0.0, adjusted - half), min(1.0, adjusted + half)]
            return s.NextBatchEstimate(
                predicted_rate=_r6(adjusted), interval90=[_r6(i) for i in interval], calibrated=True
            )
        # Not yet calibrated: treat the prediction as if k purchases had shown that rate,
        # and report the 90% interval of the matching Beta distribution.
        k = len(ids)
        a, b = 1 + predicted * k, 1 + (1 - predicted) * k
        return s.NextBatchEstimate(
            predicted_rate=_r6(predicted),
            interval90=[_r6(beta.ppf(0.05, a, b)), _r6(beta.ppf(0.95, a, b))],
            calibrated=False,
        )

    def stop_advice(self) -> s.StopAdviceResult:
        """Whether the low_yield and decay rules have fired, with the numbers behind them."""
        advice = advisor_stop_advice(self._state())
        return s.StopAdviceResult(
            purchases=advice.purchases,
            hits=advice.hits,
            cumulative_rate=_r6(advice.cumulative_rate),
            window_rate=_r6(advice.window_rate),
            best_window_rate=_r6(advice.best_window_rate),
            rules=[
                s.RuleVerdict(name=v.name, fired=v.fired, detail=_round_detail(v.detail))
                for v in advice.rules
            ],
        )

    def campaign_summary(self) -> s.CampaignSummaryResult:
        """Rounds, hits, best affinity, cumulative curve, per-class rates and predictions."""
        state = self._state()
        curve: list[int] = []
        cumulative = 0
        kds = []
        by_class: dict[str, list[int]] = {}
        for batch in state.batches:
            if not batch.status.is_purchased:
                continue
            for design_id in batch.design_ids:
                outcome = state.outcomes.get(design_id)
                if outcome is None:
                    continue
                cumulative += int(outcome.is_hit)
                curve.append(cumulative)
                if outcome.is_hit and outcome.kd_molar is not None:
                    kds.append(outcome.kd_molar)
                name = state.designs[design_id].design_class or "(none)"
                entry = by_class.setdefault(name, [0, 0])
                entry[0] += 1
                entry[1] += int(outcome.is_hit)
        return s.CampaignSummaryResult(
            rounds=sum(1 for b in state.batches if b.status.is_purchased),
            purchased=state.n_purchased,
            pending=len(state.pending_ids),
            hits=state.n_hits,
            best_kd=min(kds) if kds else None,
            cumulative_curve=curve,
            hit_rate_by_class={
                name: s.ClassRate(n=n, hits=h, rate=_r6(h / n))
                for name, (n, h) in sorted(by_class.items())
            },
            predictions_vs_realized=[
                s.PredictionVsRealized(round=r, predicted=_r6(p), realized=_r6(real))
                for r, p, real, _ in self._prediction_pairs(state)
            ],
        )


ToolMethod = Callable[..., Any]
