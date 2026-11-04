"""Response and argument models for every tool.

All responses are Pydantic models, so a tool can never return an undocumented field.
The class docstrings below become part of each tool's output schema, which an MCP
client may show to its model (the evaluation loop sends input schemas only), so they
are written for that reader and changed only deliberately.
Floats in responses are rounded to 6 decimals by the toolbox (raw metric values returned
by ``query_pool`` and ``get_history`` are not rounded).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- arguments ---------------------------------------------------------------------


class Condition(_Model):
    """One ``column op value`` filter for ``query_pool``."""

    column: str
    op: Literal[">", ">=", "<", "<=", "==", "!="]
    value: float | str


# --- campaign tools -------------------------------------------------------------------


class ColumnInfo(_Model):
    """A queryable column: name, value type, coverage and which direction is the good one."""

    name: str
    type: Literal["integer", "category", "number"]
    coverage: float
    # True when a higher raw value is the better one, False when a lower one is; None for
    # length and design_class, which have no pre-registered direction.
    higher_is_better: bool | None = None


class DescribeCampaign(_Model):
    """Budget, progress and the columns an agent may query (``describe_campaign``)."""

    target: str
    mode: Literal["replay", "live"]
    n_designs: int
    rounds_left: int
    batch_size: int
    budget: int
    budget_remaining: int
    purchased: int
    pending: int
    hits_found: int
    delay_rounds: int
    stopped: bool
    columns: list[ColumnInfo]


class NumericSummary(_Model):
    """min, median, max and missing share of one numeric column over the whole pool."""

    # True when a higher raw value is the better one, False when a lower one is; None when
    # the column has no pre-registered direction.
    higher_is_better: bool | None = None
    min: float | None
    median: float | None
    max: float | None
    missing_share: float


class CategorySummary(_Model):
    """Counts per class and the missing share of one categorical column."""

    counts: dict[str, int]
    missing_share: float


class ColumnSummary(_Model):
    """One summary per requested column (``column_summary``)."""

    columns: dict[str, NumericSummary | CategorySummary]


class QueryPoolResult(_Model):
    """Pool rows with raw values and no outcomes, plus how many matched (``query_pool``)."""

    rows: list[dict[str, float | int | str | None]]
    total_matching: int


class Result(_Model):
    """One delivered outcome; ``kd_molar`` is returned exactly as stored."""

    design_id: str
    hit: bool | None
    strength: str | None
    kd_molar: float | None


class SubmitBatchResult(_Model):
    """A bought batch, the rounds left, and every outcome delivered during the call."""

    batch_id: str
    status: str
    rounds_left: int
    results: list[Result]


class PendingBatch(_Model):
    """A submitted batch whose results have not all arrived yet."""

    batch_id: str
    design_ids: list[str]
    submitted_round: int
    status: str


class PendingBatches(_Model):
    """The batches still waiting for results (``pending_batches``)."""

    batches: list[PendingBatch]


class WaitResult(_Model):
    """Outcomes that arrived during a wait, and how many batches are still pending."""

    results: list[Result]
    pending_batches: int


class HistoryRow(_Model):
    """One delivered purchase: its visible features and its outcome (``get_history``)."""

    design_id: str
    batch_id: str
    round: int
    length: int
    design_class: str | None
    features: dict[str, float]
    hit: bool | None
    strength: str | None
    kd_molar: float | None


class History(_Model):
    """Every delivered purchase, in batch order (``get_history``)."""

    designs: list[HistoryRow]


class StopResult(_Model):
    """Confirmation that the campaign stopped, and how many batches are still pending."""

    stopped: bool
    pending_batches: int


# --- advisor tools -----------------------------------------------------------------------


class TopFeature(_Model):
    """One model input: a feature and its contribution to a design's score."""

    feature: str
    contribution: float


class Reason(_Model):
    """Why one design is in a proposed batch: score, top features and nearest hit."""

    design_id: str
    score: float | None
    top_features: list[TopFeature]
    nearest_hit: str | None
    similarity: float


# Calibration rule: once at least 3 model-based predictions have been realized, the next
# predicted rate is clip(raw + mean residual) with a 90% interval of +/- 1.645 x sd of the
# residuals (see Toolbox._next_batch_estimate).
class CalibrationInfo(_Model):
    """How the predicted rate was produced and whether it is calibrated (decision D12)."""

    basis: Literal["model", "posterior_mean"]
    n_realized: int
    mean_residual: float | None
    calibrated: bool


class ProposeBatchResult(_Model):
    """A recorded proposal: designs, predicted hit rate and one reason per design."""

    batch_id: str
    design_ids: list[str]
    predicted_hit_rate: float
    calibration: CalibrationInfo
    reasons: list[Reason]


class Weight(_Model):
    """One logistic-model coefficient, in the raw-value direction.

    A positive weight means a higher **raw** value of the column goes with binding, so the
    sign can be read directly against the values ``query_pool`` returns.
    """

    feature: str
    weight: float


class Ranked(_Model):
    """One ranked design: predicted probability and log-odds."""

    design_id: str
    score: float
    log_odds: float


class FitAndRankResult(_Model):
    """The model's weights, the ranked designs and the exploration picks (``fit_and_rank``)."""

    n_train: int
    n_hits: int
    converged: bool
    top_weights: list[Weight]
    ranked: list[Ranked]
    exploration_ids: list[str]


class Validation(_Model):
    """Leave-one-out validation of a custom model against a base-rate baseline."""

    loo_log_loss: float
    baseline_log_loss: float
    improvement: float  # baseline_log_loss - loo_log_loss; above 0: better than the baseline
    improvement_se: float
    loo_auc: float
    loo_auc_se: float


class CustomWeight(_Model):
    """One input's weight, in the raw-value direction, with its posterior sd (l2 only)."""

    feature: str
    weight: float
    sd: float | None


class CustomRanked(_Model):
    """One ranked design: predicted probability, and the optimistic bound if requested."""

    design_id: str
    score: float
    upper: float | None


class FitCustomModelResult(_Model):
    """The agent's model: its settings, validation, every weight and the ranking."""

    penalty: str
    C: float
    scaling: str
    similarity_feature: bool
    rank_by: str
    kappa: float
    n_train: int
    n_hits: int
    converged: bool
    validation: Validation
    weights: list[CustomWeight]
    ranked: list[CustomRanked]


class Neighbor(_Model):
    """One nearest neighbour: the design, its 3-mer similarity and which given design it matched."""

    design_id: str
    similarity: float
    nearest_given_id: str


class SimilarDesignsResult(_Model):
    """The nearest unpurchased designs to the given ones (``similar_designs``)."""

    neighbors: list[Neighbor]


class GroupRate(_Model):
    """Hit-rate estimate for one design class, with its 90% interval and a low-n flag."""

    n: int
    hits: int
    mean: float
    interval90: list[float]
    low_n: bool


class NextBatchEstimate(_Model):
    """Predicted hit rate of a proposed next batch, with its 90% interval."""

    predicted_rate: float
    interval90: list[float]
    calibrated: bool


class HitRateResult(_Model):
    """Posterior hit rate, optionally per class and for a proposed next batch."""

    n: int
    hits: int
    mean: float
    interval90: list[float]
    p_above_threshold: float | None = None
    groups: dict[str, GroupRate] | None = None
    next_batch: NextBatchEstimate | None = None


class RuleVerdict(_Model):
    """One stop rule: its name, whether it fired, and the numbers behind it."""

    name: str
    fired: bool
    detail: dict[str, Any]


class StopAdviceResult(_Model):
    """The hit-rate trend and the stop-rule verdicts (``stop_advice``)."""

    purchases: int
    hits: int
    cumulative_rate: float | None
    window_rate: float | None
    best_window_rate: float | None
    rules: list[RuleVerdict]


class ClassRate(_Model):
    """Delivered purchases, hits and rate for one design class."""

    n: int
    hits: int
    rate: float


class PredictionVsRealized(_Model):
    """One recorded prediction and the realized rate of its batch, once delivered."""

    round: int
    predicted: float
    realized: float | None


class CampaignSummaryResult(_Model):
    """Progress: rounds, hits, best affinity, curve, per-class rates and predictions."""

    rounds: int
    purchased: int
    pending: int
    hits: int
    best_kd: float | None
    cumulative_curve: list[int]
    hit_rate_by_class: dict[str, ClassRate]
    predictions_vs_realized: list[PredictionVsRealized]
