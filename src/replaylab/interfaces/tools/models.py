"""Models: ``fit_and_rank``, ``fit_custom_model``, ``propose_batch``.

The statistics live in ``advisor/`` (``model.fit_model``, ``analysis.analyse``) and the
policies in ``policies/``; this module checks the agent's arguments, calls them, and reports
the result. Models fit on **oriented** features (lower-is-better columns flipped), but every
weight reported here is turned back into the **raw-value** direction the agent sees. Every
fit reads ``campaign.state()``, which holds delivered outcomes only, so no model here can
learn from a design that was not bought and delivered.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from replaylab.advisor.analysis import AnalysisSettings, analyse
from replaylab.advisor.model import FittedModel, InsufficientData, fit_model
from replaylab.advisor.preprocess import Preprocessor
from replaylab.advisor.rates import estimate_hit_rate as rates_estimate
from replaylab.domain.features import FeatureConfig
from replaylab.domain.types import CampaignState, Design
from replaylab.interfaces import schemas as s
from replaylab.interfaces.tools.calibration import calibration_info
from replaylab.interfaces.tools.context import ToolContext, ToolFailure
from replaylab.interfaces.tools.formatting import r6
from replaylab.interfaces.tools.limits import FLOAT_GUARD, MAX_RANK
from replaylab.policies import make_rng
from replaylab.policies.factory import UnknownPolicy, make_policy
from replaylab.policies.purchased import purchased_scale

# --- the tools ---------------------------------------------------------------------------


def fit_and_rank(
    ctx: ToolContext, features: list[str] | None, top_k: int, exploration: float
) -> s.FitAndRankResult:
    """Fit the logistic model, rank the unpurchased designs, and pick the exploration ones."""
    if top_k < 1:
        raise ToolFailure("top_k must be at least 1")
    if not 0.0 <= exploration <= 1.0:
        raise ToolFailure("exploration must be between 0 and 1")
    top_k = min(top_k, MAX_RANK)
    state = ctx.campaign.state()
    preprocessor = preprocessor_for(ctx, state, features)
    fitted = fit_or_refuse(ctx, state, preprocessor)

    available = list(state.unpurchased_ids)
    top_k = min(top_k, len(available))
    scores = fitted.score([state.designs[d] for d in available])
    log_odds = fitted.log_odds([state.designs[d] for d in available])
    order = sorted(range(len(available)), key=lambda i: (-scores[i], available[i]))
    ranked = [
        s.Ranked(design_id=available[i], score=r6(scores[i]), log_odds=r6(log_odds[i]))
        for i in order[:top_k]
    ]
    # Same split as the scripted learners: the best share by score, the rest drawn at
    # random from the remaining designs (see policies.core._top_and_explore, which differs
    # only in not capping the draw at the designs that remain).
    n_top = min(top_k, int(math.floor((1 - exploration) * top_k + FLOAT_GUARD)))
    taken = {r.design_id for r in ranked[:n_top]}
    remaining = [d for d in available if d not in taken]
    n_explore = min(top_k - n_top, len(remaining))
    rng = make_rng(ctx.seed, state.next_round)
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
            s.Weight(feature=f, weight=r6(w * raw_direction(ctx, f)))
            for f, w in fitted.top_weights(5)
        ],
        ranked=ranked,
        exploration_ids=[remaining[i] for i in picks],
    )


def fit_custom_model(
    ctx: ToolContext,
    features: list[str] | None,
    scaling: str,
    penalty: str,
    C: float,  # noqa: N803 - the name every reference uses for this parameter
    similarity_feature: bool,
    rank_by: str,
    kappa: float,
    top_k: int,
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
    state = ctx.campaign.state()
    preprocessor = preprocessor_for(ctx, state, features)
    fitted = fit_or_refuse(ctx, state, preprocessor)  # the same data gate as fit_and_rank
    # The scaler is the only thing "purchased" changes (as in the scripted variants).
    if scaling == "purchased":
        preprocessor = purchased_scale(preprocessor, state)
    delivered = sorted(d for d, o in state.outcomes.items() if o.hit is not None)
    available = sorted(state.unpurchased_ids)
    y = np.array([int(bool(state.outcomes[d].hit)) for d in delivered])
    similarity = {}
    if similarity_feature:
        similarity = {
            "s_train": ctx.similarity.matrix(delivered, delivered),
            "s_cand": ctx.similarity.matrix(available, delivered),
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
            loo_log_loss=r6(result.loo_log_loss),
            baseline_log_loss=r6(result.baseline_log_loss),
            improvement=r6(result.baseline_log_loss - result.loo_log_loss),
            improvement_se=r6(result.improvement_se),
            loo_auc=r6(result.loo_auc),
            loo_auc_se=r6(result.loo_auc_se),
        ),
        weights=[
            s.CustomWeight(feature=f, weight=r6(w * raw_direction(ctx, f)), sd=r6(sd))
            for f, w, sd in weights
        ],
        ranked=[
            s.CustomRanked(
                design_id=available[i],
                score=r6(result.mean[i]),
                upper=r6(result.upper[i]) if result.upper is not None else None,
            )
            for i in result.order[: min(top_k, MAX_RANK)]
        ],
    )


def propose_batch(
    ctx: ToolContext, k: int, policy: str, exploration: float, feature: str | None
) -> s.ProposeBatchResult:
    """Ask a scripted policy for a batch and record the proposal with reasons; spends nothing.

    Policy ``fixed`` ranks by ``feature``; ``fixed:<feature>`` in one string also works here
    (``run-scripted`` uses it), but the MCP tool offers only the separate form.
    """
    fc = ctx.feature_config
    if policy == "fixed":
        valid = ", ".join(fc.names)
        if not feature:
            raise ToolFailure(f"policy fixed needs feature, one of: {valid}")
        if feature not in fc.names:
            raise ToolFailure(f"unknown feature {feature!r} for policy fixed; one of: {valid}")
        policy = f"fixed:{feature}"
    elif feature is not None:
        raise ToolFailure("feature is used only with policy fixed")

    # Refuse a request that cannot produce a batch, before any policy runs.
    state = ctx.campaign.state()
    if not 1 <= k <= state.config.batch_size:
        raise ToolFailure(f"k must be between 1 and the batch size {state.config.batch_size}")
    if not 0.0 <= exploration <= 1.0:
        raise ToolFailure("exploration must be between 0 and 1")
    if ctx.campaign.stopped:
        raise ToolFailure("The campaign is stopped; there is nothing to propose.")
    if state.rounds_left == 0:
        raise ToolFailure("No rounds left; there is nothing to propose.")

    # Let the policy choose.
    try:
        scripted = make_policy(policy, fc, ctx.preprocessor, ctx.similarity, eps=exploration)
    except UnknownPolicy as error:
        raise ToolFailure(str(error)) from None
    # The same generator a scripted run uses for this round, so asking a policy here
    # gives the batch that policy would have bought on this seed.
    ids = scripted.choose(state, k, make_rng(ctx.seed, state.next_round))
    if not ids:
        raise ToolFailure("No unpurchased designs are left to propose.")

    # Predict the batch's hit rate: the model's mean score once it can be fitted,
    # the Beta posterior of the hit rate before that.
    fitted = fit_model(state, ctx.preprocessor)
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
    batch = ctx.campaign.propose(ids, predicted_rate=predicted, model=f"{scripted.name}:{basis}")

    # One reason per design: its score, what drives it, and its nearest bought hit.
    nearest = ctx.similarity.max_to_hits(state)
    reasons = []
    for index, design in enumerate(chosen):
        top = [] if isinstance(fitted, InsufficientData) else top_features(ctx, design, fitted)
        similarity, hit_id = nearest[design.design_id]
        reasons.append(
            s.Reason(
                design_id=design.design_id,
                score=None if scores is None else r6(scores[index]),
                top_features=top,
                nearest_hit=hit_id if similarity > 0 else None,
                similarity=r6(similarity),
            )
        )
    return s.ProposeBatchResult(
        batch_id=batch.batch_id,
        design_ids=list(ids),
        predicted_hit_rate=r6(predicted),
        calibration=calibration_info(ctx, state, basis),
        reasons=reasons,
    )


# --- helpers -----------------------------------------------------------------------------


def fit_or_refuse(
    ctx: ToolContext, state: CampaignState, preprocessor: Preprocessor | None = None
) -> FittedModel:
    """Fit the logistic model, or tell the agent how far it is from the fitting gate."""
    fitted = fit_model(state, preprocessor or ctx.preprocessor)
    if isinstance(fitted, InsufficientData):
        raise ToolFailure(
            "insufficient_data: need at least 2 hits and 8 delivered purchases; "
            f"have {fitted.n_hits} hits and {fitted.n_purchases} purchases. Buy more, or use query_pool."
        )
    return fitted


def raw_direction(ctx: ToolContext, feature: str) -> float:
    """+1 or -1: the sign that turns an oriented weight into the raw-value direction.

    The model fits on oriented features (lower-is-better columns are flipped so that
    higher always means more likely to bind), but every value the tools show is raw.
    Reporting an oriented weight beside raw values invites reading a lower-is-better
    column backwards, so the reported weight is multiplied by the column's orientation
    and a positive weight then means "a higher **raw** value goes with binding".
    Magnitudes, and which weights l1 set to zero, are unchanged.
    """
    return float(ctx.preprocessor.orientation.get(feature, 1.0))


def preprocessor_for(
    ctx: ToolContext, state: CampaignState, features: list[str] | None
) -> Preprocessor:
    """The pool preprocessor, or one refitted on a subset of the features."""
    fc = ctx.feature_config
    if features is None:
        return ctx.preprocessor
    if not features:
        raise ToolFailure("features must not be empty; omit it to use every feature")
    unknown = [f for f in features if f not in fc.names]
    if unknown:
        raise ToolFailure(f"unknown feature(s) {unknown}; valid features: {list(fc.names)}")
    subset = FeatureConfig(
        fc.target,
        features[0],
        tuple(spec for spec in fc.specs if spec.name in features),
    )
    return Preprocessor.fit(state.designs, subset)


def top_features(ctx: ToolContext, design: Design, fitted: Any) -> list[s.TopFeature]:
    """The three features that move this design's log-odds the most.

    A feature's contribution is its weight x the design's standardized value; ties
    are broken by name, and zero contributions are left out.
    """
    row = ctx.preprocessor.transform([design])[0]
    contributions = sorted(
        zip(fitted.columns, fitted.coef, row, strict=True),
        key=lambda t: (-abs(t[1] * t[2]), t[0]),
    )
    return [
        s.TopFeature(feature=name, contribution=r6(coef * x))
        for name, coef, x in contributions[:3]
        if coef * x != 0
    ]
