"""The configurable analysis behind ``fit_custom_model``: the agent chooses the model.

A logistic regression on the delivered designs, with the options an agent may set:

* ``penalty``: ``l2`` (ridge) or ``l1`` (lasso, which can set weights to exactly zero);
* ``C``: the inverse regularization strength (smaller shrinks the weights more);
* ``similarity``: add each design's largest 3-mer similarity to a hit as an input;
* ``scaling``: how that similarity column is standardized, like the other inputs
  (``pool``: over the delivered and candidate designs; ``purchased``: over the
  delivered ones). The caller standardizes the other inputs the same way;
* ``rank_by``: ``mean`` (predicted hit probability) or ``upper``, an optimistic bound
  ``sigmoid(log-odds + kappa * sd)`` from the Laplace approximation of the L2 model,
  whose Gaussian prior has variance ``C`` on each weight (scikit-learn's objective).

Validation is leave-one-out over the delivered designs, against a base-rate baseline
that is also left-one-out. The similarity input is recomputed inside every fold from
that fold's hits, and a row is never compared with itself, so a hit's own label never
reaches its own input. One small leak remains: the similarity column's centre and spread
come from the full data, so the held-out design's label shifts the scale used in its own
fold (with the L2 penalty, scale matters). Measured on random data, it moves a fold's
prediction by about 0.0003 typically and 0.012 at most. It affects the validation numbers
only, never the ranking of the candidates.

Defaults (``l2``, ``C = 0.1``, no similarity, ranked by mean) give the same model as
``fit_and_rank``. Pure computation: no store, no ids, no sequences.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

PENALTIES = ("l2", "l1")
RANKINGS = ("mean", "upper")
SCALINGS = ("pool", "purchased")
C_RANGE = (0.001, 100.0)
KAPPA_RANGE = (0.0, 5.0)
SIMILARITY_COLUMN = "similarity_to_hits"
_CLIP = 1e-6  # probabilities are clipped to [_CLIP, 1 - _CLIP] for the log loss


@dataclass(frozen=True)
class AnalysisSettings:
    """The model options an agent may choose in ``fit_custom_model`` (ranges in :meth:`check`)."""

    penalty: str = "l2"
    C: float = 0.1
    similarity: bool = False
    scaling: str = "pool"
    rank_by: str = "mean"
    kappa: float = 1.0

    def check(self) -> None:
        """Raise ``ValueError`` naming the first setting outside the offered range."""
        if self.penalty not in PENALTIES:
            raise ValueError(f"penalty must be one of {list(PENALTIES)}")
        if not C_RANGE[0] <= self.C <= C_RANGE[1]:
            raise ValueError(f"C must be between {C_RANGE[0]} and {C_RANGE[1]}")
        if self.scaling not in SCALINGS:
            raise ValueError(f"scaling must be one of {list(SCALINGS)}")
        if self.rank_by not in RANKINGS:
            raise ValueError(f"rank_by must be one of {list(RANKINGS)}")
        if not KAPPA_RANGE[0] <= self.kappa <= KAPPA_RANGE[1]:
            raise ValueError(f"kappa must be between {KAPPA_RANGE[0]} and {KAPPA_RANGE[1]}")
        if self.rank_by == "upper" and self.penalty != "l2":
            raise ValueError("rank_by upper needs the l2 model (its uncertainty is Gaussian)")


@dataclass(frozen=True)
class Analysis:
    """One fitted analysis: model, left-one-out quality against the baseline, rankings.

    ``loo_*`` numbers describe the fit on delivered designs only; ``mean``/``upper``
    and ``order`` are what the candidate pool receives.
    """

    columns: tuple[str, ...]
    coef: tuple[float, ...]
    coef_sd: tuple[float, ...] | None  # posterior sd per weight (l2 only)
    intercept: float
    converged: bool
    loo_log_loss: float
    baseline_log_loss: float
    improvement_se: float  # standard error of the per-design (model - baseline) loss
    loo_auc: float
    loo_auc_se: float
    mean: np.ndarray  # predicted hit probability per candidate
    upper: np.ndarray | None  # optimistic bound per candidate (rank_by upper only)
    order: tuple[int, ...]  # candidate indices, best first; ties by index


def similarity_to_hits(s: np.ndarray, hits: np.ndarray, *, own: bool) -> np.ndarray:
    """Row-wise largest similarity to a hit column; 0 without hits.

    ``own=True`` means row i and column i are the same design, so that entry is skipped.
    """
    if not hits.any():
        return np.zeros(s.shape[0])
    masked = np.where(hits[None, :], np.asarray(s, dtype=float), -np.inf)
    if own:
        np.fill_diagonal(masked, -np.inf)
    best = masked.max(axis=1)
    best[~np.isfinite(best)] = 0.0
    return best


def _estimator(settings: AnalysisSettings) -> LogisticRegression:
    if settings.penalty == "l2":  # the same estimator as advisor.model (ModelConfig)
        return LogisticRegression(C=settings.C, solver="lbfgs", max_iter=1000)
    # l1_ratio=1 makes the elastic-net penalty pure lasso; saga is the solver that
    # supports it, and converges slowly, hence the large max_iter.
    return LogisticRegression(
        C=settings.C, l1_ratio=1.0, solver="saga", max_iter=20000, random_state=0
    )


def _fit(x: np.ndarray, y: np.ndarray, settings: AnalysisSettings):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = _estimator(settings).fit(x, y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model, converged


def _auc(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Mann-Whitney AUC and its Hanley-McNeil standard error."""
    # AUC = probability that a random hit is ranked above a random non-hit, computed
    # from the rank sum of the hits (ties get average ranks).
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    ranks = rankdata(p)
    auc = float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n0 - 1) * (q2 - auc**2)) / (n1 * n0)
    return auc, math.sqrt(max(var, 0.0))


def _log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _CLIP, 1 - _CLIP)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _leave_one_out(
    full_train: np.ndarray,
    y: np.ndarray,
    settings: AnalysisSettings,
    *,
    x_train: np.ndarray | None = None,
    s_train: np.ndarray | None = None,
    with_sim: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Leave-one-out predictions: for each design i, refit without it and predict it.

    Without the similarity input the folds are plain row subsets of ``full_train``. With
    it (``with_sim`` given, plus the raw ``x_train`` and ``s_train``), the input is
    rebuilt from each fold's own hits, so design i's label never reaches its own input.
    """
    n = len(y)
    hits = y == 1
    loo = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        if with_sim is not None:
            fold_hits = hits[keep]
            train_i = with_sim(
                x_train[keep], similarity_to_hits(s_train[np.ix_(keep, keep)], fold_hits, own=True)
            )
            held_i = with_sim(
                x_train[i : i + 1],
                similarity_to_hits(s_train[i : i + 1][:, keep], fold_hits, own=False),
            )
        else:
            train_i, held_i = full_train[keep], full_train[i : i + 1]

        if len(set(y[keep])) < 2:  # one class left: the fold can only predict its rate
            loo[i] = y[keep].mean()
        else:
            loo[i] = _fit(train_i, y[keep], settings)[0].predict_proba(held_i)[0, 1]
    return loo


def _laplace_covariance(model: LogisticRegression, x: np.ndarray, C: float) -> np.ndarray:
    """Approximate covariance of (intercept, weights) of an L2-penalized logistic fit.

    Laplace approximation: near its optimum the posterior is roughly Gaussian, with
    precision = X' W X (W = p(1 - p), the logistic curvature) plus 1/C on each weight
    from the prior. The intercept (first row and column) is not penalized. A tiny ridge
    keeps the inverse defined when the inputs are collinear.
    """
    design = np.c_[np.ones(len(x)), x]
    p = model.predict_proba(x)[:, 1]
    precision = design.T @ (design * (p * (1 - p))[:, None])
    precision += np.diag([0.0] + [1.0 / C] * x.shape[1])
    return np.linalg.inv(precision + 1e-9 * np.eye(len(precision)))


def analyse(
    x_train: np.ndarray,
    y: np.ndarray,
    x_cand: np.ndarray,
    columns: tuple[str, ...],
    settings: AnalysisSettings,
    *,
    s_train: np.ndarray | None = None,
    s_cand: np.ndarray | None = None,
) -> Analysis:
    """Fit, validate and rank.

    Args:
        x_train: standardized inputs of the delivered designs (n x p).
        y: their outcomes, 1 for a hit.
        x_cand: standardized inputs of the designs to rank (m x p).
        columns: the names of the p inputs.
        settings: the agent's options (checked here).
        s_train: 3-mer similarity among the delivered designs (n x n), needed with
            ``settings.similarity``.
        s_cand: similarity of each candidate to each delivered design (m x n).
    """
    settings.check()
    y = np.asarray(y, dtype=int)
    n = len(y)
    hits = y == 1
    use_sim = settings.similarity and s_train is not None and s_cand is not None
    if settings.similarity and not use_sim:
        raise ValueError("similarity needs s_train and s_cand")

    # 1. Optional similarity input: each design's largest similarity to a delivered hit,
    #    standardized like the other inputs.
    if use_sim:
        sim_train = similarity_to_hits(s_train, hits, own=True)
        sim_cand = similarity_to_hits(s_cand, hits, own=False)
        basis = sim_train if settings.scaling == "purchased" else np.r_[sim_train, sim_cand]
        centre, spread = float(basis.mean()), float(basis.std()) or 1.0

        def with_sim(x: np.ndarray, sim: np.ndarray) -> np.ndarray:
            return np.c_[x, (sim - centre) / spread]

        full_train, full_cand = with_sim(x_train, sim_train), with_sim(x_cand, sim_cand)
        columns = (*columns, SIMILARITY_COLUMN)
    else:
        full_train, full_cand = x_train, x_cand

    # 2. The model the candidates are ranked with, fitted on every delivered design.
    model, converged = _fit(full_train, y, settings)

    # 3. Leave-one-out validation, against a baseline that knows only the hit rate.
    if use_sim:
        loo = _leave_one_out(
            full_train, y, settings, x_train=x_train, s_train=s_train, with_sim=with_sim
        )
    else:
        loo = _leave_one_out(full_train, y, settings)
    # The baseline predicts, for each design, the hit rate of the other n - 1 designs:
    # the left-one-out version of "always guess the base rate".
    baseline = (y.sum() - y) / (n - 1)
    model_loss, base_loss = _log_loss(y, loo), _log_loss(y, baseline)
    difference = model_loss - base_loss
    auc, auc_se = _auc(y, loo)

    # 4. Score the candidates.
    mean = model.predict_proba(full_cand)[:, 1]
    coef = tuple(float(c) for c in model.coef_[0])
    coef_sd = upper = None
    if settings.penalty == "l2":
        # Uncertainty of each weight, from the curvature of the fit at its optimum.
        covariance = _laplace_covariance(model, full_train, settings.C)
        coef_sd = tuple(float(v) for v in np.sqrt(np.clip(np.diag(covariance)[1:], 0, None)))
        if settings.rank_by == "upper":
            # Each candidate's log-odds has sd sqrt(x' Cov x); rank on log-odds + kappa * sd
            # to favour designs the model is unsure about (optimism under uncertainty).
            cand = np.c_[np.ones(len(full_cand)), full_cand]
            sd = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", cand, covariance, cand), 0, None))
            upper = 1 / (1 + np.exp(-(model.decision_function(full_cand) + settings.kappa * sd)))
    score = upper if upper is not None else mean
    order = tuple(sorted(range(len(score)), key=lambda i: (-score[i], i)))
    return Analysis(
        columns=columns,
        coef=coef,
        coef_sd=coef_sd,
        intercept=float(model.intercept_[0]),
        converged=converged,
        loo_log_loss=float(model_loss.mean()),
        baseline_log_loss=float(base_loss.mean()),
        improvement_se=float(difference.std(ddof=1) / math.sqrt(n)),
        loo_auc=auc,
        loo_auc_se=auc_se,
        mean=mean,
        upper=upper,
        order=order,
    )
