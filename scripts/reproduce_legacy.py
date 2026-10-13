#!/usr/bin/env python
"""Reproduce the exploratory ladder numbers (T20 step 2).

Runs the library policies and a script-only *legacy* learner on a whole-pool
Nipah-G game (split "all", so it does not depend on the frozen split), 100 seeds
each, and compares the mean hits with the exploratory values from the spec:

    random                          10.2 +/- 1.5
    fixed boltz2_iptm               10.0 +/- 1
    fixed boltz2_ipsae              16.0 +/- 1
    fixed boltz2_min_ipsae          18.0 +/- 1
    legacy learner (ipSAE start)    25.9 +/- 1.5
    legacy learner + similarity w=0.5   38.9 +/- 1.5

The legacy learner exists ONLY in this script (it is not part of the library):
7 Nipah-G metrics + length + one-hot class, median imputation from the whole
pool, no orientation, StandardScaler fit on the purchased rows, LogisticRegression
(C=0.1, max_iter=1000, lbfgs), a buy-by-ipSAE fallback while fewer than 2 hits
have been bought, no purchase floor, and each round buys floor(0.8 * k) by
predicted probability with the rest at random.

It also reports the scaling comparison the freeze step needs: purchased-row
scaling (legacy) versus pool-wide scaling (the library default), both run here
from the same legacy script.

Exit code 0 when every number is inside its tolerance, 1 when any is not,
2 for a setup problem.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    import replaylab  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from replaylab.advisor.similarity import SimilarityIndex  # noqa: E402
from replaylab.config import GameConfig, load_feature_config  # noqa: E402
from replaylab.domain.types import CampaignState, Design  # noqa: E402
from replaylab.eval.ladder import run_game  # noqa: E402
from replaylab.policies.core import FixedMetric, RandomPolicy  # noqa: E402

NIPAH = "nipah-glycoprotein-g"
EPS = 0.2
GUARD = 1e-9

# label, expected mean hits, tolerance, who runs it
EXPECTED = [
    ("random", 10.2, 1.5, "library"),
    ("fixed:boltz2_iptm", 10.0, 1.0, "library"),
    ("fixed:boltz2_ipsae", 16.0, 1.0, "library"),
    ("fixed:boltz2_min_ipsae", 18.0, 1.0, "library"),
    ("legacy learner", 25.9, 1.5, "legacy"),
    ("legacy learner_sim", 38.9, 1.5, "legacy"),
]

# order is the spec's: the 7 metrics, then length, then one-hot class
METRICS = (
    "boltz2_iptm",
    "boltz2_ipsae",
    "boltz2_min_ipsae",
    "boltz2_plddt",
    "shape_complimentarity_boltz2_binder_ss",
    "esmfold_plddt",
    "proteinmpnn_score",
)


class LegacyLearner:
    """The exploratory learner, re-implemented here so the library stays clean."""

    def __init__(self, *, with_similarity: bool, scale: str = "purchased") -> None:
        if scale not in ("purchased", "pool"):
            raise ValueError(f"scale must be 'purchased' or 'pool', got {scale!r}")
        self.with_similarity = with_similarity
        self.scale = scale
        self.name = "legacy learner_sim" if with_similarity else "legacy learner"
        self._similarity: SimilarityIndex | None = None

    # --- features -----------------------------------------------------------

    @staticmethod
    def _rows(
        designs: list[Design], medians: dict[str, float], classes: tuple[str, ...]
    ) -> np.ndarray:
        rows = []
        for design in designs:
            row = [design.features.get(name, medians[name]) for name in METRICS]
            row.append(float(len(design.sequence)))
            row.extend(1.0 if design.design_class == c else 0.0 for c in classes)
            rows.append(row)
        return np.asarray(rows, dtype=float)

    def _pool_features(self, state: CampaignState) -> tuple[dict[str, Design], np.ndarray]:
        """Whole-pool rows: 7 metrics median-imputed from the pool, length, one-hot."""
        pool = dict(state.designs)
        medians = {
            name: float(np.median([d.features[name] for d in pool.values() if name in d.features]))
            for name in METRICS
        }
        classes = tuple(sorted({d.design_class for d in pool.values() if d.design_class}))
        return pool, self._rows(list(pool.values()), medians, classes)

    # --- the buy decision -----------------------------------------------------

    def choose(self, state: CampaignState, k: int, rng: np.random.Generator) -> list[str]:
        available = list(state.unpurchased_ids)
        k = min(k, len(available))
        if k == 0:
            return []
        hits = [d for d, o in state.outcomes.items() if o.is_hit]
        if len(hits) < 2:
            # the exploratory start: buy by ipSAE until the model has signal
            return FixedMetric("boltz2_ipsae", True).choose(state, k, rng)

        pool, rows = self._pool_features(state)
        order = list(pool)
        index = {design_id: i for i, design_id in enumerate(order)}
        delivered = sorted(state.outcomes)
        y = np.array([int(state.outcomes[d].is_hit) for d in delivered])

        fit_rows = rows[[index[d] for d in delivered]] if self.scale == "purchased" else rows
        mean = fit_rows.mean(axis=0)
        std = fit_rows.std(axis=0)
        std[std < 1e-12] = 1.0
        scaled = (rows - mean) / std

        model = LogisticRegression(C=0.1, max_iter=1000)
        model.fit(scaled[[index[d] for d in delivered]], y)
        probabilities = model.predict_proba(scaled[[index[d] for d in available]])[:, 1]

        if self.with_similarity:
            if self._similarity is None:
                self._similarity = SimilarityIndex(state.designs)
            nearest = self._similarity.max_to_hits(state)
            sim = np.array([nearest[d][0] for d in available])
            scores = 0.5 * rankdata(probabilities, method="dense") + 0.5 * rankdata(
                sim, method="dense"
            )
        else:
            scores = probabilities

        # top floor((1 - 0.2) * k) by score, the rest at random from the remainder
        ranked = sorted(range(len(available)), key=lambda i: (-scores[i], available[i]))
        n_top = min(k, int(math.floor((1 - EPS) * k + GUARD)))
        top = [available[i] for i in ranked[:n_top]]
        chosen = set(top)
        remaining = [d for d in available if d not in chosen]
        picks = sorted(rng.choice(len(remaining), size=k - n_top, replace=False).tolist())
        return top + [remaining[i] for i in picks]


def legacy_game() -> GameConfig:
    return GameConfig(id="legacy", target=NIPAH, split="all", rounds=4, batch_size=25, budget=100)


def mean_hits(
    make_policy: Callable[[], object],
    seeds: int,
    *,
    tables_dir: Path,
    feature_config,
    clusters_path,
) -> tuple[float, float]:
    """Mean and sd of hits over ``seeds``.

    ``make_policy`` is a factory, not an instance: policies must be built once
    per seed, because the SimilarityIndex caches alias -> sequence and aliases
    depend on the seed (decision D4). Reusing one instance across seeds scrambles
    the similarity signal -- that exact bug cost a first run 20 hits instead of ~39.
    """
    cards = [
        run_game(
            legacy_game(),
            make_policy(),
            seed,
            tables_dir=tables_dir,
            feature_config=feature_config,
            clusters_path=clusters_path,
        )
        for seed in range(seeds)
    ]
    hits = np.array([c.hits for c in cards], dtype=float)
    return float(hits.mean()), float(hits.std(ddof=1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "data/processed")
    parser.add_argument("--features", type=Path, default=REPO_ROOT / "config/features.yaml")
    parser.add_argument("--clusters", type=Path, default=REPO_ROOT / "data/splits/clusters.csv")
    args = parser.parse_args(argv)

    feature_config = load_feature_config(args.features)[NIPAH]
    clusters_path = args.clusters if args.clusters.exists() else None

    started = time.time()
    results: dict[str, tuple[float, float]] = {}
    for label, _expected, _tolerance, runner in EXPECTED:
        if runner == "library":
            if label == "random":
                make_policy: Callable[[], object] = partial(RandomPolicy)
            else:
                # the exploratory numbers are the purchased-row scaling, always
                column = label.removeprefix("fixed:")
                make_policy = partial(FixedMetric, column, True)
        else:
            make_policy = partial(
                LegacyLearner, with_similarity=label.endswith("_sim"), scale="purchased"
            )
        mean, sd = mean_hits(
            make_policy,
            args.seeds,
            tables_dir=args.tables_dir,
            feature_config=feature_config,
            clusters_path=clusters_path,
        )
        results[label] = (mean, sd)

    # the scaling comparison for the freeze step: same legacy script, pool-wide scaling
    scaling: dict[str, float] = {}
    for with_sim in (False, True):
        label = "legacy learner_sim" if with_sim else "legacy learner"
        mean, _ = mean_hits(
            partial(LegacyLearner, with_similarity=with_sim, scale="pool"),
            args.seeds,
            tables_dir=args.tables_dir,
            feature_config=feature_config,
            clusters_path=clusters_path,
        )
        scaling[label] = mean

    print(f"game legacy (all, 4 x 25), {args.seeds} seeds")
    print(f"{'policy':<24} {'expected':>10} {'got mean':>10} {'sd':>7}  verdict")
    ok = True
    for label, expected, tolerance, _runner in EXPECTED:
        mean, sd = results[label]
        verdict = "OK" if abs(mean - expected) <= tolerance else "DIFF"
        ok = ok and verdict == "OK"
        print(f"{label:<24} {expected:>10.1f} {mean:>10.2f} {sd:>7.2f}  {verdict}")

    print("\nscaling comparison (purchased-row vs pool-wide, same legacy script):")
    for label in ("legacy learner", "legacy learner_sim"):
        purchased, _ = results[label]
        print(f"  {label}: purchased {purchased:.2f} vs pool {scaling[label]:.2f}")
    purchased, _ = results["legacy learner"]
    if purchased - scaling["legacy learner"] > 2:
        print(
            "  NOTE: purchased-row scaling wins by more than 2 hits over pool-wide "
            "scaling; per the card the library must not be changed without asking first."
        )

    print(f"\ndone in {time.time() - started:.0f}s")
    if not ok:
        print(
            "a DIFF is a finding: check (1) sequences upper-case and stripped, "
            "(2) metric list and direction, (3) tie-breaking in fixed metrics, "
            "(4) scaler rows, (5) the random stream."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
