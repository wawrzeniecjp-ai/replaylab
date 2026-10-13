"""Turn designs into a numeric matrix (decision D7: orientation is applied here).

Fitted on the *pool* (all designs of the game) and never on outcomes, so it uses only
what an agent could already see through ``column_summary`` and ``query_pool``.

Columns, in this order:

1. each configured feature: missing values imputed with the pool median, multiplied by
   the orientation sign (+1 higher is better, -1 lower is better), then standardized
   with the pool mean and the population standard deviation (ddof = 0);
2. ``length``: sequence length, standardized the same way;
3. one-hot ``class=<name>`` for every class in the pool, in sorted order (0/1, not
   standardized). A design whose class is None or unseen gets all zeros.

A column with no variation in the pool, or a feature with no values at all, is dropped
and listed in ``dropped``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from replaylab.domain.features import FeatureConfig
from replaylab.domain.types import Design

_CONSTANT = 1e-12


@dataclass(frozen=True)
class Preprocessor:
    """Fitted preprocessing for one pool: medians, orientation, standardizing
    statistics and the one-hot classes -- everything needed to turn Designs
    into the numeric matrix the model fits on.

    Fit it once per pool (it never sees outcomes); see the module docstring
    for the exact column layout.
    """

    features: tuple[str, ...]  # configured features that are kept, in config order
    orientation: Mapping[str, float]
    medians: Mapping[str, float]  # raw (not oriented) pool medians
    means: Mapping[str, float]  # of the oriented, imputed values; also for 'length'
    stds: Mapping[str, float]
    classes: tuple[str, ...]  # kept class one-hot columns (class names, sorted)
    use_length: bool
    columns: tuple[str, ...]
    dropped: tuple[str, ...]

    @classmethod
    def fit(cls, designs: Mapping[str, Design], feature_config: FeatureConfig) -> Preprocessor:
        """Fit medians, means and standard deviations on the whole pool.

        Only visible information is used (features, sequence length, class),
        so what the preprocessor sees is exactly what an agent could query.
        Columns with no variation (or no values at all) are dropped and
        listed in ``dropped``.
        """
        pool = list(designs.values())
        if not pool:
            raise ValueError("cannot fit a preprocessor on an empty pool")
        orientation = feature_config.orientation
        kept, dropped = [], []
        medians: dict[str, float] = {}
        means: dict[str, float] = {}
        stds: dict[str, float] = {}

        for name in feature_config.names:
            raw = [d.features[name] for d in pool if name in d.features]
            if not raw:
                dropped.append(name)
                continue
            median = float(np.median(raw))
            values = np.array([d.features.get(name, median) for d in pool]) * orientation[name]
            if float(np.std(values)) < _CONSTANT:
                dropped.append(name)
                continue
            kept.append(name)
            medians[name], means[name], stds[name] = (
                median,
                float(values.mean()),
                float(values.std()),
            )

        lengths = np.array([len(d.sequence) for d in pool], dtype=float)
        use_length = float(lengths.std()) >= _CONSTANT
        if use_length:
            means["length"], stds["length"] = float(lengths.mean()), float(lengths.std())
        else:
            dropped.append("length")

        classes = []
        for name in sorted({d.design_class for d in pool if d.design_class is not None}):
            indicator = np.array([d.design_class == name for d in pool], dtype=float)
            if indicator.std() < _CONSTANT:
                dropped.append(f"class={name}")
            else:
                classes.append(name)

        columns = (
            tuple(kept) + (("length",) if use_length else ()) + tuple(f"class={c}" for c in classes)
        )
        return cls(
            features=tuple(kept),
            orientation=dict(orientation),
            medians=medians,
            means=means,
            stds=stds,
            classes=tuple(classes),
            use_length=use_length,
            columns=columns,
            dropped=tuple(dropped),
        )

    def transform(self, designs: Iterable[Design]) -> np.ndarray:
        """Matrix with one row per design (in the given order) and ``columns`` as columns."""
        rows = []
        for design in designs:
            row = []
            for name in self.features:
                raw = design.features.get(name, self.medians[name])
                row.append((raw * self.orientation[name] - self.means[name]) / self.stds[name])
            if self.use_length:
                row.append((len(design.sequence) - self.means["length"]) / self.stds["length"])
            row.extend(1.0 if design.design_class == c else 0.0 for c in self.classes)
            rows.append(row)
        return np.array(rows, dtype=float).reshape(len(rows), len(self.columns))
