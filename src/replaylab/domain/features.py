"""The frozen list of visible computational features and their orientation.

Week 0 finalizes this list per target (coverage and leakage review). Until then
this is the working default taken from the exploratory analysis. Orientation is a
per-column flag, applied once at load time, so every covariate points the same
way (higher is better) before any policy sees it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    """One visible computational metric."""

    name: str
    higher_is_better: bool = True


# Column names are spelled exactly as in the Proteinbase export, including the
# misspelling in the shape-complementarity metric.
DEFAULT_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("boltz2_iptm"),
    FeatureSpec("boltz2_ipsae"),
    FeatureSpec("boltz2_min_ipsae"),
    FeatureSpec("boltz2_plddt"),
    FeatureSpec("shape_complimentarity_boltz2_binder_ss"),
    FeatureSpec("esmfold_plddt"),
    FeatureSpec("proteinmpnn_score", higher_is_better=False),
)


@dataclass(frozen=True)
class FeatureConfig:
    """The visible features of one target and the pre-registered first column.

    ``first_column`` is the metric a policy sorts by in round 1. It must be one of
    the listed features. Sequence length and ``design_class`` are always visible
    and are not listed here.
    """

    target: str
    first_column: str
    specs: tuple[FeatureSpec, ...]

    def __post_init__(self) -> None:
        specs = tuple(self.specs)
        if not specs:
            raise ValueError(f"target {self.target}: at least one feature is required")
        names = [spec.name for spec in specs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"target {self.target}: duplicate feature(s): {duplicates}")
        if self.first_column not in names:
            raise ValueError(
                f"target {self.target}: first_column {self.first_column!r} "
                "is not in the feature list"
            )
        object.__setattr__(self, "specs", specs)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    @property
    def orientation(self) -> dict[str, float]:
        """+1.0 for higher-is-better features and -1.0 for lower-is-better ones."""
        return orientation_signs(self.specs)


def feature_names(specs: Iterable[FeatureSpec] = DEFAULT_FEATURES) -> tuple[str, ...]:
    return tuple(spec.name for spec in specs)


def orientation_signs(specs: Iterable[FeatureSpec] = DEFAULT_FEATURES) -> dict[str, float]:
    """Return +1.0 for higher-is-better columns and -1.0 for lower-is-better ones."""
    return {spec.name: (1.0 if spec.higher_is_better else -1.0) for spec in specs}
