"""T17: preprocessing (orientation, imputation, standardization, one-hot classes)."""

from __future__ import annotations

import numpy as np
import pytest

from replaylab.advisor.preprocess import Preprocessor
from replaylab.domain.features import FeatureConfig, FeatureSpec
from replaylab.domain.types import Design

CONFIG = FeatureConfig(
    "t",
    "f1",
    (FeatureSpec("f1"), FeatureSpec("f2"), FeatureSpec("p", higher_is_better=False)),
)


def worked_example() -> dict[str, Design]:
    return {
        "a": Design("a", "A" * 10, "A", {"f1": 1.0, "f2": 5.0, "p": 1.0}),
        "b": Design("b", "A" * 10, "B", {"f1": 2.0, "f2": 5.0, "p": 3.0}),
        "c": Design("c", "A" * 10, "A", {"f1": 3.0, "f2": 5.0, "p": 2.0}),
        "d": Design("d", "A" * 10, None, {"f2": 5.0, "p": 4.0}),  # no f1, no class
    }


def test_worked_example_matrix_and_columns():
    designs = worked_example()
    pre = Preprocessor.fit(designs, CONFIG)
    # f2 and length are constant in the pool, so they are dropped.
    assert pre.columns == ("f1", "p", "class=A", "class=B")
    assert pre.dropped == ("f2", "length")
    matrix = pre.transform(designs.values())
    # f1: raw [1, 2, 3, missing] -> median 2 fills d; mean 2, population std sqrt(0.5).
    # p: oriented [-1, -3, -2, -4]: mean -2.5, population std sqrt(1.25).
    expected = np.array(
        [
            [-1.41421356, 1.34164079, 1.0, 0.0],
            [0.0, -0.4472136, 0.0, 1.0],
            [1.41421356, 0.4472136, 1.0, 0.0],
            [0.0, -1.34164079, 0.0, 0.0],  # class None: all zeros
        ]
    )
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix, expected, atol=1e-7)
    assert pre.medians == {"f1": 2.0, "p": 2.5}  # medians are raw, not oriented
    assert pre.means["p"] == pytest.approx(-2.5) and pre.stds["f1"] == pytest.approx(0.5**0.5)


def test_standardized_columns_have_mean_zero_and_unit_population_std():
    designs = worked_example()
    pre = Preprocessor.fit(designs, CONFIG)
    matrix = pre.transform(designs.values())
    assert np.allclose(matrix[:, :2].mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(matrix[:, :2].std(axis=0), 1.0, atol=1e-12)  # ddof = 0


def test_orientation_is_applied_here_and_only_here():
    designs = worked_example()
    pre = Preprocessor.fit(designs, CONFIG)
    column = pre.columns.index("p")
    matrix = pre.transform(designs.values())
    # Lower p is better, so the design with the lowest raw p has the highest value.
    assert int(np.argmax(matrix[:, column])) == 0 and designs["a"].features["p"] == 1.0
    assert designs["a"].features["p"] == 1.0  # the design itself is untouched


def test_transform_uses_pool_statistics_for_any_subset_and_order():
    designs = worked_example()
    pre = Preprocessor.fit(designs, CONFIG)
    full = pre.transform(designs.values())
    subset = pre.transform([designs["c"], designs["a"]])
    assert np.allclose(subset, full[[2, 0]])


def test_unseen_class_gives_all_zero_one_hots_and_missing_features_use_the_median():
    pre = Preprocessor.fit(worked_example(), CONFIG)
    stranger = Design("z", "A" * 10, "NEW", {"p": 2.5})  # unseen class, f1 missing
    row = pre.transform([stranger])[0]
    assert row[pre.columns.index("class=A")] == 0.0 and row[pre.columns.index("class=B")] == 0.0
    assert row[pre.columns.index("f1")] == pytest.approx(0.0)  # median imputed: (2 - 2) / std
    assert row[pre.columns.index("p")] == pytest.approx(0.0)  # raw 2.5 -> -2.5 = the mean


def test_median_of_an_even_count_is_the_mean_of_the_middle_two():
    designs = {
        f"d{i}": Design(f"d{i}", "A" * 10, None, {"f1": v} if v is not None else {})
        for i, v in enumerate([1.0, 2.0, 3.0, 10.0, None])
    }
    config = FeatureConfig("t", "f1", (FeatureSpec("f1"),))
    pre = Preprocessor.fit(designs, config)
    assert pre.medians["f1"] == 2.5  # median of [1, 2, 3, 10]


def test_a_feature_with_no_values_is_dropped_and_reported():
    designs = {
        "a": Design("a", "A" * 8, None, {"f1": 1.0}),
        "b": Design("b", "A" * 9, None, {"f1": 2.0}),
    }
    config = FeatureConfig("t", "f1", (FeatureSpec("f1"), FeatureSpec("never")))
    pre = Preprocessor.fit(designs, config)
    assert pre.columns == ("f1", "length") and pre.dropped == ("never",)


def test_length_is_a_standardized_column_when_it_varies():
    designs = {
        "a": Design("a", "A" * 10, None, {"f1": 1.0}),
        "b": Design("b", "A" * 20, None, {"f1": 2.0}),
    }
    pre = Preprocessor.fit(designs, FeatureConfig("t", "f1", (FeatureSpec("f1"),)))
    assert pre.columns == ("f1", "length")
    assert np.allclose(pre.transform(designs.values())[:, 1], [-1.0, 1.0])


def test_a_class_shared_by_every_design_is_dropped():
    designs = {
        "a": Design("a", "A" * 8, "X", {"f1": 1.0}),
        "b": Design("b", "A" * 9, "X", {"f1": 2.0}),
    }
    pre = Preprocessor.fit(designs, FeatureConfig("t", "f1", (FeatureSpec("f1"),)))
    assert "class=X" not in pre.columns and "class=X" in pre.dropped


def test_column_order_is_features_then_length_then_classes_sorted():
    designs = {
        "a": Design("a", "A" * 8, "Z", {"f1": 1.0, "f2": 3.0}),
        "b": Design("b", "A" * 9, "A", {"f1": 2.0, "f2": 1.0}),
        "c": Design("c", "A" * 7, "M", {"f1": 3.0, "f2": 2.0}),
    }
    config = FeatureConfig("t", "f2", (FeatureSpec("f2"), FeatureSpec("f1")))
    pre = Preprocessor.fit(designs, config)
    assert pre.columns == ("f2", "f1", "length", "class=A", "class=M", "class=Z")


def test_an_empty_pool_is_rejected():
    with pytest.raises(ValueError, match="empty pool"):
        Preprocessor.fit({}, CONFIG)


def test_transform_of_nothing_has_the_right_shape():
    pre = Preprocessor.fit(worked_example(), CONFIG)
    assert pre.transform([]).shape == (0, 4)
