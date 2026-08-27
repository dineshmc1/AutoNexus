"""Regression tests for train/test isolation in the production pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_loader import DataBundle, load_dataset
from feature_processing import build_preprocessor
from main import _clean_tabular_bundle


def test_loader_splits_before_identifier_filtering(tmp_path):
    rows = 60
    frame = pd.DataFrame(
        {
            "row_id": np.arange(rows),
            "signal": np.tile([0.0, 1.0], rows // 2),
            "label": np.tile(["no", "yes"], rows // 2),
        }
    )
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)

    bundle = load_dataset(str(path), "label", random_state=7)

    assert "row_id" not in bundle.X_train
    assert "row_id" not in bundle.X_test
    assert "signal" in bundle.X_train
    assert set(bundle.y_train.unique()) == {0, 1}
    assert set(bundle.y_test.unique()) == {0, 1}


def test_highly_predictive_feature_is_not_mistaken_for_leakage(tmp_path):
    rows = 50
    target = np.tile([0, 1], rows // 2)
    frame = pd.DataFrame(
        {
            "legitimate_signal": target,
            "noise": np.random.default_rng(42).normal(size=rows),
            "target": target,
        }
    )
    path = tmp_path / "predictive.csv"
    frame.to_csv(path, index=False)

    bundle = load_dataset(str(path), "target")

    assert "legitimate_signal" in bundle.feature_names


def test_frequency_encoding_preserves_medium_cardinality_column():
    frame = pd.DataFrame(
        {"category": ["a", "a", "b", "c"], "number": [1, 2, 3, 4]}
    )
    preprocessor, _, _ = build_preprocessor(
        frame, encoding_map={"category": "frequency"}
    )

    transformed = preprocessor.fit_transform(frame, [0, 1, 0, 1])

    assert transformed.shape == (4, 2)
    assert np.isfinite(np.asarray(transformed)).all()


def test_duplicate_cleaning_keeps_analytics_metadata_aligned():
    bundle = DataBundle(
        X_train=pd.DataFrame({"feature": [1, 1, 2]}),
        X_test=pd.DataFrame({"feature": [3, 3, 4]}),
        y_train=pd.Series([0, 0, 1], name="label"),
        y_test=pd.Series([0, 0, 1], name="label"),
        problem_type="classification",
        feature_names=["feature"],
        target_name="label",
        groups_train=pd.Series(["a", "a", "b"]),
        groups_test=pd.Series(["c", "c", "d"]),
        row_ids_train=pd.Series(["train-0", "train-1", "train-2"]),
        row_ids_test=pd.Series(["test-0", "test-1", "test-2"]),
    )

    X_train, y_train, X_test, y_test = _clean_tabular_bundle(bundle)

    assert len(X_train) == len(y_train) == len(bundle.row_ids_train) == 2
    assert len(X_test) == len(y_test) == len(bundle.row_ids_test) == 2
    assert bundle.row_ids_train.tolist() == ["train-0", "train-2"]
    assert bundle.row_ids_test.tolist() == ["test-0", "test-2"]
    assert bundle.groups_train.tolist() == ["a", "b"]
    assert bundle.groups_test.tolist() == ["c", "d"]
    pd.testing.assert_frame_equal(bundle.X_train, X_train)
    pd.testing.assert_frame_equal(bundle.X_test, X_test)
