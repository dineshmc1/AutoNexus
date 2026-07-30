"""Fold-safe numeric and categorical preprocessing."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Map categories to development-fold relative frequencies."""

    def fit(self, X, y=None):
        frame = pd.DataFrame(X).astype(str)
        self.maps_ = [
            frame[column].value_counts(normalize=True).to_dict()
            for column in frame.columns
        ]
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).astype(str)
        encoded = np.column_stack(
            [
                frame[column].map(mapping).fillna(0.0).to_numpy()
                for column, mapping in zip(frame.columns, self.maps_)
            ]
        )
        return encoded.astype(np.float32, copy=False)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [
                f"feature_{index}" for index in range(self.n_features_in_)
            ]
        return np.asarray(
            [f"{feature}_frequency" for feature in input_features],
            dtype=object,
        )


def hash_features(data):
    """Convert categorical values to named tokens for FeatureHasher."""
    from sklearn.feature_extraction import FeatureHasher

    frame = pd.DataFrame(data).astype(str).fillna("missing")
    for column in frame.columns:
        frame[column] = str(column) + "=" + frame[column]
    return FeatureHasher(
        n_features=2048, input_type="string"
    ).transform(frame.values)


def detect_column_types(
    X: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
    return numeric, categorical


def build_preprocessor(
    X: pd.DataFrame,
    scaler_map: Optional[Dict[str, str]] = None,
    encoding_map: Optional[Dict[str, str]] = None,
) -> Tuple[object, List[str], List[str]]:
    """Build a cloneable transformer from development-data column metadata."""
    numeric_columns, categorical_columns = detect_column_types(X)
    unknown_columns = [
        column
        for column in X.columns
        if column not in numeric_columns and column not in categorical_columns
    ]
    if unknown_columns:
        raise ValueError(
            "Unsupported feature dtype for column(s): "
            f"{unknown_columns}. Convert them to numeric, string, category, "
            "or bool."
        )
    if not numeric_columns and not categorical_columns:
        raise ValueError("No supported feature columns were found.")

    if not categorical_columns and len(numeric_columns) >= 50:
        print(
            f"[Features] Detected dense numeric embeddings "
            f"({len(numeric_columns)}D); preserving embedding geometry."
        )
        return (
            FunctionTransformer(validate=False),
            numeric_columns,
            categorical_columns,
        )

    transformers = []
    if numeric_columns:
        if scaler_map:
            standard_columns = [
                column
                for column in numeric_columns
                if scaler_map.get(column, "standard") == "standard"
            ]
            robust_columns = [
                column
                for column in numeric_columns
                if scaler_map.get(column) == "robust"
            ]
            if standard_columns:
                transformers.append(
                    (
                        "num_standard",
                        Pipeline(
                            [
                                ("imputer", SimpleImputer(strategy="median")),
                                ("scaler", StandardScaler()),
                            ]
                        ),
                        standard_columns,
                    )
                )
            if robust_columns:
                transformers.append(
                    (
                        "num_robust",
                        Pipeline(
                            [
                                ("imputer", SimpleImputer(strategy="median")),
                                ("scaler", RobustScaler()),
                            ]
                        ),
                        robust_columns,
                    )
                )
        else:
            transformers.append(
                (
                    "numeric",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_columns,
                )
            )

    hash_columns: list[str] = []
    if categorical_columns:
        strategy = encoding_map or {}
        onehot_columns = [
            column
            for column in categorical_columns
            if strategy.get(column, "onehot") == "onehot"
        ]
        frequency_columns = [
            column
            for column in categorical_columns
            if strategy.get(column) == "frequency"
        ]
        hash_columns = [
            column
            for column in categorical_columns
            if strategy.get(column) == "hash"
        ]
        if onehot_columns:
            transformers.append(
                (
                    "categorical_onehot",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="constant",
                                    fill_value="missing",
                                ),
                            ),
                            (
                                "encoder",
                                OneHotEncoder(
                                    handle_unknown="ignore",
                                    sparse_output=False,
                                ),
                            ),
                        ]
                    ),
                    onehot_columns,
                )
            )
        if frequency_columns:
            transformers.append(
                (
                    "categorical_frequency",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="constant",
                                    fill_value="missing",
                                ),
                            ),
                            ("encoder", FrequencyEncoder()),
                        ]
                    ),
                    frequency_columns,
                )
            )
        if hash_columns:
            transformers.append(
                (
                    "categorical_hash",
                    FunctionTransformer(hash_features, validate=False),
                    hash_columns,
                )
            )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )
    if not hash_columns:
        preprocessor.set_output(transform="pandas")
    print(
        f"[Features] {len(numeric_columns)} numeric, "
        f"{len(categorical_columns)} categorical column(s)."
    )
    return preprocessor, numeric_columns, categorical_columns
