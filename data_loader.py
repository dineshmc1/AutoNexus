"""Validated loading and leakage-safe splitting for tabular datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class DataBundle:
    """Development/test data and task metadata."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    problem_type: str
    feature_names: list[str]
    target_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    groups_train: pd.Series | None = None
    groups_test: pd.Series | None = None
    row_ids_train: pd.Series | None = None
    row_ids_test: pd.Series | None = None


def detect_problem_type(y: pd.Series, threshold: int = 10) -> str:
    """Infer classification for categorical and low-cardinality targets."""
    if (
        not pd.api.types.is_numeric_dtype(y)
        or pd.api.types.is_bool_dtype(y)
        or y.nunique() <= threshold
    ):
        return "classification"
    return "regression"


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(
        f"Unsupported file format '{suffix}'. Use .csv, .xlsx, or .xls."
    )


def load_dataset(
    path: str,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    problem_type: Optional[str] = None,
) -> DataBundle:
    """Load, validate, encode, and split a tabular dataset."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = _read_table(dataset_path)
    if df.empty:
        raise ValueError("Dataset is empty.")
    if len(df) < 10:
        raise ValueError("Dataset must contain at least 10 rows.")
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    missing_targets = int(df[target_col].isna().sum())
    if missing_targets:
        raise ValueError(
            f"Target column '{target_col}' contains "
            f"{missing_targets} missing value(s)."
        )

    X = df.drop(columns=[target_col])
    if X.shape[1] == 0:
        raise ValueError("Dataset must contain at least one feature column.")
    y = df[target_col]

    if problem_type is None:
        problem_type = detect_problem_type(y)
    elif problem_type not in {"classification", "regression"}:
        raise ValueError(
            "problem_type must be 'classification', 'regression', or None."
        )

    if problem_type == "classification":
        y = pd.Series(
            LabelEncoder().fit_transform(y.astype(str)),
            name=y.name,
            index=y.index,
        )
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            raise ValueError(
                "Classification target must contain at least two classes."
            )
        if class_counts.min() < 2:
            raise ValueError(
                "Every classification class needs at least two rows."
            )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if problem_type == "classification" else None,
        )
    except ValueError as exc:
        raise ValueError(
            "The requested split cannot preserve every class. Add samples "
            "to rare classes or adjust --test-size."
        ) from exc
    if (
        problem_type == "classification"
        and y_train.value_counts().min() < 2
    ):
        raise ValueError(
            "Each class needs at least two development rows after the test "
            "split. Add samples to rare classes or reduce --test-size."
        )

    # Infer identifier columns from development data only. We intentionally do
    # not remove target-correlated features automatically: doing so before the
    # split leaks test labels and can discard legitimate predictors.
    id_columns = [
        column
        for column in X_train.columns
        if (
            column.lower() == "id" or column.lower().endswith("_id")
        )
        and X_train[column].nunique(dropna=False) > 0.5 * len(X_train)
    ]
    if id_columns:
        X_train = X_train.drop(columns=id_columns)
        X_test = X_test.drop(columns=id_columns)
        print(f"[DataLoader] Dropped ID column(s): {id_columns}")
    if X_train.shape[1] == 0:
        raise ValueError("No usable feature columns remain after validation.")

    print(
        f"[DataLoader] Loaded {len(df)} rows, "
        f"{X_train.shape[1]} features."
    )
    print(f"[DataLoader] Problem type: {problem_type}")
    class_names = (
        [str(value) for value in sorted(df[target_col].astype(str).unique())]
        if problem_type == "classification"
        else []
    )
    return DataBundle(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        problem_type=problem_type,
        feature_names=list(X_train.columns),
        target_name=target_col,
        metadata={"class_names": class_names},
        row_ids_train=pd.Series(
            X_train.index.astype(str), name="source_row"
        ).reset_index(drop=True),
        row_ids_test=pd.Series(
            X_test.index.astype(str), name="source_row"
        ).reset_index(drop=True),
    )
