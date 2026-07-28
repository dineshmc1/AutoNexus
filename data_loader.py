"""
data_loader.py
Loads tabular datasets (CSV / Excel), detects the problem type, and
splits data into train / test sets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

# Data bundle dataclass returned by the loader
@dataclass
class DataBundle:
    """Container for loaded and split data."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    problem_type: str          
    feature_names: list[str]
    target_name: str


def detect_problem_type(y: pd.Series, threshold: int = 10) -> str:
    """Infer classification for categorical targets and low-cardinality numerics."""
    n_unique = y.nunique()
    if (
        not pd.api.types.is_numeric_dtype(y)
        or pd.api.types.is_bool_dtype(y)
        or n_unique <= threshold
    ):
        return "classification"
    return "regression"


def load_local_dataset(csv_path, target_column):
    """Load and preprocess a local CSV based on onboarding config."""
    df = pd.read_csv(csv_path)
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {csv_path}")
        
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    # Detect problem type and encode labels
    problem_type = detect_problem_type(y)
    if problem_type == 'classification':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), name=y.name, index=y.index)
        
    return X, y, problem_type


def load_dataset(
    path: str,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    problem_type: Optional[str] = None,
) -> DataBundle:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. Use .csv, .xlsx, or .xls."
        )

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    if df.empty:
        raise ValueError("Dataset is empty.")
    if len(df) < 10:
        raise ValueError("Dataset must contain at least 10 rows.")
    if df[target_col].isna().any():
        missing_targets = int(df[target_col].isna().sum())
        raise ValueError(
            f"Target column '{target_col}' contains {missing_targets} missing value(s)."
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    id_cols = [
        c for c in X.columns
        if (c.lower().endswith("_id") or c.lower() == "id")
        and X[c].nunique() > 0.5 * len(X)
    ]
    if id_cols:
        X = X.drop(columns=id_cols)
        print(f"[DataLoader] Dropped ID column(s): {id_cols}")


    leaky: list[str] = []
    if pd.api.types.is_numeric_dtype(y):
        numeric_cols = X.select_dtypes(include="number").columns.tolist()

        if numeric_cols:
            corr = X[numeric_cols].corrwith(y).abs()
            leaky += corr[corr > 0.95].index.tolist()

        remaining = [c for c in numeric_cols if c not in leaky]
        if remaining and y.nunique() == 2:
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.metrics import roc_auc_score as _auc
            _sample = min(2000, len(X))
            _idx = np.random.RandomState(42).choice(
                len(X), _sample, replace=False,
            )
            for col in remaining:
                try:
                    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
                    dt.fit(X[col].values[_idx].reshape(-1, 1), y.values[_idx])
                    prob = dt.predict_proba(
                        X[col].values[_idx].reshape(-1, 1)
                    )[:, 1]
                    if _auc(y.values[_idx], prob) > 0.98:
                        leaky.append(col)
                except Exception:
                    pass

    if leaky:
        X = X.drop(columns=leaky)
        print(
            f"[DataLoader] ⚠ Dropped likely leakage column(s): {leaky}"
        )

    if problem_type is None:
        problem_type = detect_problem_type(y)
    elif problem_type not in ("classification", "regression"):
        raise ValueError(
            "problem_type must be 'classification', 'regression', or None."
        )

    if problem_type == "classification":
        from sklearn.preprocessing import LabelEncoder
        y = pd.Series(LabelEncoder().fit_transform(y), name=y.name, index=y.index)
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            raise ValueError("Classification target must contain at least two classes.")

    print(f"[DataLoader] Loaded {len(df)} rows, {X.shape[1]} features.")
    print(f"[DataLoader] Problem type: {problem_type}")

    stratify = None
    if problem_type == "classification" and y.value_counts().min() >= 2:
        stratify = y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify,
    )

    return DataBundle(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        problem_type=problem_type,
        feature_names=list(X.columns),
        target_name=target_col,
    )
