"""
dataset_embedding.py
====================
Phase 4.1 — Rich Dataset Embedding Vectors.

Computes fixed-length, FAISS-compatible ``float32`` vectors that
summarise a dataset's statistical fingerprint.  Designed for
meta-learning, dataset similarity search, and pipeline warm-starting.

Public API
----------
- ``compute_dataset_embedding(X, y=None) -> np.ndarray``
- ``build_embedding_matrix(datasets) -> np.ndarray``
- ``save_embeddings(results, path)``

Embedding dimensions (9):
    0  n_samples        — log-scaled row count
    1  n_features       — log-scaled column count
    2  missing_ratio    — fraction of NaN cells
    3  mean_skewness    — avg skewness across numeric columns
    4  mean_kurtosis    — avg kurtosis across numeric columns
    5  entropy          — target entropy (clf) or avg feature entropy
    6  corr_density     — % of pairs with |corr| > 0.5
    7  interaction_score— normalised RF importance variance
    8  task_type         — 0=binary clf, 1=regression, 2=multiclass
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 10
SEARCH_EMBEDDING_VERSION = 2
_RF_ESTIMATORS = 50
_RF_MAX_DEPTH = 5
_CORR_THRESHOLD = 0.5
_RANDOM_STATE = 42
_MAX_ROWS_FOR_RF = 10_000          # cap rows fed to the RF probe
_MAX_FEATURES_FOR_CORR = 200       # cap features for correlation matrix


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _preprocess(X: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values and label-encode categoricals.

    Returns a fully numeric DataFrame with no NaNs.
    """
    X = X.copy()

    # --- Numeric imputation (median) ---
    num_cols = X.select_dtypes(include="number").columns
    if len(num_cols):
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    # --- Categorical imputation (mode) + label encoding ---
    cat_cols = X.select_dtypes(include=["str", "object", "category", "bool"]).columns
    for col in cat_cols:
        mode_val = X[col].mode()
        fill = mode_val.iloc[0] if len(mode_val) else "missing"
        X[col] = X[col].fillna(fill)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Convert any remaining non-numeric columns
    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            try:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
            except Exception:
                X[col] = 0

    return X


def _infer_task_type(y: Optional[pd.Series]) -> int:
    """Return 0 (binary classification), 1 (regression), or 2 (multiclass).

    If *y* is ``None`` the task type defaults to regression (1).
    """
    if y is None:
        return 1  # no target → assume regression
    nunique = int(pd.Series(y).nunique())
    if nunique <= 2:
        return 0  # binary classification
    if nunique <= 20 or pd.api.types.is_object_dtype(y):
        return 2  # multiclass
    return 1      # regression


def _safe_entropy(values: np.ndarray) -> float:
    """Shannon entropy in nats, safe for empty / constant arrays."""
    values = values[~np.isnan(values)] if np.issubdtype(values.dtype, np.floating) else values
    if len(values) == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    return float(stats.entropy(counts))


def _normalize_embedding(vec: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1] and replace any residual NaN with 0."""
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    vmin, vmax = vec.min(), vec.max()
    if vmax - vmin > 1e-12:
        vec = (vec - vmin) / (vmax - vmin)
    return vec


# ---------------------------------------------------------------------------
# Core embedding function
# ---------------------------------------------------------------------------

def compute_dataset_embedding(
    X: pd.DataFrame,
    y: Optional[Union[pd.Series, np.ndarray]] = None,
) -> np.ndarray:
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if y is not None and not isinstance(y, (pd.Series, np.ndarray)):
        y = np.asarray(y)

    features = []

    # --- NEW: Structural features ---
    
    # 1. Log-scaled sample count (captures scale differences better)
    features.append(np.log1p(X.shape[0]) / 10.0)
    
    # 2. Log-scaled feature count
    features.append(np.log1p(X.shape[1]) / 10.0)
    
    # 3. Samples-to-features ratio
    features.append(np.log1p(X.shape[0] / max(X.shape[1], 1)) / 10.0)
    
    # --- NEW: Statistical diversity ---
    numeric_cols = X.select_dtypes(include=[np.number])
    
    if numeric_cols.shape[1] > 0:
        col_means = numeric_cols.mean()
        col_stds  = numeric_cols.std().fillna(0)
        
        # 4. Mean of column-wise skewness
        skewness = numeric_cols.apply(lambda c: stats.skew(c.dropna()))
        features.append(np.clip(skewness.mean() / 5.0, -1, 1))
        
        # 5. Fraction of columns that are highly skewed (|skew| > 1)
        features.append((skewness.abs() > 1).mean())
        
        # 6. Mean pairwise correlation (captures feature redundancy)
        if numeric_cols.shape[1] > 1:
            corr_matrix = numeric_cols.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            features.append(upper.stack().mean())
        else:
            features.append(0.0)
        
        # 7. Coefficient of variation mean (std/mean ratio)
        cv = (col_stds / (col_means.abs() + 1e-8)).clip(-10, 10)
        features.append(np.tanh(cv.mean()))
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])

    # --- NEW: Target features ---
    if y is not None:
        y_series = pd.Series(y)
        # 8. Number of unique classes (normalized)
        n_unique = y_series.nunique()
        features.append(np.log1p(n_unique) / 5.0)
        
        # 9. Target entropy (measures class balance more richly than ratio)
        value_counts = y_series.value_counts(normalize=True)
        entropy = stats.entropy(value_counts)
        features.append(np.clip(entropy / np.log(max(n_unique, 2)), 0, 1))
    else:
        features.extend([0.0, 0.0])
    
    # 10. Missing rate
    features.append(X.isnull().mean().mean())

    embedding = np.nan_to_num(np.array(features, dtype=np.float32), nan=0.0)
    return embedding


def compute_search_embedding(
    X: pd.DataFrame,
    y: Optional[Union[pd.Series, np.ndarray]],
    baseline_scores: Dict[str, Dict[str, float]],
) -> np.ndarray:
    """Combine statistical meta-features with fixed family landmarks.

    This is deliberately versioned separately from the legacy 10D/32D FAISS
    representation so incompatible historical indexes are never queried.
    """
    base = compute_dataset_embedding(X, y)
    families = {
        "linear": ("logistic", "sgd_clf", "ridge", "sgd_reg"),
        "gbdt": ("xgb_clf", "lgbm_clf", "gb", "xgb_reg", "lgbm_reg", "gb_reg"),
        "nonlinear": (
            "et_clf", "rf", "mlp_clf", "knn_clf",
            "et_reg", "rf_reg", "mlp_reg", "knn_reg",
        ),
    }

    def score_for(names: tuple[str, ...]) -> float:
        values = [
            float(baseline_scores[name]["score"])
            for name in names
            if name in baseline_scores
        ]
        return max(values) if values else -1.0

    all_scores = [
        float(result["score"]) for result in baseline_scores.values()
    ]
    all_times = [
        float(result.get("time", 0.0))
        for result in baseline_scores.values()
    ]
    landmarks = np.array(
        [
            np.tanh(score_for(families["linear"])),
            np.tanh(score_for(families["gbdt"])),
            np.tanh(score_for(families["nonlinear"])),
            np.tanh(max(all_scores)) if all_scores else -1.0,
            np.tanh(np.ptp(all_scores)) if all_scores else 0.0,
            np.tanh(np.mean(all_times)) if all_times else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, landmarks]).astype(np.float32)


# ---------------------------------------------------------------------------
# Batch builder
# ---------------------------------------------------------------------------

def build_embedding_matrix(
    datasets: Sequence[Dict[str, Any]],
) -> np.ndarray:
    """Compute embeddings for multiple datasets at once.

    Parameters
    ----------
    datasets : list of dict
        Each dict must contain ``"X"`` (pd.DataFrame) and optionally
        ``"y"`` (array-like).

    Returns
    -------
    np.ndarray
        2-D ``float32`` array of shape ``(len(datasets), EMBEDDING_DIM)``.
    """
    embeddings = []
    for i, ds in enumerate(datasets):
        X = ds["X"]
        y = ds.get("y", None)
        emb = compute_dataset_embedding(X, y)
        embeddings.append(emb)
        logger.info("Dataset %d/%d embedded.", i + 1, len(datasets))

    matrix = np.vstack(embeddings).astype(np.float32)
    return matrix


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_embeddings(
    results: List[Dict[str, Any]],
    path: str,
) -> str:
    """Save embedding results with metadata to a JSON file.

    Parameters
    ----------
    results : list of dict
        Each dict should follow the schema::

            {
                "dataset_id": str | int,
                "embedding": list[float],
                "task_type": int,
                "n_samples": int,
                "n_features": int,
            }

    path : str
        Output JSON file path.

    Returns
    -------
    str
        The absolute path of the written file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Ensure embeddings are plain Python lists (JSON-serialisable)
    serialisable = []
    for r in results:
        entry = dict(r)
        if isinstance(entry.get("embedding"), np.ndarray):
            entry["embedding"] = entry["embedding"].tolist()
        serialisable.append(entry)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2)

    abs_path = os.path.abspath(path)
    logger.info("Embeddings saved to %s (%d entries).", abs_path, len(serialisable))
    return abs_path
