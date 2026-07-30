"""Held-out evaluation, validation-ranked tuning, and persistence."""

from __future__ import annotations

import os
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    RandomizedSearchCV,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline

# Evaluation

def evaluate_models(
    trained: Dict[str, Pipeline],
    X_test: np.ndarray,
    y_test: np.ndarray,
    problem_type: str,
) -> pd.DataFrame:
    """Compute final metrics on data that was not used for model selection."""
    # Compute metrics for every trained model on the held‑out test set.
    rows: list[dict] = []

    for name, pipe in trained.items():
        y_pred = pipe.predict(X_test)
        row: dict = {"model": name}

        if problem_type == "classification":
            row["accuracy"] = accuracy_score(y_test, y_pred)
            row["precision"] = precision_score(
                y_test, y_pred, average="weighted", zero_division=0,
            )
            row["recall"] = recall_score(
                y_test, y_pred, average="weighted", zero_division=0,
            )
            row["f1"] = f1_score(
                y_test, y_pred, average="weighted", zero_division=0,
            )
            # ROC‑AUC (only for binary or when predict_proba is available)
            try:
                if hasattr(pipe, "predict_proba"):
                    y_prob = pipe.predict_proba(X_test)
                    if y_prob.shape[1] == 2:
                        row["roc_auc"] = roc_auc_score(y_test, y_prob[:, 1])
                    else:
                        row["roc_auc"] = roc_auc_score(
                            y_test, y_prob, multi_class="ovr", average="weighted",
                        )
                else:
                    row["roc_auc"] = None
            except Exception:
                row["roc_auc"] = None
        else:
            row["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            row["mae"] = mean_absolute_error(y_test, y_pred)
            row["r2"] = r2_score(y_test, y_pred)

        rows.append(row)

    results = pd.DataFrame(rows)
    return results


# Default param grids (kept small for speed)
_PARAM_GRIDS: Dict[str, Dict[str, list]] = {
    "logistic": {
        "model__C": [0.01, 0.1, 0.5, 1, 5, 10],
        "model__solver": ["lbfgs", "saga"],
    },
    "rf": {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
    },
    "gb": {
        "model__max_iter": [100, 200, 300],
        "model__learning_rate": [0.01, 0.1, 0.2],
        "model__max_depth": [None, 6, 12],
        "model__l2_regularization": [0.0, 0.1, 1.0],
    },
    "et_clf": {
        "model__n_estimators": [150, 250, 400],
        "model__max_depth": [12, 18, 24, 32],
        "model__min_samples_leaf": [2, 4, 8],
        "model__min_samples_split": [4, 8, 16],
        "model__max_features": ["sqrt", 0.25, 0.5],
        "model__max_samples": [0.7, 0.85, 1.0],
    },
    "et_reg": {
        "model__n_estimators": [150, 250, 400],
        "model__max_depth": [12, 18, 24, 32],
        "model__min_samples_leaf": [2, 4, 8],
        "model__min_samples_split": [4, 8, 16],
        "model__max_features": ["sqrt", 0.25, 0.5],
        "model__max_samples": [0.7, 0.85, 1.0],
    },
    "linear": {},
}


def tune_top_models(
    trained: Dict[str, Pipeline],
    X_train: np.ndarray,
    y_train: np.ndarray,
    problem_type: str,
    validation_scores: Dict[str, float],
    top_n: int = 2,
    method: str = "randomized",
    n_iter: int = 20,
    cv: int = 5,
    groups: np.ndarray | None = None,
) -> tuple[Dict[str, Pipeline], Dict[str, float]]:
    """Tune candidates ranked only by development-set validation scores."""
    if problem_type == "classification":
        scoring = "accuracy"
    else:
        scoring = "neg_root_mean_squared_error"

    search_cv: int | Any = cv
    fit_kwargs: dict[str, Any] = {}
    if groups is not None:
        groups_array = np.asarray(groups)
        fit_kwargs["groups"] = groups_array
        if problem_type == "classification":
            y_array = np.asarray(y_train)
            class_group_counts = [
                len(np.unique(groups_array[y_array == label]))
                for label in np.unique(y_array)
            ]
            group_cv = min(cv, min(class_group_counts, default=2))
            if group_cv >= 2:
                search_cv = StratifiedGroupKFold(
                    n_splits=group_cv,
                    shuffle=True,
                    random_state=42,
                )
            else:
                fit_kwargs = {}
        else:
            group_cv = min(cv, len(np.unique(groups_array)))
            if group_cv >= 2:
                search_cv = GroupKFold(n_splits=group_cv)
            else:
                fit_kwargs = {}

    top_names = sorted(
        trained,
        key=lambda name: validation_scores.get(name, -np.inf),
        reverse=True,
    )[:top_n]
    tuned: Dict[str, Pipeline] = {}
    tuned_scores: Dict[str, float] = {}

    print(f"\n[Tuner] Tuning top {top_n} model(s): {top_names}")

    for name in top_names:
        pipe = trained[name]
        param_grid = _PARAM_GRIDS.get(name, {})

        if not param_grid:
            print(f"  {name}: no param grid defined – skipping tuning.")
            tuned[name] = pipe
            tuned_scores[name] = validation_scores.get(name, -np.inf)
            continue

        if method == "grid":
            searcher = GridSearchCV(
                pipe, param_grid, scoring=scoring, cv=search_cv, n_jobs=-1,
                refit=True,
            )
        else:
            searcher = RandomizedSearchCV(
                pipe, param_grid, scoring=scoring, cv=search_cv, n_jobs=-1,
                n_iter=min(n_iter, _grid_size(param_grid)),
                refit=True, random_state=42,
            )

        searcher.fit(X_train, y_train, **fit_kwargs)
        tuned[name] = searcher.best_estimator_
        tuned_scores[name] = float(searcher.best_score_)
        print(
            f"  {name}: best params = {searcher.best_params_}, "
            f"best score = {searcher.best_score_:.4f}"
        )

    return tuned, tuned_scores


def _grid_size(param_grid: dict) -> int:
    """Calculate total number of combinations in a param grid."""
    size = 1
    for vals in param_grid.values():
        size *= len(vals)
    return size

# Persistence helpers

def save_model(model: Any, path: str) -> None:
    """Persist a model atomically."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp"
    joblib.dump(model, temporary_path)
    os.replace(temporary_path, path)
    print(f"[Selector] Best model saved → {path}")


def save_metrics(results: pd.DataFrame, path: str) -> None:
    """Save metrics atomically."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    temporary_path = f"{path}.tmp"
    results.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)
    print(f"[Selector] Metrics saved → {path}")
