"""
model_trainer.py
Provides model catalogues, baseline screening on a data subsample, and
full training with cross‑validation.  Supports parallel execution and an
optional wall‑clock time budget.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet,
    SGDClassifier, SGDRegressor, LinearRegression
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    BaggingClassifier, BaggingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = LGBMRegressor = None

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = XGBRegressor = None


# Model catalogue

CLASSIFICATION_MODELS = {
    "logistic":    LogisticRegression(max_iter=5000, random_state=42),
    "sgd_clf":     SGDClassifier(max_iter=5000, random_state=42),
    "knn_clf":     KNeighborsClassifier(n_neighbors=5),
    "naive_bayes": GaussianNB(),
    "dt_clf":      DecisionTreeClassifier(random_state=42),
    "svc":         SVC(probability=True, random_state=42),
    "mlp_clf":     MLPClassifier(max_iter=5000, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=42),
    "rf":          RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    "et_clf":      ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    "ada_clf":     AdaBoostClassifier(n_estimators=100, random_state=42),
    "bag_clf":     BaggingClassifier(n_estimators=20, n_jobs=-1, random_state=42),
    "gb":          HistGradientBoostingClassifier(random_state=42),
}  # 14 classification models

REGRESSION_MODELS = {
    "ridge":       Ridge(),
    "lasso":       Lasso(max_iter=5000),
    "elastic":     ElasticNet(max_iter=5000),
    "sgd_reg":     SGDRegressor(max_iter=5000, random_state=42),
    "knn_reg":     KNeighborsRegressor(n_neighbors=5),
    "dt_reg":      DecisionTreeRegressor(random_state=42),
    "svr":         SVR(),
    "mlp_reg":     MLPRegressor(max_iter=5000, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=42),
    "rf_reg":      RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
    "et_reg":      ExtraTreesRegressor(n_estimators=100, n_jobs=-1, random_state=42),
    "ada_reg":     AdaBoostRegressor(n_estimators=100, random_state=42),
    "bag_reg":     BaggingRegressor(n_estimators=20, n_jobs=-1, random_state=42),
    "gb_reg":      HistGradientBoostingRegressor(random_state=42),
}  # 15 regression models

if LGBMClassifier is not None:
    CLASSIFICATION_MODELS["lgbm_clf"] = LGBMClassifier(
        n_estimators=100, random_state=42, verbose=-1
    )
    REGRESSION_MODELS["lgbm_reg"] = LGBMRegressor(
        n_estimators=100, random_state=42, verbose=-1
    )

if XGBClassifier is not None:
    CLASSIFICATION_MODELS["xgb_clf"] = XGBClassifier(
        n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0
    )
    REGRESSION_MODELS["xgb_reg"] = XGBRegressor(
        n_estimators=100, random_state=42, verbosity=0
    )


def get_models(
    problem_type: str,
    model_names: Optional[List[str]] = None,
    n_samples: int = 0
) -> Dict[str, Any]:
    # Return a dict of ``{name: estimator_instance}`` for the requested problem type.
    if problem_type == "regression":
        catalogue = REGRESSION_MODELS
    else:
        catalogue = CLASSIFICATION_MODELS

    if model_names is None or "all" in model_names:
        selected = {k: clone(v) for k, v in catalogue.items()}
    else:
        selected = {k: clone(v) for k, v in catalogue.items() if k in model_names}
        if not selected:
            print(f"  [Trainer] WARNING: None of {model_names} found in "
                  f"{problem_type} catalogue. Falling back to full catalogue.")
            selected = {k: clone(v) for k, v in catalogue.items()}

    # Skip SVC for large datasets
    if n_samples > 5000:
        selected.pop("svc", None)
        selected.pop("svr", None)

    print(f"[Trainer] Selected models: {list(selected.keys())}")
    return selected


def create_model(model_name: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """Create an un-fitted instance of a model by its name, with optional custom parameters."""
    catalogue = CLASSIFICATION_MODELS if model_name in CLASSIFICATION_MODELS else REGRESSION_MODELS
    if model_name not in catalogue:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model = clone(catalogue[model_name])
    if params:
        model.set_params(**params)
    return model


def _configure_estimator(name: str, estimator: Any, y: Any) -> Any:
    """Apply target-dependent settings that cannot live in the static catalog."""
    if name == "xgb_clf":
        n_classes = len(np.unique(y))
        estimator.set_params(
            eval_metric="mlogloss" if n_classes > 2 else "logloss"
        )
    return estimator


# Helper for training with custom CV and early stopping
def _train_and_evaluate(
    name: str,
    estimator: Any,
    preprocessor: ColumnTransformer,
    X: np.ndarray,
    y: np.ndarray,
    problem_type: str,
    cv: int,
    start: float,
    max_time_seconds: Optional[float],
    refit_full: bool = False,
) -> Tuple[float, Optional[Pipeline]]:
    from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
    from sklearn.base import clone
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.pipeline import Pipeline
    import scipy.sparse
    
    cv_scores = []
    is_classification = problem_type == "classification"
    
    # Removed redundant LabelEncoder here, doing it per-fold to handle missing classes in CV splits
            
    cv_val = min(cv, len(X))
    
    if cv_val <= 1:
        # Single validation split for large datasets (speed and early stopping)
        if is_classification:
            unique, counts = np.unique(y, return_counts=True)
            if counts.min() < 2:
                splits = [train_test_split(np.arange(len(X)), test_size=0.15, random_state=42)]
            else:
                splits = [train_test_split(np.arange(len(X)), test_size=0.15, random_state=42, stratify=y)]
        else:
            splits = [train_test_split(np.arange(len(X)), test_size=0.15, random_state=42)]
    else:
        if is_classification:
            unique, counts = np.unique(y, return_counts=True)
            if counts.min() < cv_val:
                kf = KFold(n_splits=cv_val, shuffle=True, random_state=42)
            else:
                kf = StratifiedKFold(n_splits=cv_val, shuffle=True, random_state=42)
        else:
            kf = KFold(n_splits=cv_val, shuffle=True, random_state=42)
            
        splits = list(kf.split(X, y))
            
    best_pipe = None
    cv_scores = []
    fold_times = []
    
    print(f"  [Trainer] Commencing training loop for '{name}'...")
    for fold, (train_idx, val_idx) in enumerate(splits):
        if max_time_seconds and (time.time() - start) > max_time_seconds:
            print(f"    [Timeout] Stopping {name} early due to time budget.")
            break
            
        X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        y_tr = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
        X_va = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
        y_va = y.iloc[val_idx] if hasattr(y, "iloc") else y[val_idx]
        
        prep = clone(preprocessor)
        X_tr_prep = prep.fit_transform(X_tr, y_tr)
        
        if scipy.sparse.issparse(X_tr_prep) and name == "gb":
            print(f"    Skipping '{name}' as transformations yielded sparse matrix.")
            return -float('inf'), None
            
        X_va_prep = prep.transform(X_va)
        
        fit_kwargs = {}
        if is_classification:
            from sklearn.preprocessing import LabelEncoder
            le_fold = LabelEncoder()
            y_tr = le_fold.fit_transform(y_tr)
            
        if name in ["lgbm_clf", "lgbm_reg", "xgb_clf", "xgb_reg"]:
            if is_classification:
                # Filter eval_set to only contain classes present in y_tr
                valid_mask = np.isin(y_va, le_fold.classes_)
                y_va_eval = le_fold.transform(np.array(y_va)[valid_mask])
                
                if scipy.sparse.issparse(X_va_prep):
                    X_va_eval = X_va_prep.tocsr()[valid_mask]
                else:
                    X_va_eval = X_va_prep[valid_mask]
                fit_kwargs["eval_set"] = [(X_va_eval, y_va_eval)]
            else:
                fit_kwargs["eval_set"] = [(X_va_prep, y_va)]
            if name in ["lgbm_clf", "lgbm_reg"]:
                try:
                    from lightgbm import early_stopping, log_evaluation
                    fit_kwargs["callbacks"] = [early_stopping(stopping_rounds=10, verbose=True), log_evaluation(period=10)]
                except ImportError:
                    pass
            elif name in ["xgb_clf", "xgb_reg"]:
                fit_kwargs["verbose"] = 10
                
        est = _configure_estimator(name, clone(estimator), y_tr)
        start_fold = time.time()
        est.fit(X_tr_prep, y_tr, **fit_kwargs)
        fold_times.append(time.time() - start_fold)
        
        y_pred = est.predict(X_va_prep)
        if is_classification:
            y_pred = le_fold.inverse_transform(y_pred)
            sc = accuracy_score(y_va, y_pred)
        else:
            sc = -np.sqrt(mean_squared_error(y_va, y_pred))
        cv_scores.append(sc)
        
        if cv_val <= 1:
            best_pipe = Pipeline([("preprocessor", prep), ("model", est)])
            
    if not cv_scores:
        return -float('inf'), None, 0.0
        
    mean_score = float(np.mean(cv_scores))
    avg_fit_time = sum(fold_times) / len(fold_times) if fold_times else 0.0
    
    if cv_val > 1 and refit_full:
        print(f"  [Trainer] Refitting {name} on ALL data...")
        prep = clone(preprocessor)
        X_prep = prep.fit_transform(X, y)
        est = _configure_estimator(name, clone(estimator), y)
        est.fit(X_prep, y)
        best_pipe = Pipeline([("preprocessor", prep), ("model", est)])
        
    return mean_score, best_pipe, avg_fit_time


# Baseline screening

def baseline_screen(
    models: Dict[str, Any],
    preprocessor: ColumnTransformer,
    X: np.ndarray,
    y: np.ndarray,
    problem_type: str,
    sample_frac: float = 0.3,
    cv: int = 5,
    random_state: int = 42,
    max_time_seconds: Optional[float] = None,
) -> Tuple[Dict[str, Any], Dict[str, dict]]:
    # Quick evaluation of all candidate models on a data subsample.
    
    rng = np.random.RandomState(random_state)
    n_sample = max(int(len(X) * sample_frac), 50)
    idx = rng.choice(len(X), size=min(n_sample, len(X)), replace=False)
    X_sub = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
    y_sub = y.iloc[idx] if hasattr(y, "iloc") else y[idx]

    scores: Dict[str, dict] = {}
    start = time.time()

    print(f"\n[Baseline] Screening on {len(X_sub)} samples ({sample_frac:.0%} subsample)…")
    for name, estimator in models.items():
        if max_time_seconds and (time.time() - start) > max_time_seconds:
            print(f"[Baseline] Time budget exhausted – skipping '{name}'.")
            break

        try:
            mean_score, _, avg_fit_time = _train_and_evaluate(
                name, estimator, preprocessor, X_sub, y_sub, problem_type,
                cv, start, max_time_seconds, refit_full=False,
            )
        except Exception as exc:
            print(f"  [Baseline] Skipping '{name}' after training failed: {exc}")
            continue
        if mean_score > -float('inf'):
            scores[name] = {'score': mean_score, 'time': avg_fit_time}
            print(f"  {name:>12s}  baseline score = {mean_score:.4f}, time = {avg_fit_time:.3f}s")

    if not scores:
        return models, scores

    best_score = max(v['score'] for v in scores.values())
    worst_score = min(v['score'] for v in scores.values())
    score_range = best_score - worst_score

    if score_range == 0:
        return {n: models[n] for n in scores}, scores

    threshold = best_score - 0.70 * score_range

    promising = {
        name: models[name]
        for name, res in scores.items()
        if res['score'] >= threshold
    }
    dropped = set(scores) - set(promising)
    if dropped:
        print(f"[Baseline] Dropped underperforming model(s): {dropped}")

    return promising, scores


# Full training

def full_train(
    models: Dict[str, Any],
    preprocessor: ColumnTransformer,
    X: np.ndarray,
    y: np.ndarray,
    problem_type: str,
    cv: int = 5,
    max_time_seconds: Optional[float] = None,
) -> Tuple[Dict[str, Pipeline], Dict[str, float]]:
    # Train each model on the full training set with cross‑validation or single-split.
    trained: Dict[str, Pipeline] = {}
    scores: Dict[str, float] = {}
    start = time.time()

    print(f"\n[FullTrain] Training on {len(X)} samples…")
    for name, estimator in models.items():
        if max_time_seconds and (time.time() - start) > max_time_seconds:
            print(f"[FullTrain] Time budget exhausted – skipping '{name}'.")
            break

        try:
            mean_score, pipe, _ = _train_and_evaluate(
                name, estimator, preprocessor, X, y, problem_type,
                cv, start, max_time_seconds, refit_full=True,
            )
        except Exception as exc:
            print(f"  [FullTrain] Skipping '{name}' after training failed: {exc}")
            continue
        if pipe is not None and mean_score > -float('inf'):
            scores[name] = mean_score
            trained[name] = pipe
            metric_name = "validation accuracy" if problem_type == "classification" else "validation score"
            print(f"  {name:>12s}  {metric_name} = {mean_score:.4f}")

    return trained, scores
