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
from sklearn.model_selection import train_test_split
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


def _prepare_fold(
    preprocessor: ColumnTransformer,
    X_train,
    y_train,
    X_validation,
):
    """Fit preprocessing on one fold and transform both fold partitions."""
    fitted = clone(preprocessor)
    transformed_train = fitted.fit_transform(X_train, y_train)
    transformed_validation = fitted.transform(X_validation)
    return fitted, transformed_train, transformed_validation


def _prepare_full(preprocessor: ColumnTransformer, X, y):
    fitted = clone(preprocessor)
    return fitted, fitted.fit_transform(X, y)


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
    "et_clf":      ExtraTreesClassifier(
        n_estimators=200,
        max_depth=24,
        min_samples_split=6,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        max_samples=0.85,
        oob_score=True,
        n_jobs=-1,
        random_state=42,
    ),
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
    "et_reg":      ExtraTreesRegressor(
        n_estimators=200,
        max_depth=24,
        min_samples_split=6,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        max_samples=0.85,
        oob_score=True,
        n_jobs=-1,
        random_state=42,
    ),
    "ada_reg":     AdaBoostRegressor(n_estimators=100, random_state=42),
    "bag_reg":     BaggingRegressor(n_estimators=20, n_jobs=-1, random_state=42),
    "gb_reg":      HistGradientBoostingRegressor(random_state=42),
}  # 15 regression models

_CUSTOM_MODELS: Dict[str, Dict[str, Any]] = {
    "classification": {},
    "regression": {},
}


def register_model(
    name: str,
    estimator: Any,
    *,
    problem_type: str = "classification",
) -> None:
    """Register a cloneable custom estimator for AutoNexus runs."""
    if problem_type not in _CUSTOM_MODELS:
        raise ValueError("problem_type must be classification or regression")
    if not name or not isinstance(name, str):
        raise ValueError("Custom model name must be a non-empty string.")
    clone(estimator)
    _CUSTOM_MODELS[problem_type][name] = estimator

def _get_catalogue(
    problem_type: str,
    requested: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build the catalogue while importing optional boosters only on demand."""
    catalogue = dict(
        REGRESSION_MODELS
        if problem_type == "regression"
        else CLASSIFICATION_MODELS
    )
    catalogue.update(_CUSTOM_MODELS[problem_type])
    wanted = set(requested or ["all"])
    load_all = "all" in wanted

    if load_all or wanted & {"lgbm_clf", "lgbm_reg"}:
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor

            if problem_type == "classification":
                catalogue["lgbm_clf"] = LGBMClassifier(
                    n_estimators=300, random_state=42, verbose=-1
                )
            else:
                catalogue["lgbm_reg"] = LGBMRegressor(
                    n_estimators=300, random_state=42, verbose=-1
                )
        except ImportError:
            pass

    if load_all or wanted & {"xgb_clf", "xgb_reg"}:
        try:
            from xgboost import XGBClassifier, XGBRegressor

            if problem_type == "classification":
                catalogue["xgb_clf"] = XGBClassifier(
                    n_estimators=300,
                    random_state=42,
                    eval_metric="logloss",
                    verbosity=0,
                )
            else:
                catalogue["xgb_reg"] = XGBRegressor(
                    n_estimators=300, random_state=42, verbosity=0
                )
        except ImportError:
            pass
    return catalogue


def get_models(
    problem_type: str,
    model_names: Optional[List[str]] = None,
    n_samples: int = 0
) -> Dict[str, Any]:
    # Return a dict of ``{name: estimator_instance}`` for the requested problem type.
    catalogue = _get_catalogue(problem_type, model_names)

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
    problem_type = (
        "classification"
        if model_name.endswith("_clf")
        or model_name in CLASSIFICATION_MODELS
        else "regression"
    )
    catalogue = _get_catalogue(problem_type, [model_name])
    if model_name not in catalogue:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model = clone(catalogue[model_name])
    if params:
        model.set_params(**params)
    return model


def _configure_estimator(
    name: str,
    estimator: Any,
    y: Any,
    use_early_stopping: bool = False,
) -> Any:
    """Apply target-dependent settings that cannot live in the static catalog."""
    if name == "xgb_clf":
        n_classes = len(np.unique(y))
        estimator.set_params(
            eval_metric="mlogloss" if n_classes > 2 else "logloss"
        )
    if use_early_stopping and name in {"xgb_clf", "xgb_reg"}:
        estimator.set_params(early_stopping_rounds=20)
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
    preprocessing_memory: Any = None,
    groups: Optional[np.ndarray] = None,
    random_state: int = 42,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Optional[Pipeline], float]:
    from sklearn.model_selection import (
        GroupKFold,
        KFold,
        StratifiedGroupKFold,
        StratifiedKFold,
        train_test_split,
    )
    from sklearn.base import clone
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.pipeline import Pipeline
    import scipy.sparse
    
    model_started = time.monotonic()
    cv_scores = []
    is_classification = problem_type == "classification"
    
    # Removed redundant LabelEncoder here, doing it per-fold to handle missing classes in CV splits
            
    cv_val = min(cv, len(X))
    groups_array = None if groups is None else np.asarray(groups)
    if groups_array is not None and len(groups_array) != len(X):
        raise ValueError("CV groups must align with the training rows.")
    if is_classification:
        _, class_counts = np.unique(y, return_counts=True)
        cv_val = min(cv_val, int(class_counts.min()))
        if groups_array is not None:
            y_array = np.asarray(y)
            class_group_counts = [
                len(np.unique(groups_array[y_array == label]))
                for label in np.unique(y_array)
            ]
            cv_val = min(cv_val, min(class_group_counts, default=1))
            if cv_val <= 1:
                groups_array = None
    elif groups_array is not None:
        cv_val = min(cv_val, len(np.unique(groups_array)))
        if cv_val <= 1:
            groups_array = None
    
    if cv_val <= 1:
        # Single validation split for large datasets (speed and early stopping)
        if is_classification:
            splits = [
                train_test_split(
                    np.arange(len(X)),
                    test_size=0.15,
                    random_state=random_state,
                    stratify=y,
                )
            ]
        else:
            splits = [
                train_test_split(
                    np.arange(len(X)),
                    test_size=0.15,
                    random_state=random_state,
                )
            ]
    else:
        if is_classification and groups_array is not None:
            kf = StratifiedGroupKFold(
                n_splits=cv_val,
                shuffle=True,
                random_state=random_state,
            )
            splits = list(kf.split(X, y, groups_array))
        elif is_classification:
            kf = StratifiedKFold(
                n_splits=cv_val, shuffle=True, random_state=random_state
            )
            splits = list(kf.split(X, y))
        elif groups_array is not None:
            kf = GroupKFold(n_splits=cv_val)
            splits = list(kf.split(X, y, groups_array))
        else:
            kf = KFold(
                n_splits=cv_val, shuffle=True, random_state=random_state
            )
            splits = list(kf.split(X, y))
            
    best_pipe = None
    cv_scores = []
    fold_times = []
    
    print(f"  [Trainer] Commencing training loop for '{name}'...")
    for fold, (train_idx, val_idx) in enumerate(splits):
        if max_time_seconds and (time.monotonic() - start) > max_time_seconds:
            print(f"    [Timeout] Stopping {name} early due to time budget.")
            break
            
        X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        y_tr = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
        X_va = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
        y_va = y.iloc[val_idx] if hasattr(y, "iloc") else y[val_idx]
        
        prepare_fold = (
            preprocessing_memory.cache(_prepare_fold)
            if preprocessing_memory is not None
            else _prepare_fold
        )
        prep, X_tr_prep, X_va_prep = prepare_fold(
            preprocessor, X_tr, y_tr, X_va
        )
        
        if scipy.sparse.issparse(X_tr_prep) and name == "gb":
            print(f"    Skipping '{name}' as transformations yielded sparse matrix.")
            return -float('inf'), None, 0.0
            
        fit_kwargs = {}
        if name in ["lgbm_clf", "lgbm_reg", "xgb_clf", "xgb_reg"]:
            if is_classification:
                fit_kwargs["eval_set"] = [(X_va_prep, y_va)]
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
                
        est = _configure_estimator(
            name,
            clone(estimator),
            y_tr,
            use_early_stopping=name in {"xgb_clf", "xgb_reg"},
        )
        start_fold = time.monotonic()
        est.fit(X_tr_prep, y_tr, **fit_kwargs)
        fold_times.append(time.monotonic() - start_fold)
        
        y_pred = est.predict(X_va_prep)
        if is_classification:
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
    
    refit_seconds = 0.0
    if cv_val > 1 and refit_full:
        print(f"  [Trainer] Refitting {name} on ALL data...")
        prepare_full = (
            preprocessing_memory.cache(_prepare_full)
            if preprocessing_memory is not None
            else _prepare_full
        )
        prep, X_prep = prepare_full(preprocessor, X, y)
        est = _configure_estimator(name, clone(estimator), y)
        refit_started = time.monotonic()
        est.fit(X_prep, y)
        refit_seconds = time.monotonic() - refit_started
        best_pipe = Pipeline([("preprocessor", prep), ("model", est)])

    if diagnostics is not None:
        observed_ram_mb = None
        try:
            import psutil

            observed_ram_mb = psutil.Process().memory_info().rss / (1024**2)
        except ImportError:
            pass
        diagnostics.update(
            cv_mean=mean_score,
            cv_std=float(np.std(cv_scores)),
            cv_scores=[float(value) for value in cv_scores],
            folds_completed=len(cv_scores),
            average_fold_fit_seconds=avg_fit_time,
            refit_seconds=refit_seconds,
            total_seconds=time.monotonic() - model_started,
            observed_process_ram_mb=observed_ram_mb,
        )
        
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
    preprocessing_cache_dir: Optional[str] = None,
    groups: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Any], Dict[str, dict]]:
    # Quick evaluation of all candidate models on a data subsample.
    
    n_sample = max(int(len(X) * sample_frac), 50)
    n_sample = min(n_sample, len(X))
    if problem_type == "classification" and n_sample < len(X):
        all_indices = np.arange(len(X))
        try:
            idx, _ = train_test_split(
                all_indices,
                train_size=n_sample,
                random_state=random_state,
                stratify=y,
            )
        except ValueError:
            idx = np.random.RandomState(random_state).choice(
                len(X), size=n_sample, replace=False
            )
    else:
        idx = np.random.RandomState(random_state).choice(
            len(X), size=n_sample, replace=False
        )
    X_sub = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
    y_sub = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
    groups_sub = None if groups is None else np.asarray(groups)[idx]

    scores: Dict[str, dict] = {}
    start = time.monotonic()
    preprocessing_memory = None
    if preprocessing_cache_dir:
        from joblib import Memory

        preprocessing_memory = Memory(
            preprocessing_cache_dir, verbose=0
        )

    print(f"\n[Baseline] Screening on {len(X_sub)} samples ({sample_frac:.0%} subsample)…")
    for name, estimator in models.items():
        if max_time_seconds and (time.monotonic() - start) > max_time_seconds:
            print(f"[Baseline] Time budget exhausted – skipping '{name}'.")
            break

        try:
            mean_score, _, avg_fit_time = _train_and_evaluate(
                name, estimator, preprocessor, X_sub, y_sub, problem_type,
                cv, start, max_time_seconds, refit_full=False,
                preprocessing_memory=preprocessing_memory,
                groups=groups_sub,
                random_state=random_state,
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
    preprocessing_cache_dir: Optional[str] = None,
    groups: Optional[np.ndarray] = None,
    random_state: int = 42,
    diagnostics: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Pipeline], Dict[str, float]]:
    # Train each model on the full training set with cross‑validation or single-split.
    trained: Dict[str, Pipeline] = {}
    scores: Dict[str, float] = {}
    start = time.monotonic()
    preprocessing_memory = None
    if preprocessing_cache_dir:
        from joblib import Memory

        preprocessing_memory = Memory(
            preprocessing_cache_dir, verbose=0
        )

    print(f"\n[FullTrain] Training on {len(X)} samples…")
    for name, estimator in models.items():
        if max_time_seconds and (time.monotonic() - start) > max_time_seconds:
            print(f"[FullTrain] Time budget exhausted – skipping '{name}'.")
            break

        model_diagnostics: Dict[str, Any] = {}
        try:
            mean_score, pipe, _ = _train_and_evaluate(
                name, estimator, preprocessor, X, y, problem_type,
                cv, start, max_time_seconds, refit_full=True,
                preprocessing_memory=preprocessing_memory,
                groups=groups,
                random_state=random_state,
                diagnostics=model_diagnostics,
            )
        except Exception as exc:
            print(f"  [FullTrain] Skipping '{name}' after training failed: {exc}")
            continue
        if pipe is not None and mean_score > -float('inf'):
            if diagnostics is not None:
                diagnostics[name] = model_diagnostics
            scores[name] = mean_score
            trained[name] = pipe
            metric_name = "validation accuracy" if problem_type == "classification" else "validation score"
            print(f"  {name:>12s}  {metric_name} = {mean_score:.4f}")

    return trained, scores
