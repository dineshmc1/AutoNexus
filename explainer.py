"""
explainer.py
Model explainability: built‑in / permutation feature importance and
optional SHAP explanations.

Plots are saved as PNG images in a configurable output directory
(default ``reports/explanations/``).
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline


# Feature importance (built‑in or permutation)

def _unwrap_estimator(pipeline: Any) -> Any:
    """Remove calibration wrappers while preserving sklearn pipelines."""
    estimator = pipeline
    seen: set[int] = set()
    while (
        not hasattr(estimator, "named_steps")
        and hasattr(estimator, "estimator")
        and id(estimator) not in seen
    ):
        seen.add(id(estimator))
        estimator = estimator.estimator
    return estimator


def _extract_model(pipeline: Pipeline) -> Any:
    """Get the final estimator from a Pipeline."""
    estimator = _unwrap_estimator(pipeline)
    if hasattr(estimator, "named_steps"):
        return estimator.named_steps.get("model", estimator)
    return estimator


def _ensemble_members(pipeline: Any) -> list[tuple[str, Any]]:
    """Return fitted ensemble members hidden behind calibration wrappers."""
    estimator = _unwrap_estimator(pipeline)
    members = getattr(estimator, "models", None)
    if not isinstance(members, list):
        return []
    return [
        (str(name), member)
        for name, member in members
        if hasattr(member, "predict")
    ]


def _shap_reference_pipeline(pipeline: Any) -> tuple[Any, str, str]:
    """Choose and disclose a compatible reference for ensemble SHAP."""
    members = _ensemble_members(pipeline)
    if not members:
        return pipeline, "selected_model", ""
    for capability in ("feature_importances_", "coef_"):
        for name, member in members:
            if hasattr(_extract_model(member), capability):
                return member, "ensemble_member_reference", name
    name, member = members[0]
    return member, "ensemble_member_reference", name


def _postfit_permutation_importance(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_repeats: int = 3,
) -> np.ndarray:
    """Permutation importance for immutable fitted wrappers without ``fit``."""
    target = np.asarray(y)
    is_classifier = hasattr(estimator, "predict_proba") or hasattr(
        estimator, "classes_"
    )
    scorer = accuracy_score if is_classifier else r2_score
    baseline = float(scorer(target, estimator.predict(X)))
    random = np.random.RandomState(42)
    importances = np.zeros(X.shape[1], dtype=float)
    for column_index, column in enumerate(X.columns):
        decreases = []
        original = X[column].to_numpy(copy=True)
        for _ in range(n_repeats):
            shuffled = X.copy()
            shuffled[column] = random.permutation(original)
            score = float(scorer(target, estimator.predict(shuffled)))
            decreases.append(baseline - score)
        importances[column_index] = float(np.mean(decreases))
    return importances


def _get_feature_names(pipeline: Pipeline, fallback_n: int) -> List[str]:
    """Try to extract feature names from the preprocessor."""
    try:
        estimator = _unwrap_estimator(pipeline)
        preprocessor = estimator.named_steps.get("preprocessor", estimator[0])
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"feature_{i}" for i in range(fallback_n)]


def plot_feature_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    path: str,
    top_n: int = 20,
) -> str:
    # Get the final estimator from a Pipeline.
    estimator = _unwrap_estimator(pipeline)
    model = _extract_model(pipeline)
    if hasattr(estimator, "named_steps"):
        X_transformed = estimator.named_steps.get(
            "preprocessor", estimator[0]
        ).transform(X_test)
    else:
        model = pipeline
        X_transformed = X_test
    feature_names = _get_feature_names(pipeline, X_transformed.shape[1])

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        method = "Built‑in"
    elif not hasattr(estimator, "named_steps"):
        print("[Explainer] Using post-fit permutation importance...")
        n_samples = min(2000, X_test.shape[0])
        X_sample = X_test.iloc[:n_samples].copy()
        y_sample = y_test.iloc[:n_samples]
        importances = _postfit_permutation_importance(
            pipeline, X_sample, y_sample
        )
        feature_names = [str(column) for column in X_sample.columns]
        method = "Calibrated ensemble permutation"
    else:
        print("[Explainer] Using permutation importance (may take a moment)...")
        # Subsample for speed and memory safety when dense conversion is required
        n_samples = min(10000, X_transformed.shape[0])
        idx = np.random.RandomState(42).choice(X_transformed.shape[0], n_samples, replace=False)
        X_sample = X_transformed.iloc[idx] if hasattr(X_transformed, "iloc") else X_transformed[idx]
        y_sample = y_test.iloc[idx] if hasattr(y_test, "iloc") else y_test.values[idx]
        
        import scipy.sparse
        if scipy.sparse.issparse(X_sample):
            X_sample = X_sample.toarray()
            
        result = permutation_importance(
            model, X_sample, y_sample,
            n_repeats=5, random_state=42, n_jobs=-1,
        )
        importances = result.importances_mean
        method = "Permutation"

    # Sort and take top N
    idx = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in idx]
    top_vals = importances[idx]

    fig, ax = plt.subplots(figsize=(8, max(4, len(top_names) * 0.35)))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_names)))[::-1]
    ax.barh(range(len(top_names)), top_vals[::-1], color=colors)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance ({method})", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Explainer] Saved feature importance -> {path}")
    return path


# SHAP explanations

def _shap_explain(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    max_samples: int = 500,
):
    # Compute SHAP values (returns explainer, shap_values, X_sample).
    import shap

    reference, scope, reference_name = _shap_reference_pipeline(pipeline)
    estimator = _unwrap_estimator(reference)
    if not hasattr(estimator, "named_steps"):
        raise TypeError(
            "SHAP requires a fitted preprocessing pipeline for this model."
        )
    preprocessor = estimator.named_steps.get("preprocessor", estimator[0])
    model = _extract_model(reference)

    X_sample = X_test.iloc[:max_samples].copy()
    X_transformed = preprocessor.transform(X_sample)
    feature_names = _get_feature_names(reference, X_transformed.shape[1])

    if isinstance(X_transformed, np.ndarray):
        X_df = pd.DataFrame(X_transformed, columns=feature_names)
    else:
        X_df = pd.DataFrame(
            X_transformed.toarray() if hasattr(X_transformed, "toarray")
            else X_transformed,
            columns=feature_names,
        )

    background = X_df.iloc[: min(100, len(X_df))]
    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
        explained = explainer(X_df)
        shap_values = explained.values
        base_values = explained.base_values
    elif hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, background)
        explained = explainer(X_df)
        shap_values = explained.values
        base_values = explained.base_values
    else:
        predict = (
            model.predict_proba
            if hasattr(model, "predict_proba")
            else model.predict
        )
        explainer = shap.Explainer(predict, background)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explained = explainer(X_df)
            shap_values = explained.values
            base_values = explained.base_values

    values = np.asarray(shap_values)
    if values.ndim == 3:
        class_strength = np.mean(np.abs(values), axis=(0, 1))
        class_index = int(np.argmax(class_strength))
        values = values[:, :, class_index]
        base_array = np.asarray(base_values)
        base_value = float(
            base_array[0, class_index]
            if base_array.ndim == 2
            else base_array[class_index]
        )
    else:
        base_array = np.asarray(base_values, dtype=float)
        base_value = float(base_array.reshape(-1)[0])

    return (
        values,
        X_df,
        feature_names,
        base_value,
        scope,
        reference_name,
    )


def plot_shap_summary(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    path: str,
    max_samples: int = 500,
) -> str:
    # Save a SHAP summary (beeswarm) plot.
    import shap

    shap_values, X_df, _, _, scope, reference = _shap_explain(
        pipeline, X_test, max_samples
    )

    fig = plt.figure(figsize=(10, 6))
    # For multi-class, shap_values is a list — use the first class or mean
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1] if len(shap_values) == 2
                          else shap_values, X_df, show=False,
                          max_display=20)
    else:
        shap.summary_plot(shap_values, X_df, show=False, max_display=20)
    plt.title("SHAP Summary", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[Explainer] Saved SHAP summary -> {path}")
    return path


def plot_shap_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    path: str,
    max_samples: int = 500,
) -> str:
    # Save a SHAP feature‑importance bar plot.
    import shap

    shap_values, X_df, _, _, scope, reference = _shap_explain(
        pipeline, X_test, max_samples
    )

    fig = plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1] if len(shap_values) == 2
                          else shap_values, X_df, plot_type="bar",
                          show=False, max_display=20)
    else:
        shap.summary_plot(shap_values, X_df, plot_type="bar",
                          show=False, max_display=20)
    plt.title("SHAP Feature Importance", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[Explainer] Saved SHAP importance -> {path}")
    return path


def generate_shap_artifacts(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    output_dir: str,
    max_samples: int = 500,
) -> Dict[str, str]:
    """Compute SHAP once and render global plus local evidence."""
    import shap

    (
        values,
        X_df,
        feature_names,
        base_value,
        scope,
        reference_name,
    ) = _shap_explain(pipeline, X_test, max_samples)
    paths = {
        "shap_summary_path": os.path.join(output_dir, "shap_summary.png"),
        "shap_importance_path": os.path.join(output_dir, "shap_importance.png"),
        "shap_dependence_path": os.path.join(output_dir, "shap_dependence.png"),
        "shap_waterfall_path": os.path.join(output_dir, "shap_waterfall.png"),
        "shap_decision_path": os.path.join(output_dir, "shap_decision.png"),
    }
    shap.summary_plot(values, X_df, show=False, max_display=20)
    title = (
        f"Reference-member SHAP Summary: {reference_name}"
        if scope == "ensemble_member_reference"
        else "SHAP Summary"
    )
    plt.title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(paths["shap_summary_path"], dpi=150, bbox_inches="tight")
    plt.close("all")

    shap.summary_plot(
        values,
        X_df,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    title = (
        f"Reference-member SHAP Importance: {reference_name}"
        if scope == "ensemble_member_reference"
        else "SHAP Feature Importance"
    )
    plt.title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(paths["shap_importance_path"], dpi=150, bbox_inches="tight")
    plt.close("all")

    top_index = int(np.argmax(np.mean(np.abs(values), axis=0)))
    shap.dependence_plot(
        feature_names[top_index],
        values,
        X_df,
        show=False,
        interaction_index=None,
    )
    plt.title(f"SHAP Dependence: {feature_names[top_index]}", fontsize=13)
    plt.tight_layout()
    plt.savefig(paths["shap_dependence_path"], dpi=150, bbox_inches="tight")
    plt.close("all")

    local = shap.Explanation(
        values=values[0],
        base_values=base_value,
        data=X_df.iloc[0].to_numpy(),
        feature_names=feature_names,
    )
    shap.plots.waterfall(local, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(paths["shap_waterfall_path"], dpi=150, bbox_inches="tight")
    plt.close("all")

    shap.decision_plot(
        base_value,
        values[0],
        X_df.iloc[0],
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(paths["shap_decision_path"], dpi=150, bbox_inches="tight")
    plt.close("all")
    for path in paths.values():
        print(f"[Explainer] Saved SHAP evidence -> {path}")
    paths["shap_scope"] = scope
    paths["shap_reference_model"] = reference_name
    return paths


# Orchestrator
def run_explanations(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = "reports/explanations",
    use_shap: bool = True,
    shap_samples: int = 500,
) -> Dict[str, str]:
    # Generate all explanation plots and return paths.
    os.makedirs(output_dir, exist_ok=True)

    paths: Dict[str, str] = {}

    try:
        paths["feature_importance_path"] = plot_feature_importance(
            pipeline, X_test, y_test,
            os.path.join(output_dir, "feature_importance.png"),
        )
        paths["feature_importance_status"] = "generated"
        paths["feature_importance_error"] = ""
    except Exception as exc:
        print(f"[Explainer] Feature importance failed: {exc}")
        paths["feature_importance_path"] = ""
        paths["feature_importance_status"] = "failed"
        paths["feature_importance_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    if use_shap:
        try:
            paths.update(
                generate_shap_artifacts(
                    pipeline,
                    X_test,
                    output_dir,
                    max_samples=shap_samples,
                )
            )
            paths["shap_status"] = "generated"
            paths["shap_error"] = ""
        except Exception as e:
            print(f"[Explainer] SHAP failed: {e}")
            paths["shap_summary_path"] = ""
            paths["shap_importance_path"] = ""
            paths["shap_dependence_path"] = ""
            paths["shap_waterfall_path"] = ""
            paths["shap_decision_path"] = ""
            paths["shap_status"] = "failed"
            paths["shap_error"] = f"{type(e).__name__}: {e}"
    else:
        print("[Explainer] SHAP skipped (--no-shap).")
        paths["shap_summary_path"] = ""
        paths["shap_importance_path"] = ""
        paths["shap_dependence_path"] = ""
        paths["shap_waterfall_path"] = ""
        paths["shap_decision_path"] = ""
        paths["shap_status"] = "disabled"
        paths["shap_error"] = "SHAP was disabled for this run."

    return paths
