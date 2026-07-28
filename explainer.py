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
from sklearn.pipeline import Pipeline


# Feature importance (built‑in or permutation)

def _extract_model(pipeline: Pipeline) -> Any:
    """Get the final estimator from a Pipeline."""
    estimator = getattr(pipeline, "estimator", pipeline)
    if hasattr(estimator, "named_steps"):
        return estimator.named_steps.get("model", estimator)
    return estimator


def _get_feature_names(pipeline: Pipeline, fallback_n: int) -> List[str]:
    """Try to extract feature names from the preprocessor."""
    try:
        estimator = getattr(pipeline, "estimator", pipeline)
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
    estimator = getattr(pipeline, "estimator", pipeline)
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
    else:
        print("[Explainer] Using permutation importance (may take a moment)…")
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
    print(f"[Explainer] Saved feature importance → {path}")
    return path


# SHAP explanations

def _shap_explain(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    max_samples: int = 500,
):
    # Compute SHAP values (returns explainer, shap_values, X_sample).
    import shap

    preprocessor = pipeline.named_steps.get("preprocessor", pipeline[0])
    model = _extract_model(pipeline)

    X_sample = X_test.iloc[:max_samples].copy()
    X_transformed = preprocessor.transform(X_sample)
    feature_names = _get_feature_names(pipeline, X_transformed.shape[1])

    if isinstance(X_transformed, np.ndarray):
        X_df = pd.DataFrame(X_transformed, columns=feature_names)
    else:
        X_df = pd.DataFrame(
            X_transformed.toarray() if hasattr(X_transformed, "toarray")
            else X_transformed,
            columns=feature_names,
        )

    # Choose appropriate SHAP explainer
    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
    else:
        # Use a small background sample for KernelExplainer
        bg = shap.sample(X_df, min(100, len(X_df)))
        explainer = shap.KernelExplainer(model.predict, bg)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(X_df)

    return shap_values, X_df, feature_names


def plot_shap_summary(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    path: str,
    max_samples: int = 500,
) -> str:
    # Save a SHAP summary (beeswarm) plot.
    import shap

    shap_values, X_df, _ = _shap_explain(pipeline, X_test, max_samples)

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
    print(f"[Explainer] Saved SHAP summary → {path}")
    return path


def plot_shap_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    path: str,
    max_samples: int = 500,
) -> str:
    # Save a SHAP feature‑importance bar plot.
    import shap

    shap_values, X_df, _ = _shap_explain(pipeline, X_test, max_samples)

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
    print(f"[Explainer] Saved SHAP importance → {path}")
    return path


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

    paths["feature_importance_path"] = plot_feature_importance(
        pipeline, X_test, y_test,
        os.path.join(output_dir, "feature_importance.png"),
    )

    if use_shap:
        try:
            paths["shap_summary_path"] = plot_shap_summary(
                pipeline, X_test,
                os.path.join(output_dir, "shap_summary.png"),
                max_samples=shap_samples,
            )
            paths["shap_importance_path"] = plot_shap_importance(
                pipeline, X_test,
                os.path.join(output_dir, "shap_importance.png"),
                max_samples=shap_samples,
            )
        except Exception as e:
            print(f"[Explainer] SHAP failed: {e}")
            paths["shap_summary_path"] = ""
            paths["shap_importance_path"] = ""
    else:
        print("[Explainer] SHAP skipped (--no-shap).")
        paths["shap_summary_path"] = ""
        paths["shap_importance_path"] = ""

    return paths
