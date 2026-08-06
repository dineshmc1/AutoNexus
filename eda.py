"""
eda.py
Exploratory Data Analysis: dataset summary, target and feature
distribution plots, and correlation heatmap.

All visualisations are saved as PNG images in a configurable output
directory (default ``reports/eda/``).
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Dataset summary

def generate_summary(
    df: pd.DataFrame,
    target_col: str,
) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"],
    ).columns.tolist()

    missing = df.isnull().sum()
    missing = missing[missing > 0].to_dict()

    return {
        "rows": len(df),
        "cols": df.shape[1],
        "missing": missing,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "target_col": target_col,
        "target_unique": int(df[target_col].nunique()),
    }


# Plots

def plot_target_distribution(
    y: pd.Series,
    path: str,
    problem_type: str = "classification",
) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    if problem_type == "classification":
        counts = y.value_counts().sort_index()
        colors = sns.color_palette("viridis", len(counts))
        counts.plot.bar(ax=ax, color=colors, edgecolor="white")
        ax.set_ylabel("Count")
    else:
        ax.hist(y, bins=40, color="#4c72b0", edgecolor="white")
        ax.set_ylabel("Frequency")
    ax.set_title("Target Distribution")
    ax.set_xlabel(y.name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[EDA] Saved target distribution -> {path}")
    return path


def plot_feature_distributions(
    X: pd.DataFrame,
    path: str,
    max_features: int = 20,
) -> str:
    numeric = X.select_dtypes(include="number")
    cols_to_plot = numeric.columns[:max_features]
    n = len(cols_to_plot)
    if n == 0:
        print("[EDA] No numeric features to plot.")
        return ""

    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    palette = sns.color_palette("viridis", n)
    for i, col in enumerate(cols_to_plot):
        axes[i].hist(numeric[col].dropna(), bins=30, color=palette[i],
                      edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=10)
        axes[i].tick_params(labelsize=8)

    # hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions (numeric)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Saved feature distributions -> {path}")
    return path


def plot_correlation_heatmap(X: pd.DataFrame, path: str) -> str:
    numeric = X.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        print("[EDA] Not enough numeric features for a correlation heatmap.")
        return ""

    corr = numeric.corr()
    size = min(12, max(6, numeric.shape[1] * 0.6))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(
        corr, annot=numeric.shape[1] <= 15, fmt=".2f",
        cmap="coolwarm", center=0, square=True,
        linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Saved correlation heatmap -> {path}")
    return path


# Orchestrator

def run_eda(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    output_dir: str = "reports/eda",
) -> Dict[str, Any]:
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    summary = generate_summary(df, target_col)
    print(f"[EDA] Dataset: {summary['rows']} rows × {summary['cols']} cols")
    print(
        f"[EDA] Numeric: {len(summary['numeric_cols'])}, "
        f"Categorical: {len(summary['categorical_cols'])}"
    )
    if summary["missing"]:
        print(f"[EDA] Missing values: {summary['missing']}")
    else:
        print("[EDA] No missing values.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    target_path = plot_target_distribution(
        y, os.path.join(output_dir, "target_distribution.png"), problem_type,
    )
    feat_path = plot_feature_distributions(
        X, os.path.join(output_dir, "feature_distributions.png"),
    )
    corr_path = plot_correlation_heatmap(
        X, os.path.join(output_dir, "correlation_heatmap.png"),
    )

    return {
        "summary": summary,
        "target_dist_path": target_path,
        "feature_dist_path": feat_path,
        "corr_heatmap_path": corr_path,
    }
