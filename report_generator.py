"""
report_generator.py
Generates a self‑contained HTML report that embeds EDA plots, model
results, feature importance, SHAP explanations, and feature‑engineering
logs as base64 images.  Branded as **MLBuilder**.
"""

from __future__ import annotations

import base64
import html
import os
from typing import Any, Dict, List, Optional

import pandas as pd


# Helpers

def _img_to_base64(path: str) -> str:
    """Read an image file and return a base64‑encoded data URI."""
    if not path or not os.path.isfile(path):
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "svg": "image/svg+xml"}.get(ext, "image/png")
    return f"data:{mime};base64,{data}"


def _embed_image(path: str, alt: str = "", width: str = "100%") -> str:
    """Return an <img> tag with embedded base64 data, or empty string."""
    uri = _img_to_base64(path)
    if not uri:
        return ""
    return (
        f'<img src="{uri}" alt="{alt}" '
        f'style="max-width:{width}; height:auto; border-radius:8px; '
        f'margin:12px 0; box-shadow:0 2px 8px rgba(0,0,0,0.10);">'
    )


def _results_to_html(results: pd.DataFrame, problem_type: str) -> str:
    """Convert the metrics DataFrame to a styled HTML table."""
    if problem_type == "classification":
        cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    else:
        cols = ["model", "rmse", "mae", "r2"]
    display = results[[c for c in cols if c in results.columns]].copy()
    for col in display.columns:
        if col != "model" and display[col].dtype in ("float64", "float32"):
            display[col] = display[col].map(
                lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
            )
    return display.to_html(index=False, classes="metrics", border=0)


def _fe_log_to_html(fe_log: List[str]) -> str:
    """Render the feature‑engineering log as styled HTML."""
    if not fe_log:
        return '<p style="color:#94a3b8;">Feature engineering was not enabled.</p>'
    items = "".join(f"<li>{line}</li>" for line in fe_log)
    return f'<ul class="fe-log">{items}</ul>'


def _shap_fallback(explanation_paths: Dict[str, str]) -> str:
    status = explanation_paths.get("shap_status", "unavailable")
    reason = explanation_paths.get("shap_error", "")
    labels = {
        "disabled": "SHAP was not requested for this run.",
        "failed": "SHAP generation failed for the selected estimator.",
        "unavailable": "SHAP artifacts were not produced.",
    }
    message = labels.get(status, labels["unavailable"])
    detail = f"<br><code>{html.escape(reason)}</code>" if reason else ""
    return f'<p class="evidence-note"><strong>{message}</strong>{detail}</p>'


def _feature_importance_fallback(explanation_paths: Dict[str, str]) -> str:
    reason = explanation_paths.get("feature_importance_error", "")
    detail = f"<br><code>{html.escape(reason)}</code>" if reason else ""
    return (
        '<p class="evidence-note"><strong>Feature importance was not '
        f"produced.</strong>{detail}</p>"
    )


def _shap_scope_note(explanation_paths: Dict[str, str]) -> str:
    if explanation_paths.get("shap_scope") != "ensemble_member_reference":
        return ""
    member = html.escape(
        explanation_paths.get("shap_reference_model", "reference member")
    )
    return (
        '<p class="evidence-note"><strong>Scope:</strong> These SHAP plots '
        f"explain the fitted <code>{member}</code> ensemble member. They are "
        "not exact additive SHAP values for the calibrated probability "
        "ensemble.</p>"
    )


# CSS

_CSS = """
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #f4f6f9; color: #1e293b; line-height: 1.6;
  }
  .container { max-width: 960px; margin: 0 auto; padding: 24px; }
  header {
    background: linear-gradient(135deg, #1e3a5f, #3b82f6);
    color: #fff; padding: 36px 24px; text-align: center;
    border-radius: 0 0 16px 16px;
  }
  header h1 { font-size: 2rem; font-weight: 700; }
  header p  { opacity: 0.85; margin-top: 6px; }
  section {
    background: #fff; border-radius: 12px; padding: 28px;
    margin-top: 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  section h2 {
    font-size: 1.3rem; margin-bottom: 16px;
    border-bottom: 2px solid #3b82f6; padding-bottom: 6px;
    color: #1e3a5f;
  }
  table.metrics {
    width: 100%; border-collapse: collapse; margin-top: 12px;
    font-size: 0.95rem;
  }
  table.metrics th {
    background: #1e3a5f; color: #fff; padding: 10px 14px;
    text-align: left; text-transform: uppercase; font-size: 0.8rem;
    letter-spacing: 0.5px;
  }
  table.metrics td {
    padding: 10px 14px; border-bottom: 1px solid #e2e8f0;
  }
  table.metrics tr:nth-child(even) { background: #f8fafc; }
  table.metrics tr:hover { background: #eef2ff; }
  .summary-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 12px; margin-top: 8px;
  }
  .summary-card {
    background: #f8fafc; border-radius: 8px; padding: 14px;
    border-left: 4px solid #3b82f6;
  }
  .summary-card .label { font-size: 0.82rem; color: #64748b; }
  .summary-card .value { font-size: 1.15rem; font-weight: 600; }
  .best-badge {
    display: inline-block; background: #22c55e; color: #fff;
    padding: 4px 14px; border-radius: 20px; font-weight: 600;
    font-size: 0.95rem;
  }
  .img-center { text-align: center; }
  ul.fe-log {
    list-style: none; padding: 0; margin-top: 8px;
  }
  ul.fe-log li {
    background: #f0fdf4; border-left: 4px solid #22c55e;
    padding: 8px 14px; margin-bottom: 6px; border-radius: 6px;
    font-size: 0.92rem; font-family: 'Consolas', monospace;
  }
  .evidence-note {
    color: #475569; background: #f8fafc; border: 1px solid #cbd5e1;
    border-left: 4px solid #f59e0b; border-radius: 8px; padding: 14px;
    text-align: left;
  }
  .evidence-note code { color: #9a3412; overflow-wrap: anywhere; }
  footer {
    text-align: center; padding: 24px; color: #94a3b8;
    font-size: 0.82rem;
  }
</style>
"""


# Report builder

def generate_report(
    summary: Dict[str, Any],
    results: pd.DataFrame,
    best_name: str,
    problem_type: str,
    eda_paths: Dict[str, str],
    explanation_paths: Dict[str, str],
    output_path: str = "reports/report.html",
    fe_log: Optional[List[str]] = None,
) -> str:
    """
    Build a self‑contained HTML report and write it to *output_path*.

    Parameters
    ----------
    summary : dict
        From ``eda.generate_summary``.
    results : pd.DataFrame
        Model metrics table.
    best_name : str
        Name of the best model.
    problem_type : str
    eda_paths : dict
        Paths to EDA images.
    explanation_paths : dict
        Paths to explanation images.
    output_path : str
    fe_log : list[str] or None
        Log lines from ``FeatureEngineer``.

    Returns
    -------
    str  – absolute path to the written report.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    missing_info = (
        "<p>No missing values detected.</p>"
        if not summary.get("missing")
        else "<ul>" + "".join(
            f"<li><strong>{k}</strong>: {v} missing</li>"
            for k, v in summary["missing"].items()
        ) + "</ul>"
    )

    metric_label = "F1 Score" if problem_type == "classification" else "RMSE"
    best_score_col = "f1" if problem_type == "classification" else "rmse"
    best_row = results[results["model"] == best_name].iloc[0]
    best_score = f"{best_row[best_score_col]:.4f}"

    fe_section = ""
    if fe_log is not None:
        fe_section = f"""
  <!-- Feature Engineering -->
  <section>
    <h2>🔧 Feature Engineering</h2>
    {_fe_log_to_html(fe_log)}
  </section>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MLBuilder Report</title>
  {_CSS}
</head>
<body>

<header>
  <h1>MLBuilder Report</h1>
  <p>Automated Machine Learning · {problem_type.title()}</p>
</header>

<div class="container">

  <!-- Dataset Summary -->
  <section>
    <h2>📊 Dataset Summary</h2>
    <div class="summary-grid">
      <div class="summary-card">
        <div class="label">Rows</div>
        <div class="value">{summary['rows']:,}</div>
      </div>
      <div class="summary-card">
        <div class="label">Columns</div>
        <div class="value">{summary['cols']}</div>
      </div>
      <div class="summary-card">
        <div class="label">Numeric Features</div>
        <div class="value">{len(summary['numeric_cols'])}</div>
      </div>
      <div class="summary-card">
        <div class="label">Categorical Features</div>
        <div class="value">{len(summary['categorical_cols'])}</div>
      </div>
      <div class="summary-card">
        <div class="label">Target Column</div>
        <div class="value">{summary['target_col']}</div>
      </div>
      <div class="summary-card">
        <div class="label">Target Unique Values</div>
        <div class="value">{summary['target_unique']}</div>
      </div>
    </div>
    <h3 style="margin-top:18px; font-size:1rem;">Missing Values</h3>
    {missing_info}
  </section>

  <!-- EDA Visualisations -->
  <section>
    <h2>📈 Exploratory Data Analysis</h2>
    <div class="img-center">
      {_embed_image(eda_paths.get('target_dist_path', ''), 'Target Distribution')}
    </div>
    <div class="img-center">
      {_embed_image(eda_paths.get('feature_dist_path', ''), 'Feature Distributions')}
    </div>
    <div class="img-center">
      {_embed_image(eda_paths.get('corr_heatmap_path', ''), 'Correlation Heatmap')}
    </div>
  </section>

  {fe_section}

  <!-- Model Comparison -->
  <section>
    <h2>🏆 Model Comparison</h2>
    {_results_to_html(results, problem_type)}
  </section>

  <!-- Best Model -->
  <section>
    <h2>⭐ Best Model</h2>
    <p><span class="best-badge">{best_name}</span></p>
    <p style="margin-top:10px;">
      Primary metric ({metric_label}): <strong>{best_score}</strong>
    </p>
  </section>

  <!-- Feature Importance -->
  <section>
    <h2>📌 Feature Importance</h2>
    <div class="img-center">
      {_embed_image(explanation_paths.get('feature_importance_path', ''), 'Feature Importance') or _feature_importance_fallback(explanation_paths)}
    </div>
  </section>

  <!-- SHAP Explanations -->
  <section>
    <h2>🔍 SHAP Explanations</h2>
    <div class="img-center">
      {_embed_image(explanation_paths.get('shap_summary_path', ''), 'SHAP Summary') or _shap_fallback(explanation_paths)}
    </div>
    {_shap_scope_note(explanation_paths)}
    <div class="img-center">
      {_embed_image(explanation_paths.get('shap_importance_path', ''), 'SHAP Importance')}
    </div>
    <div class="img-center">
      {_embed_image(explanation_paths.get('shap_dependence_path', ''), 'SHAP Dependence')}
    </div>
  </section>

  <!-- Local Explanation -->
  <section>
    <h2>Local Prediction Explanation</h2>
    <p>The waterfall and decision plots explain the first bounded held-out sample used by the evidence layer.</p>
    <div class="img-center">
      {_embed_image(explanation_paths.get('shap_waterfall_path', ''), 'SHAP Local Waterfall')}
    </div>
    <div class="img-center">
      {_embed_image(explanation_paths.get('shap_decision_path', ''), 'SHAP Decision Plot')}
    </div>
  </section>

</div>

<footer>
  Generated by MLBuilder
</footer>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    abs_path = os.path.abspath(output_path)
    print(f"[Report] HTML report saved -> {abs_path}")
    return abs_path
