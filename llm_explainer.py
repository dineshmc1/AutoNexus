"""LLM-assisted Markdown reporting with a deterministic offline fallback."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _offline_report(context: dict[str, Any], error: str | None = None) -> str:
    dataset = context.get("dataset", {})
    performance = context.get("performance", {})
    model = context.get("model", {})
    resources = context.get("resources", {})
    image_input = context.get("image_input") or {}
    backbone_note = (
        "\n- Vision backbone: "
        f"`{image_input.get('backbone_key')}` "
        f"(`{image_input.get('backbone')}`)"
        "\n- Representation: "
        f"{image_input.get('selected_representation')}"
        if image_input
        else ""
    )
    note = (
        f"\n> LLM generation was unavailable: {error}\n"
        if error else ""
    )
    return f"""# Model Explanation
{note}
## 1. Dataset

- Dataset: `{dataset.get("path", "unknown")}`
- Modality: {dataset.get("modality", "tabular")}
- Samples: {dataset.get("samples", "unknown")}
- Features: {dataset.get("features", "unknown")}
- Problem type: {dataset.get("problem_type", "unknown")}

## 2. Selected Model

The pipeline selected **{model.get("name", "unknown")}** after baseline
screening and cross-validation.{backbone_note}

## 3. Performance

- Fitted training metric (diagnostic): {performance.get("training", "N/A")}
- Cross-validated metric (primary): {performance.get("validation", "N/A")}
- Held-out testing metric: {performance.get("testing", "N/A")}

Cross-validation is the primary model-selection estimate. The testing metric
is the final held-out estimate and should be used for deployment decisions.
The fitted training metric is diagnostic only; a large fit-to-validation gap
can indicate excessive model capacity.

## 4. Resources

- Training time: {resources.get("training_seconds", "N/A")} seconds
- Peak RAM: {resources.get("ram_peak_mb", "N/A")} MiB
- Peak VRAM: {resources.get("vram_peak_mb", "N/A")} MiB

## 5. Recommendations

Review the class distribution and confusion matrix before deployment. Validate
the model on recent production-like data, monitor drift, and retrain when the
input distribution or target behavior changes.
"""


def generate_comprehensive_report(
    master_context: dict[str, Any],
    dataset_id: str,
    output_path: str | None = None,
    use_llm: bool = True,
) -> str:
    """Generate and save a Markdown report, falling back safely offline."""
    report_md: str
    failure: str | None = None

    if use_llm:
        try:
            import litellm
            from config import LLM_MODEL

            if not LLM_MODEL:
                raise RuntimeError(
                    "LLM_MODEL is not configured; using the offline report."
                )
            system_prompt = """You are an expert ML consultant. Produce a
professional Markdown report with exactly five sections: Dataset, Model
Selection, Performance, Explainability, and Deployment Recommendations. Use
only values in the supplied JSON. Clearly distinguish training, validation,
and testing metrics. Treat fitted training performance as diagnostic and
cross-validation as the primary selection estimate. Do not invent facts."""
            response = litellm.completion(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(master_context, indent=2),
                    },
                ],
                temperature=0.3,
                max_tokens=2500,
            )
            report_md = response.choices[0].message.content.strip()
            print(f"[LLM] Consultant report generated with {LLM_MODEL}.")
        except Exception as exc:
            failure = str(exc)
            print(f"[LLM] Unavailable; writing offline explanation: {failure}")
            report_md = _offline_report(master_context, failure)
    else:
        report_md = _offline_report(master_context)

    path = Path(output_path or f"reports/{dataset_id}_consultant_report.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_md, encoding="utf-8")
    print(f"[Report] Markdown explanation saved to: {path}")
    return str(path)
