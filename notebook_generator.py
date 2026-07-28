"""Generate a standalone notebook summarizing a completed ML-Builder run."""

from __future__ import annotations

import json
from pathlib import Path


def generate_advanced_notebook(
    config: dict,
    results: dict,
    output_path: str,
) -> str:
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(results["metrics_path"]).resolve()
    model_path = Path(results["model_path"]).resolve()
    plot_paths = [
        str(Path(path).resolve())
        for path in results.get("plot_paths", [])
        if path
    ]
    summary = results.get("summary", {})

    notebook = new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.cells = [
        new_markdown_cell(
            "# ML-Builder Analysis\n\n"
            f"**Dataset:** `{config.get('data_path', 'unknown')}`  \n"
            f"**Modality:** {config.get('modality', 'tabular')}  \n"
            f"**Best model:** {results.get('best_model', 'unknown')}"
        ),
        new_code_cell(
            "import json\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            f"summary = json.loads({json.dumps(json.dumps(summary))})\n"
            "pd.Series(summary, name='value').to_frame()"
        ),
        new_markdown_cell("## Model Metrics"),
        new_code_cell(
            f"metrics_path = Path(r'{metrics_path}')\n"
            "metrics = pd.read_csv(metrics_path)\n"
            "metrics"
        ),
        new_markdown_cell("## Saved Model"),
        new_code_cell(
            f"model_path = Path(r'{model_path}')\n"
            "print(f'Model artifact: {model_path}')\n"
            "print(f'Exists: {model_path.exists()}')"
        ),
        new_markdown_cell("## Generated Plots"),
        new_code_cell(
            f"plot_paths = {plot_paths!r}\n"
            "for plot in plot_paths:\n"
            "    path = Path(plot)\n"
            "    if path.exists():\n"
            "        print(path.name)\n"
            "        display(Image(filename=str(path)))"
        ),
    ]
    nbformat.write(notebook, output)
    print(f"[Notebook] Analysis notebook saved to: {output}")
    return str(output)
