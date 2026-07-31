"""Tests for the persisted analytics bundle and generated notebook."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from analytics_artifacts import audit_image_files, persist_run_analytics
from data_loader import DataBundle
from notebook_generator import generate_advanced_notebook


def test_image_audit_finds_unreadable_and_exact_duplicates(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    image_a = tmp_path / "train" / "cats" / "a.png"
    image_b = tmp_path / "test" / "cats" / "a-copy.png"
    corrupt = tmp_path / "train" / "dogs" / "broken.png"
    image_a.parent.mkdir(parents=True)
    image_b.parent.mkdir(parents=True)
    corrupt.parent.mkdir(parents=True)
    Image.new("RGB", (20, 10), color=(20, 40, 60)).save(image_a)
    image_b.write_bytes(image_a.read_bytes())
    corrupt.write_bytes(b"not an image")

    inventory = audit_image_files(
        [str(image_a), str(image_b), str(corrupt)],
        ["cats", "cats", "dogs"],
        ["train", "test", "train"],
        groups=None,
        output_dir=tmp_path / "analysis_data",
        random_state=42,
        quality_sample_limit=10,
    )

    assert inventory["readable"].tolist() == [True, True, False]
    assert inventory["exact_duplicate"].tolist() == [True, True, False]
    assert (tmp_path / "analysis_data" / "data_index.csv").is_file()


def test_notebook_is_pretraining_first_and_all_code_cells_parse(tmp_path):
    analysis_dir = tmp_path / "analysis_data"
    analysis_dir.mkdir()
    output = tmp_path / "analysis.ipynb"

    generate_advanced_notebook(
        {
            "data_path": "images",
            "modality": "vision",
            "problem_type": "classification",
        },
        {
            "best_model": "logistic",
            "metrics_path": str(tmp_path / "metrics.csv"),
            "model_path": str(tmp_path / "best_model.joblib"),
            "plot_paths": [],
            "analytics_paths": {"analysis_dir": str(analysis_dir)},
        },
        str(output),
    )

    notebook = nbformat.read(output, as_version=4)
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    assert markdown.index("Part I: Pre-Training Analysis") < markdown.index(
        "Part II: Post-Training Analysis"
    )
    for expected in (
        "Data Quality Audit",
        "Representative Images for Every Class",
        "Embedding Geometry: PCA and UMAP",
        "Backbone Tournament",
        "Calibration Before and After",
        "Learning Curves",
        "Group-Level Generalization",
        "Model Card",
    ):
        assert expected in markdown
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)


def test_persisted_classification_bundle_supports_notebook_cells(tmp_path):
    rng = np.random.default_rng(42)
    X_train = pd.DataFrame(rng.normal(size=(80, 6)))
    X_test = pd.DataFrame(rng.normal(size=(20, 6)))
    y_train = pd.Series((X_train[0] > 0).astype(int))
    y_test = pd.Series((X_test[0] > 0).astype(int))
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    bundle = DataBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        problem_type="classification",
        feature_names=[str(column) for column in X_train.columns],
        target_name="label",
        metadata={"class_names": ["negative", "positive"]},
    )

    @dataclass
    class Config:
        dataset: Path
        output_dir: Path
        random_state: int = 42

    config = Config(tmp_path / "synthetic.csv", tmp_path)
    paths = persist_run_analytics(
        output_dir=tmp_path,
        config=config,
        bundle=bundle,
        best_name="logistic",
        best_model=model,
        X_train_final=X_train,
        X_test_final=X_test,
        y_train=y_train,
        y_test=y_test,
        results=pd.DataFrame(
            [{"model": "logistic", "accuracy": 1.0, "f1": 1.0}]
        ),
        baseline_scores={"logistic": {"score": 0.95, "time": 0.01}},
        validation_scores={"logistic": 0.96},
        training_diagnostics={
            "logistic": {
                "cv_std": 0.02,
                "folds_completed": 3,
                "total_seconds": 0.1,
                "observed_process_ram_mb": 100.0,
            }
        },
        summary={
            "fitted_training_metric": 1.0,
            "primary_cross_validated_metric": 0.96,
            "held_out_testing_metric": 1.0,
            "input_metadata": {},
        },
        max_embedding_samples=100,
        max_learning_samples=100,
    )

    assert Path(paths["probabilities"]).is_file()
    assert Path(paths["leaderboard"]).is_file()
    assert Path(paths["run_context"]).is_file()

    notebook_path = tmp_path / "analysis.ipynb"
    generate_advanced_notebook(
        {
            "data_path": str(config.dataset),
            "modality": "tabular",
            "problem_type": "classification",
        },
        {
            "best_model": "logistic",
            "metrics_path": str(tmp_path / "metrics.csv"),
            "model_path": str(tmp_path / "best_model.joblib"),
            "plot_paths": [],
            "analytics_paths": paths,
        },
        str(notebook_path),
    )
    notebook = nbformat.read(notebook_path, as_version=4)
    namespace: dict[str, object] = {}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, "<notebook-cell>", "exec"), namespace)
