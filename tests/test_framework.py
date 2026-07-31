"""Public AutoNexus framework API and lifecycle tests."""

from __future__ import annotations

import importlib
import json

import numpy as np
import pandas as pd

from autonexus import (
    AutoNexus,
    DriftBaseline,
    DriftDetector,
    FAISSMetaMemory,
    ModelRegistry,
)


def _classification_frame(rows: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    signal = rng.normal(size=rows)
    return pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.normal(size=rows),
            "label": np.where(signal > 0, "yes", "no"),
        }
    )


def test_capitalized_import_shim_exposes_framework():
    module = importlib.import_module("AutoNexus")

    assert module.AutoNexus is AutoNexus


def test_drift_detector_finds_schema_and_feature_shift():
    reference = _classification_frame(100)
    baseline = DriftBaseline.from_frame(reference, target_name="label")
    detector = DriftDetector(baseline, minimum_samples=20)
    shifted = reference.copy()
    shifted["signal"] += 20
    shifted["new_column"] = 1

    report = detector.detect(shifted)

    assert report.drifted
    assert report.schema_errors
    assert any(signal.drifted for signal in report.signals)


def test_local_meta_memory_contribution_and_search(tmp_path):
    memory = FAISSMetaMemory(tmp_path / "memory")
    first = memory.contribute(
        {
            "dataset_fingerprint": "dataset-a",
            "embedding": [0.0, 0.0, 1.0],
            "best_model": "logistic",
        }
    )
    duplicate = memory.contribute(
        {
            "dataset_fingerprint": "dataset-a",
            "embedding": [0.0, 0.0, 1.0],
            "best_model": "logistic",
        }
    )

    assert first.contributed
    assert not duplicate.contributed
    assert memory.search(np.asarray([0.0, 0.0, 1.0]))[0][
        "best_model"
    ] == "logistic"


def test_five_line_training_artifacts_inference_monitoring_and_update(tmp_path):
    frame = _classification_frame(120)
    output = tmp_path / "run"
    trainer = AutoNexus(
        preset="fast",
        output_dir=output,
        models=["sgd_clf"],
        llm=False,
        report=False,
        contribute_memory=False,
    )

    model = trainer.fit(frame, target="label")

    for path in model.artifacts.values():
        assert path.is_file(), path
    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert manifest["label_column"] == "label"
    assert manifest["model_used"]
    assert manifest["faiss_memory"]["status"] == "disabled"
    predictions = model.predict(frame.drop(columns=["label"]).head(5))
    assert set(predictions).issubset({"yes", "no"})

    monitor = model.monitor(minimum_samples=20)
    report = monitor.observe(frame.head(40))
    assert report.sample_count == 40

    update = model.update(frame.tail(40), target="label")
    assert update.action == "incremental_partial_fit"
    assert model.supports_incremental_learning

    one_class = frame.loc[frame["label"] == "yes"].head(20)
    one_class_update = model.update(one_class, target="label")
    assert one_class_update.action == "incremental_partial_fit"

    registry = ModelRegistry(tmp_path / "registry")
    registered = model.register("demo", registry=registry)
    for relative in (
        "run.json",
        "model.pkl",
        "analysis.ipynb",
        "report/explanation.md",
        "search_profile.json",
    ):
        assert (registered.path / relative).is_file()
