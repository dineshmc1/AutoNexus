"""Public AutoNexus framework API and lifecycle tests."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import socket
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from autonexus import (
    AutoNexus,
    DriftBaseline,
    DriftDetector,
    FAISSMetaMemory,
    ModelRegistry,
)
from autonexus.llm import CallableLLMProvider
from autonexus.memory import apply_search_advice, retrieve_search_advice


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

    assert module is importlib.import_module("autonexus")
    assert module.AutoNexus is AutoNexus


def test_capitalized_import_first_uses_canonical_submodules():
    command = (
        "import AutoNexus, autonexus; "
        "assert AutoNexus.AutoNexus is autonexus.AutoNexus; "
        "assert AutoNexus.AutoNexus.__module__ == 'autonexus.api'"
    )

    subprocess.run([sys.executable, "-c", command], check=True)


def test_llm_provider_rejects_empty_output():
    provider = CallableLLMProvider(lambda **_: None)

    with pytest.raises(RuntimeError, match="empty response"):
        provider.generate("report", context={})


def test_custom_llm_report_stamps_provenance_and_retries_contradiction(
    tmp_path,
):
    output = tmp_path / "run"
    report_dir = output / "report"
    report_dir.mkdir(parents=True)
    manifest_path = output / "run.json"
    manifest_path.write_text(
        json.dumps({"run_summary": {"llm_seconds": 0.0}}),
        encoding="utf-8",
    )
    fallback = "# Deterministic report\n"
    (report_dir / "explanation.md").write_text(
        fallback, encoding="utf-8"
    )
    responses = iter(
        [
            "# Draft\n\nNo LLM was used for this report.",
            "# Grounded report\n\nThe held-out evidence is documented.",
        ]
    )
    provider = CallableLLMProvider(lambda **_: next(responses))
    provider.report_provenance = {
        "provider": "openrouter",
        "model": "openrouter/openai/gpt-4.1-mini",
        "credential_storage": "ephemeral_memory",
    }
    trainer = AutoNexus(
        output_dir=output,
        llm=provider,
        contribute_memory=False,
    )
    fake_model = SimpleNamespace(
        output_dir=output,
        manifest_path=manifest_path,
        manifest={},
        explain=lambda: fallback,
    )

    trainer._write_custom_llm_report(fake_model)

    report = (report_dir / "explanation.md").read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "openrouter/openai/gpt-4.1-mini" in report
    assert "Grounded report" in report
    assert "No LLM was used" not in report
    assert manifest["llm"] is True
    assert manifest["llm_report"]["status"] == "generated"
    assert manifest["run_summary"]["custom_llm_seconds"] >= 0


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


@pytest.mark.parametrize("sample_count", [1, 29])
def test_drift_detector_requires_enough_rows_for_population_signals(
    sample_count,
):
    reference = _classification_frame(100)
    baseline = DriftBaseline.from_frame(reference, target_name="label")
    baseline.prediction_frequencies = {"yes": 0.5, "no": 0.5}
    baseline.expected_metric = 0.9
    batch = reference.head(sample_count)

    report = DriftDetector(baseline, minimum_samples=30).detect(
        batch,
        predictions=np.asarray(["wrong"] * sample_count),
        y_true=batch["label"],
    )

    assert not report.drifted
    assert report.severity == "insufficient_data"
    assert not report.sufficient_samples
    assert report.minimum_samples == 30
    assert report.metrics["observed_accuracy"] == 0.0
    assert {signal.kind for signal in report.signals} == {
        "insufficient_samples"
    }


def test_drift_detector_activates_population_signals_at_minimum_batch():
    reference = _classification_frame(100)
    baseline = DriftBaseline.from_frame(reference, target_name="label")
    repeated = pd.concat([reference.head(1)] * 30, ignore_index=True)

    report = DriftDetector(baseline, minimum_samples=30).detect(repeated)

    assert report.sufficient_samples
    assert report.severity != "insufficient_data"
    assert any(
        signal.kind == "constant_column" and signal.drifted
        for signal in report.signals
    )


def test_drift_detector_keeps_schema_errors_critical_for_one_row():
    reference = _classification_frame(100)
    baseline = DriftBaseline.from_frame(reference, target_name="label")

    report = DriftDetector(baseline).detect(
        reference.drop(columns=["noise"]).head(1)
    )

    assert report.drifted
    assert report.severity == "critical"
    assert not report.sufficient_samples
    assert report.schema_errors == ["missing columns: ['noise']"]


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


def test_memory_advice_uses_validation_evidence_and_current_veto(tmp_path):
    memory = FAISSMetaMemory(tmp_path / "memory")
    memory.contribute(
        {
            "dataset_fingerprint": "similar-a",
            "embedding_version": 2,
            "embedding": [0.0, 0.0, 1.0],
            "best_model": "logistic",
            "problem_type": "classification",
            "selection_evidence": {
                "baseline_scores": {
                    "logistic": {"score": 0.9},
                    "rf": {"score": 0.5},
                    "et_clf": {"score": 0.7},
                }
            },
        }
    )
    current = {
        "logistic": {"score": 0.88},
        "rf": {"score": 0.80},
        "et_clf": {"score": 0.84},
    }

    advice = retrieve_search_advice(
        np.asarray([0.0, 0.0, 1.0]),
        current,
        problem_type="classification",
        embedding_version=2,
        memory_dir=tmp_path / "memory",
    )
    selected, changes = apply_search_advice(
        {name: name for name in current},
        {name: name for name in current},
        current,
        advice,
    )

    assert advice["status"] == "applied"
    assert advice["recommended_models"][0] == "logistic"
    assert "rf" in advice["penalized_models"]
    assert changes["pruned"] == ["rf"]
    assert list(selected)[0] == "logistic"


def test_one_line_deployment_serves_and_stops():
    fastapi = pytest.importorskip("fastapi")
    del fastapi
    from autonexus.deployment import deploy_model

    fake = SimpleNamespace(
        best_model="fake",
        problem_type="classification",
        predict=lambda frame: np.asarray(["yes"] * len(frame)),
        predict_proba=lambda frame: np.asarray([[0.1, 0.9]] * len(frame)),
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    deployment = deploy_model(fake, port=port, api_key="test-key")
    try:
        assert deployment.running
        health = json.loads(
            urllib.request.urlopen(deployment.health_url).read()
        )
        assert health["model"] == "fake"
        request = urllib.request.Request(
            deployment.predict_url,
            data=json.dumps({"records": [{"signal": 1.0}]}).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer test-key",
            },
            method="POST",
        )
        response = json.loads(urllib.request.urlopen(request).read())
        assert response["predictions"] == ["yes"]
    finally:
        deployment.stop()
    assert not deployment.running


def test_five_line_training_artifacts_inference_monitoring_and_update(tmp_path):
    frame = _classification_frame(120)
    output = tmp_path / "run"
    trainer = AutoNexus(
        preset="fast",
        output_dir=output,
        models=["sgd_clf"],
        llm=False,
        report=False,
        use_memory=False,
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
    assert "missing_rate" in report.metrics
    assert "duplicate_rate" in report.metrics

    update = model.update(frame.tail(40), target="label")
    assert update.action == "incremental_partial_fit"
    assert update.version
    assert Path(update.model_path).is_file()
    assert (Path(update.model_path).parent / "update.json").is_file()
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


def test_auto_update_routes_tree_model_to_retrained_challenger(tmp_path):
    frame = _classification_frame(100)
    output = tmp_path / "tree-run"
    model = AutoNexus(
        preset="fast",
        output_dir=output,
        models=["rf"],
        llm=False,
        report=False,
        use_memory=False,
        contribute_memory=False,
    ).fit(frame, target="label")
    incoming = _classification_frame(40)
    incoming["signal"] += 0.05

    result = model.update(
        incoming,
        target="label",
        strategy="auto",
    )

    assert not model.supports_incremental_learning
    assert result.action == "challenger_retrain"
    assert result.version
    version_dir = output / "updates" / result.version
    assert (version_dir / "model.pkl").is_file()
    assert (version_dir / "update.json").is_file()


def test_drift_accuracy_normalizes_decoded_class_label_types():
    frame = pd.DataFrame({"feature": [0.0, 1.0], "label": [1, 2]})
    baseline = DriftBaseline.from_frame(
        frame,
        target_name="label",
        problem_type="classification",
    )
    baseline.expected_metric = 1.0
    report = DriftDetector(baseline, minimum_samples=1).detect(
        frame,
        predictions=np.asarray(["1", "2"]),
        y_true=frame["label"],
    )

    assert report.metrics["observed_accuracy"] == 1.0


def test_vision_fit_recovers_legacy_embedding_before_completion(
    tmp_path, monkeypatch
):
    import autonexus.api as api_module
    import main as main_module

    image_dir = tmp_path / "images" / "class-a"
    image_dir.mkdir(parents=True)
    output = tmp_path / "run"
    feature_names = ["embedding_0000", "embedding_0001"]
    completion_calls = []

    class FakeModel:
        def __init__(self, output_dir, callbacks=None):
            del callbacks
            self.output_dir = output_dir
            self.manifest_path = output_dir / "run.json"
            self.manifest = json.loads(
                self.manifest_path.read_text(encoding="utf-8")
            )
            self.predictor = SimpleNamespace(
                feature_names=feature_names,
                class_names=["class-a", "class-b"],
            )

        @property
        def problem_type(self):
            return "classification"

        @property
        def best_model(self):
            return "logistic"

        @property
        def supports_incremental_learning(self):
            return False

        @property
        def artifacts(self):
            return {"run": self.manifest_path}

    def fake_run(config, *, render_completion=True):
        assert render_completion is False
        analysis_dir = config.output_dir / "analysis_data"
        analysis_dir.mkdir(parents=True)
        (config.output_dir / "run.json").write_text(
            json.dumps(
                {
                    "best_model": "logistic",
                    "run_summary": {"held_out_testing_metric": 0.5},
                }
            ),
            encoding="utf-8",
        )
        np.savez_compressed(
            analysis_dir / "embedding_sample.npz",
            X=np.ones((4, 2), dtype=np.float16),
            feature_names=np.asarray(feature_names, dtype=object),
        )
        return {"_completion_dashboard": {"run": "complete"}}

    def fake_render_completion(payload):
        assert payload == {"run": "complete"}
        assert (output / "framework.json").is_file()
        assert (output / "monitoring" / "baseline.json").is_file()
        completion_calls.append(payload)

    monkeypatch.setattr(api_module, "NexusModel", FakeModel)
    monkeypatch.setattr(main_module, "run", fake_run)
    monkeypatch.setattr(
        main_module, "render_run_completion", fake_render_completion
    )

    model = AutoNexus(
        output_dir=output,
        llm=False,
        contribute_memory=False,
    ).fit(image_dir.parent)

    assert isinstance(model, FakeModel)
    assert completion_calls == [{"run": "complete"}]
