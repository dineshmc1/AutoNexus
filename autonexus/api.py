"""The high-level AutoNexus training API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .callbacks import Callback, CallbackManager
from .config import NexusConfig
from .data import DataSource, as_source
from .drift import DriftBaseline
from .llm import CallableLLMProvider, LLMProvider
from .model import NexusModel
from .plugins import plugins


def _load_embedding_representation(
    path: Path, fallback_names: Iterable[str]
) -> pd.DataFrame:
    """Load a safe numeric representation, including legacy 0.1.0 bundles."""
    with np.load(path, allow_pickle=False) as embedding:
        values = embedding["X"].astype(np.float32)
        try:
            feature_names = embedding["feature_names"].astype(str).tolist()
        except (KeyError, TypeError, ValueError):
            # AutoNexus 0.1.0 accidentally persisted this field as an object
            # array. Never enable pickle for an artifact that may be untrusted.
            feature_names = [str(name) for name in fallback_names]

    if len(feature_names) != values.shape[1]:
        feature_names = [
            f"embedding_{index:04d}" for index in range(values.shape[1])
        ]
    return pd.DataFrame(values, columns=feature_names)


class AutoNexus:
    """Train production ML models with a compact, customizable API."""

    def __init__(
        self,
        *,
        task: str = "auto",
        preset: str = "balanced",
        output_dir: str | Path = "artifacts",
        models: list[str] | None = None,
        contribute_memory: bool = True,
        memory_dir: str | Path | None = None,
        llm: bool | LLMProvider | Any = True,
        callbacks: Iterable[Callback | Any] = (),
        **options: Any,
    ):
        use_llm = bool(llm) if isinstance(llm, bool) else False
        self.llm_provider = (
            None
            if isinstance(llm, bool)
            else llm
            if isinstance(llm, LLMProvider)
            else CallableLLMProvider(llm)
        )
        self.config = NexusConfig.create(
            preset=preset,
            task=task,
            output_dir=Path(output_dir),
            models=list(models or []),
            contribute_memory=contribute_memory,
            memory_dir=Path(memory_dir) if memory_dir else None,
            use_llm=use_llm,
            **options,
        )
        self.callbacks = CallbackManager(callbacks)
        self.plugins = plugins

    def register_model(
        self,
        name: str,
        estimator: Any,
        *,
        problem_type: str = "classification",
    ) -> "AutoNexus":
        self.plugins.register_model(
            name, estimator, problem_type=problem_type
        )
        if name not in self.config.models:
            self.config = self.config.with_overrides(
                models=[*self.config.models, name]
            )
        return self

    def _materialize(
        self, data: Any, output_dir: Path
    ) -> tuple[Path, pd.DataFrame | None]:
        if isinstance(data, pd.DataFrame):
            input_dir = output_dir / ".inputs"
            input_dir.mkdir(parents=True, exist_ok=True)
            path = input_dir / "training.csv"
            data.to_csv(path, index=False)
            return path, data.copy()
        if isinstance(data, DataSource):
            frame = pd.concat(list(data.batches()), ignore_index=True)
            return self._materialize(frame, output_dir)
        path = Path(data).expanduser().resolve()
        if path.is_file():
            if path.suffix.lower() == ".csv":
                return path, pd.read_csv(path)
            if path.suffix.lower() in {".xlsx", ".xls"}:
                return path, pd.read_excel(path)
        return path, None

    def fit(
        self,
        data: Any,
        *,
        target: str | None = None,
        label_column: str | None = None,
        **overrides: Any,
    ) -> NexusModel:
        target = target or label_column
        config = (
            self.config.with_overrides(**overrides)
            if overrides
            else self.config
        )
        output_dir = config.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset, frame = self._materialize(data, output_dir)
        if dataset.is_file() and not target:
            raise ValueError(
                "target or label_column is required for tabular training."
            )
        self.callbacks.emit(
            "training_started",
            dataset=str(dataset),
            target=target,
            output_dir=str(output_dir),
        )
        from main import render_run_completion, run

        result = run(
            config.to_run_config(dataset, target), render_completion=False
        )
        model = NexusModel(output_dir, callbacks=self.callbacks)
        self._write_framework_metadata(
            model, frame=frame, target=target, config=config
        )
        if self.llm_provider is not None:
            self._write_custom_llm_report(model)
        completion = result.pop("_completion_dashboard", None)
        if completion is not None:
            render_run_completion(completion)
        self.callbacks.emit(
            "training_completed",
            best_model=model.best_model,
            artifacts={
                key: str(value) for key, value in model.artifacts.items()
            },
            result=result,
        )
        return model

    def fit_source(
        self,
        source: DataSource | Any,
        *,
        target: str,
        initial_batches: int | None = None,
        **overrides: Any,
    ) -> NexusModel:
        batches = []
        for index, batch in enumerate(as_source(source)):
            if initial_batches is not None and index >= initial_batches:
                break
            batches.append(batch)
        if not batches:
            raise ValueError("The data source produced no training batches.")
        return self.fit(
            pd.concat(batches, ignore_index=True),
            target=target,
            **overrides,
        )

    def _write_framework_metadata(
        self,
        model: NexusModel,
        *,
        frame: pd.DataFrame | None,
        target: str | None,
        config: NexusConfig,
    ) -> None:
        manifest = json.loads(
            model.manifest_path.read_text(encoding="utf-8")
        )
        manifest.update(
            framework="AutoNexus",
            framework_version="0.1.1",
            label_column=target,
            model_used=manifest.get("best_model"),
            contribute_memory=config.contribute_memory,
            artifact_contract={
                key: str(path) for key, path in model.artifacts.items()
            },
        )
        model.manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        model.manifest = manifest

        baseline_path = model.output_dir / "monitoring" / "baseline.json"
        if frame is not None:
            baseline = DriftBaseline.from_frame(
                frame,
                target_name=target,
                problem_type=model.problem_type,
                random_state=config.random_state,
            )
            features = frame.drop(columns=[target], errors="ignore")
            sample = features.head(min(2048, len(features)))
            if model.problem_type == "classification":
                predictions = model.predict(sample)
                baseline.prediction_frequencies = {
                    str(key): float(value)
                    for key, value in pd.Series(predictions)
                    .value_counts(normalize=True)
                    .items()
                }
        else:
            representation = _load_embedding_representation(
                model.output_dir
                / "analysis_data"
                / "embedding_sample.npz",
                model.predictor.feature_names,
            )
            baseline = DriftBaseline.from_frame(
                representation,
                problem_type=model.problem_type,
                random_state=config.random_state,
            )
        baseline.expected_metric = manifest.get("run_summary", {}).get(
            "held_out_testing_metric"
        )
        baseline.save(baseline_path)
        framework_context = {
            "public_api": "autonexus.AutoNexus",
            "task": config.task,
            "preset": config.preset,
            "label_column": target,
            "feature_names": model.predictor.feature_names,
            "class_names": model.predictor.class_names,
            "supports_incremental_learning": (
                model.supports_incremental_learning
            ),
            "monitoring_baseline": str(baseline_path),
        }
        (model.output_dir / "framework.json").write_text(
            json.dumps(framework_context, indent=2), encoding="utf-8"
        )

    def _write_custom_llm_report(self, model: NexusModel) -> None:
        context = json.loads(
            model.manifest_path.read_text(encoding="utf-8")
        )
        prompt = (
            "Write a concise Markdown model report covering the dataset, "
            "selection evidence, generalization, drift risks, limitations, "
            "and deployment recommendations."
        )
        try:
            report = self.llm_provider.generate(prompt, context=context)
        except Exception as exc:
            report = model.explain() + (
                f"\n\n> Custom LLM provider failed: {type(exc).__name__}: {exc}\n"
            )
        path = model.output_dir / "report" / "explanation.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(report), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> NexusModel:
        return NexusModel.load(path)
