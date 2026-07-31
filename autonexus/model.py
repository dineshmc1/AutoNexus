"""Loaded AutoNexus model lifecycle: inference, monitoring, updates, serving."""

from __future__ import annotations

import json
import pickle
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from nexus_predictor import NexusPredictor

from .callbacks import CallbackManager
from .drift import DriftBaseline
from .exceptions import ArtifactError, CapabilityError
from .registry import ModelRegistry


@dataclass
class UpdateResult:
    action: str
    promoted: bool
    previous_score: float | None
    candidate_score: float | None
    reason: str
    samples_seen: int
    model_path: str


@dataclass(frozen=True)
class UpdatePolicy:
    strategy: str = "auto"
    minimum_batch_size: int = 20
    validation_fraction: float = 0.2
    max_allowed_drop: float = 0.005


class NexusModel:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        callbacks: CallbackManager | None = None,
    ):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.model_path = self.output_dir / "model.pkl"
        self.manifest_path = self.output_dir / "run.json"
        if not self.model_path.is_file() or not self.manifest_path.is_file():
            raise ArtifactError(
                f"Invalid AutoNexus run bundle: {self.output_dir}"
            )
        self.predictor: NexusPredictor = joblib.load(self.model_path)
        self.manifest = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )
        self.callbacks = callbacks or CallbackManager()

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        callbacks: CallbackManager | None = None,
    ) -> "NexusModel":
        candidate = Path(path)
        return cls(
            candidate.parent if candidate.is_file() else candidate,
            callbacks=callbacks,
        )

    @property
    def best_model(self) -> str:
        return str(self.manifest.get("best_model", "unknown"))

    @property
    def problem_type(self) -> str:
        return self.predictor.problem_type

    @property
    def artifacts(self) -> dict[str, Path]:
        return {
            "run": self.output_dir / "run.json",
            "model": self.model_path,
            "notebook": self.output_dir / "analysis.ipynb",
            "markdown": self.output_dir / "report" / "explanation.md",
            "search_profile": self.output_dir / "search_profile.json",
        }

    def _read_tabular(self, data: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            frame = data.copy()
        elif isinstance(data, (str, Path)):
            path = Path(data)
            if path.suffix.lower() == ".csv":
                frame = pd.read_csv(path)
            elif path.suffix.lower() in {".xlsx", ".xls"}:
                frame = pd.read_excel(path)
            elif path.suffix.lower() in {".parquet", ".pq"}:
                frame = pd.read_parquet(path)
            else:
                raise ValueError(f"Unsupported inference file: {path.suffix}")
        else:
            frame = pd.DataFrame(data)
        return frame.drop(columns=[self.predictor.target_name], errors="ignore")

    def _embed_vision(self, data: Any) -> pd.DataFrame:
        try:
            import torch
            from multimodal_extractor import (
                IMAGE_EXTENSIONS,
                UniversalEmbedder,
            )
        except ImportError as exc:
            raise CapabilityError(
                "Vision inference requires: pip install AutoNexus[vision]"
            ) from exc
        if isinstance(data, (str, Path)):
            path = Path(data)
            files = (
                [str(path.resolve())]
                if path.is_file()
                else [
                    str(item.resolve())
                    for item in sorted(path.rglob("*"))
                    if item.is_file()
                    and item.suffix.lower() in IMAGE_EXTENSIONS
                ]
            )
        else:
            files = [str(Path(item).resolve()) for item in data]
        if not files:
            raise ValueError("No supported images found for inference.")
        metadata = self.manifest.get("run_summary", {}).get(
            "input_metadata", {}
        )
        representation = str(metadata.get("selected_representation", ""))
        adapter = None
        if representation.startswith("adapted-"):
            adapter_candidate = (
                self.output_dir
                / "lora_adapter"
                / str(metadata.get("backbone_key"))
            )
            if adapter_candidate.is_dir():
                adapter = str(adapter_candidate)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedder = UniversalEmbedder(
            device=device,
            batch_size=32,
            domain="general",
            modality="vision",
            adapter_path=adapter,
            cache_dir=str(self.output_dir / ".cache" / "inference"),
            model_id=metadata.get("backbone"),
            model_revision=metadata.get("backbone_revision", "main"),
        )
        try:
            embeddings, _ = embedder.embed_files(
                files,
                ["inference"] * len(files),
                cache_key=f"inference:{hash(tuple(files))}",
            )
            return embeddings
        finally:
            embedder.release()

    def _features(self, data: Any) -> pd.DataFrame:
        return (
            self._embed_vision(data)
            if self.predictor.modality == "vision"
            else self._read_tabular(data)
        )

    def predict(self, data: Any) -> np.ndarray:
        return self.predictor.predict(self._features(data))

    def predict_proba(self, data: Any) -> np.ndarray:
        return self.predictor.predict_proba(self._features(data))

    def explain(self) -> str:
        path = self.output_dir / "report" / "explanation.md"
        if not path.is_file():
            raise ArtifactError("The run has no Markdown explanation.")
        return path.read_text(encoding="utf-8")

    def save(self, destination: str | Path) -> Path:
        destination = Path(destination)
        if destination.suffix:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.model_path, destination)
            return destination
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copytree(self.output_dir, destination)
        return destination

    def _incremental_components(
        self, predictor: NexusPredictor
    ) -> tuple[Any, Any | None, Any]:
        model = predictor.model
        if hasattr(model, "temperature") and hasattr(model, "estimator"):
            model.temperature = 1.0
            model = model.estimator
        named_steps = getattr(model, "named_steps", None)
        if named_steps is None:
            if not hasattr(model, "partial_fit"):
                raise CapabilityError("Selected model has no partial_fit().")
            return model, None, model
        estimator = named_steps.get("model")
        preprocessor = named_steps.get("preprocessor")
        if estimator is None or not hasattr(estimator, "partial_fit"):
            raise CapabilityError("Selected estimator has no partial_fit().")
        return model, preprocessor, estimator

    @property
    def supports_incremental_learning(self) -> bool:
        try:
            self._incremental_components(
                pickle.loads(pickle.dumps(self.predictor))
            )
            return True
        except Exception:
            return False

    def _encode_target(self, values: pd.Series) -> np.ndarray:
        if self.problem_type != "classification":
            return values.to_numpy()
        mapping = {
            str(label): index
            for index, label in enumerate(self.predictor.class_names)
        }
        if all(str(value) in mapping for value in values):
            return np.asarray([mapping[str(value)] for value in values])
        return pd.to_numeric(values, errors="raise").to_numpy(dtype=int)

    def update(
        self,
        data: pd.DataFrame | str | Path,
        *,
        target: str,
        strategy: str = "auto",
        validation: pd.DataFrame | None = None,
        max_allowed_drop: float = 0.005,
        policy: UpdatePolicy | None = None,
    ) -> UpdateResult:
        if policy is not None:
            strategy = policy.strategy
            max_allowed_drop = policy.max_allowed_drop
            minimum_batch_size = policy.minimum_batch_size
            validation_fraction = policy.validation_fraction
        else:
            minimum_batch_size = 20
            validation_fraction = 0.2
        if self.predictor.modality != "tabular":
            return UpdateResult(
                "adapter_or_retrain_required",
                False,
                None,
                None,
                "Online vision updates require a gated adapter training job.",
                0,
                str(self.model_path),
            )
        batch = (
            data.copy()
            if isinstance(data, pd.DataFrame)
            else self._read_input_with_target(data)
        )
        if target not in batch:
            raise ValueError(f"Target column {target!r} is missing.")
        if strategy not in {"auto", "incremental"}:
            raise ValueError("strategy must be 'auto' or 'incremental'")
        if not self.supports_incremental_learning:
            return UpdateResult(
                "retrain_required",
                False,
                None,
                None,
                "The selected model does not support safe partial_fit().",
                len(batch),
                str(self.model_path),
            )

        X = batch.drop(columns=[target])
        y = self._encode_target(batch[target])
        if validation is None and len(batch) >= minimum_batch_size:
            stratify = None
            if self.problem_type == "classification":
                counts = pd.Series(y).value_counts()
                if len(counts) > 1 and int(counts.min()) >= 2:
                    stratify = y
            X_fit, X_gate, y_fit, y_gate = train_test_split(
                X,
                y,
                test_size=validation_fraction,
                random_state=42,
                stratify=stratify,
            )
        elif validation is not None:
            X_fit, y_fit = X, y
            X_gate = validation.drop(columns=[target])
            y_gate = self._encode_target(validation[target])
        else:
            return UpdateResult(
                "deferred",
                False,
                None,
                None,
                f"At least {minimum_batch_size} rows or an explicit "
                "validation batch is required.",
                len(batch),
                str(self.model_path),
            )

        candidate: NexusPredictor = pickle.loads(pickle.dumps(self.predictor))
        _, preprocessor, estimator = self._incremental_components(candidate)
        transformed_fit = candidate.transform(X_fit)
        if preprocessor is not None:
            transformed_fit = preprocessor.transform(transformed_fit)
        fit_options = {}
        if self.problem_type == "classification":
            fit_options["classes"] = np.arange(len(candidate.class_names))
        estimator.partial_fit(transformed_fit, y_fit, **fit_options)

        champion_predictions = self.predictor.predict_encoded(X_gate)
        candidate_predictions = candidate.predict_encoded(X_gate)
        scoring = (
            accuracy_score
            if self.problem_type == "classification"
            else r2_score
        )
        previous_score = float(scoring(y_gate, champion_predictions))
        candidate_score = float(scoring(y_gate, candidate_predictions))
        promoted = candidate_score >= previous_score - max_allowed_drop
        reason = (
            "Candidate passed the holdout non-regression gate."
            if promoted
            else "Candidate exceeded the permitted holdout performance drop."
        )
        if promoted:
            temporary = self.model_path.with_suffix(".tmp")
            joblib.dump(candidate, temporary)
            temporary.replace(self.model_path)
            joblib.dump(
                candidate.model, self.output_dir / "best_model.joblib"
            )
            self.predictor = candidate
        result = UpdateResult(
            "incremental_partial_fit",
            promoted,
            previous_score,
            candidate_score,
            reason,
            len(batch),
            str(self.model_path),
        )
        history = self.output_dir / "monitoring" / "update_history.jsonl"
        history.parent.mkdir(parents=True, exist_ok=True)
        with history.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps({"timestamp": time.time(), **asdict(result)}) + "\n"
            )
        self.manifest.setdefault("updates", []).append(asdict(result))
        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=2), encoding="utf-8"
        )
        self.callbacks.emit("model_updated", result=asdict(result))
        return result

    def retrain(
        self,
        new_data: pd.DataFrame | str | Path,
        *,
        target: str | None = None,
        output_dir: str | Path | None = None,
        models: list[str] | None = None,
        promote_to: tuple[ModelRegistry, str] | None = None,
    ) -> "NexusModel":
        """Train a gated replacement when native incremental update is absent."""
        if self.predictor.modality != "tabular":
            raise CapabilityError(
                "Vision replacement training must be started with AutoNexus.fit()."
            )
        target = target or self.predictor.target_name
        original_path = Path(str(self.manifest.get("dataset", "")))
        if not original_path.is_file():
            raise ArtifactError(
                "The original tabular dataset is unavailable; provide a "
                "complete retraining dataset to AutoNexus.fit()."
            )
        original = self._read_input_with_target(original_path)
        incoming = (
            new_data.copy()
            if isinstance(new_data, pd.DataFrame)
            else self._read_input_with_target(new_data)
        )
        combined = pd.concat([original, incoming], ignore_index=True)
        destination = Path(
            output_dir
            or self.output_dir
            / "updates"
            / f"retrain-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        from .api import AutoNexus

        trainer = AutoNexus(
            task=self.problem_type,
            preset="balanced",
            output_dir=destination,
            models=models or list(self.manifest.get("models", [])),
            contribute_memory=bool(
                self.manifest.get("contribute_memory", True)
            ),
            llm=False,
        )
        challenger = trainer.fit(combined, target=target)
        if promote_to is not None:
            registry, name = promote_to
            registered = registry.register(
                challenger.output_dir, name=name, stage="challenger"
            )
            registry.promote(name, registered.version)
        return challenger

    @staticmethod
    def _read_input_with_target(data: str | Path) -> pd.DataFrame:
        path = Path(data)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported update file: {path.suffix}")

    def monitor(
        self,
        *,
        baseline_path: str | Path | None = None,
        sinks: list[Any] | None = None,
        **detector_options: Any,
    ):
        from .drift import DriftDetector
        from .monitoring import NexusMonitor

        baseline = DriftBaseline.load(
            baseline_path
            or self.output_dir / "monitoring" / "baseline.json"
        )
        return NexusMonitor(
            self,
            baseline,
            detector=DriftDetector(baseline, **detector_options),
            sinks=sinks,
        )

    def register(
        self,
        name: str,
        *,
        registry: ModelRegistry | None = None,
        version: str | None = None,
        promote: bool = False,
    ):
        registry = registry or ModelRegistry()
        registered = registry.register(
            self.output_dir, name=name, version=version
        )
        return (
            registry.promote(name, registered.version)
            if promote
            else registered
        )

    def serve(self, *, host: str = "127.0.0.1", port: int = 8000) -> None:
        try:
            from fastapi import FastAPI
            from pydantic import BaseModel
            import uvicorn
        except ImportError as exc:
            raise RuntimeError(
                "Serving requires: pip install AutoNexus[serve]"
            ) from exc

        nexus_model = self

        class PredictionRequest(BaseModel):
            records: list[dict[str, Any]]

        app = FastAPI(title="AutoNexus Inference", version="1")

        @app.get("/health")
        def health():
            return {"status": "ok", "model": nexus_model.best_model}

        @app.post("/predict")
        def predict(request: PredictionRequest):
            predictions = nexus_model.predict(
                pd.DataFrame(request.records)
            )
            return {"predictions": predictions.tolist()}

        uvicorn.run(app, host=host, port=port)
