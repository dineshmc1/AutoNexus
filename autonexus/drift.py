"""Deterministic schema, feature, prediction, and performance drift checks."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance


@dataclass
class DriftSignal:
    kind: str
    feature: str
    score: float
    threshold: float
    drifted: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftReport:
    drifted: bool
    severity: str
    sample_count: int
    signals: list[DriftSignal]
    schema_errors: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "signals": [asdict(signal) for signal in self.signals],
        }


@dataclass
class DriftBaseline:
    columns: list[str]
    dtypes: dict[str, str]
    numeric_samples: dict[str, list[float]]
    categorical_frequencies: dict[str, dict[str, float]]
    prediction_frequencies: dict[str, float] = field(default_factory=dict)
    expected_metric: float | None = None
    target_name: str | None = None
    problem_type: str = "classification"
    metric_name: str = "accuracy"
    sample_count: int = 0

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        target_name: str | None = None,
        problem_type: str = "classification",
        sample_limit: int = 2048,
        random_state: int = 42,
    ) -> "DriftBaseline":
        features = frame.drop(
            columns=[target_name], errors="ignore"
        ).copy()
        numeric_samples: dict[str, list[float]] = {}
        categorical_frequencies: dict[str, dict[str, float]] = {}
        for column in features.columns:
            series = features[column]
            if pd.api.types.is_numeric_dtype(series):
                values = series.replace([np.inf, -np.inf], np.nan).dropna()
                if len(values) > sample_limit:
                    values = values.sample(
                        sample_limit, random_state=random_state
                    )
                numeric_samples[str(column)] = [
                    float(value) for value in values
                ]
            else:
                frequencies = (
                    series.fillna("<missing>")
                    .astype(str)
                    .value_counts(normalize=True)
                    .head(100)
                )
                categorical_frequencies[str(column)] = {
                    str(key): float(value)
                    for key, value in frequencies.items()
                }
        return cls(
            columns=[str(column) for column in features.columns],
            dtypes={
                str(column): str(dtype)
                for column, dtype in features.dtypes.items()
            },
            numeric_samples=numeric_samples,
            categorical_frequencies=categorical_frequencies,
            target_name=target_name,
            problem_type=problem_type,
            metric_name=(
                "accuracy"
                if problem_type == "classification"
                else "r2"
            ),
            sample_count=len(frame),
        )

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> "DriftBaseline":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


class DriftDetector:
    """Compare a current batch with a persisted training baseline."""

    def __init__(
        self,
        baseline: DriftBaseline,
        *,
        feature_threshold: float = 0.2,
        categorical_threshold: float = 0.15,
        prediction_threshold: float = 0.15,
        performance_drop_threshold: float = 0.03,
        minimum_samples: int = 30,
    ) -> None:
        self.baseline = baseline
        self.feature_threshold = feature_threshold
        self.categorical_threshold = categorical_threshold
        self.prediction_threshold = prediction_threshold
        self.performance_drop_threshold = performance_drop_threshold
        self.minimum_samples = minimum_samples

    @staticmethod
    def _categorical_js(
        reference: dict[str, float], current: pd.Series
    ) -> float:
        current_frequencies = (
            current.fillna("<missing>").astype(str).value_counts(normalize=True)
        )
        keys = sorted(set(reference) | set(current_frequencies.index))
        reference_values = np.asarray(
            [reference.get(key, 0.0) for key in keys], dtype=float
        )
        current_values = np.asarray(
            [float(current_frequencies.get(key, 0.0)) for key in keys],
            dtype=float,
        )
        if not reference_values.sum() or not current_values.sum():
            return 0.0
        return float(
            jensenshannon(reference_values, current_values, base=2) ** 2
        )

    def detect(
        self,
        frame: pd.DataFrame,
        *,
        predictions: np.ndarray | None = None,
        y_true: np.ndarray | pd.Series | None = None,
    ) -> DriftReport:
        features = frame.drop(
            columns=[self.baseline.target_name], errors="ignore"
        )
        schema_errors = []
        missing = sorted(set(self.baseline.columns) - set(features.columns))
        extra = sorted(set(features.columns) - set(self.baseline.columns))
        if missing:
            schema_errors.append(f"missing columns: {missing}")
        if extra:
            schema_errors.append(f"unexpected columns: {extra}")

        signals: list[DriftSignal] = []
        if len(frame) >= self.minimum_samples:
            for column, reference in self.baseline.numeric_samples.items():
                if column not in features or not reference:
                    continue
                current = pd.to_numeric(
                    features[column], errors="coerce"
                ).dropna()
                if len(current) < self.minimum_samples:
                    continue
                ks = float(ks_2samp(reference, current).statistic)
                scale = max(float(np.std(reference)), 1e-8)
                normalized_wasserstein = float(
                    wasserstein_distance(reference, current) / scale
                )
                score = max(ks, min(normalized_wasserstein / 5.0, 1.0))
                signals.append(
                    DriftSignal(
                        kind="numeric_feature",
                        feature=column,
                        score=score,
                        threshold=self.feature_threshold,
                        drifted=score >= self.feature_threshold,
                        details={
                            "ks": ks,
                            "normalized_wasserstein": normalized_wasserstein,
                        },
                    )
                )
            for column, reference in (
                self.baseline.categorical_frequencies.items()
            ):
                if column not in features:
                    continue
                score = self._categorical_js(reference, features[column])
                signals.append(
                    DriftSignal(
                        kind="categorical_feature",
                        feature=column,
                        score=score,
                        threshold=self.categorical_threshold,
                        drifted=score >= self.categorical_threshold,
                    )
                )

        metrics: dict[str, float] = {}
        if predictions is not None and len(predictions):
            frequencies = pd.Series(predictions).astype(str).value_counts(
                normalize=True
            )
            if self.baseline.prediction_frequencies:
                score = self._categorical_js(
                    self.baseline.prediction_frequencies,
                    pd.Series(predictions),
                )
                signals.append(
                    DriftSignal(
                        kind="prediction",
                        feature="prediction",
                        score=score,
                        threshold=self.prediction_threshold,
                        drifted=score >= self.prediction_threshold,
                    )
                )
            metrics["prediction_classes"] = float(len(frequencies))

        if y_true is not None and predictions is not None:
            true_values = np.asarray(y_true)
            prediction_values = np.asarray(predictions)
            if len(true_values) == len(prediction_values):
                observed = (
                    float(np.mean(true_values == prediction_values))
                    if self.baseline.problem_type == "classification"
                    else float(r2_score(true_values, prediction_values))
                )
                metrics[f"observed_{self.baseline.metric_name}"] = observed
                if self.baseline.expected_metric is not None:
                    drop = self.baseline.expected_metric - observed
                    signals.append(
                        DriftSignal(
                            kind="performance",
                            feature=self.baseline.metric_name,
                            score=max(drop, 0.0),
                            threshold=self.performance_drop_threshold,
                            drifted=drop >= self.performance_drop_threshold,
                            details={"drop": drop},
                        )
                    )

        drifted_signals = sum(signal.drifted for signal in signals)
        drifted = bool(schema_errors or drifted_signals)
        ratio = drifted_signals / max(len(signals), 1)
        severity = (
            "critical"
            if schema_errors or ratio >= 0.5
            else "warning"
            if drifted
            else "stable"
        )
        return DriftReport(
            drifted=drifted,
            severity=severity,
            sample_count=len(frame),
            signals=signals,
            schema_errors=schema_errors,
            metrics=metrics,
        )
