"""Typed public configuration and developer-friendly presets."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .exceptions import ConfigurationError


def _seconds(value: float | int | str | None) -> float | None:
    if value is None or isinstance(value, (float, int)):
        return None if value is None else float(value)
    normalized = value.strip().lower()
    multiplier = 60.0
    if normalized.endswith("s"):
        multiplier, normalized = 1.0, normalized[:-1]
    elif normalized.endswith("m"):
        multiplier, normalized = 60.0, normalized[:-1]
    elif normalized.endswith("h"):
        multiplier, normalized = 3600.0, normalized[:-1]
    return float(normalized) * multiplier


PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "cv": 2,
        "sample_fraction": 0.1,
        "baseline_seconds": 10.0,
        "tune": False,
        "backbone_time_seconds": 300.0,
    },
    "balanced": {},
    "accurate": {
        "cv": 5,
        "sample_fraction": 0.2,
        "baseline_seconds": 60.0,
        "tune": True,
        "tune_iterations": 30,
        "backbone_time_seconds": 1800.0,
    },
    "low_memory": {
        "cv": 2,
        "sample_fraction": 0.05,
        "baseline_seconds": 15.0,
        "models": ["logistic", "sgd_clf", "ridge", "sgd_reg"],
        "backbones": ["clip", "dinov2"],
    },
    "online": {
        "cv": 3,
        "models": ["sgd_clf", "sgd_reg"],
        "tune": False,
    },
}


@dataclass(frozen=True)
class NexusConfig:
    task: str = "auto"
    preset: str = "balanced"
    output_dir: Path = Path("artifacts")
    models: list[str] = field(default_factory=list)
    test_size: float = 0.2
    sample_fraction: float = 0.1
    baseline_seconds: float = 15.0
    cv: int = 5
    max_time_seconds: float | None = None
    random_state: int = 42
    feature_engineering: bool = False
    interactions: int | None = None
    ratios: bool = False
    outlier_strategy: str = "cap"
    tune: bool = False
    tune_method: str = "randomized"
    tune_iterations: int = 20
    report: bool = True
    shap: bool = False
    use_llm: bool = True
    adapt_lora: bool = False
    backbones: list[str] = field(default_factory=lambda: ["auto"])
    backbone_time_seconds: float = 900.0
    contribute_memory: bool = True
    memory_dir: Path | None = None

    @classmethod
    def create(cls, *, preset: str = "balanced", **options: Any):
        if preset not in PRESETS:
            raise ConfigurationError(
                f"Unknown preset {preset!r}; choose {sorted(PRESETS)}"
            )
        merged = {**PRESETS[preset], **options, "preset": preset}
        if "max_time" in merged:
            merged["max_time_seconds"] = _seconds(merged.pop("max_time"))
        if "backbone_time" in merged:
            merged["backbone_time_seconds"] = _seconds(
                merged.pop("backbone_time")
            )
        if "output_dir" in merged:
            merged["output_dir"] = Path(merged["output_dir"])
        if merged.get("memory_dir") is not None:
            merged["memory_dir"] = Path(merged["memory_dir"])
        return cls(**merged)

    def with_overrides(self, **overrides: Any) -> "NexusConfig":
        normalized = dict(overrides)
        if "max_time" in normalized:
            normalized["max_time_seconds"] = _seconds(
                normalized.pop("max_time")
            )
        if "output_dir" in normalized:
            normalized["output_dir"] = Path(normalized["output_dir"])
        return replace(self, **normalized)

    def to_run_config(self, dataset: Path, target: str | None):
        from main import RunConfig

        problem_type = None if self.task in {"auto", "vision"} else self.task
        return RunConfig(
            dataset=dataset.resolve(),
            target=target,
            output_dir=self.output_dir.expanduser().resolve(),
            problem_type=problem_type,
            models=list(self.models),
            test_size=self.test_size,
            sample_fraction=self.sample_fraction,
            baseline_seconds=self.baseline_seconds,
            cv=self.cv,
            max_time_seconds=self.max_time_seconds,
            random_state=self.random_state,
            feature_engineering=self.feature_engineering,
            interactions=self.interactions,
            ratios=self.ratios,
            outlier_strategy=self.outlier_strategy,
            tune=self.tune,
            tune_method=self.tune_method,
            tune_iterations=self.tune_iterations,
            report=self.report,
            shap=self.shap,
            llm=self.use_llm,
            notebook=True,
            adapt_lora=self.adapt_lora,
            backbones=list(self.backbones),
            backbone_time_seconds=self.backbone_time_seconds,
            contribute_memory=self.contribute_memory,
            memory_dir=self.memory_dir,
        )

