"""Registration points for custom estimators and framework extensions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class PluginRegistry:
    models: dict[str, tuple[str, Any]] = field(default_factory=dict)
    drift_detectors: dict[str, Callable[..., Any]] = field(default_factory=dict)
    data_sources: dict[str, Callable[..., Any]] = field(default_factory=dict)
    llm_providers: dict[str, Callable[..., Any]] = field(default_factory=dict)

    def register_model(
        self,
        name: str,
        estimator: Any,
        *,
        problem_type: str = "classification",
    ) -> "PluginRegistry":
        if problem_type not in {"classification", "regression"}:
            raise ValueError("problem_type must be classification or regression")
        from model_trainer import register_model

        register_model(name, estimator, problem_type=problem_type)
        self.models[name] = (problem_type, estimator)
        return self

    def register_drift_detector(
        self, name: str, factory: Callable[..., Any]
    ) -> "PluginRegistry":
        self.drift_detectors[name] = factory
        return self

    def register_data_source(
        self, name: str, factory: Callable[..., Any]
    ) -> "PluginRegistry":
        self.data_sources[name] = factory
        return self

    def register_llm_provider(
        self, name: str, factory: Callable[..., Any]
    ) -> "PluginRegistry":
        self.llm_providers[name] = factory
        return self


plugins = PluginRegistry()

