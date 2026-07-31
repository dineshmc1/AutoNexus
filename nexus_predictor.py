"""Serializable inference wrapper shared by the CLI and AutoNexus SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class NexusPredictor:
    """Keep fitted feature engineering and the selected estimator together."""

    model: Any
    problem_type: str
    target_name: str
    feature_names: list[str]
    class_names: list[str] = field(default_factory=list)
    feature_engineer: Any | None = None
    modality: str = "tabular"
    metadata: dict[str, Any] = field(default_factory=dict)

    def transform(self, X: Any) -> pd.DataFrame:
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if self.feature_engineer is not None:
            frame = self.feature_engineer.transform(frame)
        return frame

    def predict_encoded(self, X: Any) -> np.ndarray:
        return np.asarray(self.model.predict(self.transform(X)))

    def predict(self, X: Any) -> np.ndarray:
        encoded = self.predict_encoded(X)
        if self.problem_type != "classification" or not self.class_names:
            return encoded
        labels = []
        for value in encoded:
            try:
                index = int(value)
                labels.append(
                    self.class_names[index]
                    if 0 <= index < len(self.class_names)
                    else str(value)
                )
            except (TypeError, ValueError):
                labels.append(str(value))
        return np.asarray(labels)

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("The selected estimator has no predict_proba().")
        return np.asarray(self.model.predict_proba(self.transform(X)))

