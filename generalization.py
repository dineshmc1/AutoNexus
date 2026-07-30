"""Leakage-safe calibration and structurally diverse ensembling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split


MODEL_FAMILIES = {
    "gbdt": ("xgb_clf", "lgbm_clf", "gb"),
    "linear": ("logistic", "sgd_clf"),
    "nonlinear": ("et_clf", "rf", "mlp_clf", "knn_clf"),
}


def _aligned_probabilities(model: Any, X: Any, classes: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X), dtype=float)
        model_classes = np.asarray(model.classes_)
        aligned = np.zeros((len(X), len(classes)), dtype=float)
        for source, label in enumerate(model_classes):
            target = int(np.flatnonzero(classes == label)[0])
            aligned[:, target] = probabilities[:, source]
        return aligned

    predictions = np.asarray(model.predict(X))
    return (predictions[:, None] == classes[None, :]).astype(float)


class DiverseProbabilityEnsemble(BaseEstimator, ClassifierMixin):
    """Average probabilities from fitted models belonging to distinct families."""

    def __init__(self, models: list[tuple[str, Any]], classes: np.ndarray):
        self.models = models
        self.classes = classes
        self.classes_ = np.asarray(classes)

    def predict_proba(self, X):
        probabilities = [
            _aligned_probabilities(model, X, self.classes_)
            for _, model in self.models
        ]
        return np.mean(probabilities, axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class TemperatureScaledClassifier(BaseEstimator, ClassifierMixin):
    """Apply one learned temperature to a fitted classifier's probabilities."""

    def __init__(self, estimator: Any, temperature: float = 1.0):
        self.estimator = estimator
        self.temperature = float(temperature)
        self.classes_ = np.asarray(estimator.classes_)

    def predict_proba(self, X):
        probabilities = np.clip(
            self.estimator.predict_proba(X), 1e-12, 1.0
        )
        logits = np.log(probabilities) / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def predict(self, X):
        # Positive scalar temperature preserves the probability argmax.
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


@dataclass
class GeneralizationSelection:
    model: Any
    name: str
    gate_accuracy: float | None
    primary_cv_accuracy: float
    temperature: float
    ensemble_used: bool
    members: list[str]
    nll_before: float | None
    nll_after: float | None


def _optimize_temperature(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    classes: np.ndarray,
) -> tuple[float, float, float]:
    probabilities = np.clip(probabilities, 1e-12, 1.0)

    def calibrated(temperature: float) -> np.ndarray:
        logits = np.log(probabilities) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    result = minimize_scalar(
        lambda value: log_loss(
            y_true, calibrated(value), labels=classes
        ),
        bounds=(0.25, 10.0),
        method="bounded",
    )
    temperature = float(result.x) if result.success else 1.0
    before = float(log_loss(y_true, probabilities, labels=classes))
    after = float(log_loss(y_true, calibrated(temperature), labels=classes))
    if after > before:
        return 1.0, before, before
    return temperature, before, after


def select_generalized_classifier(
    trained: dict[str, Any],
    validation_scores: dict[str, float],
    X,
    y,
    random_state: int = 42,
    groups: np.ndarray | None = None,
) -> GeneralizationSelection:
    """Select a single model or diverse ensemble using training-only validation."""
    available = {
        family: max(
            (name for name in names if name in trained),
            key=lambda name: validation_scores.get(name, -np.inf),
            default=None,
        )
        for family, names in MODEL_FAMILIES.items()
    }
    best_name = max(
        trained, key=lambda name: validation_scores.get(name, -np.inf)
    )
    selected_names = [name for name in available.values() if name is not None]
    _, class_counts = np.unique(y, return_counts=True)
    if class_counts.min() < 2:
        return GeneralizationSelection(
            model=trained[best_name],
            name=best_name,
            gate_accuracy=None,
            primary_cv_accuracy=float(validation_scores[best_name]),
            temperature=1.0,
            ensemble_used=False,
            members=[best_name],
            nll_before=None,
            nll_after=None,
        )

    indices = np.arange(len(X))
    validation_count = max(len(np.unique(y)), int(np.ceil(0.2 * len(X))))
    validation_count = min(
        validation_count, len(X) - len(np.unique(y))
    )
    if groups is not None:
        from image_splitting import split_labeled_indices

        fit_idx, val_idx, _ = split_labeled_indices(
            y,
            test_size=validation_count / len(X),
            random_state=random_state,
            groups=groups,
        )
    else:
        fit_idx, val_idx = train_test_split(
            indices,
            test_size=validation_count,
            random_state=random_state,
            stratify=y,
        )
    X_fit = X.iloc[fit_idx] if hasattr(X, "iloc") else X[fit_idx]
    y_fit = y.iloc[fit_idx] if hasattr(y, "iloc") else y[fit_idx]
    X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
    y_val = np.asarray(
        y.iloc[val_idx] if hasattr(y, "iloc") else y[val_idx]
    )
    classes = np.unique(y)

    validation_models: dict[str, Any] = {}
    for name in set(selected_names + [best_name]):
        candidate = clone(trained[name])
        candidate.fit(X_fit, y_fit)
        validation_models[name] = candidate

    best_probabilities = _aligned_probabilities(
        validation_models[best_name], X_val, classes
    )
    best_accuracy = accuracy_score(
        y_val, classes[np.argmax(best_probabilities, axis=1)]
    )
    chosen_model = trained[best_name]
    chosen_name = best_name
    chosen_probabilities = best_probabilities
    ensemble_used = False

    if len(selected_names) == len(MODEL_FAMILIES):
        validation_ensemble = DiverseProbabilityEnsemble(
            [(name, validation_models[name]) for name in selected_names],
            classes,
        )
        ensemble_probabilities = validation_ensemble.predict_proba(X_val)
        ensemble_accuracy = accuracy_score(
            y_val, classes[np.argmax(ensemble_probabilities, axis=1)]
        )
        cv_reference = validation_scores.get(best_name, best_accuracy)
        improves_same_split = ensemble_accuracy >= best_accuracy + 0.002
        consistent_with_cv = ensemble_accuracy >= cv_reference - 0.005
        if improves_same_split and consistent_with_cv:
            chosen_model = DiverseProbabilityEnsemble(
                [(name, trained[name]) for name in selected_names], classes
            )
            chosen_name = "diverse_ensemble"
            chosen_probabilities = ensemble_probabilities
            best_accuracy = float(ensemble_accuracy)
            ensemble_used = True

    temperature, nll_before, nll_after = _optimize_temperature(
        chosen_probabilities, y_val, classes
    )
    calibrated = TemperatureScaledClassifier(chosen_model, temperature)
    return GeneralizationSelection(
        model=calibrated,
        name=chosen_name,
        gate_accuracy=float(best_accuracy),
        primary_cv_accuracy=float(validation_scores[best_name]),
        temperature=temperature,
        ensemble_used=ensemble_used,
        members=selected_names if ensemble_used else [best_name],
        nll_before=nll_before,
        nll_after=nll_after,
    )
