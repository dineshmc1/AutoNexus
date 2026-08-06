"""Regression tests for calibrated ensemble explainability."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from explainer import _shap_reference_pipeline, run_explanations
from generalization import DiverseProbabilityEnsemble, TemperatureScaledClassifier


def _pipeline(model):
    return Pipeline([("model", model)])


def test_calibrated_ensemble_produces_postfit_global_importance(tmp_path):
    random = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "signal": random.normal(size=80),
            "noise": random.normal(size=80),
        }
    )
    y = pd.Series((X["signal"] > 0).astype(int))
    members = [
        ("logistic", _pipeline(LogisticRegression()).fit(X, y)),
        (
            "et_clf",
            _pipeline(
                ExtraTreesClassifier(n_estimators=20, random_state=42)
            ).fit(X, y),
        ),
    ]
    ensemble = TemperatureScaledClassifier(
        DiverseProbabilityEnsemble(members, np.asarray([0, 1])),
        temperature=1.1,
    )

    paths = run_explanations(
        ensemble,
        X,
        y,
        output_dir=str(tmp_path),
        use_shap=False,
    )
    reference, scope, name = _shap_reference_pipeline(ensemble)

    assert paths["feature_importance_status"] == "generated"
    assert (tmp_path / "feature_importance.png").is_file()
    assert scope == "ensemble_member_reference"
    assert name == "et_clf"
    assert reference is members[1][1]
