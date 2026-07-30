"""Fast smoke tests for the unified CLI and calibration wrappers."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from generalization import TemperatureScaledClassifier
from main import _config_from_args, build_parser


def test_cli_allows_interactive_dataset_and_target():
    args = build_parser().parse_args([])

    assert args.dataset is None
    assert args.target is None


def test_temperature_scaling_preserves_predictions():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 4))
    y = (X[:, 0] > 0).astype(int)
    estimator = LogisticRegression().fit(X, y)
    calibrated = TemperatureScaledClassifier(estimator, temperature=2.5)

    np.testing.assert_array_equal(
        estimator.predict(X), calibrated.predict(X)
    )
    np.testing.assert_allclose(
        calibrated.predict_proba(X).sum(axis=1), 1.0
    )


def test_prompt_accepts_trailing_flags_and_train_folder(monkeypatch, tmp_path):
    dataset_root = tmp_path / "actions"
    train_dir = dataset_root / "train"
    (train_dir / "walking").mkdir(parents=True)
    (dataset_root / "test" / "walking").mkdir(parents=True)
    entered = (
        f"{train_dir} --adapt-lora --no-report "
        "--backbones clip,dinov2 --backbone-time 2m"
    )
    monkeypatch.setattr("builtins.input", lambda _: entered)

    config = _config_from_args(build_parser().parse_args([]))

    assert config.dataset == dataset_root.resolve()
    assert config.adapt_lora is True
    assert config.report is False
    assert config.backbones == ["clip", "dinov2"]
    assert config.backbone_time_seconds == 120
