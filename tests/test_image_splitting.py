"""Tests for automatic image grouping and representation gating."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

from image_splitting import infer_image_groups, split_labeled_indices
from backbone_selector import (
    _stage_indices,
    select_vision_backbone,
)
from main import _probe_image_representation
from model_trainer import CLASSIFICATION_MODELS, _train_and_evaluate
from vision_backbones import VISION_BACKBONES


def test_generic_image_ids_do_not_create_false_groups(tmp_path):
    files = [
        str(tmp_path / label / f"Image_{index}.jpg")
        for label in ("walk", "run")
        for index in range(10)
    ]
    labels = [label for label in ("walk", "run") for _ in range(10)]

    groups, method = infer_image_groups(files, labels, tmp_path)

    assert groups is None
    assert method.startswith("stratified-")


def test_video_folders_are_kept_on_one_side_of_split(tmp_path):
    files: list[str] = []
    labels: list[str] = []
    for label in ("walk", "run"):
        for video in range(5):
            for frame in range(3):
                files.append(
                    str(
                        tmp_path
                        / label
                        / f"video_{video}"
                        / f"frame_{frame}.jpg"
                    )
                )
                labels.append(label)

    groups, _ = infer_image_groups(files, labels, tmp_path)
    train_indices, test_indices, method = split_labeled_indices(
        labels, test_size=0.2, random_state=42, groups=groups
    )

    assert method == "stratified-group"
    assert groups is not None
    assert set(np.asarray(groups)[train_indices]).isdisjoint(
        set(np.asarray(groups)[test_indices])
    )
    assert set(np.asarray(labels)[train_indices]) == {"walk", "run"}
    assert set(np.asarray(labels)[test_indices]) == {"walk", "run"}


def test_representation_probe_rewards_generalizable_signal():
    rng = np.random.default_rng(42)
    y_fit = pd.Series(np.tile(["a", "b"], 50))
    y_gate = pd.Series(np.tile(["a", "b"], 20))
    frozen_fit = pd.DataFrame(rng.normal(size=(100, 4)))
    frozen_gate = pd.DataFrame(rng.normal(size=(40, 4)))
    adapted_fit = frozen_fit.copy()
    adapted_gate = frozen_gate.copy()
    adapted_fit[0] = (y_fit == "b").astype(float) * 4 - 2
    adapted_gate[0] = (y_gate == "b").astype(float) * 4 - 2

    frozen = _probe_image_representation(
        frozen_fit, y_fit, frozen_gate, y_gate
    )
    adapted = _probe_image_representation(
        adapted_fit, y_fit, adapted_gate, y_gate
    )

    assert adapted["accuracy"] > frozen["accuracy"]
    assert adapted["nll"] < frozen["nll"]


def test_extra_trees_defaults_are_regularized():
    model = CLASSIFICATION_MODELS["et_clf"]

    assert model.max_depth == 24
    assert model.min_samples_leaf == 2
    assert model.bootstrap is True
    assert model.oob_score is True


def test_downstream_group_cv_blocks_video_identity_memorization():
    group_count = 20
    groups = np.repeat([f"video-{index}" for index in range(group_count)], 2)
    labels = np.repeat(np.arange(group_count) % 2, 2)
    values = np.zeros((len(groups), group_count), dtype=float)
    values[np.arange(len(groups)), np.repeat(np.arange(group_count), 2)] = 1.0
    X = pd.DataFrame(
        values, columns=[f"group_{index}" for index in range(group_count)]
    )
    y = pd.Series(labels)
    preprocessor = ColumnTransformer(
        [("numeric", "passthrough", list(X.columns))]
    )
    estimator = LogisticRegression(max_iter=1000)

    ordinary_score, _, _ = _train_and_evaluate(
        "logistic",
        estimator,
        preprocessor,
        X,
        y,
        "classification",
        cv=5,
        start=0.0,
        max_time_seconds=None,
    )
    grouped_score, _, _ = _train_and_evaluate(
        "logistic",
        estimator,
        preprocessor,
        X,
        y,
        "classification",
        cv=5,
        start=0.0,
        max_time_seconds=None,
        groups=groups,
    )

    assert ordinary_score > grouped_score + 0.2


def test_backbone_stage_samples_are_nested_and_group_preserving():
    labels = np.repeat(["walk", "run"], 30)
    groups = np.asarray(
        [
            f"{label}-video-{group}"
            for label in ("walk", "run")
            for group in range(10)
            for _ in range(3)
        ]
    )

    stage_one = _stage_indices(labels, groups, 0.1, random_state=42)
    stage_two = _stage_indices(labels, groups, 0.3, random_state=42)

    assert set(stage_one).issubset(set(stage_two))
    for indices in (stage_one, stage_two):
        selected_groups = set(groups[indices])
        for group in selected_groups:
            assert set(np.flatnonzero(groups == group)).issubset(set(indices))


class _FakeEmbedder:
    def __init__(self, *, model_id, **_):
        self.model_id = model_id

    def embed_files(self, files, labels, cache_key):
        del cache_key
        if self.model_id == VISION_BACKBONES["dinov2"].model_id:
            signal = np.asarray([label == "run" for label in labels], dtype=float)
            values = np.column_stack([signal * 4 - 2, np.ones(len(labels))])
        else:
            values = np.zeros((len(labels), 2), dtype=float)
        return pd.DataFrame(values), pd.Series(labels)

    def release(self):
        return None


def test_successive_halving_selects_generalizable_backbone():
    files = [f"image-{index}.jpg" for index in range(80)]
    labels = ["walk" if index % 2 == 0 else "run" for index in range(80)]

    selection = select_vision_backbone(
        files,
        labels,
        groups=None,
        requested=["clip", "dinov2", "resnet"],
        device=SimpleNamespace(type="cpu"),
        cache_dir="unused",
        time_budget_seconds=60,
        random_state=42,
        embedder_factory=_FakeEmbedder,
    )

    assert selection.spec.key == "dinov2"
    assert len(selection.embeddings) == len(files)
    assert selection.metrics["test_images_touched"] is False
    assert len(selection.metrics["stages"]) == 3


class _FinalizeFailingEmbedder(_FakeEmbedder):
    def embed_files(self, files, labels, cache_key):
        if (
            self.model_id == VISION_BACKBONES["dinov2"].model_id
            and len(files) > 20
        ):
            raise RuntimeError("simulated full-data failure")
        return super().embed_files(files, labels, cache_key)


def test_backbone_finalization_falls_back_to_clip():
    files = [f"image-{index}.jpg" for index in range(80)]
    labels = ["walk" if index % 2 == 0 else "run" for index in range(80)]

    selection = select_vision_backbone(
        files,
        labels,
        groups=None,
        requested=["dinov2"],
        device=SimpleNamespace(type="cpu"),
        cache_dir="unused",
        time_budget_seconds=1e-9,
        random_state=42,
        embedder_factory=_FinalizeFailingEmbedder,
    )

    assert selection.spec.key == "clip"
    assert selection.metrics["fallback_used"] == "clip"


def test_clip_model_output_is_normalized_to_tensor():
    torch = pytest.importorskip("torch")
    from multimodal_extractor import extract_vision_features

    class Processor:
        def __call__(self, *, images, return_tensors):
            assert images == ["image"]
            assert return_tensors == "pt"
            return {"pixel_values": torch.ones((1, 3, 2, 2))}

    class ClipModel:
        def get_image_features(self, **inputs):
            assert "pixel_values" in inputs
            return SimpleNamespace(pooler_output=torch.ones((1, 8)))

    features = extract_vision_features(
        ClipModel(), Processor(), ["image"], torch.device("cpu")
    )

    assert torch.is_tensor(features)
    assert tuple(features.shape) == (1, 8)
