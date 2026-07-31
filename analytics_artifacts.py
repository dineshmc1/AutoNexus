"""Persist bounded, reproducible inputs for the generated analysis notebook."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections import Counter
from dataclasses import asdict, is_dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from data_loader import DataBundle


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value) if np.isscalar(value) and not isinstance(value, str) else False:
        return None
    return value


def _stable_sample_indices(
    labels: Sequence[Any],
    limit: int,
    random_state: int,
) -> np.ndarray:
    """Choose a deterministic, class-covering sample without random drift."""
    labels_array = np.asarray(labels).astype(str)
    if len(labels_array) <= limit:
        return np.arange(len(labels_array))
    rng = np.random.default_rng(random_state)
    selected: list[int] = []
    classes = np.unique(labels_array)
    quota = max(1, limit // max(len(classes), 1))
    for label in classes:
        candidates = np.flatnonzero(labels_array == label)
        rng.shuffle(candidates)
        selected.extend(candidates[:quota].tolist())
    selected = list(dict.fromkeys(selected))
    if len(selected) < limit:
        remaining = np.setdiff1d(
            np.arange(len(labels_array)),
            np.asarray(selected, dtype=int),
            assume_unique=False,
        )
        rng.shuffle(remaining)
        selected.extend(remaining[: limit - len(selected)].tolist())
    return np.asarray(sorted(selected[:limit]), dtype=int)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_quality(image: Any) -> dict[str, float | str]:
    """Compute inexpensive thumbnail-level quality and perceptual features."""
    from PIL import Image

    rgb = image.convert("RGB")
    rgb.thumbnail((128, 128), Image.Resampling.BILINEAR)
    values = np.asarray(rgb, dtype=np.float32)
    gray = np.asarray(rgb.convert("L"), dtype=np.float32)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    gradient_y, gradient_x = np.gradient(gray)
    blur_score = float(np.var(gradient_x) + np.var(gradient_y))
    histogram = np.bincount(gray.astype(np.uint8).ravel(), minlength=256)
    probabilities = histogram[histogram > 0] / max(histogram.sum(), 1)
    entropy = float(-(probabilities * np.log2(probabilities)).sum())

    hash_image = np.asarray(
        rgb.convert("L").resize((8, 8), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    bits = hash_image >= hash_image.mean()
    perceptual_hash = f"{int(''.join('1' if bit else '0' for bit in bits.ravel()), 2):016x}"
    return {
        "brightness": brightness,
        "contrast": contrast,
        "blur_score": blur_score,
        "entropy": entropy,
        "red_mean": float(values[..., 0].mean()),
        "green_mean": float(values[..., 1].mean()),
        "blue_mean": float(values[..., 2].mean()),
        "perceptual_hash": perceptual_hash,
    }


def audit_image_files(
    files: Sequence[str],
    labels: Sequence[str],
    splits: Sequence[str],
    groups: Sequence[str | None] | None,
    output_dir: str | Path,
    *,
    random_state: int,
    quality_sample_limit: int = 5000,
) -> pd.DataFrame:
    """Validate every image and persist a bounded pre-training quality audit."""
    from PIL import Image

    if not (len(files) == len(labels) == len(splits)):
        raise ValueError("Image audit paths, labels, and splits must align.")
    if groups is not None and len(groups) != len(files):
        raise ValueError("Image audit groups must align with paths.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    quality_indices = set(
        _stable_sample_indices(
            [
                f"{split}:{label}"
                for split, label in zip(splits, labels)
            ],
            quality_sample_limit,
            random_state,
        )
    )
    rows: list[dict[str, Any]] = []
    for index, (filename, label, split) in enumerate(
        zip(files, labels, splits)
    ):
        path = Path(filename)
        row: dict[str, Any] = {
            "path": str(path.resolve()),
            "label": str(label),
            "split": str(split),
            "group": (
                ""
                if groups is None or groups[index] is None
                else str(groups[index])
            ),
            "extension": path.suffix.lower(),
            "file_size_bytes": path.stat().st_size if path.exists() else np.nan,
            "readable": False,
            "width": np.nan,
            "height": np.nan,
            "aspect_ratio": np.nan,
            "mode": "",
            "format": "",
            "quality_sampled": index in quality_indices,
            "error": "",
        }
        try:
            with Image.open(path) as image:
                row.update(
                    width=int(image.width),
                    height=int(image.height),
                    aspect_ratio=float(image.width / max(image.height, 1)),
                    mode=str(image.mode),
                    format=str(image.format or path.suffix.lstrip(".")),
                )
                image.verify()
            row["readable"] = True
            if index in quality_indices:
                with Image.open(path) as image:
                    row.update(_image_quality(image))
        except (OSError, ValueError, SyntaxError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    inventory = pd.DataFrame(rows)
    inventory["sha256"] = ""
    readable = inventory[inventory["readable"]]
    duplicate_sizes = {
        int(size)
        for size, count in Counter(readable["file_size_bytes"]).items()
        if count > 1 and pd.notna(size)
    }
    for index in readable.index[
        readable["file_size_bytes"].isin(duplicate_sizes)
    ]:
        try:
            inventory.at[index, "sha256"] = _sha256(
                Path(inventory.at[index, "path"])
            )
        except OSError as exc:
            inventory.at[index, "error"] = f"{type(exc).__name__}: {exc}"
            inventory.at[index, "readable"] = False

    inventory["exact_duplicate"] = (
        inventory["sha256"].ne("")
        & inventory["sha256"].duplicated(keep=False)
    )
    inventory["near_duplicate_of"] = ""
    sampled = inventory[
        inventory.get("perceptual_hash", pd.Series(index=inventory.index)).notna()
    ]
    if len(sampled) >= 2:
        try:
            from sklearn.neighbors import NearestNeighbors

            bit_vectors = np.asarray(
                [
                    [int(bit) for bit in f"{int(value, 16):064b}"]
                    for value in sampled["perceptual_hash"].astype(str)
                ],
                dtype=np.uint8,
            )
            neighbors = NearestNeighbors(
                n_neighbors=2, metric="hamming", algorithm="brute"
            ).fit(bit_vectors)
            distances, indices = neighbors.kneighbors(bit_vectors)
            sampled_indices = sampled.index.to_numpy()
            for position, distance in enumerate(distances[:, 1]):
                if distance <= 5 / 64:
                    source = sampled_indices[position]
                    neighbor = sampled_indices[indices[position, 1]]
                    if inventory.at[source, "sha256"] != inventory.at[
                        neighbor, "sha256"
                    ]:
                        inventory.at[source, "near_duplicate_of"] = inventory.at[
                            neighbor, "path"
                        ]
        except (ImportError, ValueError):
            pass

    inventory.to_csv(output / "data_index.csv", index=False)
    quality_summary = {
        "files_scanned": len(inventory),
        "readable_files": int(inventory["readable"].sum()),
        "unreadable_files": int((~inventory["readable"]).sum()),
        "quality_sample_limit": quality_sample_limit,
        "quality_samples": int(inventory["quality_sampled"].sum()),
        "exact_duplicate_files": int(inventory["exact_duplicate"].sum()),
        "near_duplicate_candidates": int(
            inventory["near_duplicate_of"].ne("").sum()
        ),
    }
    (output / "data_quality.json").write_text(
        json.dumps(quality_summary, indent=2), encoding="utf-8"
    )
    return inventory


def _aligned_probabilities(
    model: Any,
    X: pd.DataFrame,
    classes: np.ndarray,
) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X), dtype=float)
        model_classes = np.asarray(model.classes_)
        aligned = np.zeros((len(X), len(classes)), dtype=float)
        for source, label in enumerate(model_classes):
            matches = np.flatnonzero(classes == label)
            if len(matches):
                aligned[:, int(matches[0])] = probabilities[:, source]
        row_sums = aligned.sum(axis=1, keepdims=True)
        return aligned / np.where(row_sums == 0, 1.0, row_sums)
    predictions = np.asarray(model.predict(X))
    return (predictions[:, None] == classes[None, :]).astype(float)


def _fingerprint(rows: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in rows.fillna("").astype(str).itertuples(index=False, name=None):
        digest.update("\x1f".join(row).encode("utf-8", errors="replace"))
        digest.update(b"\n")
    return digest.hexdigest()


def persist_run_analytics(
    *,
    output_dir: str | Path,
    config: Any,
    bundle: DataBundle,
    best_name: str,
    best_model: Any,
    X_train_final: pd.DataFrame,
    X_test_final: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    results: pd.DataFrame,
    baseline_scores: dict[str, dict[str, float]],
    validation_scores: dict[str, float],
    training_diagnostics: dict[str, dict[str, Any]],
    summary: dict[str, Any],
    max_embedding_samples: int = 5000,
    max_learning_samples: int = 5000,
) -> dict[str, str]:
    """Save notebook inputs without serializing large arrays into run.json."""
    output = Path(output_dir) / "analysis_data"
    output.mkdir(parents=True, exist_ok=True)
    class_names = bundle.metadata.get("class_names", [])
    is_classification = bundle.problem_type == "classification"
    if is_classification:
        classes = np.unique(
            np.concatenate([np.asarray(y_train), np.asarray(y_test)])
        )
        calibrated_test = _aligned_probabilities(
            best_model, X_test_final, classes
        )
        raw_model = getattr(best_model, "estimator", best_model)
        raw_test = _aligned_probabilities(raw_model, X_test_final, classes)
        test_predictions = classes[np.argmax(calibrated_test, axis=1)]
    else:
        classes = np.asarray([], dtype=float)
        calibrated_test = np.empty((len(y_test), 0), dtype=float)
        raw_test = calibrated_test.copy()
        test_predictions = np.asarray(best_model.predict(X_test_final))

    def label_name(value: Any) -> str:
        try:
            index = int(value)
            if 0 <= index < len(class_names):
                return str(class_names[index])
        except (TypeError, ValueError):
            pass
        return str(value)

    test_groups = (
        [""] * len(y_test)
        if bundle.groups_test is None
        else bundle.groups_test.astype(str).tolist()
    )
    test_row_ids = (
        [str(index) for index in range(len(y_test))]
        if bundle.row_ids_test is None
        else bundle.row_ids_test.astype(str).tolist()
    )
    if is_classification:
        confidence = calibrated_test.max(axis=1)
        raw_confidence = raw_test.max(axis=1)
        probability_entropy = -np.sum(
            calibrated_test * np.log(np.clip(calibrated_test, 1e-12, 1.0)),
            axis=1,
        )
    else:
        confidence = np.full(len(y_test), np.nan)
        raw_confidence = np.full(len(y_test), np.nan)
        probability_entropy = np.full(len(y_test), np.nan)
    prediction_index = pd.DataFrame(
        {
            "split": "test",
            "row_id": test_row_ids,
            "group": test_groups,
            "y_true": np.asarray(y_test),
            "y_pred": test_predictions,
            "true_label": [label_name(value) for value in y_test],
            "predicted_label": [
                label_name(value) for value in test_predictions
            ],
            "correct": (
                np.asarray(y_test) == test_predictions
                if is_classification
                else np.full(len(y_test), np.nan)
            ),
            "absolute_error": (
                np.full(len(y_test), np.nan)
                if is_classification
                else np.abs(np.asarray(y_test) - test_predictions)
            ),
            "confidence": confidence,
            "raw_confidence": raw_confidence,
            "uncertainty_entropy": probability_entropy,
        }
    )
    prediction_index.to_csv(output / "prediction_index.csv", index=False)
    np.savez_compressed(
        output / "test_probabilities.npz",
        y_true=np.asarray(y_test),
        y_pred=test_predictions,
        classes=classes,
        class_names=np.asarray([label_name(value) for value in classes]),
        calibrated=calibrated_test.astype(np.float32),
        uncalibrated=raw_test.astype(np.float32),
    )

    leaderboard_rows: list[dict[str, Any]] = []
    model_names = sorted(
        set(baseline_scores) | set(validation_scores) | {best_name}
    )
    held_out_row = (
        results.loc[results["model"] == best_name].iloc[0].to_dict()
        if best_name in set(results["model"])
        else {}
    )
    for name in model_names:
        baseline = baseline_scores.get(name, {})
        diagnostic = training_diagnostics.get(name, {})
        row = {
            "model": name,
            "selected": name == best_name,
            "baseline_score": baseline.get("score"),
            "baseline_fit_seconds": baseline.get("time"),
            "cv_mean": validation_scores.get(name),
            "cv_std": diagnostic.get("cv_std"),
            "cv_folds_completed": diagnostic.get("folds_completed"),
            "training_seconds": diagnostic.get("total_seconds"),
            "observed_process_ram_mb": diagnostic.get(
                "observed_process_ram_mb"
            ),
        }
        if name == best_name:
            row.update(
                {
                    f"test_{key}": value
                    for key, value in held_out_row.items()
                    if key != "model"
                }
            )
        leaderboard_rows.append(row)
    pd.DataFrame(leaderboard_rows).to_csv(
        output / "model_leaderboard.csv", index=False
    )

    raw_embeddings = pd.concat(
        [bundle.X_train, bundle.X_test], ignore_index=True
    )
    all_embeddings = raw_embeddings.select_dtypes(include=[np.number])
    if all_embeddings.shape[1] == 0:
        encoded_columns = {}
        for column in raw_embeddings.columns[:128]:
            encoded_columns[str(column)] = pd.factorize(
                raw_embeddings[column].astype(str), sort=True
            )[0]
        all_embeddings = pd.DataFrame(encoded_columns)
    all_labels = np.concatenate([np.asarray(y_train), np.asarray(y_test)])
    all_splits = np.asarray(
        ["development_cv"] * len(y_train) + ["test"] * len(y_test)
    )
    train_groups = (
        [""] * len(y_train)
        if bundle.groups_train is None
        else bundle.groups_train.astype(str).tolist()
    )
    all_groups = np.asarray(train_groups + test_groups)
    all_row_ids = np.asarray(
        (
            [str(index) for index in range(len(y_train))]
            if bundle.row_ids_train is None
            else bundle.row_ids_train.astype(str).tolist()
        )
        + test_row_ids
    )
    sample_indices = _stable_sample_indices(
        [f"{split}:{label}" for split, label in zip(all_splits, all_labels)],
        max_embedding_samples,
        int(getattr(config, "random_state", 42)),
    )
    np.savez_compressed(
        output / "embedding_sample.npz",
        X=all_embeddings.iloc[sample_indices].to_numpy(dtype=np.float16),
        y=all_labels[sample_indices],
        split=all_splits[sample_indices],
        group=all_groups[sample_indices],
        row_id=all_row_ids[sample_indices],
        feature_names=np.asarray(all_embeddings.columns.astype(str)),
        source_count=np.asarray(len(all_embeddings)),
    )

    learning_indices = _stable_sample_indices(
        y_train,
        max_learning_samples,
        int(getattr(config, "random_state", 42)),
    )
    learning_sample = X_train_final.iloc[learning_indices].copy()
    learning_sample["__target__"] = np.asarray(y_train)[learning_indices]
    learning_sample.to_csv(output / "learning_curve_sample.csv", index=False)

    fingerprints: dict[str, str] = {}
    data_index_path = output / "data_index.csv"
    if data_index_path.is_file():
        data_index = pd.read_csv(data_index_path)
        for split, frame in data_index.groupby("split", dropna=False):
            fingerprints[str(split)] = _fingerprint(
                frame[["path", "label", "group"]]
            )
    else:
        fingerprints["development_cv"] = _fingerprint(
            pd.DataFrame(
                {
                    "row_id": all_row_ids[: len(y_train)],
                    "label": np.asarray(y_train),
                    "group": train_groups,
                }
            )
        )
        fingerprints["test"] = _fingerprint(
            pd.DataFrame(
                {
                    "row_id": test_row_ids,
                    "label": np.asarray(y_test),
                    "group": test_groups,
                }
            )
        )

    package_versions = {}
    for package in (
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "torch",
        "transformers",
        "peft",
        "Pillow",
    ):
        try:
            package_versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            package_versions[package] = None
    hardware: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version,
    }
    try:
        import psutil

        hardware.update(
            logical_cpus=psutil.cpu_count(),
            physical_cpus=psutil.cpu_count(logical=False),
            total_ram_mb=round(psutil.virtual_memory().total / (1024**2), 1),
        )
    except ImportError:
        pass
    try:
        import torch

        hardware["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            hardware["gpu"] = torch.cuda.get_device_name(0)
            hardware["cuda_version"] = torch.version.cuda
    except ImportError:
        hardware["cuda_available"] = False

    cache_state: dict[str, Any] = {}
    for name, directory in {
        "embeddings": Path(output_dir) / ".cache" / "embeddings",
        "preprocessing": Path(output_dir) / ".cache" / "preprocessing",
    }.items():
        cache_files = (
            [path for path in directory.rglob("*") if path.is_file()]
            if directory.is_dir()
            else []
        )
        cache_state[name] = {
            "path": str(directory.resolve()),
            "exists": directory.is_dir(),
            "files": len(cache_files),
            "bytes": sum(path.stat().st_size for path in cache_files),
        }

    context = {
        "config": _json_safe(config),
        "summary": _json_safe(summary),
        "best_model": best_name,
        "problem_type": bundle.problem_type,
        "target_name": bundle.target_name,
        "class_names": class_names,
        "split_fingerprints": fingerprints,
        "package_versions": package_versions,
        "hardware": hardware,
        "cache_state": cache_state,
        "artifacts": {
            "model": str((Path(output_dir) / "best_model.joblib").resolve()),
            "metrics": str((Path(output_dir) / "metrics.csv").resolve()),
            "analysis_data": str(output.resolve()),
        },
    }
    (output / "run_context.json").write_text(
        json.dumps(context, indent=2), encoding="utf-8"
    )
    return {
        "analysis_dir": str(output.resolve()),
        "run_context": str((output / "run_context.json").resolve()),
        "prediction_index": str(
            (output / "prediction_index.csv").resolve()
        ),
        "probabilities": str(
            (output / "test_probabilities.npz").resolve()
        ),
        "embedding_sample": str(
            (output / "embedding_sample.npz").resolve()
        ),
        "leaderboard": str((output / "model_leaderboard.csv").resolve()),
        "learning_sample": str(
            (output / "learning_curve_sample.csv").resolve()
        ),
        "data_index": (
            str(data_index_path.resolve()) if data_index_path.is_file() else ""
        ),
    }
