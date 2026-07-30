"""Successive-halving selection of frozen image backbones."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from vision_backbones import (
    DEFAULT_BACKBONE_KEY,
    VISION_BACKBONES,
    VisionBackboneSpec,
    filter_backbones_for_resources,
    resolve_backbones,
)


@dataclass
class BackboneSelection:
    """Selected frozen representation plus a serializable audit trail."""

    spec: VisionBackboneSpec
    embeddings: pd.DataFrame
    labels: pd.Series
    metrics: dict[str, Any]


def _stage_indices(
    labels: Sequence[str],
    groups: Sequence[str] | None,
    fraction: float,
    random_state: int,
) -> np.ndarray:
    """Build deterministic, nested, class-covering stage samples."""
    labels_array = np.asarray(labels)
    if fraction >= 1.0:
        return np.arange(len(labels_array))
    rng = np.random.default_rng(random_state)
    selected: set[int] = set()
    groups_array = None if groups is None else np.asarray(groups)

    for label in np.unique(labels_array):
        class_indices = np.flatnonzero(labels_array == label)
        if groups_array is None:
            ordered = class_indices.copy()
            rng.shuffle(ordered)
            count = min(
                len(ordered),
                max(2, int(math.ceil(fraction * len(ordered)))),
            )
            selected.update(int(index) for index in ordered[:count])
            continue

        class_groups = np.unique(groups_array[class_indices])
        ordered_groups = class_groups.copy()
        rng.shuffle(ordered_groups)
        count = min(
            len(ordered_groups),
            max(2, int(math.ceil(fraction * len(ordered_groups)))),
        )
        chosen = set(ordered_groups[:count])
        selected.update(
            int(index)
            for index in class_indices
            if groups_array[index] in chosen
        )

    return np.asarray(sorted(selected), dtype=int)


def _probe_frozen_embeddings(
    X: pd.DataFrame,
    labels: Sequence[str],
    groups: Sequence[str] | None,
    cv: int,
    random_state: int,
) -> dict[str, float]:
    """Evaluate one representation with the same regularized linear probe."""
    y = np.asarray(labels)
    class_counts = np.unique(y, return_counts=True)[1]
    fold_count = min(cv, int(class_counts.min()))
    groups_array = None if groups is None else np.asarray(groups)
    if groups_array is not None:
        class_group_counts = [
            len(np.unique(groups_array[y == label]))
            for label in np.unique(y)
        ]
        fold_count = min(fold_count, min(class_group_counts, default=1))
    if fold_count < 2:
        raise ValueError("Backbone probe requires two samples/groups per class.")

    if groups_array is not None:
        splitter = StratifiedGroupKFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=random_state,
        )
        splits = splitter.split(X, y, groups_array)
    else:
        splitter = StratifiedKFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=random_state,
        )
        splits = splitter.split(X, y)

    accuracies: list[float] = []
    nll_values: list[float] = []
    started = time.monotonic()
    for fit_indices, validation_indices in splits:
        probe = LogisticRegression(
            C=1.0,
            max_iter=3000,
            class_weight="balanced",
            random_state=random_state,
        )
        probe.fit(X.iloc[fit_indices], y[fit_indices])
        predictions = probe.predict(X.iloc[validation_indices])
        probabilities = probe.predict_proba(X.iloc[validation_indices])
        accuracies.append(
            float(accuracy_score(y[validation_indices], predictions))
        )
        nll_values.append(
            float(
                log_loss(
                    y[validation_indices],
                    probabilities,
                    labels=probe.classes_,
                )
            )
        )
    return {
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_nll": float(np.mean(nll_values)),
        "folds": fold_count,
        "probe_seconds": time.monotonic() - started,
    }


def selection_score(
    mean_accuracy: float,
    std_accuracy: float,
    mean_nll: float,
    class_count: int,
    embedding_seconds: float,
    sample_count: int,
    ram_mb: float,
    vram_mb: float | None,
    available_ram_mb: float,
    available_vram_mb: float | None,
) -> float:
    """Combine quality, uncertainty, calibration, speed, and memory."""
    normalized_nll = mean_nll / max(math.log(max(class_count, 2)), 1e-6)
    seconds_per_thousand = embedding_seconds * 1000 / max(sample_count, 1)
    latency_penalty = 0.01 * math.log1p(seconds_per_thousand)
    ram_penalty = 0.005 * min(ram_mb / max(available_ram_mb, 1.0), 2.0)
    vram_penalty = 0.0
    if vram_mb is not None and available_vram_mb:
        vram_penalty = 0.01 * min(vram_mb / available_vram_mb, 2.0)
    return float(
        mean_accuracy
        - 0.25 * std_accuracy
        - 0.02 * normalized_nll
        - latency_penalty
        - ram_penalty
        - vram_penalty
    )


def _resource_snapshot(device: Any) -> dict[str, float | None]:
    import psutil

    process = psutil.Process()
    ram_mb = process.memory_info().rss / (1024**2)
    vram_mb = None
    if getattr(device, "type", str(device)) == "cuda":
        import torch

        vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    return {"ram_mb": ram_mb, "vram_mb": vram_mb}


def _winner_from_finalists(
    finalists: Sequence[tuple[VisionBackboneSpec, dict[str, Any]]],
) -> VisionBackboneSpec:
    """Use score normally and model size only for statistically tied quality."""
    best_accuracy = max(item[1]["mean_accuracy"] for item in finalists)
    best_std = min(
        item[1]["std_accuracy"]
        for item in finalists
        if item[1]["mean_accuracy"] == best_accuracy
    )
    folds = max(
        item[1]["folds"]
        for item in finalists
        if item[1]["mean_accuracy"] == best_accuracy
    )
    tie_margin = max(0.002, min(0.01, best_std / math.sqrt(max(folds, 1))))
    accuracy_ties = [
        item
        for item in finalists
        if item[1]["mean_accuracy"] >= best_accuracy - tie_margin
    ]
    highest_score = max(item[1]["selection_score"] for item in accuracy_ties)
    score_ties = [
        item
        for item in accuracy_ties
        if item[1]["selection_score"] >= highest_score - 0.002
    ]
    selected = min(
        score_ties,
        key=lambda item: (
            item[0].parameters_millions,
            item[1]["embedding_seconds"],
            item[0].key,
        ),
    )
    fastest = min(
        accuracy_ties,
        key=lambda item: (
            item[1]["embedding_seconds"],
            item[0].parameters_millions,
        ),
    )
    selected_is_slower = (
        selected[1]["embedding_seconds"]
        > fastest[1]["embedding_seconds"] * 1.05
    )
    meaningful_accuracy_gain = (
        selected[1]["mean_accuracy"] - fastest[1]["mean_accuracy"] >= 0.002
    )
    meaningful_nll_gain = (
        fastest[1]["mean_nll"] - selected[1]["mean_nll"] >= 0.01
    )
    if (
        selected_is_slower
        and not meaningful_accuracy_gain
        and not meaningful_nll_gain
    ):
        selected = fastest
    return selected[0]


def select_vision_backbone(
    files: Sequence[str],
    labels: Sequence[str],
    groups: Sequence[str] | None,
    requested: Sequence[str],
    device: Any,
    cache_dir: str,
    time_budget_seconds: float,
    random_state: int,
    embedder_factory: Callable[..., Any] | None = None,
) -> BackboneSelection:
    """Select a backbone on development data without touching test images."""
    import psutil
    import torch

    if len(files) != len(labels):
        raise ValueError("Backbone search files and labels must align.")
    if groups is not None and len(groups) != len(files):
        raise ValueError("Backbone search groups must align with files.")
    if len(set(labels)) < 2:
        raise ValueError("Backbone search requires at least two classes.")

    if embedder_factory is None:
        from multimodal_extractor import UniversalEmbedder

        embedder_factory = UniversalEmbedder

    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    available_vram_gb = None
    if getattr(device, "type", str(device)) == "cuda":
        available_vram_gb = (
            torch.cuda.get_device_properties(device).total_memory / (1024**3)
        )
    resolved = resolve_backbones(requested)
    candidates, resource_rejections = filter_backbones_for_resources(
        resolved,
        available_ram_gb=available_ram_gb,
        available_vram_gb=available_vram_gb,
        device_type=getattr(device, "type", str(device)),
    )
    fallback = VISION_BACKBONES[DEFAULT_BACKBONE_KEY]

    audit: dict[str, Any] = {
        "strategy": "successive-halving-frozen-linear-probe",
        "test_images_touched": False,
        "test_images_used_for_selection": False,
        "requested": list(requested),
        "resource_rejections": resource_rejections,
        "available_ram_gb": round(available_ram_gb, 3),
        "available_vram_gb": (
            None if available_vram_gb is None else round(available_vram_gb, 3)
        ),
        "stages": [],
        "failures": [],
    }
    stage_plan = ((0.10, 2, 3), (0.30, 3, 2), (1.0, 5, 1))
    vector_keys = {spec.key for spec in candidates} | {fallback.key}
    vectors: dict[str, dict[str, np.ndarray]] = {
        key: {} for key in vector_keys
    }
    embedding_costs = {key: 0.0 for key in vector_keys}
    resolved_revisions = {
        spec.key: spec.revision for spec in candidates
    }
    resolved_revisions.setdefault(fallback.key, fallback.revision)
    surviving = list(candidates)
    final_stage_results: list[tuple[VisionBackboneSpec, dict[str, Any]]] = []
    started = time.monotonic()

    for stage_number, (fraction, folds, keep_count) in enumerate(
        stage_plan, start=1
    ):
        if (
            stage_number > 1
            and time.monotonic() - started >= time_budget_seconds
        ):
            audit["budget_exhausted_after_stage"] = stage_number - 1
            break
        indices = _stage_indices(labels, groups, fraction, random_state)
        stage_files = [files[index] for index in indices]
        stage_labels = [labels[index] for index in indices]
        stage_groups = (
            None if groups is None else [groups[index] for index in indices]
        )
        stage_audit = {
            "stage": stage_number,
            "fraction": fraction,
            "samples": len(indices),
            "requested_folds": folds,
            "candidates": [],
        }
        successful: list[tuple[VisionBackboneSpec, dict[str, Any]]] = []
        attempted = 0

        for spec in surviving:
            if (
                attempted
                and time.monotonic() - started >= time_budget_seconds
            ):
                audit["budget_exhausted_during_stage"] = stage_number
                break
            attempted += 1
            missing_files = [
                filename
                for filename in stage_files
                if filename not in vectors[spec.key]
            ]
            missing_labels = [
                stage_labels[index]
                for index, filename in enumerate(stage_files)
                if filename not in vectors[spec.key]
            ]
            candidate_started = time.monotonic()
            embedder = None
            try:
                if getattr(device, "type", str(device)) == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                if missing_files:
                    embedder = embedder_factory(
                        device=device,
                        batch_size=spec.batch_size,
                        domain="general",
                        modality="vision",
                        adapter_path=None,
                        cache_dir=cache_dir,
                        model_id=spec.model_id,
                        model_revision=spec.revision,
                    )
                    missing_X, embedded_labels = embedder.embed_files(
                        missing_files,
                        missing_labels,
                        cache_key=(
                            f"backbone-search:{spec.key}:stage-{stage_number}:"
                            f"seed-{random_state}"
                        ),
                    )
                    embedding_costs[spec.key] += float(
                        getattr(
                            embedder,
                            "last_embedding_seconds",
                            time.monotonic() - candidate_started,
                        )
                    )
                    resolved_revisions[spec.key] = getattr(
                        embedder,
                        "resolved_model_revision",
                        spec.revision,
                    )
                    embedder.release()
                    embedder = None
                    if len(missing_X) != len(missing_files):
                        raise ValueError(
                            "Unreadable images changed backbone-search alignment."
                        )
                    if list(embedded_labels.astype(str)) != list(
                        map(str, missing_labels)
                    ):
                        raise ValueError(
                            "Backbone-search labels changed during embedding."
                        )
                    for filename, row in zip(
                        missing_files, missing_X.to_numpy()
                    ):
                        vectors[spec.key][filename] = row

                X = pd.DataFrame(
                    np.vstack(
                        [vectors[spec.key][filename] for filename in stage_files]
                    )
                )
                embedding_seconds = embedding_costs[spec.key]
                probe = _probe_frozen_embeddings(
                    X,
                    stage_labels,
                    stage_groups,
                    cv=folds,
                    random_state=random_state,
                )
                resources = _resource_snapshot(device)
                result = {
                    **probe,
                    "embedding_seconds": embedding_seconds,
                    "observed_ram_mb": resources["ram_mb"],
                    "observed_vram_mb": resources["vram_mb"],
                    "estimated_ram_mb": spec.estimated_ram_gb * 1024,
                    "estimated_vram_mb": (
                        spec.estimated_vram_gb * 1024
                        if getattr(device, "type", str(device)) == "cuda"
                        else None
                    ),
                    "resolved_revision": resolved_revisions[spec.key],
                }
                result["selection_score"] = selection_score(
                    result["mean_accuracy"],
                    result["std_accuracy"],
                    result["mean_nll"],
                    len(set(labels)),
                    embedding_seconds,
                    len(stage_files),
                    spec.estimated_ram_gb * 1024,
                    (
                        None
                        if getattr(device, "type", str(device)) != "cuda"
                        else spec.estimated_vram_gb * 1024
                    ),
                    available_ram_gb * 1024,
                    (
                        None
                        if available_vram_gb is None
                        else available_vram_gb * 1024
                    ),
                )
                serializable = {
                    key: (
                        round(value, 6)
                        if isinstance(value, float)
                        else value
                    )
                    for key, value in result.items()
                }
                stage_audit["candidates"].append(
                    {"key": spec.key, **serializable}
                )
                successful.append((spec, result))
                print(
                    f"[Backbone {stage_number}] {spec.key}: "
                    f"accuracy={result['mean_accuracy']:.4f} "
                    f"(+/-{result['std_accuracy']:.4f}), "
                    f"NLL={result['mean_nll']:.4f}, "
                    f"score={result['selection_score']:.4f}."
                )
            except Exception as exc:
                failure = {
                    "stage": stage_number,
                    "key": spec.key,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                audit["failures"].append(failure)
                stage_audit["candidates"].append(failure)
                print(f"[Backbone] Skipping {spec.key}: {exc}")
            finally:
                if embedder is not None:
                    embedder.release()

        if not successful:
            break
        successful.sort(
            key=lambda item: item[1]["selection_score"], reverse=True
        )
        surviving = [
            item[0] for item in successful[: min(keep_count, len(successful))]
        ]
        stage_audit["survivors"] = [item.key for item in surviving]
        audit["stages"].append(stage_audit)
        final_stage_results = successful

    if final_stage_results:
        winner = _winner_from_finalists(final_stage_results)
    else:
        winner = fallback
        audit["fallback_used"] = DEFAULT_BACKBONE_KEY
        audit["fallback_reason"] = "every requested candidate failed"
    finalization_order = [winner]
    if winner.key != fallback.key:
        finalization_order.append(fallback)
    finalized = False
    for final_candidate in finalization_order:
        embedder = None
        try:
            missing_files = [
                filename
                for filename in files
                if filename not in vectors[final_candidate.key]
            ]
            missing_labels = [
                labels[index]
                for index, filename in enumerate(files)
                if filename not in vectors[final_candidate.key]
            ]
            if missing_files:
                embedder = embedder_factory(
                    device=device,
                    batch_size=final_candidate.batch_size,
                    domain="general",
                    modality="vision",
                    adapter_path=None,
                    cache_dir=cache_dir,
                    model_id=final_candidate.model_id,
                    model_revision=final_candidate.revision,
                )
                missing_X, embedded_labels = embedder.embed_files(
                    missing_files,
                    missing_labels,
                    cache_key=(
                        f"backbone-search:{final_candidate.key}:finalize:"
                        f"seed-{random_state}"
                    ),
                )
                resolved_revisions[final_candidate.key] = getattr(
                    embedder,
                    "resolved_model_revision",
                    final_candidate.revision,
                )
                if len(missing_X) != len(missing_files) or list(
                    embedded_labels.astype(str)
                ) != list(map(str, missing_labels)):
                    raise RuntimeError(
                        "Backbone could not produce aligned full embeddings."
                    )
                for filename, row in zip(
                    missing_files, missing_X.to_numpy()
                ):
                    vectors[final_candidate.key][filename] = row
            winner = final_candidate
            finalized = True
            break
        except Exception as exc:
            audit["failures"].append(
                {
                    "stage": "finalize",
                    "key": final_candidate.key,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if final_candidate.key != fallback.key:
                audit["fallback_used"] = DEFAULT_BACKBONE_KEY
        finally:
            if embedder is not None:
                embedder.release()
    if not finalized:
        raise RuntimeError(
            "Selected backbone and CLIP fallback both failed during full "
            "development embedding."
        )

    final_X = pd.DataFrame(
        np.vstack([vectors[winner.key][filename] for filename in files])
    )
    final_X.columns = [
        f"feat_{index}" for index in range(final_X.shape[1])
    ]
    audit["selected"] = {
        **winner.to_dict(),
        "resolved_revision": resolved_revisions[winner.key],
    }
    audit["elapsed_seconds"] = round(time.monotonic() - started, 3)
    audit["test_images_touched"] = False
    audit["test_images_used_for_selection"] = False
    print(
        f"[Backbone] Selected {winner.key} ({winner.model_id}) after "
        f"{audit['elapsed_seconds']:.1f}s of development-only search."
    )
    return BackboneSelection(
        spec=winner,
        embeddings=final_X,
        labels=pd.Series(labels, name="label"),
        metrics=audit,
    )
