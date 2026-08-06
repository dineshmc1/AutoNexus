"""Local, privacy-conscious FAISS meta-memory contribution and search."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


MODEL_FAMILY_NAMES = (
    {"logistic", "sgd_clf", "ridge", "sgd_reg"},
    {"xgb_clf", "lgbm_clf", "gb", "xgb_reg", "lgbm_reg", "gb_reg"},
    {
        "et_clf", "rf", "mlp_clf", "knn_clf",
        "et_reg", "rf_reg", "mlp_reg", "knn_reg",
    },
)


@dataclass
class MemoryContribution:
    contributed: bool
    status: str
    memory_dir: str
    entry_count: int
    index_backend: str
    dataset_fingerprint: str | None = None


def _score_value(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("score")
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if np.isfinite(score) else None


@contextmanager
def _file_lock(path: Path, timeout: float = 10.0) -> Iterator[None]:
    started = time.monotonic()
    descriptor = None
    while descriptor is None:
        try:
            descriptor = os.open(
                path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
        except FileExistsError:
            if time.monotonic() - started >= timeout:
                raise TimeoutError(f"Could not acquire memory lock: {path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            path.unlink()
        except FileNotFoundError:
            pass


class FAISSMetaMemory:
    """Store run metadata and vectors locally; raw data never leaves the run."""

    def __init__(self, directory: str | Path | None = None):
        self.directory = Path(
            directory or Path.home() / ".autonexus" / "memory"
        ).expanduser()
        self.metadata_path = self.directory / "entries.json"
        self.faiss_path = self.directory / "memory.faiss"
        self.numpy_path = self.directory / "memory_vectors.npz"
        self.lock_path = self.directory / ".lock"

    def _load(self) -> list[dict[str, Any]]:
        if not self.metadata_path.is_file():
            return []
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def _write(self, entries: list[dict[str, Any]]) -> str:
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.metadata_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(entries, indent=2), encoding="utf-8"
        )
        temporary.replace(self.metadata_path)
        vectors = np.asarray(
            [entry["embedding"] for entry in entries], dtype=np.float32
        )
        np.savez_compressed(self.numpy_path, vectors=vectors)
        try:
            import faiss

            index = faiss.IndexFlatL2(vectors.shape[1])
            index.add(np.ascontiguousarray(vectors))
            faiss.write_index(index, str(self.faiss_path))
            return "faiss"
        except ImportError:
            return "numpy"

    def contribute(self, entry: dict[str, Any]) -> MemoryContribution:
        embedding = np.asarray(entry["embedding"], dtype=np.float32).ravel()
        if not len(embedding):
            raise ValueError("Memory contribution requires an embedding.")
        self.directory.mkdir(parents=True, exist_ok=True)
        with _file_lock(self.lock_path):
            entries = self._load()
            fingerprint = str(entry["dataset_fingerprint"])
            duplicate_index = next(
                (
                    index
                    for index, item in enumerate(entries)
                    if item.get("dataset_fingerprint") == fingerprint
                    and item.get("best_model") == entry.get("best_model")
                ),
                None,
            )
            if duplicate_index is not None:
                existing = entries[duplicate_index]
                refresh = {
                    key: value
                    for key, value in entry.items()
                    if key not in {"embedding", "dataset_fingerprint"}
                    and value not in (None, [], {})
                }
                changed = any(existing.get(key) != value for key, value in refresh.items())
                if changed:
                    entries[duplicate_index] = {**existing, **refresh}
                    backend = self._write(entries)
                else:
                    backend = (
                        "faiss" if self.faiss_path.is_file() else "numpy"
                    )
                return MemoryContribution(
                    contributed=False,
                    status=(
                        "duplicate-metadata-refreshed"
                        if changed
                        else "duplicate-skipped"
                    ),
                    memory_dir=str(self.directory.resolve()),
                    entry_count=len(entries),
                    index_backend=backend,
                    dataset_fingerprint=fingerprint,
                )
            dimensions = {
                len(item.get("embedding", [])) for item in entries
            }
            if dimensions and dimensions != {len(embedding)}:
                raise ValueError(
                    "Memory embedding version/dimension mismatch. Use a "
                    "separate memory directory for the new schema."
                )
            sanitized = {
                **entry,
                "embedding": embedding.astype(float).tolist(),
                "contributed_at": time.time(),
            }
            entries.append(sanitized)
            backend = self._write(entries)
        return MemoryContribution(
            contributed=True,
            status="contributed",
            memory_dir=str(self.directory.resolve()),
            entry_count=len(entries),
            index_backend=backend,
            dataset_fingerprint=fingerprint,
        )

    def search(
        self, embedding: np.ndarray, *, k: int = 5
    ) -> list[dict[str, Any]]:
        entries = self._load()
        if not entries:
            return []
        query = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        vectors = np.asarray(
            [entry["embedding"] for entry in entries], dtype=np.float32
        )
        if query.shape[1] != vectors.shape[1]:
            raise ValueError("Query and memory embedding dimensions differ.")
        count = min(k, len(entries))
        try:
            import faiss

            index = (
                faiss.read_index(str(self.faiss_path))
                if self.faiss_path.is_file()
                else faiss.IndexFlatL2(vectors.shape[1])
            )
            if index.ntotal == 0:
                index.add(np.ascontiguousarray(vectors))
            distances, indices = index.search(query, count)
            pairs = zip(distances[0], indices[0])
        except ImportError:
            distances = np.sum((vectors - query) ** 2, axis=1)
            selected = np.argsort(distances)[:count]
            pairs = ((distances[index], index) for index in selected)
        return [
            {**entries[int(index)], "distance": float(distance)}
            for distance, index in pairs
            if 0 <= int(index) < len(entries)
        ]


def retrieve_search_advice(
    embedding: np.ndarray,
    current_scores: dict[str, Any],
    *,
    problem_type: str,
    embedding_version: int,
    memory_dir: str | Path | None = None,
    k: int = 7,
    max_distance: float = 0.35,
) -> dict[str, Any]:
    """Build leakage-safe model advice from compatible nearby runs."""
    empty = {
        "status": "no-compatible-neighbors",
        "neighbors_considered": 0,
        "max_distance": max_distance,
        "model_scores": {},
        "recommended_models": [],
        "penalized_models": [],
        "neighbors": [],
        "rationale": (
            "Current-dataset validation remains authoritative; memory is advisory."
        ),
    }
    try:
        neighbors = FAISSMetaMemory(memory_dir).search(embedding, k=k)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            **empty,
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
        }

    compatible = [
        neighbor
        for neighbor in neighbors
        if neighbor.get("problem_type") == problem_type
        and neighbor.get("embedding_version") == embedding_version
        and float(neighbor.get("distance", float("inf"))) <= max_distance
    ]
    if not compatible:
        return empty

    available = {
        name: score
        for name, value in current_scores.items()
        if (score := _score_value(value)) is not None
    }
    weighted: dict[str, float] = {name: 0.0 for name in available}
    weights: dict[str, float] = {name: 0.0 for name in available}
    neighbor_summaries = []
    for neighbor in compatible:
        distance = max(float(neighbor["distance"]), 0.0)
        similarity = max(0.0, 1.0 - distance / max(max_distance, 1e-12))
        evidence = neighbor.get("selection_evidence", {}) or {}
        historical = {
            name: score
            for name, value in (evidence.get("baseline_scores", {}) or {}).items()
            if (score := _score_value(value)) is not None and name in available
        }
        if historical:
            low, high = min(historical.values()), max(historical.values())
            span = max(high - low, 1e-12)
            for name, score in historical.items():
                relative = 2.0 * ((score - low) / span) - 1.0
                weighted[name] += similarity * relative
                weights[name] += similarity

        winners = set(neighbor.get("ensemble_members", []) or [])
        best_model = str(neighbor.get("best_model") or "")
        if best_model in available:
            winners.add(best_model)
        for name in winners & available.keys():
            weighted[name] += similarity * 0.5
            weights[name] += similarity * 0.5
        neighbor_summaries.append(
            {
                "distance": round(distance, 6),
                "similarity": round(similarity, 6),
                "best_model": best_model or None,
                "ensemble_members": sorted(winners),
                "dataset_fingerprint": neighbor.get("dataset_fingerprint"),
            }
        )

    model_scores = {
        name: round(weighted[name] / weights[name], 6)
        for name in available
        if weights[name] > 0
    }
    recommended = sorted(
        (name for name, score in model_scores.items() if score >= 0.2),
        key=lambda name: (model_scores[name], available[name]),
        reverse=True,
    )
    penalized = sorted(
        (name for name, score in model_scores.items() if score <= -0.35),
        key=lambda name: model_scores[name],
    )
    return {
        **empty,
        "status": "applied" if model_scores else "neighbors-without-evidence",
        "neighbors_considered": len(compatible),
        "model_scores": model_scores,
        "recommended_models": recommended,
        "penalized_models": penalized,
        "neighbors": neighbor_summaries,
    }


def apply_search_advice(
    promising: dict[str, Any],
    catalogue: dict[str, Any],
    current_scores: dict[str, Any],
    advice: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Reorder/prune a shortlist while letting current evidence veto memory."""
    selected = dict(promising)
    baseline = {
        name: score
        for name, value in current_scores.items()
        if (score := _score_value(value)) is not None
    }
    if not baseline or advice.get("status") != "applied":
        return selected, {"promoted": [], "pruned": []}
    best_current = max(baseline.values())
    promoted = []
    for name in advice.get("recommended_models", []):
        if (
            name in catalogue
            and name not in selected
            and baseline.get(name, -float("inf")) >= best_current - 0.05
        ):
            selected[name] = catalogue[name]
            promoted.append(name)

    pruned = []
    for name in advice.get("penalized_models", []):
        family = next(
            (members for members in MODEL_FAMILY_NAMES if name in members),
            set(),
        )
        only_family_member = bool(family) and len(family & selected.keys()) <= 1
        if (
            name in selected
            and len(selected) > 2
            and not only_family_member
            and baseline.get(name, best_current) < best_current - 0.03
        ):
            selected.pop(name)
            pruned.append(name)

    memory_scores = advice.get("model_scores", {})
    ordered_names = sorted(
        selected,
        key=lambda name: (
            memory_scores.get(name, 0.0),
            baseline.get(name, -float("inf")),
        ),
        reverse=True,
    )
    return (
        {name: selected[name] for name in ordered_names},
        {"promoted": promoted, "pruned": pruned},
    )


def contribute_run(
    run_dir: str | Path,
    *,
    enabled: bool = True,
    memory_dir: str | Path | None = None,
) -> MemoryContribution:
    run_path = Path(run_dir)
    if not enabled:
        return MemoryContribution(
            contributed=False,
            status="disabled",
            memory_dir=str(
                Path(memory_dir or Path.home() / ".autonexus" / "memory")
            ),
            entry_count=0,
            index_backend="none",
        )
    profile = json.loads(
        (run_path / "search_profile.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (run_path / "run.json").read_text(encoding="utf-8")
    )
    fingerprints = (
        json.loads(
            (run_path / "analysis_data" / "run_context.json").read_text(
                encoding="utf-8"
            )
        ).get("split_fingerprints", {})
        if (run_path / "analysis_data" / "run_context.json").is_file()
        else {}
    )
    fingerprint = "|".join(
        f"{key}:{value}" for key, value in sorted(fingerprints.items())
    )
    dataset_path_hash = __import__("hashlib").sha256(
        str(manifest.get("dataset", "")).encode("utf-8")
    ).hexdigest()
    fingerprint = fingerprint or f"path-sha256:{dataset_path_hash}"
    entry = {
        "dataset_fingerprint": fingerprint,
        "embedding_version": profile.get("version"),
        "embedding": profile["embedding"],
        "best_model": manifest.get("best_model"),
        "ensemble_members": manifest.get("run_summary", {}).get(
            "ensemble_members", []
        ),
        "problem_type": manifest.get("problem_type"),
        "label_column": manifest.get("label_column") or manifest.get("target"),
        "selection_evidence": {
            "baseline_scores": profile.get("baseline_scores", {}),
            "selected_cross_validated_metric": manifest.get(
                "run_summary", {}
            ).get("primary_cross_validated_metric"),
            "scope": manifest.get("run_summary", {}).get(
                "primary_cross_validated_metric_scope"
            ),
        },
        "training_seconds": manifest.get("run_summary", {}).get(
            "training_seconds"
        ),
        "dataset_path_hash": dataset_path_hash,
    }
    return FAISSMetaMemory(memory_dir).contribute(entry)


def contribution_to_dict(
    contribution: MemoryContribution,
) -> dict[str, Any]:
    return asdict(contribution)
