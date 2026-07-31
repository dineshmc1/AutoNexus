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


@dataclass
class MemoryContribution:
    contributed: bool
    status: str
    memory_dir: str
    entry_count: int
    index_backend: str
    dataset_fingerprint: str | None = None


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
            if any(
                item.get("dataset_fingerprint") == fingerprint
                and item.get("best_model") == entry.get("best_model")
                for item in entries
            ):
                backend = (
                    "faiss" if self.faiss_path.is_file() else "numpy"
                )
                return MemoryContribution(
                    contributed=False,
                    status="duplicate-skipped",
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
    metrics_path = run_path / "metrics.csv"
    metrics = []
    if metrics_path.is_file():
        import pandas as pd

        metrics = pd.read_csv(metrics_path).to_dict("records")
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
        "problem_type": manifest.get("problem_type"),
        "label_column": manifest.get("label_column") or manifest.get("target"),
        "metrics": metrics,
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
