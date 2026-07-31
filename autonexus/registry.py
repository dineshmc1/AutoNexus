"""Filesystem model registry with champion, challenger, and rollback support."""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RegisteredVersion:
    name: str
    version: str
    path: Path
    stage: str


class ModelRegistry:
    def __init__(self, root: str | Path = "autonexus_registry"):
        self.root = Path(root).expanduser().resolve()
        self.index_path = self.root / "registry.json"

    def _load(self) -> dict:
        if not self.index_path.is_file():
            return {"models": {}}
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _save(self, index: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.index_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(index, indent=2), encoding="utf-8")
        temporary.replace(self.index_path)

    def register(
        self,
        run_dir: str | Path,
        *,
        name: str,
        version: str | None = None,
        stage: str = "challenger",
    ) -> RegisteredVersion:
        source = Path(run_dir).resolve()
        version = version or (
            f"{time.strftime('%Y%m%d-%H%M%S')}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        destination = self.root / name / version
        destination.mkdir(parents=True, exist_ok=False)
        for relative in (
            "model.pkl",
            "best_model.joblib",
            "run.json",
            "metrics.csv",
            "search_profile.json",
            "analysis.ipynb",
            "framework.json",
        ):
            path = source / relative
            if path.is_file():
                shutil.copy2(path, destination / path.name)
        for relative in (
            Path("report") / "explanation.md",
            Path("monitoring") / "baseline.json",
        ):
            path = source / relative
            if path.is_file():
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        index = self._load()
        model = index["models"].setdefault(
            name, {"versions": {}, "champion": None, "history": []}
        )
        model["versions"][version] = {
            "path": str(destination),
            "stage": stage,
            "registered_at": time.time(),
        }
        self._save(index)
        return RegisteredVersion(name, version, destination, stage)

    def promote(self, name: str, version: str) -> RegisteredVersion:
        index = self._load()
        model = index["models"][name]
        if version not in model["versions"]:
            raise KeyError(f"Unknown model version: {name}/{version}")
        previous = model.get("champion")
        if previous:
            model["history"].append(previous)
            model["versions"][previous]["stage"] = "archived"
        model["champion"] = version
        model["versions"][version]["stage"] = "champion"
        self._save(index)
        return RegisteredVersion(
            name,
            version,
            Path(model["versions"][version]["path"]),
            "champion",
        )

    def rollback(self, name: str) -> RegisteredVersion:
        index = self._load()
        model = index["models"][name]
        if not model["history"]:
            raise RuntimeError(f"No rollback version exists for {name}.")
        version = model["history"].pop()
        current = model.get("champion")
        if current:
            model["versions"][current]["stage"] = "challenger"
        model["champion"] = version
        model["versions"][version]["stage"] = "champion"
        self._save(index)
        return RegisteredVersion(
            name,
            version,
            Path(model["versions"][version]["path"]),
            "champion",
        )

    def champion(self, name: str) -> RegisteredVersion | None:
        index = self._load()
        model = index["models"].get(name)
        if not model or not model.get("champion"):
            return None
        version = model["champion"]
        return RegisteredVersion(
            name,
            version,
            Path(model["versions"][version]["path"]),
            "champion",
        )
