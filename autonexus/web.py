"""Local-first web control plane for AutoNexus training runs."""

import argparse
import json
import logging
import os
import re
import secrets
import shutil
import threading
import time
import traceback
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import urlparse

import pandas as pd
import numpy as np

from .config import PRESETS
from .exceptions import CapabilityError, ConfigurationError
from .llm import LLMProvider, LiteLLMProvider, OllamaProvider
from .web_auth import (
    AuthenticationError,
    StudioAuthenticator,
    authenticator_from_env,
)
from .web_storage import FirebaseStorageMirror
from .web_store import SQLiteRunStore


LOGGER = logging.getLogger("autonexus.web")
STATIC_DIR = Path(__file__).with_name("web_static")
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
}
TABULAR_EXTENSIONS = {".csv", ".xlsx", ".xls"}
VALID_TASKS = {"auto", "classification", "regression", "vision"}
VALID_BACKBONES = {"auto", "clip", "dinov2", "resnet", "siglip"}
VALID_LLM_MODES = {"offline", "environment", "byok", "ollama"}
VALID_EXECUTION_TARGETS = {"cloud", "local_agent"}
HOSTED_LLM_PROVIDERS = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "openrouter": "openrouter",
    "groq": "groq",
    "mistral": "mistral",
    "custom": "",
}
ARTIFACT_PATHS = {
    "manifest": Path("run.json"),
    "model": Path("model.pkl"),
    "notebook": Path("analysis.ipynb"),
    "analytics_bundle": Path("analysis_bundle.zip"),
    "explanation": Path("report") / "explanation.md",
    "html_report": Path("report") / "report.html",
    "search_profile": Path("search_profile.json"),
    "framework": Path("framework.json"),
    "drift_baseline": Path("monitoring") / "baseline.json",
}
ARTIFACT_MEDIA_TYPES = {
    "manifest": "application/json",
    "model": "application/octet-stream",
    "notebook": "application/x-ipynb+json",
    "analytics_bundle": "application/zip",
    "explanation": "text/markdown; charset=utf-8",
    "html_report": "text/html; charset=utf-8",
    "search_profile": "application/json",
    "framework": "application/json",
    "drift_baseline": "application/json",
}
EVIDENCE_PATHS = {
    "feature_importance": Path("report") / "explanations" / "feature_importance.png",
    "shap_summary": Path("report") / "explanations" / "shap_summary.png",
    "shap_importance": Path("report") / "explanations" / "shap_importance.png",
    "shap_dependence": Path("report") / "explanations" / "shap_dependence.png",
    "shap_waterfall": Path("report") / "explanations" / "shap_waterfall.png",
    "shap_decision": Path("report") / "explanations" / "shap_decision.png",
}
DOCUMENT_PATHS = {
    "thesis": STATIC_DIR / "documents" / "thesis.pdf",
    "research-paper": STATIC_DIR / "documents" / "research-paper.pdf",
    "framework-guide": STATIC_DIR / "documents" / "framework-guide.pdf",
}


def default_workspace() -> Path:
    """Return a user-scoped Studio data directory outside the source tree."""
    explicit = os.getenv("AUTONEXUS_WEB_WORKSPACE")
    if explicit:
        return Path(explicit).expanduser().resolve()
    if os.name == "nt" and os.getenv("LOCALAPPDATA"):
        data_home = Path(os.environ["LOCALAPPDATA"])
    else:
        data_home = Path(
            os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share")
        )
    return (data_home / "AutoNexus" / "studio-runs").resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _safe_upload_path(filename: str) -> Path:
    """Normalize a browser filename while preserving safe folder structure."""
    normalized = filename.replace("\\", "/")
    parts = [part for part in PurePosixPath(normalized).parts if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        raise ValueError("Upload contains an unsafe file path.")
    safe_parts = []
    for part in parts:
        cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", part).strip(" .")
        if not cleaned:
            raise ValueError("Upload contains an invalid file name.")
        safe_parts.append(cleaned)
    return Path(*safe_parts)


def _target_candidates(sample: pd.DataFrame) -> list[str]:
    if sample.empty:
        return list(sample.columns[-5:])
    rows = max(len(sample), 1)
    scored: list[tuple[float, str]] = []
    for position, column in enumerate(sample.columns):
        unique = sample[column].nunique(dropna=True)
        ratio = unique / rows
        score = position / max(len(sample.columns), 1)
        if 1 < unique <= 50:
            score += 1.5
        elif ratio < 0.2:
            score += 0.6
        name = str(column).lower()
        if any(token in name for token in ("target", "label", "class", "outcome")):
            score += 3.0
        scored.append((score, str(column)))
    return [column for _, column in sorted(scored, reverse=True)[:8]]


class _RedactingLLMProvider(LLMProvider):
    """Prevent ephemeral credentials from appearing in provider failures."""

    def __init__(
        self,
        provider: LLMProvider,
        secrets_to_redact: list[str],
        *,
        provenance: dict[str, Any] | None = None,
    ):
        self.provider = provider
        self.secrets_to_redact = [secret for secret in secrets_to_redact if secret]
        self.report_provenance = dict(provenance or {})

    def generate(self, prompt: str, *, context: dict[str, Any]) -> str:
        try:
            return self.provider.generate(prompt, context=context)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            for secret in self.secrets_to_redact:
                message = message.replace(secret, "[REDACTED]")
            raise RuntimeError(message) from None


def _validated_url(value: str, *, label: str) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigurationError(f"{label} must be a valid HTTP(S) URL.")
    return normalized


def _validate_llm_config(raw: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """Separate persistable LLM metadata from ephemeral credentials."""
    if isinstance(raw, bool):
        raw = {"mode": "environment" if raw else "offline"}
    if not isinstance(raw, dict):
        raise ConfigurationError("Invalid LLM configuration.")
    mode = str(raw.get("mode", "environment")).strip().lower()
    if mode not in VALID_LLM_MODES:
        raise ConfigurationError(f"Unknown LLM mode: {mode}")
    public = {
        "mode": mode,
        "provider": None,
        "model": None,
        "api_base": None,
        "credential_storage": "none",
    }
    secrets_for_run: dict[str, str] = {}
    if mode == "offline":
        return public, secrets_for_run
    if mode == "environment":
        public.update(provider="environment", credential_storage="server_environment")
        return public, secrets_for_run
    if mode == "ollama":
        model = str(raw.get("model") or "").strip()
        if not model:
            raise ConfigurationError("An Ollama model name is required.")
        base_url = _validated_url(
            str(raw.get("api_base") or "http://localhost:11434"),
            label="Ollama endpoint",
        )
        public.update(provider="ollama", model=model, api_base=base_url)
        return public, secrets_for_run

    provider = str(raw.get("provider") or "").strip().lower()
    if provider not in HOSTED_LLM_PROVIDERS:
        raise ConfigurationError("Choose a supported hosted LLM provider.")
    model = str(raw.get("model") or "").strip()
    api_key = str(raw.get("api_key") or "").strip()
    if not model:
        raise ConfigurationError("An LLM model identifier is required for BYOK.")
    if not api_key:
        raise ConfigurationError("An API key is required for hosted BYOK.")
    prefix = HOSTED_LLM_PROVIDERS[provider]
    if prefix and not model.startswith(f"{prefix}/"):
        model = f"{prefix}/{model}"
    api_base = None
    if str(raw.get("api_base") or "").strip():
        api_base = _validated_url(str(raw["api_base"]), label="API endpoint")
    if provider == "custom" and not api_base:
        raise ConfigurationError("A custom provider requires an API endpoint.")
    public.update(
        provider=provider,
        model=model,
        api_base=api_base,
        credential_storage="ephemeral_memory",
    )
    secrets_for_run["api_key"] = api_key
    return public, secrets_for_run


def _build_llm_provider(
    config: dict[str, Any],
    secrets_for_run: dict[str, str],
) -> bool | LLMProvider:
    mode = config["mode"]
    if mode == "offline":
        return False
    if mode == "environment":
        return True
    if mode == "ollama":
        return OllamaProvider(config["model"], base_url=config["api_base"])
    options: dict[str, Any] = {
        "api_key": secrets_for_run["api_key"],
        "temperature": 0.2,
        "max_tokens": 2500,
    }
    if config.get("api_base"):
        options["api_base"] = config["api_base"]
    provider = LiteLLMProvider(config["model"], **options)
    return _RedactingLLMProvider(
        provider,
        [secrets_for_run["api_key"]],
        provenance={
            "mode": config.get("mode"),
            "provider": config.get("provider"),
            "model": config.get("model"),
            "api_base": config.get("api_base"),
            "credential_storage": config.get("credential_storage"),
        },
    )


def inspect_dataset(path: str | Path) -> dict[str, Any]:
    """Return a lightweight, JSON-safe dataset profile for the web UI."""
    dataset = Path(path).expanduser().resolve()
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")
    if dataset.is_file():
        suffix = dataset.suffix.lower()
        if suffix not in TABULAR_EXTENSIONS:
            raise ValueError("Tabular uploads must be CSV, XLSX, or XLS files.")
        sample = (
            pd.read_csv(dataset, nrows=500)
            if suffix == ".csv"
            else pd.read_excel(dataset, nrows=500)
        )
        return {
            "path": str(dataset),
            "name": dataset.name,
            "modality": "tabular",
            "size_bytes": dataset.stat().st_size,
            "sample_rows": len(sample),
            "columns": [str(column) for column in sample.columns],
            "target_candidates": _target_candidates(sample),
            "missing_cells_in_sample": int(sample.isna().sum().sum()),
        }

    image_count = 0
    formats: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    for item in dataset.rglob("*"):
        if not item.is_file() or item.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image_count += 1
        suffix = item.suffix.lower().lstrip(".")
        formats[suffix] = formats.get(suffix, 0) + 1
        relative = item.relative_to(dataset)
        class_name = relative.parts[0] if len(relative.parts) > 1 else "unassigned"
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    if not image_count:
        raise ValueError("No supported images were found in this folder.")
    return {
        "path": str(dataset),
        "name": dataset.name,
        "modality": "vision",
        "images": image_count,
        "classes": len(class_counts),
        "class_counts": dict(sorted(class_counts.items())),
        "formats": formats,
        "target_candidates": [],
    }


def _validate_config(
    config: dict[str, Any], modality: str
) -> tuple[dict[str, Any], dict[str, str]]:
    preset = str(config.get("preset", "balanced"))
    if preset not in PRESETS:
        raise ConfigurationError(f"Unknown preset: {preset}")
    task = str(config.get("task", "auto"))
    if task not in VALID_TASKS:
        raise ConfigurationError(f"Unknown task: {task}")
    if modality == "vision":
        task = "vision"
    models = config.get("models") or []
    if isinstance(models, str):
        models = [item.strip() for item in models.split(",") if item.strip()]
    backbones = config.get("backbones") or ["auto"]
    if isinstance(backbones, str):
        backbones = [item.strip() for item in backbones.split(",") if item.strip()]
    if not backbones or any(item not in VALID_BACKBONES for item in backbones):
        raise ConfigurationError("Invalid vision backbone selection.")
    llm_config, secrets_for_run = _validate_llm_config(
        config.get("llm_config", config.get("llm", True))
    )
    execution_target = str(config.get("execution_target", "cloud")).strip()
    if execution_target not in VALID_EXECUTION_TARGETS:
        raise ConfigurationError("Invalid execution target.")
    normalized = {
        "preset": preset,
        "task": task,
        "target": str(config.get("target") or "").strip() or None,
        "models": models,
        "backbones": backbones,
        "test_size": float(config.get("test_size", 0.2)),
        "cv": int(config.get("cv", PRESETS[preset].get("cv", 5))),
        "max_time": str(config.get("max_time") or "").strip() or None,
        "feature_engineering": bool(config.get("feature_engineering", False)),
        "tune": bool(config.get("tune", PRESETS[preset].get("tune", False))),
        "shap": bool(config.get("shap", False)),
        "llm": llm_config,
        "adapt_lora": bool(config.get("adapt_lora", False)),
        "preprocessing_cache": bool(config.get("preprocessing_cache", True)),
        "use_memory": bool(config.get("use_memory", True)),
        "contribute_memory": bool(config.get("contribute_memory", True)),
        "execution_target": execution_target,
        "local_gpu_consent": bool(config.get("local_gpu_consent", False)),
    }
    if not 0.05 <= normalized["test_size"] <= 0.4:
        raise ConfigurationError("test_size must be between 0.05 and 0.4.")
    if not 2 <= normalized["cv"] <= 10:
        raise ConfigurationError("cv must be between 2 and 10.")
    if modality == "tabular" and not normalized["target"]:
        raise ConfigurationError("A target column is required for tabular data.")
    return normalized, secrets_for_run


class RunManager:
    """Persist and execute web-initiated AutoNexus runs."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_workers: int = 1,
        trainer_factory: Callable[..., Any] | None = None,
        store: SQLiteRunStore | None = None,
        blob_mirror: FirebaseStorageMirror | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        database_path = os.getenv("AUTONEXUS_WEB_DB") or self.root / "studio.sqlite3"
        self.store = store or SQLiteRunStore(database_path)
        self.blob_mirror = blob_mirror
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="autonexus-web",
        )
        self._lock = threading.RLock()
        self._runs: dict[str, dict[str, Any]] = {}
        self._secrets: dict[str, dict[str, str]] = {}
        self._trainer_factory = trainer_factory
        self._load_existing()

    def _load_existing(self) -> None:
        loaded = {str(state["id"]): state for state in self.store.load_all()}
        for path in sorted(self.root.glob("*/web_run.json")):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
                loaded.setdefault(str(state["id"]), state)
            except (OSError, ValueError, KeyError):
                LOGGER.warning("Ignoring invalid web run state: %s", path)
        for state in loaded.values():
            try:
                state.setdefault("owner_id", "local-user")
                if state.get("status") in {"queued", "running"}:
                    state.update(
                        status="interrupted",
                        message="The web process stopped before this run completed.",
                        finished_at=_utc_now(),
                    )
                    self._persist(state)
                if state.get("deployment", {}).get("status") == "active":
                    state["deployment"] = {
                        "status": "inactive",
                        "reason": "Studio process restarted",
                    }
                    self._persist(state)
                self._runs[str(state["id"])] = state
            except (OSError, ValueError, KeyError):
                LOGGER.warning("Ignoring invalid stored web run state")

    def new_run_id(self) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{stamp}-{secrets.token_hex(3)}"

    def run_dir(self, run_id: str) -> Path:
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{6}", run_id):
            raise KeyError(run_id)
        return self.root / run_id

    def input_dir(self, run_id: str) -> Path:
        path = self.run_dir(run_id) / "input"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _persist(self, state: dict[str, Any]) -> None:
        safe_state = _json_safe(state)
        path = self.run_dir(str(state["id"])) / "web_run.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(safe_state, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
        self.store.upsert(safe_state)

    def _update(self, run_id: str, **changes: Any) -> None:
        with self._lock:
            state = self._runs[run_id]
            state.update(_json_safe(changes), updated_at=_utc_now())
            self._persist(state)

    def enqueue(
        self,
        run_id: str,
        dataset: str | Path,
        config: dict[str, Any],
        profile: dict[str, Any],
        secrets_for_run: dict[str, str] | None = None,
        owner_id: str = "local-user",
    ) -> dict[str, Any]:
        output_dir = self.run_dir(run_id) / "artifacts"
        state = {
            "id": run_id,
            "status": "queued",
            "progress": 4,
            "message": "Mission accepted. Waiting for compute.",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "started_at": None,
            "finished_at": None,
            "dataset": str(Path(dataset).resolve()),
            "owner_id": owner_id,
            "dataset_profile": profile,
            "config": config,
            "output_dir": str(output_dir),
            "best_model": None,
            "problem_type": None,
            "summary": {},
            "artifacts": [],
            "error": None,
            "storage": {
                "metadata": "sqlite",
                "working_copy": "local_filesystem",
                "firebase_mirror": None,
            },
            "events": [
                {"time": _utc_now(), "name": "queued", "message": "Run queued"}
            ],
        }
        with self._lock:
            if run_id in self._runs:
                raise ValueError(f"Duplicate run id: {run_id}")
            self._runs[run_id] = state
            self._secrets[run_id] = dict(secrets_for_run or {})
            self._persist(state)
        self._executor.submit(self._execute, run_id)
        return self.get(run_id)

    def _event(self, run_id: str, name: str, message: str) -> None:
        with self._lock:
            events = list(self._runs[run_id].get("events", []))
            events.append({"time": _utc_now(), "name": name, "message": message})
            self._update(run_id, events=events[-30:])

    def _execute(self, run_id: str) -> None:
        state = self.get(run_id)
        config = state["config"]
        try:
            self._update(
                run_id,
                status="running",
                progress=18,
                message="Profiling data and constructing the search space.",
                started_at=_utc_now(),
            )
            self._event(run_id, "training_started", "AutoNexus pipeline started")
            if self.blob_mirror is not None:
                try:
                    mirrored_dataset = self.blob_mirror.mirror_dataset(state)
                    storage = dict(state.get("storage", {}))
                    storage["firebase_mirror"] = {"dataset": mirrored_dataset}
                    self._update(run_id, storage=storage)
                    self._event(
                        run_id,
                        "dataset_mirrored",
                        "Dataset mirrored to Firebase Storage",
                    )
                except Exception as exc:
                    LOGGER.warning("Dataset mirror failed for %s: %s", run_id, exc)
                    storage = dict(state.get("storage", {}))
                    storage["firebase_mirror"] = {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    self._update(run_id, storage=storage)

            def callback(event: Any) -> None:
                if event.name == "training_started":
                    self._update(
                        run_id,
                        progress=28,
                        message="Training pipeline online. Searching candidates.",
                    )

            if self._trainer_factory is None:
                from .api import AutoNexus

                trainer_factory = AutoNexus
            else:
                trainer_factory = self._trainer_factory
            options: dict[str, Any] = {
                "preset": config["preset"],
                "task": config["task"],
                "output_dir": state["output_dir"],
                "models": config["models"],
                "contribute_memory": config["contribute_memory"],
                "llm": _build_llm_provider(
                    config["llm"], self._secrets.get(run_id, {})
                ),
                "callbacks": [callback],
                "test_size": config["test_size"],
                "cv": config["cv"],
                "feature_engineering": config["feature_engineering"],
                "tune": config["tune"],
                "shap": config["shap"],
                "adapt_lora": config["adapt_lora"],
                "backbones": config["backbones"],
                "preprocessing_cache": config["preprocessing_cache"],
                "use_memory": config["use_memory"],
            }
            if config["max_time"]:
                options["max_time"] = config["max_time"]
            trainer = trainer_factory(**options)
            model = trainer.fit(state["dataset"], target=config["target"])
            manifest = json.loads(model.manifest_path.read_text(encoding="utf-8"))
            artifacts = [
                name
                for name, relative in ARTIFACT_PATHS.items()
                if (Path(state["output_dir"]) / relative).is_file()
            ]
            self._event(run_id, "artifacts_sealed", "Artifact bundle validated")
            self._update(
                run_id,
                status="completed",
                progress=100,
                message="Mission complete. Model and evidence bundle are ready.",
                finished_at=_utc_now(),
                best_model=manifest.get("best_model"),
                problem_type=manifest.get("problem_type"),
                summary=manifest.get("run_summary", {}),
                artifacts=artifacts,
            )
            if self.blob_mirror is not None:
                try:
                    completed = self.get(run_id)
                    mirrored_artifacts = self.blob_mirror.mirror_artifacts(completed)
                    storage = dict(completed.get("storage", {}))
                    firebase_mirror = dict(storage.get("firebase_mirror") or {})
                    firebase_mirror["artifacts"] = mirrored_artifacts
                    storage["firebase_mirror"] = firebase_mirror
                    self._update(run_id, storage=storage)
                    self._event(
                        run_id,
                        "artifacts_mirrored",
                        "Artifacts mirrored to Firebase Storage",
                    )
                except Exception as exc:
                    LOGGER.warning("Artifact mirror failed for %s: %s", run_id, exc)
                    self._event(
                        run_id,
                        "artifact_mirror_failed",
                        f"Firebase Storage mirror failed: {type(exc).__name__}",
                    )
        except Exception as exc:
            LOGGER.exception("AutoNexus web run failed: %s", run_id)
            self._event(run_id, "failed", f"{type(exc).__name__}: {exc}")
            self._update(
                run_id,
                status="failed",
                progress=100,
                message="Mission failed. Review the diagnostic message.",
                finished_at=_utc_now(),
                error=f"{type(exc).__name__}: {exc}",
                diagnostic=traceback.format_exc(limit=12),
            )
        finally:
            with self._lock:
                self._secrets.pop(run_id, None)

    def get(
        self,
        run_id: str,
        *,
        owner_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if run_id not in self._runs:
                raise KeyError(run_id)
            state = self._runs[run_id]
            if owner_id is not None and state.get("owner_id") != owner_id:
                raise KeyError(run_id)
            return deepcopy(state)

    def list(self, *, owner_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            values = [
                deepcopy(item)
                for item in self._runs.values()
                if owner_id is None or item.get("owner_id") == owner_id
            ]
        return sorted(values, key=lambda item: item["created_at"], reverse=True)

    def artifact(
        self,
        run_id: str,
        artifact_name: str,
        *,
        owner_id: str | None = None,
    ) -> Path:
        state = self.get(run_id, owner_id=owner_id)
        if state["status"] != "completed" or artifact_name not in ARTIFACT_PATHS:
            raise FileNotFoundError(artifact_name)
        output = Path(state["output_dir"]).resolve()
        candidate = (output / ARTIFACT_PATHS[artifact_name]).resolve()
        if output not in candidate.parents or not candidate.is_file():
            raise FileNotFoundError(artifact_name)
        return candidate

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)


async def _save_uploads(form: Any, destination: Path, max_bytes: int) -> Path:
    uploads = [item for item in form.getlist("files") if getattr(item, "filename", None)]
    if not uploads:
        raise ValueError("No dataset files were uploaded.")
    relative_paths = [_safe_upload_path(item.filename) for item in uploads]
    first_parts = {path.parts[0] for path in relative_paths if len(path.parts) > 1}
    strip_root = len(first_parts) == 1 and all(len(path.parts) > 1 for path in relative_paths)
    total = 0
    saved: list[Path] = []
    for upload, relative in zip(uploads, relative_paths):
        if strip_root:
            relative = Path(*relative.parts[1:])
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("Upload exceeds the configured size limit.")
                stream.write(chunk)
        await upload.close()
        saved.append(target)
    if len(saved) == 1 and saved[0].suffix.lower() in TABULAR_EXTENSIONS:
        return saved[0]
    return destination


def _potential_target_proxies(columns: list[str], target: str | None) -> list[str]:
    if not target:
        return []
    normalized_target = re.sub(r"[^a-z0-9]+", "", target.lower())
    if len(normalized_target) < 3:
        return []
    proxies = []
    for column in columns:
        if column == target:
            continue
        normalized = re.sub(r"[^a-z0-9]+", "", column.lower())
        if normalized_target in normalized and any(
            token in column.lower()
            for token in ("target", "label", "class", "category")
        ):
            proxies.append(column)
    return proxies


def _geometry_numeric_candidates(
    frame: pd.DataFrame,
    *,
    excluded: set[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Return continuous columns that are safe enough for a response slice."""
    excluded = excluded or set()
    candidates: list[str] = []
    rejected: dict[str, str] = {}
    minimum_finite = max(20, int(len(frame) * 0.8))
    for column in frame.select_dtypes(include=[np.number]).columns:
        if column in excluded:
            rejected[column] = "excluded target or target proxy"
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) < minimum_finite:
            rejected[column] = "too many non-finite values"
            continue
        if len(np.unique(finite)) < 5:
            rejected[column] = "fewer than five distinct values"
            continue
        low, high = np.quantile(finite, [0.05, 0.95])
        if not np.isfinite(low + high) or high - low <= 1e-12:
            rejected[column] = "negligible robust range"
            continue
        extreme = float(np.quantile(np.abs(finite), 0.995))
        if extreme > 1e15:
            rejected[column] = "numerically extreme values"
            continue
        candidates.append(str(column))
    return candidates, rejected


def _unwrap_fitted_pipeline(estimator: Any) -> Any:
    seen: set[int] = set()
    while (
        not hasattr(estimator, "named_steps")
        and hasattr(estimator, "estimator")
        and id(estimator) not in seen
    ):
        seen.add(id(estimator))
        estimator = estimator.estimator
    return estimator


def _raw_feature_importance(
    predictor: Any,
    candidates: list[str],
    *,
    excluded_sources: set[str] | None = None,
) -> dict[str, float]:
    """Aggregate fitted transformed-feature importance back to raw columns."""
    scores = {column: 0.0 for column in candidates}
    pipeline = _unwrap_fitted_pipeline(predictor.model)
    members = getattr(pipeline, "models", None)
    if isinstance(members, list) and members:
        for _, member in members:
            member_scores = _raw_feature_importance(
                SimpleNamespace(model=member),
                candidates,
                excluded_sources=excluded_sources,
            )
            for column, value in member_scores.items():
                scores[column] += value / len(members)
        return scores
    if not hasattr(pipeline, "named_steps"):
        return scores
    preprocessor = pipeline.named_steps.get("preprocessor")
    estimator = pipeline.named_steps.get("model")
    if preprocessor is None or estimator is None:
        return scores
    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        importance = np.mean(
            np.abs(np.atleast_2d(estimator.coef_)), axis=0
        )
    else:
        return scores
    try:
        feature_names = [
            str(value).lower()
            for value in preprocessor.get_feature_names_out()
        ]
    except Exception:
        feature_names = [""] * len(importance)
        output_indices = getattr(preprocessor, "output_indices_", {})
        for name, _, columns in getattr(preprocessor, "transformers_", []):
            output_slice = output_indices.get(name)
            if not isinstance(output_slice, slice):
                continue
            column_names = [str(column).lower() for column in columns]
            start = int(output_slice.start or 0)
            stop = int(output_slice.stop or start)
            if stop - start != len(column_names):
                continue
            feature_names[start:stop] = column_names
    if len(feature_names) != len(importance):
        return scores
    excluded_tokens = {
        re.sub(r'[^a-z0-9_]+', '_', value.lower())
        for value in (excluded_sources or set())
    }
    for column in candidates:
        token = re.sub(r'[^a-z0-9_]+', '_', column.lower())
        pattern = re.compile(rf"(^|__){re.escape(token)}($|__|_)")
        scores[column] = float(
            np.sum([
                value
                for name, value in zip(feature_names, importance)
                if pattern.search(name)
                and not any(
                    re.search(rf"(^|__){re.escape(excluded)}($|__|_)", name)
                    for excluded in excluded_tokens
                )
            ])
        )
    return scores


def _read_geometry_frame(
    state: dict[str, Any],
    *,
    max_rows: int = 10000,
) -> pd.DataFrame:
    dataset = Path(str(state.get("dataset", "")))
    if not dataset.is_file():
        raise FileNotFoundError("The persisted tabular dataset is unavailable.")
    suffix = dataset.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(dataset, nrows=max_rows)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(dataset, nrows=max_rows)
    raise ValueError(f"Unsupported geometry source: {suffix}")


def _balanced_geometry_sample(
    frame: pd.DataFrame,
    target: str,
    *,
    max_rows: int = 700,
) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.copy()
    labels = frame[target].astype(str).fillna("<missing>")
    class_count = max(labels.nunique(dropna=False), 1)
    per_class = max(max_rows // class_count, 1)
    selected_parts = []
    for label in labels.drop_duplicates():
        group = frame.loc[labels == label]
        selected_parts.append(
            group.sample(min(len(group), per_class), random_state=42)
        )
    selected = pd.concat(selected_parts) if selected_parts else frame.head(0)
    if len(selected) < max_rows:
        remaining = frame.drop(index=selected.index, errors="ignore")
        if not remaining.empty:
            selected = pd.concat(
                [
                    selected,
                    remaining.sample(
                        min(max_rows - len(selected), len(remaining)),
                        random_state=42,
                    ),
                ]
            )
    return selected.head(max_rows).sort_index().copy()


def _tree_response_geometry(
    state: dict[str, Any],
    output: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """Create a bounded raw-feature response slice through the full predictor."""
    import joblib

    predictor = joblib.load(output / "model.pkl")
    target = str(predictor.target_name)
    raw = _read_geometry_frame(state)
    if target not in raw.columns:
        raise ValueError(f"Target column '{target}' is absent from the saved data.")
    proxies = _potential_target_proxies(list(raw.columns), target)
    excluded = {target, *proxies}
    candidates, rejected = _geometry_numeric_candidates(raw, excluded=excluded)
    if len(candidates) < 2:
        raise ValueError("Fewer than two safe continuous raw features are available.")

    importance = _raw_feature_importance(
        predictor,
        candidates,
        excluded_sources=excluded,
    )
    ranked = sorted(
        candidates,
        key=lambda column: (
            importance.get(column, 0.0),
            raw[column].nunique(dropna=True),
        ),
        reverse=True,
    )
    first, second = ranked[:2]
    sample = _balanced_geometry_sample(raw, target)
    features = sample.drop(columns=[target])
    actual = sample[target].astype(str).fillna("<missing>")

    reference: dict[str, Any] = {}
    for column in features.columns:
        series = features[column]
        if pd.api.types.is_numeric_dtype(series):
            finite = pd.to_numeric(series, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            reference[column] = float(finite.median())
        else:
            modes = series.mode(dropna=True)
            reference[column] = modes.iloc[0] if len(modes) else "<missing>"

    x_values = np.linspace(
        float(pd.to_numeric(features[first], errors="coerce").quantile(0.05)),
        float(pd.to_numeric(features[first], errors="coerce").quantile(0.95)),
        22,
    )
    z_values = np.linspace(
        float(pd.to_numeric(features[second], errors="coerce").quantile(0.05)),
        float(pd.to_numeric(features[second], errors="coerce").quantile(0.95)),
        22,
    )
    grid_rows: list[dict[str, Any]] = []
    for z_value in z_values:
        for x_value in x_values:
            row = dict(reference)
            row[first] = float(x_value)
            row[second] = float(z_value)
            grid_rows.append(row)
    grid = pd.DataFrame(grid_rows, columns=features.columns)

    if predictor.problem_type == "classification":
        sample_probability = np.asarray(predictor.predict_proba(features), dtype=float)
        grid_probability = np.asarray(predictor.predict_proba(grid), dtype=float)
        sample_score = sample_probability.max(axis=1)
        grid_score = grid_probability.max(axis=1)
        predicted = predictor.predict(features).astype(str)
        grid_prediction = predictor.predict(grid).astype(str)
        score_label = "model confidence"
    else:
        sample_score = np.asarray(predictor.predict(features), dtype=float)
        grid_score = np.asarray(predictor.predict(grid), dtype=float)
        predicted = np.asarray(sample_score).astype(str)
        grid_prediction = np.asarray(grid_score).astype(str)
        score_label = "predicted response"

    points = [
        {
            "x": round(float(x), 8),
            "y": round(float(score), 8),
            "z": round(float(z), 8),
            "label": label,
            "predicted": prediction,
            "confidence": round(float(score), 8),
            "split": "analysis_sample",
            "group": "",
            "correct": label == prediction if predictor.problem_type == "classification" else None,
            "row_id": str(index),
        }
        for index, x, score, z, label, prediction in zip(
            sample.index,
            features[first],
            sample_score,
            features[second],
            actual,
            predicted,
        )
        if np.isfinite(float(x) + float(score) + float(z))
    ]
    surface = {
        "feature_x": first,
        "feature_z": second,
        "score_label": score_label,
        "selection_method": "aggregated fitted-model importance",
        "importance": {
            first: round(float(importance.get(first, 0.0)), 8),
            second: round(float(importance.get(second, 0.0)), 8),
        },
        "rows": len(z_values),
        "columns": len(x_values),
        "vertices": [
            {
                "x": round(float(x), 8),
                "y": round(float(score), 8),
                "z": round(float(z), 8),
                "predicted": str(prediction),
            }
            for x, score, z, prediction in zip(
                np.tile(x_values, len(z_values)),
                grid_score,
                np.repeat(z_values, len(x_values)),
                grid_prediction,
            )
        ],
        "axes": [
            {"key": "x", "label": first, "role": "raw feature"},
            {"key": "y", "label": score_label, "role": "model output"},
            {"key": "z", "label": second, "role": "raw feature"},
        ],
    }
    notes = []
    extreme_rejections = [
        column for column, reason in rejected.items()
        if reason == "numerically extreme values"
    ]
    if extreme_rejections:
        notes.append(
            "Geometry excluded numerically extreme columns: "
            + ", ".join(extreme_rejections[:8])
        )
    return points, surface, notes


def _finite_projection_matrix(values: np.ndarray) -> np.ndarray:
    """Median-fill non-finite projection inputs without leaking labels."""
    matrix = np.asarray(values, dtype=np.float32).copy()
    if matrix.ndim != 2 or not matrix.size:
        raise ValueError("The saved representation is empty or not two-dimensional.")
    matrix[~np.isfinite(matrix)] = np.nan
    with np.errstate(all="ignore"):
        medians = np.nanmedian(matrix, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    rows, columns = np.where(np.isnan(matrix))
    matrix[rows, columns] = medians[columns]
    return matrix


def _bounded_geometry_error(prefix: str, exc: Exception) -> str:
    message = " ".join(str(exc).split())
    if len(message) > 240:
        message = message[:237] + "..."
    return f"{prefix}: {type(exc).__name__}: {message}"


def _run_insights(state: dict[str, Any]) -> dict[str, Any]:
    """Build bounded, scientifically labelled visualization data."""
    output = Path(state["output_dir"])
    summary = state.get("summary", {})
    config = state.get("config", {})
    profile = state.get("dataset_profile", {})
    best_model = str(state.get("best_model") or "unknown")
    input_metadata = summary.get("input_metadata", {}) or {}
    modality = profile.get("modality", "tabular")
    nodes = [
        {"id": "ingest", "label": "Ingestion", "detail": profile.get("name", "dataset")},
        {
            "id": "audit",
            "label": "Data audit",
            "detail": f"{profile.get('modality', 'auto')} / schema + quality",
        },
        {
            "id": "split",
            "label": "Split firewall",
            "detail": f"{int(float(config.get('test_size', 0.2)) * 100)}% held out",
        },
        {
            "id": "search",
            "label": "Search",
            "detail": f"{summary.get('search_models_screened', 0)} candidates screened",
        },
        {"id": "fit", "label": "Validated fit", "detail": best_model},
        {
            "id": "calibrate",
            "label": "Calibration",
            "detail": f"T={summary.get('temperature', 1.0):.3f}",
        },
        {
            "id": "evidence",
            "label": "Evidence bundle",
            "detail": f"{len(state.get('artifacts', []))} sealed artifacts",
        },
        {
            "id": "deploy",
            "label": "Deployment",
            "detail": state.get("deployment", {}).get("status", "not deployed"),
        },
    ]

    points: list[dict[str, Any]] = []
    geometry_error = None
    embedding_path = output / "analysis_data" / "embedding_sample.npz"
    if embedding_path.is_file():
        try:
            from sklearn.decomposition import PCA

            with np.load(embedding_path, allow_pickle=False) as payload:
                values = np.asarray(payload["X"], dtype=np.float32)
                count = min(len(values), 700)
                indices = np.linspace(0, len(values) - 1, count, dtype=int)
                selected = _finite_projection_matrix(values[indices])
                dimensions = min(3, selected.shape[1], len(selected))
                coordinates = PCA(n_components=dimensions).fit_transform(selected)
                if dimensions < 3:
                    coordinates = np.pad(
                        coordinates,
                        ((0, 0), (0, 3 - dimensions)),
                    )
                labels = np.asarray(payload["y"]).astype(str)[indices]
                splits = np.asarray(payload["split"]).astype(str)[indices]
                groups = np.asarray(payload["group"]).astype(str)[indices]
                row_ids = np.asarray(payload["row_id"]).astype(str)[indices]
            prediction_details: dict[str, dict[str, Any]] = {}
            prediction_path = output / "analysis_data" / "prediction_index.csv"
            if prediction_path.is_file():
                prediction_index = pd.read_csv(prediction_path)
                prediction_details = {
                    str(row.row_id): {
                        "correct": bool(row.correct),
                        "predicted": str(row.predicted_label),
                        "confidence": (
                            None
                            if pd.isna(row.confidence)
                            else float(row.confidence)
                        ),
                    }
                    for row in prediction_index.itertuples()
                    if pd.notna(row.correct)
                }
            points = [
                {
                    "x": round(float(x), 6),
                    "y": round(float(y), 6),
                    "z": round(float(z), 6),
                    "label": label,
                    "predicted": prediction_details.get(row_id, {}).get(
                        "predicted"
                    ),
                    "confidence": prediction_details.get(row_id, {}).get(
                        "confidence"
                    ),
                    "split": split,
                    "group": group,
                    "correct": prediction_details.get(row_id, {}).get(
                        "correct"
                    ),
                    "row_id": row_id,
                }
                for (x, y, z), label, split, group, row_id in zip(
                    coordinates, labels, splits, groups, row_ids
                )
            ]
        except Exception as exc:
            geometry_error = _bounded_geometry_error(
                "Embedding projection unavailable", exc
            )

    adaptation_movement: list[dict[str, Any]] = []
    movement_path = output / "analysis_data" / "lora_movement.npz"
    if movement_path.is_file():
        try:
            from sklearn.decomposition import PCA

            with np.load(movement_path, allow_pickle=False) as payload:
                frozen = np.asarray(payload["frozen"], dtype=np.float32)
                adapted = np.asarray(payload["adapted"], dtype=np.float32)
                labels = np.asarray(payload["labels"]).astype(str)
            count = min(len(frozen), len(adapted), 400)
            indices = np.linspace(0, len(frozen) - 1, count, dtype=int)
            paired = _finite_projection_matrix(
                np.vstack([frozen[indices], adapted[indices]])
            )
            dimensions = min(3, paired.shape[1], len(paired))
            coordinates = PCA(n_components=dimensions).fit_transform(paired)
            if dimensions < 3:
                coordinates = np.pad(
                    coordinates,
                    ((0, 0), (0, 3 - dimensions)),
                )
            before = coordinates[:count]
            after = coordinates[count:]
            adaptation_movement = [
                {
                    "label": label,
                    "from": [round(float(value), 6) for value in start],
                    "to": [round(float(value), 6) for value in end],
                }
                for start, end, label in zip(before, after, labels[indices])
            ]
            adapter_selected = bool(
                input_metadata.get("lora_gate", {}).get("adapter_selected")
            )
            selected_coordinates = after if adapter_selected else before
            points = [
                {
                    "x": round(float(row[0]), 6),
                    "y": round(float(row[1]), 6),
                    "z": round(float(row[2]), 6),
                    "label": label,
                    "split": "lora_gate",
                    "group": "",
                    "correct": None,
                }
                for row, label in zip(selected_coordinates, labels[indices])
            ]
        except Exception as exc:
            geometry_error = _bounded_geometry_error(
                "LoRA movement unavailable", exc
            )

    family = (
        "linear"
        if best_model in {"logistic", "ridge", "sgd_clf", "sgd_reg"}
        else "ensemble"
        if best_model == "diverse_ensemble"
        else "tree"
        if best_model in {
            "rf", "et_clf", "et_reg", "gb", "gb_reg", "xgb_clf",
            "xgb_reg", "lgbm_clf", "lgbm_reg",
        }
        else "nonlinear"
    )
    response_surface = None
    geometry_notes: list[str] = []
    if family in {"tree", "ensemble"}:
        try:
            points, response_surface, geometry_notes = _tree_response_geometry(
                state, output
            )
            geometry_error = None
        except Exception as exc:
            geometry_error = _bounded_geometry_error(
                "Model response surface unavailable", exc
            )
    geometry_kind = (
        "lora_embedding_movement"
        if adaptation_movement
        else "embedding_projection"
        if modality == "vision"
        else "model_response_slice"
        if family in {"tree", "ensemble"} and response_surface is not None
        else "feature_projection"
    )
    geometry_axes = [
        {"key": "x", "label": "PCA component 1", "role": "projection"},
        {"key": "y", "label": "PCA component 2", "role": "projection"},
        {"key": "z", "label": "PCA component 3", "role": "projection"},
    ]
    if response_surface is not None:
        geometry_axes = response_surface["axes"]
    truth_label = (
        "PCA projection of the saved embedding representation; this is not a "
        "literal decision boundary."
        if modality == "vision"
        else "PCA projection of high-dimensional model inputs; this is not the "
        "model's literal decision boundary."
    )
    if response_surface is not None:
        model_label = "ensemble" if family == "ensemble" else "tree model"
        truth_label = (
            f"Interactive {model_label} response slice over raw "
            f"{response_surface['feature_x']} "
            f"and {response_surface['feature_z']}. Height is "
            f"{response_surface['score_label']}; all other raw features are held "
            "at their observed median or mode. Dots are real samples and the mesh "
            "is the model response. This is not the full decision boundary."
        )
    if adaptation_movement:
        truth_label = (
            "PCA projection fitted jointly to paired frozen and LoRA-adapted "
            "embeddings. Arrows show representation movement; no literal "
            "classifier boundary is claimed."
        )
    exact_plane = None
    if best_model == "logistic":
        try:
            import joblib

            fitted = joblib.load(output / "best_model.joblib")
            while not hasattr(fitted, "named_steps") and hasattr(fitted, "estimator"):
                fitted = fitted.estimator
            estimator = fitted.named_steps["model"]
            if estimator.coef_.shape[1] == 3 and estimator.coef_.shape[0] == 1:
                sample = pd.read_csv(
                    output / "analysis_data" / "learning_curve_sample.csv"
                ).head(700)
                labels = sample.pop("__target__").astype(str).tolist()
                preprocessor = fitted.named_steps["preprocessor"]
                transformed = preprocessor.transform(sample)
                if hasattr(transformed, "toarray"):
                    transformed = transformed.toarray()
                transformed = np.asarray(transformed, dtype=float)
                predicted = estimator.predict(transformed).astype(str)
                confidence = (
                    estimator.predict_proba(transformed).max(axis=1)
                    if hasattr(estimator, "predict_proba")
                    else np.full(len(transformed), np.nan)
                )
                points = [
                    {
                        "x": round(float(row[0]), 6),
                        "y": round(float(row[1]), 6),
                        "z": round(float(row[2]), 6),
                        "label": label,
                        "predicted": prediction,
                        "confidence": (
                            None if not np.isfinite(score) else round(float(score), 8)
                        ),
                        "split": "development_cv",
                        "group": "",
                        "correct": label == prediction,
                        "row_id": str(index),
                    }
                    for index, (row, label, prediction, score) in enumerate(
                        zip(transformed, labels, predicted, confidence)
                    )
                ]
                try:
                    transformed_names = [
                        str(value)
                        for value in preprocessor.get_feature_names_out()
                    ]
                except Exception:
                    transformed_names = [
                        "transformed feature 1",
                        "transformed feature 2",
                        "transformed feature 3",
                    ]
                geometry_axes = [
                    {"key": key, "label": name, "role": "exact model input"}
                    for key, name in zip(
                        ("x", "y", "z"), transformed_names[:3]
                    )
                ]
                exact_plane = {
                    "coefficients": estimator.coef_[0].astype(float).tolist(),
                    "intercept": float(estimator.intercept_[0]),
                    "feature_names": transformed_names[:3],
                }
                geometry_kind = "exact_logistic_plane"
                truth_label = (
                    "Exact logistic decision plane in the three transformed "
                    "features used by the fitted classifier."
                )
        except Exception:
            exact_plane = None

    proxies = _potential_target_proxies(
        list(profile.get("columns", [])),
        config.get("target"),
    )
    warnings_found = []
    fit_gap = summary.get("fit_validation_gap")
    if fit_gap is not None and float(fit_gap) > 0.05:
        warnings_found.append(
            f"Fit-validation gap is {float(fit_gap):.3f}; inspect overfitting."
        )
    if proxies:
        warnings_found.append(
            "Potential target-proxy leakage columns: " + ", ".join(proxies)
        )
    evidence = {
        key: (output / path).is_file() for key, path in EVIDENCE_PATHS.items()
    }
    return {
        "run_id": state["id"],
        "lineage": nodes,
        "geometry": {
            "kind": geometry_kind,
            "family": family,
            "title": (
                f"{best_model} conditional response landscape"
                if response_surface is not None
                else f"{best_model} representation geometry"
            ),
            "truth_label": truth_label,
            "points": points,
            "exact_plane": exact_plane,
            "response_surface": response_surface,
            "axes": geometry_axes,
            "adaptation_movement": adaptation_movement,
            "error": geometry_error,
            "notes": geometry_notes,
        },
        "explainability": {
            "evidence": evidence,
            "local_explanations": (
                "A held-out local waterfall and cumulative decision plot are "
                "available."
                if evidence.get("shap_waterfall")
                and evidence.get("shap_decision")
                else "Local SHAP evidence was not persisted for this run."
            ),
            "vision_attention": (
                "Not claimed: backbone activations were not persisted. Nearest "
                "embedding neighbors remain the faithful explanation."
                if modality == "vision"
                else "Not applicable to tabular data."
            ),
            "adaptation_movement": (
                f"{len(adaptation_movement)} paired frozen-to-adapted vectors "
                "are available."
                if adaptation_movement
                else "Available only when frozen and adapted embeddings are "
                "both persisted."
            ),
        },
        "warnings": warnings_found,
        "selected_backbone": input_metadata.get("backbone_key"),
        "representation": input_metadata.get("selected_representation"),
    }


def _monitoring_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    output = Path(state["output_dir"])
    baseline_path = output / "monitoring" / "baseline.json"
    baseline = (
        json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline_path.is_file()
        else {}
    )
    records = []
    events_path = output / "monitoring" / "events.jsonl"
    if events_path.is_file():
        for line in events_path.read_text(encoding="utf-8").splitlines()[-200:]:
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    return {
        "run_id": state["id"],
        "baseline": {
            "sample_count": baseline.get("sample_count"),
            "target_name": baseline.get("target_name"),
            "problem_type": baseline.get("problem_type"),
            "metric_name": baseline.get("metric_name"),
            "expected_metric": baseline.get("expected_metric"),
            "feature_count": len(baseline.get("columns", [])),
            "numeric_features": len(baseline.get("numeric_samples", {})),
            "categorical_features": len(
                baseline.get("categorical_frequencies", {})
            ),
            "prediction_frequencies": baseline.get(
                "prediction_frequencies", {}
            ),
        },
        "events": records,
        "incremental_supported": None,
    }


def create_app(
    *,
    workspace: str | Path | None = None,
    manager: RunManager | None = None,
    authenticator: StudioAuthenticator | None = None,
    cors_origins: list[str] | None = None,
    blob_mirror: FirebaseStorageMirror | None = None,
) -> Any:
    """Create the optional FastAPI web application."""
    try:
        from contextlib import asynccontextmanager

        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import FileResponse, JSONResponse
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise CapabilityError(
            'Web Studio requires: pip install "AutoNexus[serve]"'
        ) from exc

    studio_auth = authenticator or authenticator_from_env()
    if blob_mirror is None and manager is None:
        blob_mirror = FirebaseStorageMirror.from_env()
    run_manager = manager or RunManager(
        workspace or default_workspace(),
        max_workers=max(1, int(os.getenv("AUTONEXUS_WEB_WORKERS", "1"))),
        blob_mirror=blob_mirror,
    )

    @asynccontextmanager
    async def lifespan(_: Any):
        yield
        if manager is None:
            run_manager.shutdown()

    app = FastAPI(
        title="Auto Nexus Studio",
        version="1",
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.run_manager = run_manager
    app.state.authenticator = studio_auth
    app.state.deployments = {}
    app.state.deployment_metrics = {}
    app.state.insights_cache = {}

    configured_origins = cors_origins
    if configured_origins is None:
        configured_origins = [
            value.strip().rstrip("/")
            for value in os.getenv("AUTONEXUS_CORS_ORIGINS", "").split(",")
            if value.strip()
        ]
    if configured_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=configured_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
            allow_private_network=studio_auth.mode == "agent",
            expose_headers=["Content-Disposition"],
            max_age=600,
        )

    def remote_local_paths_allowed() -> bool:
        return (
            studio_auth.mode != "firebase"
            or os.getenv("AUTONEXUS_ALLOW_REMOTE_LOCAL_PATHS", "false")
            .strip()
            .lower()
            in {"1", "true", "yes"}
        )

    @app.middleware("http")
    async def disable_studio_cache(request: Request, call_next):
        """Always serve the current Studio shell instead of stale browser assets."""
        response = await call_next(request)
        if request.headers.get("access-control-request-private-network") == "true":
            response.headers["Access-Control-Allow-Private-Network"] = "true"
        if request.url.path == "/" or request.url.path.startswith("/assets/"):
            response.headers["Cache-Control"] = (
                "no-store, no-cache, must-revalidate, max-age=0"
            )
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    @app.middleware("http")
    async def authenticate_studio_request(request: Request, call_next):
        public_api = {"/api/health", "/api/auth/config", "/api/documents"}
        if request.method == "OPTIONS":
            return await call_next(request)
        if request.url.path.startswith("/api/") and request.url.path not in public_api:
            try:
                request.state.principal = studio_auth.authenticate(
                    request.headers.get("authorization")
                )
            except AuthenticationError as exc:
                return JSONResponse(
                    status_code=401,
                    content={"detail": str(exc)},
                    headers={"WWW-Authenticate": "Bearer"},
                )
        return await call_next(request)

    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/documents")
    async def documents():
        return {
            "documents": {
                name: {
                    "available": path.is_file(),
                    "url": f"/documents/{name}",
                }
                for name, path in DOCUMENT_PATHS.items()
            }
        }

    @app.get("/documents/{document_name}", include_in_schema=False)
    async def document(document_name: str):
        path = DOCUMENT_PATHS.get(document_name)
        if path is None or not path.is_file():
            raise HTTPException(
                status_code=404,
                detail=(
                    "Document has not been uploaded yet. See "
                    "autonexus/web_static/documents/README.md."
                ),
            )
        return FileResponse(path, media_type="application/pdf")

    @app.get("/api/health")
    async def health():
        from . import __version__

        return {
            "status": "online",
            "version": __version__,
            "auth_mode": studio_auth.mode,
            "deployment": os.getenv("AUTONEXUS_DEPLOYMENT", "local"),
            "persistence": "sqlite+filesystem",
            "firebase_storage": run_manager.blob_mirror is not None,
        }

    @app.get("/api/auth/config")
    async def auth_configuration():
        config = studio_auth.public_config()
        config["local_paths_allowed"] = remote_local_paths_allowed()
        return config

    @app.get("/api/auth/me")
    async def current_user(request: Request):
        principal = request.state.principal
        return {
            "uid": principal.uid,
            "email": principal.email,
            "name": principal.name,
        }

    @app.get("/api/config")
    async def configuration():
        from model_trainer import CLASSIFICATION_MODELS, REGRESSION_MODELS

        return {
            "presets": list(PRESETS),
            "tasks": sorted(VALID_TASKS),
            "backbones": sorted(VALID_BACKBONES),
            "llm_modes": sorted(VALID_LLM_MODES),
            "llm_providers": list(HOSTED_LLM_PROVIDERS),
            "classification_models": [
                *CLASSIFICATION_MODELS,
                "lgbm_clf",
                "xgb_clf",
            ],
            "regression_models": [*REGRESSION_MODELS, "lgbm_reg", "xgb_reg"],
            "execution_mode": studio_auth.mode,
        }

    @app.get("/api/agent/capabilities")
    async def agent_capabilities():
        if studio_auth.mode != "agent":
            raise HTTPException(status_code=404, detail="Local agent is not enabled.")
        gpu = {"available": False, "name": None, "backend": None}
        try:
            import torch

            if torch.cuda.is_available():
                gpu = {
                    "available": True,
                    "name": torch.cuda.get_device_name(0),
                    "backend": "cuda",
                }
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                gpu = {"available": True, "name": "Apple GPU", "backend": "mps"}
        except ImportError:
            pass
        return {
            "agent": "autonexus-local",
            "gpu": gpu,
            "consent_required_for_every_run": True,
            "storage": "local_sqlite_and_filesystem",
        }

    @app.post("/api/datasets/inspect")
    async def inspect(request: Request):
        try:
            if not remote_local_paths_allowed():
                raise HTTPException(
                    status_code=403,
                    detail=(
                        "Server-local dataset paths are disabled in Firebase "
                        "mode. Upload the dataset through the browser."
                    ),
                )
            payload = await request.json()
            return inspect_dataset(payload.get("path", ""))
        except (FileNotFoundError, ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/runs")
    async def list_runs(request: Request):
        return {
            "runs": run_manager.list(
                owner_id=request.state.principal.uid
            )
        }

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str, request: Request):
        try:
            return run_manager.get(
                run_id,
                owner_id=request.state.principal.uid,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc

    @app.get("/api/runs/{run_id}/insights")
    async def run_insights(run_id: str, request: Request):
        try:
            state = run_manager.get(
                run_id,
                owner_id=request.state.principal.uid,
            )
            model_path = Path(state["output_dir"]) / "model.pkl"
            model_revision = (
                model_path.stat().st_mtime_ns if model_path.is_file() else 0
            )
            cache_key = (
                request.state.principal.uid,
                run_id,
                model_revision,
            )
            cached = app.state.insights_cache.get(cache_key)
            if cached is None:
                cached = _run_insights(state)
                app.state.insights_cache = {
                    key: value
                    for key, value in app.state.insights_cache.items()
                    if key[:2] != cache_key[:2]
                }
                app.state.insights_cache[cache_key] = cached
            return deepcopy(cached)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc

    @app.get("/api/runs/{run_id}/monitoring")
    async def run_monitoring(run_id: str, request: Request):
        try:
            state = run_manager.get(
                run_id,
                owner_id=request.state.principal.uid,
            )
            snapshot = _monitoring_snapshot(state)
            owner_id = request.state.principal.uid
            snapshot["deployment"] = state.get(
                "deployment", {"status": "inactive"}
            )
            snapshot["deployment_telemetry"] = deepcopy(
                app.state.deployment_metrics.get((owner_id, run_id), {})
            )
            if state["status"] == "completed":
                from .model import NexusModel

                snapshot["incremental_supported"] = NexusModel(
                    state["output_dir"]
                ).supports_incremental_learning
            return snapshot
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc
        except (OSError, ValueError, CapabilityError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/monitoring/observe")
    async def observe_run(run_id: str, request: Request):
        try:
            state = run_manager.get(
                run_id,
                owner_id=request.state.principal.uid,
            )
            payload = await request.json()
            records = payload.get("records", [])
            if not isinstance(records, list) or not records:
                raise ValueError("records must be a non-empty JSON array.")
            if len(records) > 10000:
                raise ValueError("Monitoring batches are limited to 10,000 rows.")
            from .model import NexusModel

            model = NexusModel(state["output_dir"])
            report = model.monitor().observe(pd.DataFrame(records))
            run_manager._event(
                run_id,
                "monitoring_observation",
                f"severity={report.severity} / samples={report.sample_count}",
            )
            return report.to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc
        except (OSError, ValueError, CapabilityError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/audit")
    async def audit_log(request: Request):
        rows = []
        for run in run_manager.list(owner_id=request.state.principal.uid):
            for event in run.get("events", []):
                rows.append(
                    {
                        **event,
                        "run_id": run["id"],
                        "dataset": Path(run.get("dataset", "")).name,
                    }
                )
        rows.sort(key=lambda item: item.get("time", ""), reverse=True)
        return {"events": rows[:500]}

    @app.get("/api/runs/{run_id}/evidence/{evidence_name}")
    async def evidence_image(
        run_id: str,
        evidence_name: str,
        request: Request,
    ):
        try:
            state = run_manager.get(
                run_id,
                owner_id=request.state.principal.uid,
            )
            relative = EVIDENCE_PATHS[evidence_name]
            output = Path(state["output_dir"]).resolve()
            candidate = (output / relative).resolve()
            if output not in candidate.parents or not candidate.is_file():
                raise FileNotFoundError(evidence_name)
            return FileResponse(candidate, media_type="image/png")
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="Evidence not found.") from exc

    @app.post("/api/runs/{run_id}/deploy")
    async def deploy_run(run_id: str, request: Request):
        try:
            owner_id = request.state.principal.uid
            state = run_manager.get(run_id, owner_id=owner_id)
            if state["status"] != "completed":
                raise ValueError("Only completed runs can be deployed.")
            from .model import NexusModel

            model = NexusModel(state["output_dir"])
            app.state.deployments[(owner_id, run_id)] = model
            app.state.deployment_metrics[(owner_id, run_id)] = {
                "request_count": 0,
                "sample_count": 0,
                "error_count": 0,
                "last_latency_ms": None,
                "mean_latency_ms": None,
                "max_latency_ms": None,
                "mean_confidence": None,
            }
            deployment = {
                "status": "active",
                "deployed_at": _utc_now(),
                "predict_url": f"/api/deployments/{run_id}/predict",
                "scope": "authenticated Studio process",
            }
            run_manager._update(run_id, deployment=deployment)
            run_manager._event(
                run_id,
                "deployment_activated",
                "Authenticated local inference endpoint activated",
            )
            return deployment
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc
        except (OSError, ValueError, CapabilityError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/what-if")
    async def what_if(run_id: str, request: Request):
        try:
            state = run_manager.get(
                run_id,
                owner_id=request.state.principal.uid,
            )
            payload = await request.json()
            record = payload.get("record")
            if not isinstance(record, dict) or not record:
                raise ValueError("record must be one non-empty JSON object.")
            if len(record) > 1000:
                raise ValueError("What-if records are limited to 1,000 fields.")
            from .model import NexusModel

            model = NexusModel(state["output_dir"])
            if model.predictor.modality != "tabular":
                raise CapabilityError(
                    "Web what-if controls currently require tabular features."
                )
            frame = pd.DataFrame([record])
            prediction = model.predict(frame).tolist()
            response: dict[str, Any] = {"prediction": _json_safe(prediction[0])}
            try:
                response["probabilities"] = _json_safe(
                    model.predict_proba(frame)[0].tolist()
                )
                response["classes"] = model.predictor.class_names
            except AttributeError:
                pass
            return response
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc
        except (OSError, ValueError, CapabilityError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/incremental-update")
    async def incremental_update(run_id: str, request: Request):
        try:
            owner_id = request.state.principal.uid
            state = run_manager.get(run_id, owner_id=owner_id)
            payload = await request.json()
            records = payload.get("records", [])
            if not isinstance(records, list) or not records:
                raise ValueError("records must be a non-empty labelled JSON array.")
            if len(records) > 10000:
                raise ValueError("Incremental batches are limited to 10,000 rows.")
            target = state.get("config", {}).get("target")
            if not target:
                raise CapabilityError(
                    "This run has no tabular label column for incremental update."
                )
            from .model import NexusModel

            model = NexusModel(state["output_dir"])
            result = model.update(pd.DataFrame(records), target=target)
            run_manager._event(
                run_id,
                "incremental_update",
                f"{result.stage or result.action} / promoted={result.promoted}",
            )
            if result.promoted and (owner_id, run_id) in app.state.deployments:
                app.state.deployments[(owner_id, run_id)] = NexusModel(
                    state["output_dir"]
                )
            return _json_safe(result.__dict__)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc
        except (OSError, ValueError, CapabilityError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/runs/{run_id}/deploy")
    async def undeploy_run(run_id: str, request: Request):
        try:
            owner_id = request.state.principal.uid
            run_manager.get(run_id, owner_id=owner_id)
            app.state.deployments.pop((owner_id, run_id), None)
            deployment = {"status": "inactive", "stopped_at": _utc_now()}
            run_manager._update(run_id, deployment=deployment)
            run_manager._event(
                run_id,
                "deployment_stopped",
                "Inference endpoint stopped",
            )
            return deployment
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found.") from exc

    @app.post("/api/deployments/{run_id}/predict")
    async def deployed_predict(run_id: str, request: Request):
        owner_id = request.state.principal.uid
        deployment_key = (owner_id, run_id)
        model = app.state.deployments.get(deployment_key)
        if model is None:
            raise HTTPException(status_code=404, detail="Deployment is not active.")
        started = time.perf_counter()
        telemetry = app.state.deployment_metrics.setdefault(
            deployment_key,
            {
                "request_count": 0,
                "sample_count": 0,
                "error_count": 0,
                "last_latency_ms": None,
                "mean_latency_ms": None,
                "max_latency_ms": None,
                "mean_confidence": None,
            },
        )
        try:
            payload = await request.json()
            records = payload.get("records", [])
            if not isinstance(records, list) or not records:
                raise ValueError("records must be a non-empty JSON array.")
            if len(records) > 10000:
                raise ValueError("Prediction batches are limited to 10,000 rows.")
            frame = pd.DataFrame(records)
            predictions = model.predict(frame)
            response: dict[str, Any] = {
                "run_id": run_id,
                "predictions": _json_safe(predictions.tolist()),
            }
            try:
                probabilities = np.asarray(model.predict_proba(frame), dtype=float)
                response["probabilities"] = _json_safe(probabilities.tolist())
                batch_confidence = float(np.max(probabilities, axis=1).mean())
                previous_samples = int(telemetry["sample_count"])
                previous_confidence = telemetry.get("mean_confidence")
                telemetry["mean_confidence"] = (
                    batch_confidence
                    if previous_confidence is None
                    else (
                        float(previous_confidence) * previous_samples
                        + batch_confidence * len(frame)
                    )
                    / max(previous_samples + len(frame), 1)
                )
            except AttributeError:
                pass
            latency_ms = (time.perf_counter() - started) * 1000.0
            previous_requests = int(telemetry["request_count"])
            previous_mean = telemetry.get("mean_latency_ms")
            telemetry.update(
                request_count=previous_requests + 1,
                sample_count=int(telemetry["sample_count"]) + len(frame),
                last_latency_ms=latency_ms,
                mean_latency_ms=(
                    latency_ms
                    if previous_mean is None
                    else (
                        float(previous_mean) * previous_requests + latency_ms
                    )
                    / (previous_requests + 1)
                ),
                max_latency_ms=max(
                    float(telemetry.get("max_latency_ms") or 0.0),
                    latency_ms,
                ),
            )
            response["telemetry"] = _json_safe(telemetry)
            return response
        except (OSError, ValueError, CapabilityError) as exc:
            telemetry["error_count"] = int(telemetry["error_count"]) + 1
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/runs", status_code=202)
    async def create_run(request: Request):
        run_id = run_manager.new_run_id()
        try:
            content_type = request.headers.get("content-type", "")
            if content_type.startswith("application/json"):
                payload = await request.json()
                raw_config = payload.get("config", payload)
                if not remote_local_paths_allowed():
                    raise ConfigurationError(
                        "Server-local dataset paths are disabled in Firebase "
                        "mode; submit a browser upload instead."
                    )
                dataset = Path(payload.get("dataset_path", "")).expanduser().resolve()
            else:
                form = await request.form()
                raw_config = json.loads(str(form.get("config", "{}")))
                dataset_path = str(form.get("dataset_path", "")).strip()
                if dataset_path:
                    if not remote_local_paths_allowed():
                        raise ConfigurationError(
                            "Server-local dataset paths are disabled in "
                            "Firebase mode; submit a browser upload instead."
                        )
                    dataset = Path(dataset_path).expanduser().resolve()
                else:
                    max_mb = int(os.getenv("AUTONEXUS_MAX_UPLOAD_MB", "2048"))
                    dataset = await _save_uploads(
                        form,
                        run_manager.input_dir(run_id),
                        max_mb * 1024 * 1024,
                    )
            profile = inspect_dataset(dataset)
            config, secrets_for_run = _validate_config(
                raw_config, profile["modality"]
            )
            if studio_auth.mode == "agent":
                if config["execution_target"] != "local_agent":
                    raise ConfigurationError(
                        "Local-agent missions must select local execution."
                    )
                if not config["local_gpu_consent"]:
                    raise ConfigurationError(
                        "Explicit local CPU/GPU permission is required for this mission."
                    )
            elif config["execution_target"] == "local_agent":
                raise ConfigurationError(
                    "Local execution must be submitted to a paired local agent."
                )
            if profile["modality"] == "tabular" and config["target"] not in profile["columns"]:
                raise ConfigurationError(
                    f"Target column not found: {config['target']}"
                )
            return run_manager.enqueue(
                run_id,
                dataset,
                config,
                profile,
                secrets_for_run,
                owner_id=request.state.principal.uid,
            )
        except (ConfigurationError, FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
            staging = run_manager.run_dir(run_id)
            if run_id not in {item["id"] for item in run_manager.list()} and staging.is_dir():
                shutil.rmtree(staging, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/artifacts/{artifact_name}")
    async def download_artifact(
        run_id: str,
        artifact_name: str,
        request: Request,
    ):
        try:
            artifact = run_manager.artifact(
                run_id,
                artifact_name,
                owner_id=request.state.principal.uid,
            )
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="Artifact not found.") from exc
        disposition = "inline" if artifact_name == "html_report" else "attachment"
        headers = {"Content-Disposition": f'{disposition}; filename="{artifact.name}"'}
        return FileResponse(
            artifact,
            media_type=ARTIFACT_MEDIA_TYPES[artifact_name],
            headers=headers,
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autonexus-web",
        description="Launch the local Auto Nexus Studio.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--workspace",
        default=None,
        help=(
            "Run storage directory (default: AUTONEXUS_WEB_WORKSPACE or the "
            "current user's application-data directory)."
        ),
    )
    parser.add_argument(
        "--open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the Studio in the default browser.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise CapabilityError(
            'Web Studio requires: pip install "AutoNexus[serve]"'
        ) from exc
    args = build_parser().parse_args(argv)
    studio_auth = authenticator_from_env()
    if (
        args.host not in {"127.0.0.1", "localhost", "::1"}
        and studio_auth.mode != "firebase"
    ):
        raise SystemExit(
            "A non-loopback Studio requires AUTONEXUS_AUTH_MODE=firebase; "
            "otherwise use --host 127.0.0.1."
        )
    if args.open:
        timer = threading.Timer(
            1.25,
            lambda: webbrowser.open(f"http://{args.host}:{args.port}"),
        )
        timer.daemon = True
        timer.start()
    uvicorn.run(
        create_app(
            workspace=args.workspace,
            authenticator=studio_auth,
        ),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
