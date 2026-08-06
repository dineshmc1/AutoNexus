"""Production command-line entry point for AutoNexus.

Examples:
    python main.py data.csv --target outcome
    python main.py --target outcome
    python main.py data.xlsx --target price --problem-type regression --tune
    autonexus data.csv --target label --models logistic,rf,gb --report
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shlex
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.logging import RichHandler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder

from autonexus.cli_ui import (
    ask,
    console,
    event,
    phase,
    render_banner,
    render_final_dashboard,
    render_launch,
    render_results,
)

from data_cleaner import clean
from data_loader import DataBundle, load_dataset
from feature_processing import build_preprocessor
from model_selector import (
    evaluate_models,
    save_metrics,
    save_model,
    tune_top_models,
)
from model_trainer import baseline_screen, full_train, get_models
from resource_manager import ResourceManager

LOGGER = logging.getLogger("ml_builder")


def _configure_stdio() -> None:
    """Make legacy progress output safe on Windows and redirected terminals."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class RunConfig:
    dataset: Path
    target: str | None
    output_dir: Path
    problem_type: str | None
    models: list[str]
    test_size: float
    sample_fraction: float
    baseline_seconds: float
    cv: int
    max_time_seconds: float | None
    random_state: int
    feature_engineering: bool
    interactions: int | None
    ratios: bool
    outlier_strategy: str
    tune: bool
    tune_method: str
    tune_iterations: int
    report: bool
    shap: bool
    llm: bool
    notebook: bool
    adapt_lora: bool
    backbones: list[str]
    backbone_time_seconds: float
    preprocessing_cache: bool
    use_memory: bool
    contribute_memory: bool
    memory_dir: Path | None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed < 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _duration(value: str) -> float:
    """Parse a duration such as 30s, 10m, or 2h into seconds."""
    normalized = value.strip().lower()
    multipliers = {"s": 1.0, "m": 60.0, "h": 3600.0}
    suffix = normalized[-1]
    try:
        if suffix in multipliers:
            seconds = float(normalized[:-1]) * multipliers[suffix]
        else:
            seconds = float(normalized) * 60.0
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "use seconds, minutes, or hours, for example: 30s, 10m, 2h"
        ) from exc
    if seconds <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return seconds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autonexus",
        description="Train and evaluate tabular or image ML from one command.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        help="CSV or Excel file, or image folder; prompted when omitted",
    )
    parser.add_argument(
        "--target",
        help="Target column for tabular data; prompted for when omitted",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts"),
        help="Directory for the model, metrics, and run manifest",
    )
    parser.add_argument(
        "--problem-type", choices=("classification", "regression"),
        help="Override automatic task detection",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated model keys; omit to use resource-aware defaults",
    )
    parser.add_argument("--test-size", type=_fraction, default=0.2)
    parser.add_argument(
        "--sample", type=_fraction, default=0.1,
        help="Data fraction used by the cheap landmark/baseline stage",
    )
    parser.add_argument(
        "--baseline-time",
        type=_duration,
        default=15.0,
        help="Separate wall-clock budget for shortlist screening",
    )
    parser.add_argument("--cv", type=_positive_int, default=5)
    parser.add_argument(
        "--max-time", type=_duration,
        help="Training budget, for example 30s, 10m, or 2h",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature-engineering", action=argparse.BooleanOptionalAction,
        default=False, help="Enable adaptive feature engineering",
    )
    parser.add_argument(
        "--interactions", type=int,
        help="Maximum source features used for interactions",
    )
    parser.add_argument(
        "--ratios", action=argparse.BooleanOptionalAction, default=False,
        help="Generate ratio features when feature engineering is enabled",
    )
    parser.add_argument(
        "--outlier-strategy", choices=("cap", "none"), default="cap",
    )
    parser.add_argument(
        "--tune", action=argparse.BooleanOptionalAction, default=False,
        help="Tune the two strongest trained models",
    )
    parser.add_argument(
        "--tune-method", choices=("grid", "randomized"), default="randomized",
    )
    parser.add_argument("--tune-iterations", type=_positive_int, default=20)
    parser.add_argument(
        "--report", action=argparse.BooleanOptionalAction, default=True,
        help="Generate EDA and an HTML report",
    )
    parser.add_argument(
        "--shap", action=argparse.BooleanOptionalAction, default=False,
        help="Include SHAP plots in the report",
    )
    parser.add_argument(
        "--llm", action=argparse.BooleanOptionalAction, default=True,
        help="Generate a Markdown explanation using the configured LLM",
    )
    parser.add_argument(
        "--notebook", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Compatibility flag; analysis.ipynb is mandatory for every "
            "successful run"
        ),
    )
    parser.add_argument(
        "--adapt-lora",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fine-tune LoRA on image development data with early stopping",
    )
    parser.add_argument(
        "--backbones",
        default="auto",
        help=(
            "Comma-separated vision backbone keys (clip,dinov2,resnet,siglip) "
            "or auto"
        ),
    )
    parser.add_argument(
        "--backbone-time",
        type=_duration,
        default=900.0,
        help="Cooperative time budget for automatic backbone search",
    )
    parser.add_argument(
        "--preprocessing-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache reusable preprocessing transforms during model search",
    )
    parser.add_argument(
        "--memory-retrieval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use compatible FAISS run memory to advise the model shortlist",
    )
    parser.add_argument(
        "--contribute-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Contribute run metadata to the local AutoNexus FAISS memory",
    )
    parser.add_argument(
        "--memory-dir",
        type=Path,
        help="Override the local AutoNexus memory directory",
    )
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> RunConfig:
    dataset = args.dataset
    if dataset is None:
        try:
            entered_path = ask(
                "Dataset path  [nexus.muted](CSV, Excel, or image folder)[/]"
            ).strip()
        except EOFError as exc:
            raise ValueError(
                "Dataset path is required when input is non-interactive."
            ) from exc
        if not entered_path:
            raise ValueError("Dataset path cannot be empty.")

        # Accept convenient input such as:
        # C:\datasets\images --adapt-lora --no-shap
        option_match = re.search(
            r"\s+(--[a-z][a-z0-9-]*)", entered_path, flags=re.IGNORECASE
        )
        if option_match:
            path_text = entered_path[: option_match.start()].strip()
            option_text = entered_path[option_match.start() :].strip()
            prompt_options = [
                token.strip('"').strip("'")
                for token in shlex.split(option_text, posix=False)
            ]
            prompt_parser = build_parser()
            parsed_prompt = prompt_parser.parse_args(
                [path_text.strip('"').strip("'"), *prompt_options]
            )
            for token in prompt_options:
                if not token.startswith("--"):
                    continue
                option_name = token.split("=", 1)[0]
                action = prompt_parser._option_string_actions.get(option_name)
                if action is not None:
                    setattr(args, action.dest, getattr(parsed_prompt, action.dest))
            dataset = parsed_prompt.dataset
        else:
            dataset = Path(entered_path.strip('"').strip("'"))

    dataset = dataset.expanduser().resolve()
    if (
        dataset.is_dir()
        and dataset.name.lower() == "train"
        and (dataset.parent / "test").is_dir()
    ):
        event(
            "input route",
            f"Sibling train/test folders detected; root={dataset.parent}",
            tone="blue",
        )
        dataset = dataset.parent
    target = args.target
    if dataset.is_file() and target is None:
        try:
            target = ask("Target column").strip()
        except EOFError as exc:
            raise ValueError(
                "Target column is required for tabular data."
            ) from exc
        if not target:
            raise ValueError("Target column cannot be empty for tabular data.")

    models = []
    if args.models:
        models = [name.strip() for name in args.models.split(",") if name.strip()]
    if args.interactions is not None and args.interactions < 0:
        raise ValueError("--interactions cannot be negative")
    backbones = [
        name.strip().lower()
        for name in args.backbones.split(",")
        if name.strip()
    ]
    from vision_backbones import resolve_backbones

    resolve_backbones(backbones)
    return RunConfig(
        dataset=dataset,
        target=target,
        output_dir=args.output_dir.expanduser().resolve(),
        problem_type=args.problem_type,
        models=models,
        test_size=args.test_size,
        sample_fraction=args.sample,
        baseline_seconds=args.baseline_time,
        cv=args.cv,
        max_time_seconds=args.max_time,
        random_state=args.seed,
        feature_engineering=args.feature_engineering,
        interactions=args.interactions,
        ratios=args.ratios,
        outlier_strategy=args.outlier_strategy,
        tune=args.tune,
        tune_method=args.tune_method,
        tune_iterations=args.tune_iterations,
        report=args.report,
        shap=args.shap,
        llm=args.llm,
        notebook=args.notebook,
        adapt_lora=args.adapt_lora,
        backbones=backbones,
        backbone_time_seconds=args.backbone_time,
        preprocessing_cache=args.preprocessing_cache,
        use_memory=args.memory_retrieval,
        contribute_memory=args.contribute_memory,
        memory_dir=(
            None
            if args.memory_dir is None
            else args.memory_dir.expanduser().resolve()
        ),
    )


def _display_results(results: pd.DataFrame, problem_type: str) -> None:
    render_results(results, problem_type)


def _write_manifest(
    config: RunConfig,
    problem_type: str,
    best_model: str,
    elapsed_seconds: float,
    run_summary: dict[str, Any],
) -> Path:
    path = config.output_dir / "run.json"
    payload = asdict(config)
    payload["dataset"] = str(config.dataset)
    payload["output_dir"] = str(config.output_dir)
    payload["memory_dir"] = (
        None if config.memory_dir is None else str(config.memory_dir)
    )
    payload.update(
        problem_type=problem_type,
        best_model=best_model,
        model_used=best_model,
        label_column=config.target,
        elapsed_seconds=round(elapsed_seconds, 3),
        run_summary=run_summary,
    )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _memory_usage() -> dict[str, float | None]:
    """Return current/peak process RAM and CUDA VRAM in MiB."""
    ram_current = None
    ram_peak = None
    try:
        import psutil

        memory = psutil.Process().memory_info()
        ram_current = memory.rss / (1024 ** 2)
        peak_bytes = getattr(memory, "peak_wset", memory.rss)
        ram_peak = peak_bytes / (1024 ** 2)
    except ImportError:
        pass

    vram_current = None
    vram_peak = None
    try:
        import torch

        if torch.cuda.is_available():
            vram_current = torch.cuda.memory_allocated() / (1024 ** 2)
            vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    except ImportError:
        pass

    return {
        "ram_current_mb": ram_current,
        "ram_peak_mb": ram_peak,
        "vram_current_mb": vram_current,
        "vram_peak_mb": vram_peak,
    }


def _format_usage(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1f} MiB"


def _oob_score(model: Any) -> float | None:
    """Extract an out-of-bag score through calibration/pipeline wrappers."""
    candidate = getattr(model, "estimator", model)
    named_steps = getattr(candidate, "named_steps", None)
    if named_steps is not None:
        candidate = named_steps.get("model", candidate)
    value = getattr(candidate, "oob_score_", None)
    return None if value is None else float(value)


def _probe_image_representation(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_gate: pd.DataFrame,
    y_gate: pd.Series,
) -> dict[str, float]:
    """Score one frozen/adapted representation on a LoRA-unseen gate."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss

    probe = LogisticRegression(
        C=1.0,
        max_iter=3000,
        class_weight="balanced",
        random_state=42,
    )
    probe.fit(X_fit, y_fit)
    predictions = probe.predict(X_gate)
    probabilities = probe.predict_proba(X_gate)
    return {
        "accuracy": float(accuracy_score(y_gate, predictions)),
        "nll": float(log_loss(y_gate, probabilities, labels=probe.classes_)),
    }


def _load_image_dataset(config: RunConfig) -> tuple[DataBundle, pd.DataFrame]:
    """Build a leakage-safe image representation and standard data bundle."""
    image_started = time.monotonic()
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_count = sum(
        1
        for path in config.dataset.rglob("*")
        if path.is_file() and path.suffix.lower() in image_extensions
    )
    if image_count < 10:
        raise ValueError(
            "Image dataset must contain at least 10 supported images "
            "(.jpg, .jpeg, .png, .bmp, or .webp)."
        )

    try:
        import torch
        from multimodal_extractor import (
            UniversalEmbedder,
            discover_labeled_files,
        )
        from image_splitting import (
            infer_image_groups,
            split_labeled_indices,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Image training dependencies are missing. Run: uv sync --extra images"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    event(
        "vision input",
        f"{image_count} images detected / compute device={device}",
        tone="blue",
    )
    train_dir = config.dataset / "train"
    val_dir = config.dataset / "val"
    test_dir = config.dataset / "test"
    explicit_split = train_dir.is_dir() and test_dir.is_dir()
    split_method = "explicit-folders"
    development_groups: list[str] | None = None
    selected_development_groups: list[str] | None = None
    selected_development_files: list[str] | None = None
    test_groups: list[str] | None = None
    grouping_method = "explicit-folders"
    if explicit_split:
        train_files, train_labels = discover_labeled_files(
            str(train_dir), "vision"
        )
        val_files, val_labels = (
            discover_labeled_files(str(val_dir), "vision")
            if val_dir.is_dir()
            else ([], [])
        )
        test_files, test_labels = discover_labeled_files(
            str(test_dir), "vision"
        )
        development_files = train_files + val_files
        development_labels = train_labels + val_labels
        development_groups, grouping_method = infer_image_groups(
            development_files,
            development_labels,
            config.dataset,
        )
        event(
            "split firewall",
            f"{len(development_files)} development / {len(test_files)} test / "
            f"grouping={grouping_method}",
            tone="green",
        )
    else:
        all_files, all_labels = discover_labeled_files(
            str(config.dataset), "vision"
        )
        if len(set(all_labels)) < 2:
            raise ValueError(
                "Image classification requires at least two class folders."
            )
        try:
            all_groups, grouping_method = infer_image_groups(
                all_files, all_labels, config.dataset
            )
            development_idx, test_idx, split_method = split_labeled_indices(
                all_labels,
                test_size=config.test_size,
                random_state=config.random_state,
                groups=all_groups,
            )
        except ValueError as exc:
            raise ValueError(
                "Automatic image splitting failed. Each class needs enough "
                "images for development and test sets."
            ) from exc
        development_files = [all_files[index] for index in development_idx]
        development_labels = [
            all_labels[index] for index in development_idx
        ]
        test_files = [all_files[index] for index in test_idx]
        test_labels = [all_labels[index] for index in test_idx]
        development_groups = (
            [all_groups[index] for index in development_idx]
            if all_groups is not None
            else None
        )
        test_groups = (
            [all_groups[index] for index in test_idx]
            if all_groups is not None
            else None
        )
        event(
            "split firewall",
            f"{split_method} / {len(development_files)} development / "
            f"{len(test_files)} sealed test / grouping={grouping_method}",
            tone="green",
        )

    from analytics_artifacts import audit_image_files

    if explicit_split:
        audit_files = train_files + val_files + test_files
        audit_labels = train_labels + val_labels + test_labels
        audit_splits = (
            ["train"] * len(train_files)
            + ["validation"] * len(val_files)
            + ["test"] * len(test_files)
        )
    else:
        audit_files = development_files + test_files
        audit_labels = development_labels + test_labels
        audit_splits = (
            ["development_cv"] * len(development_files)
            + ["test"] * len(test_files)
        )
    audit_groups = (
        None
        if development_groups is None and test_groups is None
        else (
            (development_groups or [None] * len(development_files))
            + (test_groups or [None] * len(test_files))
        )
    )
    inventory = audit_image_files(
        audit_files,
        audit_labels,
        audit_splits,
        audit_groups,
        config.output_dir / "analysis_data",
        random_state=config.random_state,
    )
    valid_files = set(inventory.loc[inventory["readable"], "path"].astype(str))
    unreadable_count = int((~inventory["readable"]).sum())
    if unreadable_count:
        event(
            "data audit",
            f"{unreadable_count} unreadable images quarantined; see data_index.csv",
            tone="amber",
        )
    duplicate_split_counts = (
        inventory.loc[inventory["sha256"].ne("")]
        .groupby("sha256")["split"]
        .nunique()
    )
    duplicate_split_leaks = int((duplicate_split_counts > 1).sum())
    if duplicate_split_leaks:
        LOGGER.warning(
            "%d exact-duplicate image group(s) cross split boundaries. "
            "Treat final metrics as potentially optimistic and inspect the "
            "notebook leakage audit.",
            duplicate_split_leaks,
        )

    def keep_valid(
        files: list[str],
        labels: list[str],
        groups: list[str] | None = None,
    ) -> tuple[list[str], list[str], list[str] | None]:
        positions = [
            index
            for index, filename in enumerate(files)
            if str(Path(filename).resolve()) in valid_files
        ]
        return (
            [files[index] for index in positions],
            [labels[index] for index in positions],
            (
                None
                if groups is None
                else [groups[index] for index in positions]
            ),
        )

    development_group_map = (
        None
        if development_groups is None
        else dict(zip(development_files, development_groups))
    )
    development_files, development_labels, development_groups = keep_valid(
        development_files, development_labels, development_groups
    )
    test_files, test_labels, test_groups = keep_valid(
        test_files, test_labels, test_groups
    )
    if explicit_split:
        explicit_train_groups = (
            None
            if development_group_map is None
            else [development_group_map[filename] for filename in train_files]
        )
        explicit_val_groups = (
            None
            if development_group_map is None
            else [development_group_map[filename] for filename in val_files]
        )
        train_files, train_labels, explicit_train_groups = keep_valid(
            train_files, train_labels, explicit_train_groups
        )
        val_files, val_labels, explicit_val_groups = keep_valid(
            val_files, val_labels, explicit_val_groups
        )
        development_files = train_files + val_files
        development_labels = train_labels + val_labels
        development_groups = (
            None
            if explicit_train_groups is None or explicit_val_groups is None
            else explicit_train_groups + explicit_val_groups
        )

    cache_dir = config.output_dir / ".cache" / "embeddings"
    from backbone_selector import select_vision_backbone

    embedding_started = time.monotonic()
    backbone_selection = select_vision_backbone(
        development_files,
        development_labels,
        development_groups,
        requested=config.backbones,
        device=device,
        cache_dir=str(cache_dir),
        time_budget_seconds=config.backbone_time_seconds,
        random_state=config.random_state,
    )
    selected_backbone = backbone_selection.spec
    selected_backbone_revision = backbone_selection.metrics["selected"][
        "resolved_revision"
    ]
    frozen_development_X = backbone_selection.embeddings
    frozen_development_labels = backbone_selection.labels.astype(str)
    image_metadata: dict[str, Any] = {
        "backbone": selected_backbone.model_id,
        "backbone_key": selected_backbone.key,
        "backbone_family": selected_backbone.family,
        "backbone_revision": selected_backbone_revision,
        "backbone_search": backbone_selection.metrics,
        "split_method": split_method,
        "grouping_method": grouping_method,
        "development_images": len(development_files),
        "test_images": len(test_files),
        "lora_requested": config.adapt_lora,
    }

    def frozen_fallback_after_lora_failure(
        exc: Exception,
        failure_started: float,
    ) -> tuple[DataBundle, pd.DataFrame]:
        failure = f"{type(exc).__name__}: {exc}"
        LOGGER.warning(
            "LoRA path failed; retrying with the selected frozen backbone: %s",
            failure,
        )
        failed_seconds = time.monotonic() - failure_started
        fallback_bundle, fallback_raw = _load_image_dataset(
            replace(config, adapt_lora=False)
        )
        fallback_bundle.metadata["lora_requested"] = True
        fallback_bundle.metadata["lora_gate"] = {
            "adapter_selected": False,
            "status": "failed-frozen-fallback",
            "error": failure,
        }
        fallback_bundle.metadata["failed_lora_seconds"] = round(
            failed_seconds, 3
        )
        fallback_bundle.metadata["total_image_input_seconds"] = round(
            fallback_bundle.metadata.get("total_image_input_seconds", 0.0)
            + failed_seconds,
            3,
        )
        return fallback_bundle, fallback_raw

    if config.adapt_lora and selected_backbone.supports_lora:
        from lora_adapter_trainer import train_universal_lora

        if explicit_split and val_files:
            probe_files = train_files
            probe_labels = train_labels
            gate_files = val_files
            gate_labels = val_labels
            if development_groups is not None:
                probe_groups = development_groups[: len(train_files)]
                gate_groups = development_groups[len(train_files) :]
                probe_grouping = grouping_method
            else:
                probe_groups, probe_grouping = infer_image_groups(
                    probe_files, probe_labels, train_dir
                )
                gate_groups = None
            gate_split_method = "explicit-val-folder"
        else:
            probe_groups_all, probe_grouping = infer_image_groups(
                development_files,
                development_labels,
                train_dir if explicit_split else config.dataset,
            )
            if development_groups is not None:
                probe_groups_all = development_groups
            probe_idx, gate_idx, gate_split_method = split_labeled_indices(
                development_labels,
                test_size=0.15,
                random_state=config.random_state + 1,
                groups=probe_groups_all,
            )
            probe_files = [
                development_files[index] for index in probe_idx
            ]
            probe_labels = [
                development_labels[index] for index in probe_idx
            ]
            gate_files = [development_files[index] for index in gate_idx]
            gate_labels = [
                development_labels[index] for index in gate_idx
            ]
            probe_groups = (
                [probe_groups_all[index] for index in probe_idx]
                if probe_groups_all is not None
                else None
            )
            gate_groups = (
                [probe_groups_all[index] for index in gate_idx]
                if probe_groups_all is not None
                else None
            )

        if probe_groups is not None and gate_groups is not None:
            selected_development_groups = probe_groups + gate_groups
        selected_development_files = probe_files + gate_files

        adapter_path = (
            config.output_dir / "lora_adapter" / selected_backbone.key
        )
        lora_started = time.monotonic()
        try:
            train_universal_lora(
                modality="vision",
                domain="general",
                data_dir=str(config.dataset),
                output_path=str(adapter_path),
                files=probe_files,
                labels=probe_labels,
                groups=probe_groups,
                random_state=config.random_state,
                model_id=selected_backbone.model_id,
                model_revision=selected_backbone_revision,
            )
        except Exception as exc:
            return frozen_fallback_after_lora_failure(exc, lora_started)
        image_metadata["lora_training_seconds"] = round(
            time.monotonic() - lora_started, 3
        )

        development_positions = {
            filename: index
            for index, filename in enumerate(development_files)
        }
        frozen_fit = frozen_development_X.iloc[
            [development_positions[filename] for filename in probe_files]
        ].reset_index(drop=True)
        frozen_fit_labels = pd.Series(probe_labels, name="label")
        frozen_gate = frozen_development_X.iloc[
            [development_positions[filename] for filename in gate_files]
        ].reset_index(drop=True)
        frozen_gate_labels = pd.Series(gate_labels, name="label")
        frozen_scores = _probe_image_representation(
            frozen_fit,
            frozen_fit_labels.astype(str),
            frozen_gate,
            frozen_gate_labels.astype(str),
        )
        adapted_embedder = UniversalEmbedder(
            device=device,
            batch_size=selected_backbone.batch_size,
            domain="general",
            modality="vision",
            adapter_path=str(adapter_path),
            cache_dir=str(cache_dir),
            model_id=selected_backbone.model_id,
            model_revision=selected_backbone_revision,
        )
        try:
            adapted_fit, adapted_fit_labels = adapted_embedder.embed_files(
                probe_files,
                probe_labels,
                cache_key=f"{config.dataset}:lora-probe-fit:adapted",
            )
            adapted_gate, adapted_gate_labels = adapted_embedder.embed_files(
                gate_files,
                gate_labels,
                cache_key=f"{config.dataset}:lora-gate:adapted",
            )
            adapted_scores = _probe_image_representation(
                adapted_fit,
                adapted_fit_labels.astype(str),
                adapted_gate,
                adapted_gate_labels.astype(str),
            )
            frozen_movement = pd.concat(
                [frozen_fit, frozen_gate], ignore_index=True
            )
            adapted_movement = pd.concat(
                [adapted_fit, adapted_gate], ignore_index=True
            )
            movement_labels = pd.concat(
                [adapted_fit_labels, adapted_gate_labels], ignore_index=True
            ).astype(str)
            movement_count = min(len(frozen_movement), 1200)
            movement_indices = np.linspace(
                0,
                len(frozen_movement) - 1,
                movement_count,
                dtype=int,
            )
            movement_path = (
                config.output_dir / "analysis_data" / "lora_movement.npz"
            )
            movement_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                movement_path,
                frozen=frozen_movement.iloc[movement_indices].to_numpy(
                    dtype=np.float16
                ),
                adapted=adapted_movement.iloc[movement_indices].to_numpy(
                    dtype=np.float16
                ),
                labels=movement_labels.iloc[movement_indices].to_numpy(
                    dtype=str
                ),
            )
            image_metadata["lora_movement_path"] = str(
                movement_path.resolve()
            )
        except Exception as exc:
            adapted_embedder.release()
            return frozen_fallback_after_lora_failure(exc, lora_started)
        accuracy_gain = (
            adapted_scores["accuracy"] - frozen_scores["accuracy"]
        )
        nll_gain = frozen_scores["nll"] - adapted_scores["nll"]
        adapter_selected = (
            accuracy_gain >= 0.002
            and adapted_scores["nll"] <= frozen_scores["nll"] + 0.01
        ) or (accuracy_gain >= 0.0 and nll_gain >= 0.01)

        image_metadata["lora_gate"] = {
            "split_method": gate_split_method,
            "grouping_method": probe_grouping,
            "probe_images": len(probe_files),
            "gate_images": len(gate_files),
            "frozen": {
                key: round(value, 6)
                for key, value in frozen_scores.items()
            },
            "adapted": {
                key: round(value, 6)
                for key, value in adapted_scores.items()
            },
            "accuracy_gain": round(accuracy_gain, 6),
            "nll_improvement": round(nll_gain, 6),
            "adapter_selected": adapter_selected,
        }
        selected_name = (
            f"adapted-{selected_backbone.key}"
            if adapter_selected
            else f"frozen-{selected_backbone.key}"
        )
        event(
            "lora gate",
            f"frozen={frozen_scores['accuracy']:.4f} / "
            f"adapted={adapted_scores['accuracy']:.4f} / "
            f"NLL {frozen_scores['nll']:.4f}->{adapted_scores['nll']:.4f} / "
            f"selected={selected_name}",
            tone="green" if adapter_selected else "blue",
        )

        if adapter_selected:
            X = pd.concat(
                [adapted_fit, adapted_gate], ignore_index=True
            )
            labels = pd.concat(
                [adapted_fit_labels, adapted_gate_labels], ignore_index=True
            )
            selected_embedder = adapted_embedder
        else:
            adapted_embedder.release()
            X = pd.concat(
                [frozen_fit, frozen_gate], ignore_index=True
            )
            labels = pd.concat(
                [frozen_fit_labels, frozen_gate_labels], ignore_index=True
            )
            selected_embedder = UniversalEmbedder(
                device=device,
                batch_size=selected_backbone.batch_size,
                domain="general",
                modality="vision",
                adapter_path=None,
                cache_dir=str(cache_dir),
                model_id=selected_backbone.model_id,
                model_revision=selected_backbone_revision,
            )
        image_metadata["selected_representation"] = selected_name
        try:
            X_test, test_labels = selected_embedder.embed_files(
                test_files,
                test_labels,
                cache_key=(
                    f"{config.dataset}:test:{selected_name}:"
                    f"{config.random_state}"
                ),
            )
        except Exception as exc:
            selected_embedder.release()
            if adapter_selected:
                return frozen_fallback_after_lora_failure(
                    exc, lora_started
                )
            raise
        selected_embedder.release()
    else:
        if config.adapt_lora and not selected_backbone.supports_lora:
            image_metadata["lora_gate"] = {
                "adapter_selected": False,
                "status": "skipped-unsupported-backbone",
                "reason": (
                    f"{selected_backbone.key} uses adaptation strategy "
                    f"'{selected_backbone.adaptation}', not transformer LoRA"
                ),
            }
            event(
                "lora bypass",
                f"{selected_backbone.key} has no q/v adapter path; frozen representation retained",
                tone="amber",
            )
        embedder = UniversalEmbedder(
            device=device,
            batch_size=selected_backbone.batch_size,
            domain="general",
            modality="vision",
            adapter_path=None,
            cache_dir=str(cache_dir),
            model_id=selected_backbone.model_id,
            model_revision=selected_backbone_revision,
        )
        X = frozen_development_X
        labels = frozen_development_labels
        X_test, test_labels = embedder.embed_files(
            test_files,
            test_labels,
            cache_key=(
                f"{config.dataset}:test:frozen-{selected_backbone.key}:"
                f"{config.random_state}"
            ),
        )
        embedder.release()
        image_metadata["selected_representation"] = (
            f"frozen-{selected_backbone.key}"
        )
        selected_development_groups = development_groups
        selected_development_files = development_files

    image_metadata["embedding_and_gate_seconds"] = round(
        time.monotonic() - embedding_started
        - image_metadata.get("lora_training_seconds", 0.0),
        3,
    )
    image_metadata["total_image_input_seconds"] = round(
        time.monotonic() - image_started, 3
    )
    groups_train: pd.Series | None = None
    if (
        selected_development_groups is not None
        and len(selected_development_groups) == len(X)
    ):
        groups_train = pd.Series(
            selected_development_groups, name="image_group"
        )
        image_metadata["downstream_cv_grouping"] = "stratified-group"
    elif selected_development_groups is not None:
        image_metadata["downstream_cv_grouping"] = (
            "stratified-fallback-after-invalid-image-drop"
        )
    else:
        image_metadata["downstream_cv_grouping"] = "stratified"
    labels = pd.Series(labels, name="label").astype(str)
    test_labels = pd.Series(test_labels, name="label").astype(str)
    class_counts = labels.value_counts()
    if len(class_counts) < 2:
        raise ValueError(
            "Image classification requires at least two class subfolders."
        )
    if class_counts.min() < 2:
        raise ValueError(
            "Each image class must contain at least two valid images."
        )

    encoder = LabelEncoder().fit(labels)
    image_metadata["class_names"] = encoder.classes_.astype(str).tolist()
    y = pd.Series(encoder.transform(labels), name="label")
    unseen = sorted(set(test_labels) - set(encoder.classes_))
    if unseen:
        raise ValueError(f"Test split contains unseen classes: {unseen}")
    y_test = pd.Series(encoder.transform(test_labels), name="label")
    bundle = DataBundle(
        X_train=X.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        problem_type="classification",
        feature_names=list(X.columns),
        target_name="label",
        metadata=image_metadata,
        groups_train=groups_train,
        groups_test=(
            None
            if test_groups is None
            else pd.Series(test_groups, name="image_group")
        ),
        row_ids_train=pd.Series(
            selected_development_files or development_files,
            name="image_path",
        ),
        row_ids_test=pd.Series(test_files, name="image_path"),
    )
    raw_df = pd.concat([X, X_test], ignore_index=True)
    raw_df["label"] = pd.concat(
        [labels, test_labels], ignore_index=True
    ).to_numpy()
    return bundle, raw_df


def _load_input(config: RunConfig) -> tuple[DataBundle, pd.DataFrame]:
    if not config.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {config.dataset}")
    if config.dataset.is_dir():
        return _load_image_dataset(config)
    if config.target is None:
        raise ValueError("Target column is required for tabular data.")

    bundle = load_dataset(
        str(config.dataset),
        config.target,
        test_size=config.test_size,
        random_state=config.random_state,
        problem_type=config.problem_type,
    )
    raw_df = pd.concat(
        [
            bundle.X_train.assign(**{bundle.target_name: bundle.y_train.to_numpy()}),
            bundle.X_test.assign(**{bundle.target_name: bundle.y_test.to_numpy()}),
        ],
        ignore_index=True,
    )
    return bundle, raw_df


def render_run_completion(dashboard: dict[str, Any]) -> None:
    """Render completion only after every caller-owned finalizer succeeds."""
    phase(5, "Mission Complete", "validated model / sealed artifact bundle")
    _display_results(dashboard["results"], dashboard["problem_type"])
    render_final_dashboard(
        best_model=dashboard["best_model"],
        representation=dashboard["representation"],
        metric_rows=dashboard["metric_rows"],
        timing_rows=dashboard["timing_rows"],
        resource_rows=dashboard["resource_rows"],
        artifacts=dashboard["artifacts"],
        elapsed_seconds=dashboard["elapsed_seconds"],
    )


def run(
    config: RunConfig, *, render_completion: bool = True
) -> dict[str, Any]:
    _configure_stdio()
    started = time.monotonic()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    phase(1, "Input Matrix", "validation / split firewall / representation")
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    bundle, raw_df = _load_input(config)
    input_seconds = time.monotonic() - started
    if config.dataset.is_dir():
        # Image embeddings are finite numeric vectors. Keeping rows intact
        # preserves the alignment of optional video/subject CV groups.
        X_train = bundle.X_train.copy()
        y_train = bundle.y_train.copy()
        X_test = bundle.X_test.copy()
        y_test = bundle.y_test.copy()
    else:
        X_train, y_train = clean(bundle.X_train, bundle.y_train)
        X_test, y_test = clean(
            bundle.X_test, bundle.y_test, verbose=False
        )
    training_groups = (
        None
        if bundle.groups_train is None
        else bundle.groups_train.to_numpy()
    )

    phase(2, "Candidate Search", "landmarking / shortlist / memory profile")
    decisions = ResourceManager().analyze(X_train, bundle.problem_type)
    cv = config.cv
    if decisions["size_category"] == "large":
        cv = min(cv, 2)

    requested_models = config.models or decisions["models_to_run"]
    models = get_models(
        bundle.problem_type, requested_models, n_samples=len(X_train)
    )
    if not models:
        raise RuntimeError("No compatible models are available in this environment")

    encoding_map = decisions["encoding_strategies"]
    baseline_preprocessor, _, _ = build_preprocessor(
        X_train, encoding_map=encoding_map
    )
    training_started = time.monotonic()
    baseline_budget = config.baseline_seconds
    if config.max_time_seconds is not None:
        baseline_budget = min(baseline_budget, config.max_time_seconds)
    promising, baseline_scores = baseline_screen(
        models,
        baseline_preprocessor,
        X_train,
        y_train,
        bundle.problem_type,
        sample_frac=config.sample_fraction,
        cv=min(cv, 2),
        random_state=config.random_state,
        max_time_seconds=baseline_budget,
        preprocessing_cache_dir=(
            str(config.output_dir / ".cache" / "preprocessing")
            if config.preprocessing_cache
            else None
        ),
        groups=training_groups,
    )
    from dataset_embedding import (
        SEARCH_EMBEDDING_VERSION,
        compute_search_embedding,
    )

    search_embedding = compute_search_embedding(
        X_train, y_train, baseline_scores
    )
    search_profile = {
        "version": SEARCH_EMBEDDING_VERSION,
        "sample_fraction": config.sample_fraction,
        "cv_folds": min(cv, 2),
        "budget_seconds": baseline_budget,
        "models_screened": list(baseline_scores),
        "baseline_scores": baseline_scores,
        "embedding": search_embedding.tolist(),
    }
    memory_advice: dict[str, Any] = {
        "status": "disabled",
        "rationale": "Memory retrieval was disabled for this run.",
    }
    if config.use_memory:
        try:
            from autonexus.memory import retrieve_search_advice

            memory_advice = retrieve_search_advice(
                search_embedding,
                baseline_scores,
                problem_type=bundle.problem_type,
                embedding_version=SEARCH_EMBEDDING_VERSION,
                memory_dir=config.memory_dir,
            )
        except Exception as exc:
            memory_advice = {
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
                "rationale": "Ordinary current-dataset search was retained.",
            }
            LOGGER.warning("FAISS memory retrieval failed: %s", exc)
    if bundle.problem_type == "classification":
        from generalization import MODEL_FAMILIES

        # Preserve the strongest screened candidate from each structural
        # family so diversity remains available for validation later.
        for family_names in MODEL_FAMILIES.values():
            family_candidates = [
                name for name in family_names if name in baseline_scores
            ]
            if family_candidates:
                family_best = max(
                    family_candidates,
                    key=lambda name: baseline_scores[name]["score"],
                )
                promising[family_best] = models[family_best]
    if config.use_memory:
        try:
            from autonexus.memory import apply_search_advice

            promising, shortlist_changes = apply_search_advice(
                promising,
                models,
                baseline_scores,
                memory_advice,
            )
            memory_advice["shortlist_changes"] = shortlist_changes
        except Exception as exc:
            memory_advice = {
                **memory_advice,
                "status": "advice-rejected",
                "error": f"{type(exc).__name__}: {exc}",
            }
            LOGGER.warning("FAISS memory advice was rejected: %s", exc)
    memory_advice["final_shortlist"] = list(promising)
    search_profile["memory_advice"] = memory_advice
    (config.output_dir / "search_profile.json").write_text(
        json.dumps(search_profile, indent=2), encoding="utf-8"
    )

    fe_log: list[str] = []
    scaler_map = None
    engineer = None
    X_train_final, X_test_final = X_train, X_test
    if config.feature_engineering:
        from feature_engineering import FeatureEngineer

        interaction_count = (
            decisions["interaction_k"]
            if config.interactions is None
            else min(config.interactions, decisions["interaction_k"])
        )
        engineer = FeatureEngineer(
            interaction_features=interaction_count,
            enable_ratios=config.ratios,
            outlier_strategy=config.outlier_strategy,
        )
        X_train_final = engineer.fit_transform(
            X_train, y_train, problem_type=bundle.problem_type
        )
        X_test_final = engineer.transform(X_test)
        scaler_map = engineer.get_scalers()
        fe_log = engineer.log.copy()

    preprocessor, _, _ = build_preprocessor(
        X_train_final, scaler_map=scaler_map, encoding_map=encoding_map
    )
    phase(3, "Model Forge", "cross-validation / tuning / generalization gate")
    training_diagnostics: dict[str, dict[str, Any]] = {}
    trained, validation_scores = full_train(
        promising,
        preprocessor,
        X_train_final,
        y_train,
        bundle.problem_type,
        cv=cv,
        max_time_seconds=config.max_time_seconds,
        preprocessing_cache_dir=(
            str(config.output_dir / ".cache" / "preprocessing")
            if config.preprocessing_cache
            else None
        ),
        groups=training_groups,
        random_state=config.random_state,
        diagnostics=training_diagnostics,
    )
    if not trained:
        raise RuntimeError(
            "No model completed training; increase --max-time or choose fewer models"
        )

    if config.tune:
        tuned, tuned_scores = tune_top_models(
            trained,
            X_train_final,
            y_train,
            bundle.problem_type,
            validation_scores,
            top_n=min(2, len(trained)),
            method=config.tune_method,
            n_iter=config.tune_iterations,
            cv=cv,
            groups=training_groups,
        )
        trained.update(tuned)
        validation_scores.update(tuned_scores)

    generalization = None
    generalization_gate_metric: float | None = None
    primary_cv_scope = "selected-model"
    if bundle.problem_type == "classification":
        from generalization import select_generalized_classifier

        generalization = select_generalized_classifier(
            trained,
            validation_scores,
            X_train_final,
            y_train,
            random_state=config.random_state,
            groups=training_groups,
        )
        trained[generalization.name] = generalization.model
        validation_scores[generalization.name] = (
            generalization.primary_cv_accuracy
        )
        generalization_gate_metric = generalization.gate_accuracy
        if generalization.ensemble_used:
            primary_cv_scope = "best-ensemble-member-reference"
        best_name = generalization.name
        calibration_message = (
            f"NLL {generalization.nll_before:.4f}->{generalization.nll_after:.4f}"
            if generalization.nll_before is not None
            and generalization.nll_after is not None
            else "calibration skipped / insufficient validation data"
        )
        event(
            "generalization",
            f"selected={best_name} / T={generalization.temperature:.3f} / "
            f"{calibration_message}",
            tone="green",
        )
    else:
        best_name = max(
            trained, key=lambda name: validation_scores.get(name, -np.inf)
        )
    # Final evaluation is the first point at which the held-out test set is
    # used; all selection, tuning, ensembling, and calibration happen above.
    results = evaluate_models(
        {best_name: trained[best_name]},
        X_test_final,
        y_test,
        bundle.problem_type,
    )
    training_seconds = time.monotonic() - training_started
    best_model = trained[best_name]
    train_predictions = best_model.predict(X_train_final)
    if bundle.problem_type == "classification":
        training_metric_name = "training_accuracy"
        validation_metric_name = "validation_accuracy"
        testing_metric_name = "testing_accuracy"
        training_metric = float(accuracy_score(y_train, train_predictions))
        validation_metric = float(validation_scores.get(best_name, np.nan))
        testing_metric = float(
            results.loc[results["model"] == best_name, "accuracy"].iloc[0]
        )
    else:
        training_metric_name = "training_r2"
        validation_metric_name = "validation_score"
        testing_metric_name = "testing_r2"
        training_metric = float(r2_score(y_train, train_predictions))
        validation_metric = float(validation_scores.get(best_name, np.nan))
        testing_metric = float(
            results.loc[results["model"] == best_name, "r2"].iloc[0]
        )

    oob_score = _oob_score(best_model)
    save_model(trained[best_name], str(config.output_dir / "best_model.joblib"))
    from nexus_predictor import NexusPredictor

    predictor_artifact = NexusPredictor(
        model=trained[best_name],
        problem_type=bundle.problem_type,
        target_name=bundle.target_name,
        feature_names=list(X_train.columns),
        class_names=list(bundle.metadata.get("class_names", [])),
        feature_engineer=engineer,
        modality="vision" if config.dataset.is_dir() else "tabular",
        metadata=dict(bundle.metadata),
    )
    save_model(predictor_artifact, str(config.output_dir / "model.pkl"))
    save_metrics(results, str(config.output_dir / "metrics.csv"))

    plot_paths: list[str] = []
    html_report_path: str | None = None
    explanations: dict[str, str] = {
        "feature_importance_status": "disabled",
        "feature_importance_error": "Reporting was disabled.",
        "shap_status": "disabled",
        "shap_error": "Reporting was disabled.",
    }
    report_errors: list[str] = []
    phase(4, "Evidence Layer", "analytics / explanations / reproducibility")
    report_started = time.monotonic()
    if config.report:
        report_dir = config.output_dir / "report"
        eda_result: dict[str, Any] = {}
        try:
            from eda import run_eda

            eda_result = run_eda(
                raw_df,
                bundle.target_name,
                bundle.problem_type,
                output_dir=str(report_dir / "eda"),
            )
        except Exception as exc:
            message = f"EDA failed: {type(exc).__name__}: {exc}"
            report_errors.append(message)
            LOGGER.warning(message)
        try:
            from explainer import run_explanations

            explanations = run_explanations(
                trained[best_name],
                X_test_final,
                y_test,
                output_dir=str(report_dir / "explanations"),
                use_shap=config.shap,
            )
        except Exception as exc:
            message = f"Explainability failed: {type(exc).__name__}: {exc}"
            report_errors.append(message)
            LOGGER.warning(message)
            explanations = {
                "feature_importance_status": "failed",
                "feature_importance_error": message,
                "shap_status": "failed",
                "shap_error": message,
            }
        if eda_result:
            try:
                from report_generator import generate_report

                html_report_path = generate_report(
                    summary=eda_result["summary"],
                    results=results,
                    best_name=best_name,
                    problem_type=bundle.problem_type,
                    eda_paths=eda_result,
                    explanation_paths=explanations,
                    fe_log=fe_log,
                    output_path=str(report_dir / "report.html"),
                )
            except Exception as exc:
                message = f"HTML report failed: {type(exc).__name__}: {exc}"
                report_errors.append(message)
                LOGGER.warning(message)
        plot_paths = [
            path
            for path in [
                eda_result.get("target_dist_path"),
                eda_result.get("feature_dist_path"),
                eda_result.get("corr_heatmap_path"),
                *(
                    value
                    for key, value in explanations.items()
                    if key.endswith("_path")
                ),
            ]
            if path
        ]
    reporting_seconds = time.monotonic() - report_started

    elapsed = time.monotonic() - started
    usage = _memory_usage()
    summary = {
        training_metric_name: round(training_metric, 6),
        validation_metric_name: (
            None if np.isnan(validation_metric) else round(validation_metric, 6)
        ),
        testing_metric_name: round(testing_metric, 6),
        "training_seconds": round(training_seconds, 3),
        "downstream_automl_seconds": round(training_seconds, 3),
        "input_preparation_seconds": round(input_seconds, 3),
        "reporting_seconds": round(reporting_seconds, 3),
        "total_pipeline_seconds": round(elapsed, 3),
        **{
            key: None if value is None else round(value, 3)
            for key, value in usage.items()
        },
        "search_models_screened": len(baseline_scores),
        "search_embedding_version": SEARCH_EMBEDDING_VERSION,
        "memory_retrieval": memory_advice,
        "input_metadata": bundle.metadata,
        "explanations": {
            key: value
            for key, value in explanations.items()
            if key.endswith("_status")
            or key.endswith("_error")
            or key in {"shap_scope", "shap_reference_model"}
        },
        "report_errors": report_errors,
    }
    summary["fitted_training_metric"] = round(training_metric, 6)
    summary["primary_cross_validated_metric"] = (
        None if np.isnan(validation_metric) else round(validation_metric, 6)
    )
    summary["held_out_testing_metric"] = round(testing_metric, 6)
    summary["primary_cross_validated_metric_scope"] = primary_cv_scope
    summary["generalization_gate_metric"] = (
        None
        if generalization_gate_metric is None
        else round(generalization_gate_metric, 6)
    )
    summary["fit_validation_gap"] = (
        None
        if np.isnan(validation_metric)
        else round(training_metric - validation_metric, 6)
    )
    summary["validation_test_gap"] = (
        None
        if np.isnan(validation_metric)
        else round(validation_metric - testing_metric, 6)
    )
    if oob_score is not None:
        summary["out_of_bag_score"] = round(oob_score, 6)
    if generalization is not None:
        summary["temperature"] = round(generalization.temperature, 6)
        summary["calibration_nll_before"] = (
            None
            if generalization.nll_before is None
            else round(generalization.nll_before, 6)
        )
        summary["calibration_nll_after"] = (
            None
            if generalization.nll_after is None
            else round(generalization.nll_after, 6)
        )
        summary["ensemble_used"] = generalization.ensemble_used
        summary["ensemble_members"] = generalization.members
    report_dir = config.output_dir / "report"
    markdown_path: str | None = None
    llm_started = time.monotonic()
    from llm_explainer import generate_comprehensive_report

    metric_row = results.loc[results["model"] == best_name].iloc[0]
    llm_context = {
        "dataset": {
            "path": str(config.dataset),
            "modality": "vision" if config.dataset.is_dir() else "tabular",
            "samples": len(raw_df),
            "features": len(bundle.feature_names),
            "problem_type": bundle.problem_type,
        },
        "model": {"name": best_name},
        "image_input": (
            bundle.metadata if config.dataset.is_dir() else None
        ),
        "performance": {
            "training": training_metric,
            "validation": (
                None if np.isnan(validation_metric) else validation_metric
            ),
            "testing": testing_metric,
            "held_out_metrics": {
                key: (
                    None
                    if pd.isna(value)
                    else value.item()
                    if hasattr(value, "item")
                    else value
                )
                for key, value in metric_row.to_dict().items()
            },
        },
        "resources": summary,
        "plots": plot_paths,
    }
    markdown_path = generate_comprehensive_report(
        llm_context,
        config.dataset.stem,
        output_path=str(report_dir / "explanation.md"),
        use_llm=config.llm,
    )
    llm_seconds = time.monotonic() - llm_started
    summary["llm_seconds"] = round(llm_seconds, 3)

    notebook_path: str | None = None
    notebook_started = time.monotonic()
    from analytics_artifacts import persist_run_analytics
    from notebook_generator import generate_advanced_notebook

    analytics_paths = persist_run_analytics(
        output_dir=config.output_dir,
        config=config,
        bundle=bundle,
        best_name=best_name,
        best_model=best_model,
        X_train_final=X_train_final,
        X_test_final=X_test_final,
        y_train=y_train,
        y_test=y_test,
        results=results,
        baseline_scores=baseline_scores,
        validation_scores=validation_scores,
        training_diagnostics=training_diagnostics,
        summary=summary,
    )
    notebook_path = generate_advanced_notebook(
        {
            "data_path": str(config.dataset),
            "modality": (
                "vision" if config.dataset.is_dir() else "tabular"
            ),
            "problem_type": bundle.problem_type,
        },
        {
            "best_model": best_name,
            "summary": summary,
            "metrics_path": str(config.output_dir / "metrics.csv"),
            "model_path": str(config.output_dir / "model.pkl"),
            "plot_paths": plot_paths,
            "analytics_paths": analytics_paths,
        },
        str(config.output_dir / "analysis.ipynb"),
    )
    notebook_seconds = time.monotonic() - notebook_started

    elapsed = time.monotonic() - started
    usage = _memory_usage()
    summary.update(
        llm_seconds=round(llm_seconds, 3),
        notebook_seconds=round(notebook_seconds, 3),
        total_pipeline_seconds=round(elapsed, 3),
        **{
            key: None if value is None else round(value, 3)
            for key, value in usage.items()
        },
    )
    manifest = _write_manifest(
        config, bundle.problem_type, best_name, elapsed, summary
    )
    try:
        from autonexus.memory import (
            contribute_run,
            contribution_to_dict,
        )

        memory_contribution = contribute_run(
            config.output_dir,
            enabled=config.contribute_memory,
            memory_dir=config.memory_dir,
        )
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        manifest_payload["faiss_memory"] = contribution_to_dict(
            memory_contribution
        )
        manifest_payload["artifact_contract"] = {
            "run_json": str(manifest.resolve()),
            "model_pkl": str((config.output_dir / "model.pkl").resolve()),
            "analysis_ipynb": str(
                (config.output_dir / "analysis.ipynb").resolve()
            ),
            "explanation_md": str(Path(markdown_path).resolve()),
            "search_profile_json": str(
                (config.output_dir / "search_profile.json").resolve()
            ),
        }
        manifest.write_text(
            json.dumps(manifest_payload, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        LOGGER.warning("FAISS memory contribution failed: %s", exc)
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        manifest_payload["faiss_memory"] = {
            "contributed": False,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "memory_dir": (
                None if config.memory_dir is None else str(config.memory_dir)
            ),
        }
        manifest_payload["artifact_contract"] = {
            "run_json": str(manifest.resolve()),
            "model_pkl": str((config.output_dir / "model.pkl").resolve()),
            "analysis_ipynb": str(
                (config.output_dir / "analysis.ipynb").resolve()
            ),
            "explanation_md": str(Path(markdown_path).resolve()),
            "search_profile_json": str(
                (config.output_dir / "search_profile.json").resolve()
            ),
        }
        manifest.write_text(
            json.dumps(manifest_payload, indent=2), encoding="utf-8"
        )
    analysis_context_path = (
        config.output_dir / "analysis_data" / "run_context.json"
    )
    if analysis_context_path.is_file():
        try:
            analysis_context = json.loads(
                analysis_context_path.read_text(encoding="utf-8")
            )
            analysis_context["summary"] = summary
            analysis_context.setdefault("artifacts", {}).update(
                notebook=notebook_path,
                html_report=html_report_path,
                markdown_explanation=markdown_path,
                manifest=str(manifest.resolve()),
            )
            analysis_context_path.write_text(
                json.dumps(analysis_context, indent=2), encoding="utf-8"
            )
        except (OSError, ValueError) as exc:
            LOGGER.warning("Could not finalize notebook context: %s", exc)
    try:
        from notebook_generator import create_notebook_bundle

        create_notebook_bundle(
            config.output_dir,
            notebook_path=Path(notebook_path),
            plot_paths=plot_paths,
        )
    except (OSError, ValueError) as exc:
        LOGGER.warning("Could not package the notebook bundle: %s", exc)
    metric_label = (
        "ACCURACY" if bundle.problem_type == "classification" else "R2"
    )
    metric_rows = [
        (f"FITTED TRAIN {metric_label}", f"{training_metric:.4f}", "nexus.value"),
        (
            f"CROSS-VALIDATED {metric_label}",
            "N/A" if np.isnan(validation_metric) else f"{validation_metric:.4f}",
            "nexus.green",
        ),
        (f"HELD-OUT TEST {metric_label}", f"{testing_metric:.4f}", "nexus.cyan"),
    ]
    if not np.isnan(validation_metric):
        fit_gap = training_metric - validation_metric
        test_gap = validation_metric - testing_metric
        metric_rows.extend(
            [
                (
                    "FIT / CV GAP",
                    f"{fit_gap:+.4f}",
                    "nexus.green" if abs(fit_gap) <= 0.03 else "nexus.amber",
                ),
                (
                    "CV / TEST GAP",
                    f"{test_gap:+.4f}",
                    "nexus.green" if abs(test_gap) <= 0.03 else "nexus.amber",
                ),
            ]
        )
    if generalization_gate_metric is not None:
        metric_rows.append(
            (
                "GENERALIZATION GATE",
                f"{generalization_gate_metric:.4f}",
                "nexus.blue",
            )
        )
    if oob_score is not None:
        metric_rows.append(("OUT-OF-BAG", f"{oob_score:.4f}", "nexus.blue"))

    timing_rows = [
        ("INPUT PREPARATION", f"{input_seconds:.1f}s", "nexus.value"),
    ]
    backbone_search = bundle.metadata.get("backbone_search", {})
    if backbone_search.get("elapsed_seconds") is not None:
        timing_rows.append(
            (
                "BACKBONE SEARCH",
                f"{backbone_search['elapsed_seconds']:.1f}s",
                "nexus.value",
            )
        )
    if bundle.metadata.get("lora_training_seconds") is not None:
        timing_rows.append(
            (
                "LORA TRAINING",
                f"{bundle.metadata['lora_training_seconds']:.1f}s",
                "nexus.value",
            )
        )
    if bundle.metadata.get("embedding_and_gate_seconds") is not None:
        timing_rows.append(
            (
                "EMBEDDING + GATE",
                f"{bundle.metadata['embedding_and_gate_seconds']:.1f}s",
                "nexus.value",
            )
        )
    timing_rows.extend(
        [
            ("DOWNSTREAM AUTOML", f"{training_seconds:.1f}s", "nexus.value"),
            ("PLOTS / HTML", f"{reporting_seconds:.1f}s", "nexus.value"),
            ("LLM EXPLANATION", f"{llm_seconds:.1f}s", "nexus.value"),
            ("ANALYSIS NOTEBOOK", f"{notebook_seconds:.1f}s", "nexus.value"),
        ]
    )

    artifacts = [
        ("MODEL", str(config.output_dir / "model.pkl")),
        ("MANIFEST", str(manifest)),
        ("ANALYTICS", str(notebook_path)),
        ("REPORT", str(markdown_path)),
        ("SEARCH", str(config.output_dir / "search_profile.json")),
    ]
    if html_report_path:
        artifacts.append(("HTML", str(html_report_path)))
    completion_dashboard = {
        "best_model": best_name,
        "representation": bundle.metadata.get("selected_representation"),
        "metric_rows": metric_rows,
        "timing_rows": timing_rows,
        "resource_rows": [
            (
                "RAM CURRENT / PEAK",
                f"{_format_usage(usage['ram_current_mb'])} / "
                f"{_format_usage(usage['ram_peak_mb'])}",
                "nexus.amber",
            ),
            (
                "VRAM CURRENT / PEAK",
                f"{_format_usage(usage['vram_current_mb'])} / "
                f"{_format_usage(usage['vram_peak_mb'])}",
                "nexus.blue",
            ),
        ],
        "artifacts": artifacts,
        "elapsed_seconds": elapsed,
        "results": results,
        "problem_type": bundle.problem_type,
    }
    if render_completion:
        render_run_completion(completion_dashboard)
    return {
        "best_model": best_name,
        "problem_type": bundle.problem_type,
        "results": results,
        "run_summary": summary,
        "output_dir": config.output_dir,
        "artifacts": {
            "run_json": manifest,
            "model_pkl": config.output_dir / "model.pkl",
            "analysis_ipynb": Path(notebook_path),
            "explanation_md": Path(markdown_path),
            "search_profile_json": (
                config.output_dir / "search_profile.json"
            ),
        },
        "_completion_dashboard": completion_dashboard,
    }


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(message)s",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                show_path=False,
                markup=False,
            )
        ],
    )
    try:
        render_banner()
        config = _config_from_args(args)
        render_launch(config)
        run(config)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted")
        return 130
    except Exception:
        LOGGER.exception("Pipeline failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
