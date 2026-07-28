"""Production command-line entry point for ML-Builder.

Examples:
    python main.py data.csv --target outcome
    python main.py --target outcome
    python main.py data.xlsx --target price --problem-type regression --tune
    ml-builder data.csv --target label --models logistic,rf,gb --report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data_cleaner import clean
from data_loader import DataBundle, load_dataset
from feature_processing import build_preprocessor
from model_selector import (
    evaluate_models,
    save_metrics,
    save_model,
    select_best,
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
        prog="ml-builder",
        description="Train and evaluate tabular ML models from one command.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        help="CSV or Excel dataset path; prompted for when omitted",
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
    parser.add_argument("--sample", type=_fraction, default=0.3)
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
        help="Generate a standalone analysis notebook",
    )
    parser.add_argument(
        "--adapt-lora",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fine-tune LoRA for image data with explicit train/test folders",
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
            entered_path = input("Dataset path (CSV, Excel, or image folder): ").strip()
        except EOFError as exc:
            raise ValueError(
                "Dataset path is required when input is non-interactive."
            ) from exc
        if not entered_path:
            raise ValueError("Dataset path cannot be empty.")
        dataset = Path(entered_path.strip('"').strip("'"))

    dataset = dataset.expanduser().resolve()
    target = args.target
    if dataset.is_file() and target is None:
        try:
            target = input("Target column: ").strip()
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
    return RunConfig(
        dataset=dataset,
        target=target,
        output_dir=args.output_dir.expanduser().resolve(),
        problem_type=args.problem_type,
        models=models,
        test_size=args.test_size,
        sample_fraction=args.sample,
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
    )


def _display_results(results: pd.DataFrame, problem_type: str) -> None:
    preferred = (
        ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
        if problem_type == "classification"
        else ["model", "rmse", "mae", "r2"]
    )
    columns = [column for column in preferred if column in results.columns]
    print("\nModel results")
    print(results[columns].round(4).to_string(index=False))


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
    payload.update(
        problem_type=problem_type,
        best_model=best_model,
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


def _load_image_dataset(config: RunConfig) -> tuple[DataBundle, pd.DataFrame]:
    """Embed a class-folder image dataset and return a standard data bundle."""
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
    except ImportError as exc:
        raise RuntimeError(
            "Image training dependencies are missing. Run: uv sync --extra images"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Input] Detected image dataset ({image_count} files); device={device}.")
    train_dir = config.dataset / "train"
    val_dir = config.dataset / "val"
    test_dir = config.dataset / "test"
    explicit_split = train_dir.is_dir() and test_dir.is_dir()
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
        lora_files, lora_labels = train_files, train_labels
        print(
            f"[Split] Using folders: {len(development_files)} development, "
            f"{len(test_files)} test images."
        )
    else:
        all_files, all_labels = discover_labeled_files(
            str(config.dataset), "vision"
        )
        if len(set(all_labels)) < 2:
            raise ValueError(
                "Image classification requires at least two class folders."
            )
        indices = np.arange(len(all_files))
        try:
            development_idx, test_idx = train_test_split(
                indices,
                test_size=config.test_size,
                random_state=config.random_state,
                stratify=all_labels,
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
        lora_files, lora_labels = development_files, development_labels
        print(
            f"[Split] Automatically reserved {len(test_files)} untouched "
            f"test images; {len(development_files)} development images."
        )

    adapter_path = None
    if config.adapt_lora:
        from lora_adapter_trainer import train_universal_lora

        adapter_path = config.output_dir / "lora_adapter"
        train_universal_lora(
            modality="vision",
            domain="general",
            data_dir=str(config.dataset),
            output_path=str(adapter_path),
            files=lora_files,
            labels=lora_labels,
        )

    embedder = UniversalEmbedder(
        device=device,
        batch_size=32,
        domain="general",
        modality="vision",
        adapter_path=str(adapter_path) if adapter_path else None,
    )
    X, labels = embedder.embed_files(
        development_files,
        development_labels,
        "vision",
        cache_key=f"{config.dataset}:development:{config.random_state}",
    )
    X_test, test_labels = embedder.embed_files(
        test_files,
        test_labels,
        "vision",
        cache_key=f"{config.dataset}:test:{config.random_state}",
    )
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


def run(config: RunConfig) -> dict[str, Any]:
    started = time.monotonic()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    bundle, raw_df = _load_input(config)
    X_train, y_train = clean(bundle.X_train, bundle.y_train)
    X_test, y_test = clean(bundle.X_test, bundle.y_test, verbose=False)

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
    promising, baseline_scores = baseline_screen(
        models,
        baseline_preprocessor,
        X_train,
        y_train,
        bundle.problem_type,
        sample_frac=config.sample_fraction,
        cv=cv,
        random_state=config.random_state,
        max_time_seconds=config.max_time_seconds,
    )
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

    fe_log: list[str] = []
    scaler_map = None
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
    trained, validation_scores = full_train(
        promising,
        preprocessor,
        X_train_final,
        y_train,
        bundle.problem_type,
        cv=cv,
        max_time_seconds=config.max_time_seconds,
    )
    if not trained:
        raise RuntimeError(
            "No model completed training; increase --max-time or choose fewer models"
        )

    results = evaluate_models(trained, X_test_final, y_test, bundle.problem_type)
    if config.tune:
        tuned = tune_top_models(
            trained,
            X_train_final,
            y_train,
            bundle.problem_type,
            results,
            top_n=min(2, len(trained)),
            method=config.tune_method,
            n_iter=config.tune_iterations,
            cv=cv,
        )
        trained.update(tuned)
        results = evaluate_models(
            trained, X_test_final, y_test, bundle.problem_type
        )

    generalization = None
    if bundle.problem_type == "classification":
        from generalization import select_generalized_classifier

        generalization = select_generalized_classifier(
            trained,
            validation_scores,
            X_train_final,
            y_train,
            random_state=config.random_state,
        )
        trained[generalization.name] = generalization.model
        validation_scores[generalization.name] = (
            generalization.validation_accuracy
        )
        best_name = generalization.name
        results = evaluate_models(
            trained, X_test_final, y_test, bundle.problem_type
        )
        print(
            f"[Generalization] Selected {best_name}; "
            f"T={generalization.temperature:.3f}, "
            f"NLL {generalization.nll_before:.4f} -> "
            f"{generalization.nll_after:.4f}."
        )
    else:
        best_name = max(
            trained, key=lambda name: validation_scores.get(name, -np.inf)
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

    save_model(trained[best_name], str(config.output_dir / "best_model.joblib"))
    save_metrics(results, str(config.output_dir / "metrics.csv"))

    plot_paths: list[str] = []
    html_report_path: str | None = None
    if config.report:
        try:
            from eda import run_eda
            from explainer import run_explanations
            from report_generator import generate_report

            report_dir = config.output_dir / "report"
            eda_result = run_eda(
                raw_df,
                bundle.target_name,
                bundle.problem_type,
                output_dir=str(report_dir / "eda"),
            )
            explanations = run_explanations(
                trained[best_name],
                X_test_final,
                y_test,
                output_dir=str(report_dir / "explanations"),
                use_shap=config.shap,
            )
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
            plot_paths = [
                path
                for path in [
                    eda_result.get("target_dist_path"),
                    eda_result.get("feature_dist_path"),
                    eda_result.get("corr_heatmap_path"),
                    *explanations.values(),
                ]
                if path
            ]
        except Exception as exc:
            LOGGER.warning("Plot/HTML report generation failed: %s", exc)

    elapsed = time.monotonic() - started
    usage = _memory_usage()
    summary = {
        training_metric_name: round(training_metric, 6),
        validation_metric_name: (
            None if np.isnan(validation_metric) else round(validation_metric, 6)
        ),
        testing_metric_name: round(testing_metric, 6),
        "training_seconds": round(training_seconds, 3),
        "total_pipeline_seconds": round(elapsed, 3),
        **{
            key: None if value is None else round(value, 3)
            for key, value in usage.items()
        },
    }
    if generalization is not None:
        summary["temperature"] = round(generalization.temperature, 6)
        summary["calibration_nll_before"] = round(
            generalization.nll_before, 6
        )
        summary["calibration_nll_after"] = round(
            generalization.nll_after, 6
        )
        summary["ensemble_used"] = generalization.ensemble_used
        summary["ensemble_members"] = generalization.members
    report_dir = config.output_dir / "report"
    markdown_path: str | None = None
    if config.llm:
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
            use_llm=True,
        )

    notebook_path: str | None = None
    if config.notebook:
        try:
            from notebook_generator import generate_advanced_notebook

            notebook_path = generate_advanced_notebook(
                {
                    "data_path": str(config.dataset),
                    "modality": (
                        "vision" if config.dataset.is_dir() else "tabular"
                    ),
                },
                {
                    "best_model": best_name,
                    "summary": summary,
                    "metrics_path": str(config.output_dir / "metrics.csv"),
                    "model_path": str(config.output_dir / "best_model.joblib"),
                    "plot_paths": plot_paths,
                },
                str(config.output_dir / "analysis.ipynb"),
            )
        except Exception as exc:
            LOGGER.warning("Notebook generation failed: %s", exc)

    elapsed = time.monotonic() - started
    summary["total_pipeline_seconds"] = round(elapsed, 3)
    manifest = _write_manifest(
        config, bundle.problem_type, best_name, elapsed, summary
    )
    _display_results(results, bundle.problem_type)
    print(f"\nBest model: {best_name}")
    print("\nRun summary")
    print(f"Training {training_metric_name.split('_', 1)[1]}:   {training_metric:.4f}")
    validation_display = (
        "N/A" if np.isnan(validation_metric) else f"{validation_metric:.4f}"
    )
    print(f"Validation {validation_metric_name.split('_', 1)[1]}: {validation_display}")
    print(f"Testing {testing_metric_name.split('_', 1)[1]}:    {testing_metric:.4f}")
    print(f"Total training time: {training_seconds:.1f}s")
    print(
        f"RAM usage: {_format_usage(usage['ram_current_mb'])} current, "
        f"{_format_usage(usage['ram_peak_mb'])} peak"
    )
    print(
        f"VRAM usage: {_format_usage(usage['vram_current_mb'])} current, "
        f"{_format_usage(usage['vram_peak_mb'])} peak"
    )
    print(f"Artifacts: {config.output_dir}")
    if html_report_path:
        print(f"HTML report: {html_report_path}")
    if markdown_path:
        print(f"Markdown explanation: {markdown_path}")
    if notebook_path:
        print(f"Notebook: {notebook_path}")
    print(f"Manifest: {manifest}")
    print(f"Completed in {elapsed:.1f}s")
    return {
        "best_model": best_name,
        "problem_type": bundle.problem_type,
        "results": results,
        "run_summary": summary,
        "output_dir": config.output_dir,
    }


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        run(_config_from_args(args))
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
