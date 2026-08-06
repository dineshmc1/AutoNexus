"""Generate a reproducible, pre-training-first ML-Builder analysis notebook."""

from __future__ import annotations

import json
import textwrap
import zipfile
from pathlib import Path


def generate_advanced_notebook(
    config: dict,
    results: dict,
    output_path: str,
) -> str:
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    analytics = results.get("analytics_paths", {})
    original_analysis_dir = Path(
        analytics.get("analysis_dir", output.parent / "analysis_data")
    ).resolve()
    run_root = output.parent.resolve()

    def portable_path(value: str) -> str:
        resolved = Path(value).resolve()
        try:
            return str(resolved.relative_to(run_root))
        except ValueError:
            return str(resolved)

    metrics_path = portable_path(results["metrics_path"])
    model_path = portable_path(results["model_path"])
    plot_paths = [
        portable_path(path)
        for path in results.get("plot_paths", [])
        if path
    ]

    def md(source: str):
        return new_markdown_cell(textwrap.dedent(source).strip())

    def code(source: str):
        return new_code_cell(textwrap.dedent(source).strip())

    cells = [
        md(
            f"""
            # ML-Builder: Complete Data and Model Investigation

            **Dataset:** `{config.get("data_path", "unknown")}`<br>
            **Modality:** {config.get("modality", "tabular")}<br>
            **Problem type:** {config.get("problem_type", "unknown")}<br>
            **Selected classifier:** {results.get("best_model", "unknown")}

            The notebook intentionally starts with **pre-training analysis**.
            Model metrics and test-set diagnostics appear only after the
            `Post-Training Analysis` boundary. The held-out test set must not
            be used to make further model-selection decisions.
            """
        ),
        code(
            f"""
            import json
            import math
            import warnings
            from pathlib import Path

            import joblib
            import matplotlib
            try:
                get_ipython
            except NameError:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            try:
                from IPython.display import (
                    Image as DisplayImage, Markdown, display
                )
            except ImportError:
                class Markdown(str):
                    pass

                class DisplayImage:
                    def __init__(self, filename):
                        self.filename = filename

                    def __repr__(self):
                        return f"Image({{self.filename!r}})"

                def display(value):
                    print(value)

            warnings.filterwarnings("ignore", category=FutureWarning)
            sns.set_theme(style="whitegrid", context="notebook")
            plt.rcParams["figure.dpi"] = 120
            pd.set_option("display.max_columns", 120)
            pd.set_option("display.max_rows", 200)

            BUNDLE_ROOT = Path.cwd()
            ORIGINAL_ANALYSIS_DIR = Path({json.dumps(str(original_analysis_dir))})
            analysis_candidates = [
                BUNDLE_ROOT / "analysis_data",
                ORIGINAL_ANALYSIS_DIR,
            ]
            ANALYSIS_DIR = next(
                (
                    path for path in analysis_candidates
                    if (path / "run_context.json").is_file()
                ),
                None,
            )
            if ANALYSIS_DIR is None:
                raise FileNotFoundError(
                    "analysis_data is missing. Download and extract the "
                    "Analytics Bundle, then open analysis.ipynb from the "
                    "extracted directory."
                )
            CONTEXT_PATH = ANALYSIS_DIR / "run_context.json"
            context = json.loads(CONTEXT_PATH.read_text(encoding="utf-8"))
            summary = context["summary"]
            input_metadata = summary.get("input_metadata", {{}})
            prediction_index = pd.read_csv(ANALYSIS_DIR / "prediction_index.csv")
            leaderboard = pd.read_csv(ANALYSIS_DIR / "model_leaderboard.csv")
            probabilities = np.load(
                ANALYSIS_DIR / "test_probabilities.npz", allow_pickle=False
            )
            data_index_path = ANALYSIS_DIR / "data_index.csv"
            data_index = (
                pd.read_csv(data_index_path)
                if data_index_path.exists()
                else pd.DataFrame()
            )
            print(f"Loaded analytics bundle: {{ANALYSIS_DIR}}")
            """
        ),
        md(
            """
            # Part I: Pre-Training Analysis

            Everything in this part describes the dataset, split policy, image
            quality, and selected representation inputs. These checks should be
            reviewed before interpreting model performance.
            """
        ),
        md("## 1. Executive Summary and Major Warnings"),
        code(
            """
            warnings_found = []
            fit_gap = summary.get("fit_validation_gap")
            if fit_gap is not None and fit_gap > 0.05:
                warnings_found.append(
                    f"Fit-validation gap is {fit_gap:.3f}; overfitting risk is elevated."
                )
            if not data_index.empty:
                unreadable = int((~data_index["readable"].astype(bool)).sum())
                if unreadable:
                    warnings_found.append(f"{unreadable} unreadable files were excluded.")
                duplicate_rows = data_index[data_index["exact_duplicate"].astype(bool)]
                if not duplicate_rows.empty:
                    cross_split = (
                        duplicate_rows[duplicate_rows["sha256"].ne("")]
                        .groupby("sha256")["split"].nunique()
                        .gt(1).sum()
                    )
                    if cross_split:
                        warnings_found.append(
                            f"{cross_split} exact-duplicate groups cross split boundaries."
                        )
                counts = data_index[data_index["readable"].astype(bool)]["label"].value_counts()
                if len(counts) and counts.max() / max(counts.min(), 1) >= 5:
                    warnings_found.append(
                        f"Class imbalance ratio is {counts.max() / max(counts.min(), 1):.1f}:1."
                    )
            backbone = input_metadata.get("backbone_key", "not applicable")
            representation = input_metadata.get(
                "selected_representation", "tabular features"
            )
            executive = pd.Series(
                {
                    "Selected backbone": backbone,
                    "Representation": representation,
                    "Classifier": context["best_model"],
                    "Fitted training metric": summary.get("fitted_training_metric"),
                    "Primary CV metric": summary.get("primary_cross_validated_metric"),
                    "Held-out test metric": summary.get("held_out_testing_metric"),
                    "Fit-validation gap": fit_gap,
                    "Validation-test gap": summary.get("validation_test_gap"),
                    "Input preparation (s)": summary.get("input_preparation_seconds"),
                    "Downstream AutoML (s)": summary.get("downstream_automl_seconds"),
                    "Total wall time (s)": summary.get("total_pipeline_seconds"),
                    "Peak RAM (MiB)": summary.get("ram_peak_mb"),
                    "Peak VRAM (MiB)": summary.get("vram_peak_mb"),
                },
                name="value",
            ).to_frame()
            display(executive)
            if warnings_found:
                display(Markdown("### Warnings\\n" + "\\n".join(
                    f"- {item}" for item in warnings_found
                )))
            else:
                display(Markdown("### Warnings\\nNo major automated warning was detected."))
            """
        ),
        md("## 2. Data Quality Audit"),
        code(
            """
            if data_index.empty:
                display(Markdown(
                    "Image-level quality checks are not applicable to this tabular run. "
                    "Use the learning sample below for schema and missing-value inspection."
                ))
                tabular_sample = pd.read_csv(ANALYSIS_DIR / "learning_curve_sample.csv")
                display(pd.DataFrame({
                    "dtype": tabular_sample.dtypes.astype(str),
                    "missing": tabular_sample.isna().sum(),
                    "unique": tabular_sample.nunique(dropna=False),
                }))
            else:
                quality_summary = pd.Series({
                    "Files scanned": len(data_index),
                    "Readable": int(data_index["readable"].astype(bool).sum()),
                    "Unreadable": int((~data_index["readable"].astype(bool)).sum()),
                    "Exact duplicate files": int(data_index["exact_duplicate"].astype(bool).sum()),
                    "Near-duplicate candidates (bounded audit)": int(
                        data_index["near_duplicate_of"].fillna("").ne("").sum()
                    ),
                    "Formats": ", ".join(sorted(data_index["format"].dropna().astype(str).unique())),
                    "Quality-sampled files": int(data_index["quality_sampled"].astype(bool).sum()),
                }, name="value").to_frame()
                display(quality_summary)
                unreadable_rows = data_index[~data_index["readable"].astype(bool)]
                if not unreadable_rows.empty:
                    display(unreadable_rows[["path", "split", "label", "error"]])

                valid = data_index[data_index["readable"].astype(bool)].copy()
                dimension_outlier = (
                    (valid["width"] < valid["width"].quantile(0.01))
                    | (valid["width"] > valid["width"].quantile(0.99))
                    | (valid["height"] < valid["height"].quantile(0.01))
                    | (valid["height"] > valid["height"].quantile(0.99))
                    | (valid["aspect_ratio"] < valid["aspect_ratio"].quantile(0.01))
                    | (valid["aspect_ratio"] > valid["aspect_ratio"].quantile(0.99))
                )
                display(Markdown("**Unusual dimensions (1st/99th percentile rule)**"))
                display(valid.loc[dimension_outlier, [
                    "path", "label", "split", "width", "height", "aspect_ratio"
                ]].head(100))
            """
        ),
        md("## 3. Split, Class, Group, and Leakage Audit"),
        code(
            """
            if not data_index.empty:
                valid = data_index[data_index["readable"].astype(bool)].copy()
                split_class = pd.crosstab(valid["label"], valid["split"])
                display(split_class)
                split_class.plot(
                    kind="bar", stacked=True, figsize=(14, 5),
                    title="Class distribution by train/validation/development-CV/test role"
                )
                plt.ylabel("Images")
                plt.tight_layout()
                plt.show()

                if valid["group"].fillna("").ne("").any():
                    group_audit = valid[valid["group"].fillna("").ne("")].groupby(
                        ["split", "label"]
                    )["group"].nunique().rename("unique_groups").to_frame()
                    display(group_audit)
                    leaking_groups = (
                        valid[valid["group"].fillna("").ne("")]
                        .groupby("group")["split"].nunique()
                    )
                    leaking_groups = leaking_groups[leaking_groups > 1]
                    print(f"Groups crossing split boundaries: {len(leaking_groups)}")
                    display(leaking_groups.head(100))
                else:
                    print("No reliable subject/video/source groups were inferred.")

                exact_leakage = (
                    valid[valid["sha256"].fillna("").ne("")]
                    .groupby("sha256")["split"].nunique()
                )
                exact_leakage = exact_leakage[exact_leakage > 1]
                print(f"Exact-content hashes crossing split boundaries: {len(exact_leakage)}")
                near_cross_split = valid[
                    valid["near_duplicate_of"].fillna("").ne("")
                ][["path", "near_duplicate_of", "split", "label"]]
                display(near_cross_split.head(100))
            else:
                print("Tabular split fingerprints:")
                display(pd.Series(context["split_fingerprints"], name="sha256").to_frame())
            """
        ),
        md("## 4. Representative Images for Every Class"),
        code(
            """
            if data_index.empty:
                print("Representative image grids are not applicable to tabular data.")
            else:
                from PIL import Image

                valid = data_index[data_index["readable"].astype(bool)].copy()
                representatives = (
                    valid.sort_values(["label", "quality_sampled"], ascending=[True, False])
                    .groupby("label", as_index=False).first()
                )
                per_page = 30
                for page_start in range(0, len(representatives), per_page):
                    page = representatives.iloc[page_start:page_start + per_page]
                    columns = 6
                    rows = math.ceil(len(page) / columns)
                    fig, axes = plt.subplots(rows, columns, figsize=(15, 2.6 * rows))
                    axes = np.atleast_1d(axes).ravel()
                    for axis, (_, item) in zip(axes, page.iterrows()):
                        with Image.open(item["path"]) as image:
                            axis.imshow(image.convert("RGB"))
                        axis.set_title(str(item["label"]), fontsize=9)
                        axis.axis("off")
                    for axis in axes[len(page):]:
                        axis.axis("off")
                    fig.suptitle(
                        f"One representative per class: "
                        f"{page_start + 1}-{page_start + len(page)} of {len(representatives)}"
                    )
                    plt.tight_layout()
                    plt.show()

                sampled_quality = valid[valid["quality_sampled"].astype(bool)].copy()
                if not sampled_quality.empty:
                    unusual = pd.concat([
                        sampled_quality.nsmallest(6, "blur_score"),
                        sampled_quality.nsmallest(6, "brightness"),
                        sampled_quality.nlargest(6, "brightness"),
                        sampled_quality.nsmallest(6, "entropy"),
                    ]).drop_duplicates("path")
                    fig, axes = plt.subplots(
                        math.ceil(len(unusual) / 6), 6,
                        figsize=(15, 2.6 * math.ceil(len(unusual) / 6))
                    )
                    axes = np.atleast_1d(axes).ravel()
                    for axis, (_, item) in zip(axes, unusual.iterrows()):
                        with Image.open(item["path"]) as image:
                            axis.imshow(image.convert("RGB"))
                        axis.set_title(
                            f"{item['label']}\\nblur={item['blur_score']:.1f}, "
                            f"light={item['brightness']:.0f}", fontsize=8
                        )
                        axis.axis("off")
                    for axis in axes[len(unusual):]:
                        axis.axis("off")
                    fig.suptitle("Unusual and potentially low-quality examples")
                    plt.tight_layout()
                    plt.show()
            """
        ),
        md("## 5. Image Statistics"),
        code(
            """
            if data_index.empty:
                print("Image statistics are not applicable to tabular data.")
            else:
                valid = data_index[data_index["readable"].astype(bool)].copy()
                sampled = valid[valid["quality_sampled"].astype(bool)].copy()
                fig, axes = plt.subplots(2, 3, figsize=(16, 9))
                sns.histplot(valid["width"], ax=axes[0, 0], bins=40)
                axes[0, 0].set_title("Width")
                sns.histplot(valid["height"], ax=axes[0, 1], bins=40)
                axes[0, 1].set_title("Height")
                sns.histplot(valid["aspect_ratio"], ax=axes[0, 2], bins=40)
                axes[0, 2].set_title("Aspect ratio")
                sns.histplot(sampled["brightness"], ax=axes[1, 0], bins=40)
                axes[1, 0].set_title("Brightness")
                sns.histplot(sampled["contrast"], ax=axes[1, 1], bins=40)
                axes[1, 1].set_title("Contrast")
                sns.histplot(sampled["entropy"], ax=axes[1, 2], bins=40)
                axes[1, 2].set_title("Entropy")
                plt.tight_layout()
                plt.show()

                fig, axes = plt.subplots(1, 2, figsize=(13, 4))
                sns.histplot(sampled["blur_score"], bins=40, ax=axes[0])
                axes[0].set_title("Gradient sharpness / blur score")
                sampled[["red_mean", "green_mean", "blue_mean"]].plot(
                    kind="density", ax=axes[1]
                )
                axes[1].set_title("Thumbnail channel-mean distributions")
                plt.tight_layout()
                plt.show()

                display(pd.crosstab(valid["format"], valid["split"]))
                display(valid.groupby("label")[
                    ["width", "height", "aspect_ratio"]
                ].agg(["count", "median", "min", "max"]))
            """
        ),
        md("## 6. Embedding Geometry: PCA and UMAP"),
        code(
            """
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import LabelEncoder

            embedding = np.load(
                ANALYSIS_DIR / "embedding_sample.npz", allow_pickle=False
            )
            embedding_frame = pd.DataFrame(
                embedding["X"].astype(np.float32)
            ).replace([np.inf, -np.inf], np.nan)
            non_finite_values = int(embedding_frame.isna().sum().sum())
            embedding_X = (
                embedding_frame
                .fillna(embedding_frame.median(numeric_only=True))
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
            )
            if non_finite_values:
                display(Markdown(
                    f"**Projection preprocessing:** median-imputed "
                    f"{non_finite_values:,} non-finite embedding values."
                ))
            embedding_y = embedding["y"]
            embedding_split = embedding["split"].astype(str)
            embedding_group = embedding["group"].astype(str)
            embedding_row_id = embedding["row_id"].astype(str)
            pca = PCA(n_components=2, random_state=context["config"]["random_state"])
            pca_xy = pca.fit_transform(embedding_X)
            correctness_map = prediction_index.set_index(
                prediction_index["row_id"].astype(str)
            )["correct"].to_dict()
            embedding_correctness = np.asarray([
                correctness_map.get(row_id, np.nan)
                if split == "test" else np.nan
                for row_id, split in zip(embedding_row_id, embedding_split)
            ])

            def scatter_categories(axis, points, values, title):
                encoded = LabelEncoder().fit_transform(pd.Series(values).astype(str))
                axis.scatter(points[:, 0], points[:, 1], c=encoded, s=10, alpha=.65, cmap="turbo")
                axis.set_title(title)
                axis.set_xticks([])
                axis.set_yticks([])

            fig, axes = plt.subplots(2, 2, figsize=(14, 11))
            scatter_categories(axes[0, 0], pca_xy, embedding_y, "PCA by class")
            scatter_categories(axes[0, 1], pca_xy, embedding_split, "PCA by split")
            group_values = np.where(embedding_group == "", "no-group", embedding_group)
            scatter_categories(axes[1, 0], pca_xy, group_values, "PCA by group")
            scatter_categories(
                axes[1, 1], pca_xy,
                np.where(pd.isna(embedding_correctness), "development", embedding_correctness),
                "PCA by test correctness"
            )
            fig.suptitle(
                f"Embedding sample: {len(embedding_X):,} of "
                f"{int(embedding['source_count']):,} rows; "
                f"PCA variance={pca.explained_variance_ratio_.sum():.1%}"
            )
            plt.tight_layout()
            plt.show()

            try:
                import umap

                umap_xy = umap.UMAP(
                    n_neighbors=15, min_dist=0.1, metric="cosine",
                    random_state=context["config"]["random_state"],
                ).fit_transform(embedding_X)
                fig, axes = plt.subplots(2, 2, figsize=(14, 11))
                scatter_categories(axes[0, 0], umap_xy, embedding_y, "UMAP by class")
                scatter_categories(
                    axes[0, 1], umap_xy, embedding_split, "UMAP by split"
                )
                scatter_categories(
                    axes[1, 0], umap_xy, group_values, "UMAP by group"
                )
                scatter_categories(
                    axes[1, 1], umap_xy,
                    np.where(pd.isna(embedding_correctness), "development", embedding_correctness),
                    "UMAP by correctness"
                )
                plt.tight_layout()
                plt.show()
            except ImportError:
                display(Markdown(
                    "**UMAP skipped:** install `umap-learn` in the notebook "
                    "environment to enable this optional projection. PCA remains available."
                ))
            """
        ),
        md("## 7. Class-Separation Diagnostics"),
        code(
            """
            from sklearn.metrics import silhouette_score
            from sklearn.neighbors import NearestNeighbors

            diagnostic_limit = min(2000, len(embedding_X))
            diagnostic_indices = np.linspace(
                0, len(embedding_X) - 1, diagnostic_limit, dtype=int
            )
            diagnostic_X = embedding_X[diagnostic_indices]
            diagnostic_y = embedding_y[diagnostic_indices]
            if len(np.unique(diagnostic_y)) > 1 and len(diagnostic_X) > len(np.unique(diagnostic_y)):
                silhouette = silhouette_score(
                    diagnostic_X, diagnostic_y, metric="cosine"
                )
            else:
                silhouette = np.nan
            neighbors = NearestNeighbors(n_neighbors=2, metric="cosine").fit(diagnostic_X)
            _, neighbor_indices = neighbors.kneighbors(diagnostic_X)
            neighbor_consistency = np.mean(
                diagnostic_y == diagnostic_y[neighbor_indices[:, 1]]
            )
            display(pd.Series({
                "Silhouette score (cosine)": silhouette,
                "Nearest-neighbor class consistency": neighbor_consistency,
                "Diagnostic sample size": diagnostic_limit,
                "Source embedding count": int(embedding["source_count"]),
            }, name="value").to_frame())
            """
        ),
        md(
            """
            # Part II: Post-Training Analysis

            The sections below use model-selection records and the held-out test
            predictions. Test findings are for final auditing and deployment
            decisions, not for repeatedly tuning the same run.
            """
        ),
        md("## 8. Backbone Tournament and Selection Rationale"),
        code(
            """
            search = input_metadata.get("backbone_search", {})
            tournament_rows = []
            for stage in search.get("stages", []):
                survivors = set(stage.get("survivors", []))
                for candidate in stage.get("candidates", []):
                    tournament_rows.append({
                        "stage": stage.get("stage"),
                        "fraction": stage.get("fraction"),
                        "samples": stage.get("samples"),
                        "candidate": candidate.get("key"),
                        "accuracy": candidate.get("mean_accuracy"),
                        "cv_std": candidate.get("std_accuracy"),
                        "nll": candidate.get("mean_nll"),
                        "selection_score": candidate.get("selection_score"),
                        "embedding_seconds": candidate.get("embedding_seconds"),
                        "observed_ram_mb": candidate.get("observed_ram_mb"),
                        "observed_vram_mb": candidate.get("observed_vram_mb"),
                        "outcome": "survived" if candidate.get("key") in survivors else "eliminated",
                        "error": candidate.get("error"),
                    })
            tournament = pd.DataFrame(tournament_rows)
            if tournament.empty:
                print("Backbone tournament is not applicable to this run.")
            else:
                display(tournament)
                display(Markdown(
                    f"**Selected:** `{search.get('selected', {}).get('key')}`. "
                    f"Strategy: `{search.get('strategy')}`. "
                    f"Elapsed: {search.get('elapsed_seconds')} seconds. "
                    "Selection combines CV accuracy, uncertainty, NLL, latency, "
                    "RAM/VRAM pressure, statistical ties, and model size."
                ))
                latest = tournament.sort_values("stage").groupby(
                    "candidate", as_index=False
                ).tail(1)
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                sns.scatterplot(
                    data=latest, x="embedding_seconds", y="accuracy",
                    hue="candidate", size="observed_ram_mb", sizes=(60, 400),
                    ax=axes[0]
                )
                axes[0].set_title("Backbone quality vs embedding latency")
                sns.scatterplot(
                    data=latest, x="nll", y="accuracy",
                    hue="candidate", size="observed_vram_mb", sizes=(60, 400),
                    ax=axes[1], legend=False
                )
                axes[1].set_title("Backbone accuracy vs NLL")
                plt.tight_layout()
                plt.show()
            """
        ),
        md("## 9. Model Leaderboard and Performance-Cost Pareto Frontier"),
        code(
            """
            display(leaderboard.sort_values("cv_mean", ascending=False))
            cost_data = leaderboard.dropna(subset=["cv_mean", "training_seconds"]).copy()
            if not cost_data.empty:
                cost_data = cost_data.sort_values("training_seconds")
                cost_data["pareto_best_so_far"] = cost_data["cv_mean"].cummax()
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                sns.scatterplot(
                    data=cost_data, x="training_seconds", y="cv_mean",
                    hue="model", size="observed_process_ram_mb",
                    sizes=(80, 450), ax=axes[0]
                )
                axes[0].plot(
                    cost_data["training_seconds"],
                    cost_data["pareto_best_so_far"],
                    color="black", linestyle="--", label="Pareto envelope"
                )
                axes[0].set_xscale("log")
                axes[0].set_title("CV quality vs training time")
                sns.scatterplot(
                    data=cost_data, x="observed_process_ram_mb", y="cv_mean",
                    hue="model", size="training_seconds",
                    sizes=(80, 450), ax=axes[1], legend=False
                )
                axes[1].set_title("CV quality vs observed process RAM")
                plt.tight_layout()
                plt.show()
            """
        ),
        md("## 10. Frozen Representation vs LoRA Adapter"),
        code(
            """
            lora = input_metadata.get("lora_gate", {})
            if not lora:
                print("LoRA was not requested for this run.")
            else:
                comparison = pd.DataFrame([
                    {"representation": "frozen", **lora.get("frozen", {})},
                    {"representation": "adapted", **lora.get("adapted", {})},
                ])
                display(comparison)
                display(pd.Series({
                    "Status": lora.get("status", "evaluated"),
                    "Adapter selected": lora.get("adapter_selected"),
                    "Accuracy gain": lora.get("accuracy_gain"),
                    "NLL improvement": lora.get("nll_improvement"),
                    "Decision reason": lora.get(
                        "reason",
                        "Accepted only when the unseen gate satisfied the "
                        "accuracy/NLL generalization thresholds."
                    ),
                }, name="value").to_frame())
                available_metrics = [
                    column for column in ("accuracy", "nll")
                    if column in comparison
                ]
                if available_metrics and not comparison[available_metrics].dropna(
                    how="all"
                ).empty:
                    comparison.set_index("representation")[available_metrics].plot(
                        kind="bar", subplots=True, figsize=(8, 6), legend=False
                    )
                    plt.tight_layout()
                    plt.show()
            """
        ),
        md("## 11. Raw and Normalized Confusion Matrices"),
        code(
            """
            from sklearn.metrics import confusion_matrix

            y_true = probabilities["y_true"]
            y_pred = probabilities["y_pred"]
            classes = probabilities["classes"]
            class_names = probabilities["class_names"].astype(str)
            confusion_raw = confusion_matrix(y_true, y_pred, labels=classes)
            confusion_normalized = confusion_matrix(
                y_true, y_pred, labels=classes, normalize="true"
            )
            figure_size = min(28, max(8, len(classes) * 0.3))
            fig, axes = plt.subplots(1, 2, figsize=(2 * figure_size, figure_size))
            sns.heatmap(
                confusion_raw, cmap="Blues", ax=axes[0],
                xticklabels=class_names, yticklabels=class_names
            )
            axes[0].set_title("Raw counts")
            sns.heatmap(
                confusion_normalized, cmap="mako", vmin=0, vmax=1, ax=axes[1],
                xticklabels=class_names, yticklabels=class_names
            )
            axes[1].set_title("Row-normalized recall")
            for axis in axes:
                axis.set_xlabel("Predicted")
                axis.set_ylabel("Actual")
            plt.tight_layout()
            plt.show()
            """
        ),
        md("## 12. Per-Class Metrics and One-vs-Rest ROC/PR"),
        code(
            """
            from sklearn.metrics import (
                average_precision_score, precision_recall_fscore_support,
                roc_auc_score, roc_curve, precision_recall_curve
            )

            calibrated = probabilities["calibrated"]
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, labels=classes, zero_division=0
            )
            one_hot = (y_true[:, None] == classes[None, :]).astype(int)
            specificity = []
            roc_auc = []
            average_precision = []
            for index, _ in enumerate(classes):
                tp = np.sum((one_hot[:, index] == 1) & (y_pred == classes[index]))
                fn = np.sum((one_hot[:, index] == 1) & (y_pred != classes[index]))
                fp = np.sum((one_hot[:, index] == 0) & (y_pred == classes[index]))
                tn = len(y_true) - tp - fn - fp
                specificity.append(tn / max(tn + fp, 1))
                if len(np.unique(one_hot[:, index])) == 2:
                    roc_auc.append(roc_auc_score(one_hot[:, index], calibrated[:, index]))
                    average_precision.append(
                        average_precision_score(one_hot[:, index], calibrated[:, index])
                    )
                else:
                    roc_auc.append(np.nan)
                    average_precision.append(np.nan)
            per_class = pd.DataFrame({
                "class": class_names,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "specificity": specificity,
                "ovr_roc_auc": roc_auc,
                "ovr_average_precision": average_precision,
            }).sort_values("f1")
            display(per_class)

            plotted = per_class.head(min(10, len(per_class)))
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for class_name in plotted["class"]:
                index = int(np.flatnonzero(class_names == class_name)[0])
                if len(np.unique(one_hot[:, index])) < 2:
                    continue
                fpr, tpr, _ = roc_curve(one_hot[:, index], calibrated[:, index])
                pr_recall, pr_precision, _ = precision_recall_curve(
                    one_hot[:, index], calibrated[:, index]
                )
                axes[0].plot(fpr, tpr, label=class_name)
                axes[1].plot(pr_recall, pr_precision, label=class_name)
            axes[0].plot([0, 1], [0, 1], "k--")
            axes[0].set_title("One-vs-rest ROC: lowest-F1 classes")
            axes[1].set_title("One-vs-rest precision-recall: lowest-F1 classes")
            for axis in axes:
                axis.legend(fontsize=7)
            plt.tight_layout()
            plt.show()
            """
        ),
        md("## 13. Most-Confused Class Pairs with Examples"),
        code(
            """
            pair_rows = []
            for true_index, true_name in enumerate(class_names):
                for predicted_index, predicted_name in enumerate(class_names):
                    if true_index != predicted_index and confusion_raw[true_index, predicted_index]:
                        pair_rows.append({
                            "actual": true_name,
                            "predicted": predicted_name,
                            "count": int(confusion_raw[true_index, predicted_index]),
                        })
            confused_pairs = pd.DataFrame(pair_rows).sort_values(
                "count", ascending=False
            ).head(12) if pair_rows else pd.DataFrame()
            display(confused_pairs)

            if not data_index.empty and not confused_pairs.empty:
                from PIL import Image

                examples = []
                for _, pair in confused_pairs.head(6).iterrows():
                    matches = prediction_index[
                        (prediction_index["true_label"] == pair["actual"])
                        & (prediction_index["predicted_label"] == pair["predicted"])
                    ].head(2)
                    examples.extend(matches.to_dict("records"))
                if examples:
                    fig, axes = plt.subplots(
                        math.ceil(len(examples) / 4), 4,
                        figsize=(13, 3 * math.ceil(len(examples) / 4))
                    )
                    axes = np.atleast_1d(axes).ravel()
                    for axis, item in zip(axes, examples):
                        with Image.open(item["row_id"]) as image:
                            axis.imshow(image.convert("RGB"))
                        axis.set_title(
                            f"Actual: {item['true_label']}\\n"
                            f"Predicted: {item['predicted_label']} "
                            f"({item['confidence']:.2f})", fontsize=8
                        )
                        axis.axis("off")
                    for axis in axes[len(examples):]:
                        axis.axis("off")
                    plt.tight_layout()
                    plt.show()
            """
        ),
        md("## 14. Error Gallery"),
        code(
            """
            errors = prediction_index[~prediction_index["correct"].astype(bool)].copy()
            gallery_groups = {
                "High-confidence mistakes": errors.nlargest(8, "confidence"),
                "Low-confidence mistakes": errors.nsmallest(8, "confidence"),
                "Ambiguous samples": prediction_index.iloc[
                    (prediction_index["confidence"] - 1 / max(len(classes), 2))
                    .abs().argsort()[:8]
                ],
                "Possible label-error candidates": errors[
                    errors["confidence"] >= max(0.9, errors["confidence"].quantile(0.9)
                    if len(errors) else 1.0)
                ].head(8),
            }
            if data_index.empty:
                for title, frame in gallery_groups.items():
                    display(Markdown(f"### {title}"))
                    display(frame)
            else:
                from PIL import Image

                for title, frame in gallery_groups.items():
                    display(Markdown(f"### {title}"))
                    if frame.empty:
                        print("No examples.")
                        continue
                    fig, axes = plt.subplots(
                        math.ceil(len(frame) / 4), 4,
                        figsize=(13, 3 * math.ceil(len(frame) / 4))
                    )
                    axes = np.atleast_1d(axes).ravel()
                    for axis, (_, item) in zip(axes, frame.iterrows()):
                        with Image.open(item["row_id"]) as image:
                            axis.imshow(image.convert("RGB"))
                        axis.set_title(
                            f"True: {item['true_label']}\\n"
                            f"Pred: {item['predicted_label']} "
                            f"p={item['confidence']:.2f}", fontsize=8
                        )
                        axis.axis("off")
                    for axis in axes[len(frame):]:
                        axis.axis("off")
                    plt.tight_layout()
                    plt.show()
            display(Markdown(
                "*Possible label errors are review candidates, not automatic relabeling decisions.*"
            ))
            """
        ),
        md("## 15. Calibration Before and After Temperature Scaling"),
        code(
            """
            from sklearn.metrics import log_loss

            uncalibrated = probabilities["uncalibrated"]
            calibrated = probabilities["calibrated"]

            def calibration_metrics(probability_matrix, bins=15):
                confidence = probability_matrix.max(axis=1)
                predicted = classes[np.argmax(probability_matrix, axis=1)]
                correct = (predicted == y_true).astype(float)
                boundaries = np.linspace(0, 1, bins + 1)
                ece = 0.0
                rows = []
                for lower, upper in zip(boundaries[:-1], boundaries[1:]):
                    mask = (confidence >= lower) & (
                        confidence <= upper if upper == 1 else confidence < upper
                    )
                    if mask.any():
                        bin_accuracy = correct[mask].mean()
                        bin_confidence = confidence[mask].mean()
                        ece += mask.mean() * abs(bin_accuracy - bin_confidence)
                        rows.append((bin_confidence, bin_accuracy, int(mask.sum())))
                brier = np.mean(np.sum((probability_matrix - one_hot) ** 2, axis=1))
                nll = log_loss(y_true, probability_matrix, labels=classes)
                return {"ece": ece, "brier": brier, "nll": nll}, pd.DataFrame(
                    rows, columns=["confidence", "accuracy", "count"]
                )

            raw_metrics, raw_curve = calibration_metrics(uncalibrated)
            calibrated_metrics, calibrated_curve = calibration_metrics(calibrated)
            display(pd.DataFrame([
                {"state": "before temperature scaling", **raw_metrics},
                {"state": "after temperature scaling", **calibrated_metrics},
            ]))
            plt.figure(figsize=(7, 6))
            plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
            plt.plot(raw_curve["confidence"], raw_curve["accuracy"], "o-", label="Before")
            plt.plot(
                calibrated_curve["confidence"], calibrated_curve["accuracy"],
                "o-", label=f"After (T={summary.get('temperature', 1.0):.3f})"
            )
            plt.xlabel("Mean confidence")
            plt.ylabel("Observed accuracy")
            plt.title("Reliability diagram")
            plt.legend()
            plt.tight_layout()
            plt.show()
            """
        ),
        md("## 16. Confidence and Uncertainty Distributions"),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            sns.histplot(
                data=prediction_index, x="confidence", hue="correct",
                bins=30, stat="density", common_norm=False, ax=axes[0]
            )
            axes[0].set_title("Confidence: correct vs incorrect")
            sns.histplot(
                data=prediction_index, x="uncertainty_entropy", hue="correct",
                bins=30, stat="density", common_norm=False, ax=axes[1]
            )
            axes[1].set_title("Predictive entropy: correct vs incorrect")
            plt.tight_layout()
            plt.show()
            display(prediction_index.groupby("correct")[
                ["confidence", "uncertainty_entropy"]
            ].agg(["count", "mean", "median", "std"]))
            """
        ),
        md("## 17. Learning Curves and Data Sufficiency"),
        code(
            """
            from sklearn.linear_model import LogisticRegression, Ridge
            from sklearn.model_selection import learning_curve

            learning_X = embedding_X
            learning_y = embedding_y
            if context["problem_type"] == "classification":
                min_class = pd.Series(learning_y).value_counts().min()
                folds = min(3, int(min_class))
                estimator = LogisticRegression(
                    C=1.0, max_iter=2000, class_weight="balanced",
                    random_state=context["config"]["random_state"]
                )
                scoring = "accuracy"
            else:
                folds = 3
                estimator = Ridge(alpha=1.0)
                scoring = "r2"

            if folds >= 2:
                sizes, train_scores, validation_scores_curve = learning_curve(
                    estimator, learning_X, learning_y,
                    train_sizes=np.array([0.15, 0.35, 0.6, 1.0]),
                    cv=folds, scoring=scoring, n_jobs=1,
                    shuffle=True, random_state=context["config"]["random_state"]
                )
                plt.figure(figsize=(8, 5))
                plt.plot(sizes, train_scores.mean(axis=1), "o-", label="Train")
                plt.fill_between(
                    sizes,
                    train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1),
                    alpha=.15
                )
                plt.plot(
                    sizes, validation_scores_curve.mean(axis=1),
                    "o-", label="Cross-validation"
                )
                plt.fill_between(
                    sizes,
                    validation_scores_curve.mean(axis=1)
                    - validation_scores_curve.std(axis=1),
                    validation_scores_curve.mean(axis=1)
                    + validation_scores_curve.std(axis=1),
                    alpha=.15
                )
                plt.xlabel("Training samples")
                plt.ylabel(scoring)
                plt.title("Representation data-sufficiency curve (regularized probe)")
                plt.legend()
                plt.tight_layout()
                plt.show()
            else:
                print("Learning curve skipped: too few samples per class.")
            display(Markdown(
                "This bounded curve uses the saved representation sample and a "
                "regularized probe. It estimates whether more data will help "
                "without retraining the expensive backbone or full AutoML search."
            ))
            """
        ),
        md("## 18. Group-Level Generalization"),
        code(
            """
            grouped_predictions = prediction_index[
                prediction_index["group"].fillna("").ne("")
            ].copy()
            if grouped_predictions.empty:
                print("No reliable subject/video/source groups are available.")
            else:
                group_metrics = grouped_predictions.groupby("group").agg(
                    samples=("correct", "size"),
                    accuracy=("correct", "mean"),
                    mean_confidence=("confidence", "mean"),
                    mean_uncertainty=("uncertainty_entropy", "mean"),
                ).sort_values(["accuracy", "samples"], ascending=[True, False])
                display(group_metrics)
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                sns.histplot(group_metrics["accuracy"], bins=20, ax=axes[0])
                axes[0].set_title("Accuracy across groups")
                sns.scatterplot(
                    data=group_metrics.reset_index(), x="samples", y="accuracy",
                    size="mean_uncertainty", ax=axes[1]
                )
                axes[1].set_title("Group accuracy vs support")
                plt.tight_layout()
                plt.show()
            """
        ),
        md("## 19. Explainability: Nearest Embedding Neighbors"),
        code(
            """
            from sklearn.neighbors import NearestNeighbors

            query_index = 0  # Change this to inspect another sampled row.
            neighbor_model = NearestNeighbors(
                n_neighbors=min(6, len(embedding_X)), metric="cosine"
            ).fit(embedding_X)
            distances, indices = neighbor_model.kneighbors(
                embedding_X[[query_index]]
            )
            neighbor_table = pd.DataFrame({
                "sample_index": indices[0],
                "cosine_distance": distances[0],
                "label": embedding_y[indices[0]],
                "split": embedding_split[indices[0]],
                "group": embedding_group[indices[0]],
                "row_id": embedding_row_id[indices[0]],
            })
            display(neighbor_table)

            if not data_index.empty:
                from PIL import Image

                fig, axes = plt.subplots(1, len(neighbor_table), figsize=(3 * len(neighbor_table), 3))
                axes = np.atleast_1d(axes)
                for axis, (_, item) in zip(axes, neighbor_table.iterrows()):
                    with Image.open(item["row_id"]) as image:
                        axis.imshow(image.convert("RGB"))
                    axis.set_title(
                        f"{item['label']}\\nd={item['cosine_distance']:.3f}\\n"
                        f"{item['split']}", fontsize=8
                    )
                    axis.axis("off")
                plt.tight_layout()
                plt.show()

            display(Markdown(
                "**Grad-CAM/attention decision:** the deployed downstream model "
                "consumes persisted embeddings rather than backbone feature maps, "
                "so generic Grad-CAM would be technically misleading. Nearest-neighbor "
                "evidence is faithful to the actual saved representation. Backbone-specific "
                "attention rollout can be added only when intermediate activations are "
                "persisted for a compatible transformer."
            ))
            """
        ),
        md("## 20. Existing Generated Plots"),
        code(
            f"""
            plot_paths = {plot_paths!r}
            for plot in plot_paths:
                path = Path(plot)
                if not path.is_absolute():
                    path = BUNDLE_ROOT / path
                if path.exists():
                    print(path.name)
                    display(DisplayImage(filename=str(path)))
            """
        ),
        md("## 21. Reproducibility Record"),
        code(
            f"""
            display(Markdown("### Configuration"))
            display(pd.Series(context["config"], name="value").to_frame())
            display(Markdown("### Hardware"))
            display(pd.Series(context["hardware"], name="value").to_frame())
            display(Markdown("### Package versions"))
            display(pd.Series(context["package_versions"], name="version").to_frame())
            display(Markdown("### Exact split fingerprints"))
            display(pd.Series(context["split_fingerprints"], name="sha256").to_frame())
            display(Markdown("### Cache state"))
            display(pd.DataFrame(context.get("cache_state", {{}})).T)
            display(Markdown("### Artifact paths"))
            display(pd.Series(context["artifacts"], name="path").to_frame())
            print("Model revision:", input_metadata.get("backbone_revision"))
            print("Embedding cache:", input_metadata.get("embedding_cache", "see .cache/embeddings"))
            print("Saved model:", BUNDLE_ROOT / Path({json.dumps(model_path)}))
            print("Metrics:", BUNDLE_ROOT / Path({json.dumps(metrics_path)}))
            """
        ),
        md("## 22. Model Card"),
        code(
            """
            modality = context["config"].get("dataset", "")
            model_card = f'''
            ### Model
            **Name:** `{context["best_model"]}`<br>
            **Representation:** `{input_metadata.get("selected_representation", "tabular features")}`<br>
            **Backbone:** `{input_metadata.get("backbone", "not applicable")}`

            ### Intended Use
            Supervised {context["problem_type"]} on data drawn from the same
            operational domain and label definition as the audited dataset.

            ### Performance
            Primary cross-validated metric:
            `{summary.get("primary_cross_validated_metric")}`. Held-out test
            metric: `{summary.get("held_out_testing_metric")}`. Calibration
            temperature: `{summary.get("temperature", "not applicable")}`.

            ### Limitations
            Performance can degrade under camera, geography, demographic,
            acquisition-device, temporal, class-prior, or labeling shifts.
            Rare classes and groups with limited support require additional
            review. Similar pretrained data may also inflate apparent transfer
            performance.

            ### Known Failure Modes
            Review the confusion-pair table, high-confidence errors, low-quality
            image gallery, calibration diagnostics, and worst-performing groups
            above. Possible label errors require human verification.

            ### Deployment Recommendations
            Validate on recent production data, define confidence-based
            abstention, monitor drift and per-class/group metrics, preserve input
            preprocessing and backbone revision, and establish rollback criteria.
            Do not tune repeatedly against this held-out test set.
            '''
            display(Markdown(model_card))
            """
        ),
    ]

    if config.get("problem_type") != "classification":
        classification_sections = (
            "## 11.", "## 12.", "## 13.", "## 14.", "## 15.", "## 16."
        )
        for index, cell in enumerate(cells[:-1]):
            if (
                cell.cell_type == "markdown"
                and cell.source.startswith(classification_sections)
                and cells[index + 1].cell_type == "code"
            ):
                cells[index + 1] = code(
                    "display(Markdown("
                    "'This classification-specific diagnostic is not applicable "
                    "to a regression run.'))"
                )

    notebook = new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3"}
    notebook.metadata["ml_builder"] = {
        "analysis_order": "pre-training-first",
        "analytics_directory": "analysis_data",
        "original_analytics_directory": str(original_analysis_dir),
        "expensive_analyses_bounded": True,
    }
    nbformat.write(notebook, output)
    print(f"[Notebook] Analysis notebook saved to: {output}")
    return str(output)


def create_notebook_bundle(
    output_dir: str | Path,
    *,
    notebook_path: str | Path,
    plot_paths: list[str] | None = None,
) -> str:
    """Package the notebook with every bounded input it needs to execute."""
    root = Path(output_dir).resolve()
    notebook = Path(notebook_path).resolve()
    destination = root / "analysis_bundle.zip"
    members = [notebook, root / "metrics.csv"]
    analysis_dir = root / "analysis_data"
    if analysis_dir.is_dir():
        members.extend(
            path for path in analysis_dir.rglob("*") if path.is_file()
        )
    for value in plot_paths or []:
        path = Path(value).resolve()
        if path.is_file() and root in path.parents:
            members.append(path)
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in dict.fromkeys(members):
            if path.is_file() and root in path.parents:
                archive.write(path, path.relative_to(root))
    return str(destination)
