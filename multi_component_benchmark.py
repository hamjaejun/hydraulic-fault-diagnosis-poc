"""Nested generalization benchmark for four hydraulic components."""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV

from config import FEATURE_DIR, RANDOM_SEED
from generalization_audit import inner_splitter, load_profile, make_outer_splits, parameter_grid
from multi_component import (
    COMPONENT_SPECS,
    MULTI_RESULT_DIR,
    ensure_multi_directories,
    load_component_table,
)


OBSERVATION_TIMES = (5, 60)
MODEL_NAMES = ("logistic_regression", "random_forest")
SCHEME_ORDER = ("random_cycle", "block_aware", "condition_holdout")


def configure_logging() -> logging.Logger:
    ensure_multi_directories()
    logger = logging.getLogger("multi_component_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(MULTI_RESULT_DIR / "execution_log.txt", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def metric_values(y_true, y_pred, component: str) -> dict:
    spec = COMPONENT_SPECS[component]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=spec["classes"], average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=spec["classes"], average="weighted", zero_division=0),
        "severe_recall": float(
            precision_recall_fscore_support(y_true, y_pred, labels=[spec["severe_class"]], zero_division=0)[1][0]
        ),
    }


def class_metric_rows(y_true, y_pred, component: str) -> list[dict]:
    spec = COMPONENT_SPECS[component]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=spec["classes"], zero_division=0
    )
    return [
        {
            "class_id": class_id,
            "class_label": spec["labels"][class_id],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
            "is_severe_class": class_id == spec["severe_class"],
        }
        for index, class_id in enumerate(spec["classes"])
    ]


def save_label_distribution(profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for component, spec in COMPONENT_SPECS.items():
        counts = profile[spec["column"]].value_counts().reindex(spec["classes"], fill_value=0)
        for class_id, count in counts.items():
            rows.append(
                {
                    "component": component,
                    "class_id": int(class_id),
                    "class_label": spec["labels"][int(class_id)],
                    "count": int(count),
                    "proportion": float(count / len(profile)),
                    "is_severe_class": int(class_id) == spec["severe_class"],
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv(MULTI_RESULT_DIR / "component_label_distribution.csv", index=False)
    return output


def load_pump_cache() -> pd.DataFrame:
    path = MULTI_RESULT_DIR.parent / "generalization_audit" / "validation_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    cache = pd.read_csv(path)
    required = {"scheme", "observation_seconds", "model", "fold", "best_params", "inner_best_macro_f1"}
    return cache if required.issubset(cache.columns) else pd.DataFrame()


def fit_outer_model(
    component: str,
    seconds: int,
    scheme: str,
    fold: int,
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    train_profile: pd.DataFrame,
    model_name: str,
    pump_cache: pd.DataFrame,
):
    estimator, grid = parameter_grid(model_name)
    if component == "pump" and not pump_cache.empty:
        cached = pump_cache.query(
            "scheme == @scheme and observation_seconds == @seconds and model == @model_name and fold == @fold"
        )
        if len(cached) == 1:
            row = cached.iloc[0]
            params = json.loads(row["best_params"])
            estimator.set_params(**params)
            started = time.perf_counter()
            estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
            return estimator, params, float(row["inner_best_macro_f1"]), time.perf_counter() - started, "generalization_audit_cache"

    groups = None
    if scheme == "block_aware":
        groups = train_profile["inferred_block_id"].to_numpy()
    elif scheme == "condition_holdout":
        groups = train_profile["condition_id"].to_numpy()
    search = GridSearchCV(
        estimator,
        grid,
        scoring="f1_macro",
        cv=inner_splitter(scheme),
        refit=True,
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )
    started = time.perf_counter()
    search.fit(X.iloc[train_idx], y.iloc[train_idx], groups=groups)
    return search.best_estimator_, search.best_params_, float(search.best_score_), time.perf_counter() - started, "nested_inner_cv"


def run_nested_benchmark(logger: logging.Logger):
    profile = load_profile()
    outer_splits = make_outer_splits(profile)
    pump_cache = load_pump_cache()
    fold_rows: list[dict] = []
    class_rows: list[dict] = []
    prediction_rows: list[dict] = []
    confusion_rows: list[dict] = []
    pooled_matrices: dict[tuple, np.ndarray] = {}

    for component, spec in COMPONENT_SPECS.items():
        for seconds in OBSERVATION_TIMES:
            X, y, cycle_ids, _ = load_component_table(seconds, component)
            if not np.array_equal(cycle_ids.to_numpy(), profile["cycle_id"].to_numpy()):
                raise AssertionError(f"{component} {seconds}s cycle IDs do not align")
            for scheme in SCHEME_ORDER:
                for fold, (train_idx, test_idx) in enumerate(outer_splits[scheme], start=1):
                    train_profile = profile.iloc[train_idx]
                    test_profile = profile.iloc[test_idx]
                    expected_classes = set(spec["classes"])
                    if set(y.iloc[train_idx].unique()) != expected_classes or set(y.iloc[test_idx].unique()) != expected_classes:
                        raise AssertionError(f"{component} {scheme} fold {fold}: missing class")
                    shared_blocks = len(set(train_profile["inferred_block_id"]) & set(test_profile["inferred_block_id"]))
                    shared_conditions = len(set(train_profile["condition_id"]) & set(test_profile["condition_id"]))
                    if scheme == "block_aware" and shared_blocks:
                        raise AssertionError("Block leakage detected")
                    if scheme == "condition_holdout" and shared_conditions:
                        raise AssertionError("Condition leakage detected")

                    for model_name in MODEL_NAMES:
                        model, params, inner_score, fit_seconds, parameter_source = fit_outer_model(
                            component,
                            seconds,
                            scheme,
                            fold,
                            X,
                            y,
                            train_idx,
                            train_profile,
                            model_name,
                            pump_cache,
                        )
                        prediction = model.predict(X.iloc[test_idx])
                        probability = model.predict_proba(X.iloc[test_idx])
                        classes_from_model = [int(value) for value in model.classes_]
                        metrics = metric_values(y.iloc[test_idx], prediction, component)
                        fold_rows.append(
                            {
                                "component": component,
                                "scheme": scheme,
                                "observation_seconds": seconds,
                                "model": model_name,
                                "fold": fold,
                                **metrics,
                                "train_cycles": len(train_idx),
                                "evaluation_cycles": len(test_idx),
                                "train_class_counts": json.dumps(y.iloc[train_idx].value_counts().sort_index().to_dict()),
                                "evaluation_class_counts": json.dumps(y.iloc[test_idx].value_counts().sort_index().to_dict()),
                                "train_blocks": train_profile["inferred_block_id"].nunique(),
                                "evaluation_blocks": test_profile["inferred_block_id"].nunique(),
                                "shared_blocks": shared_blocks,
                                "train_conditions": train_profile["condition_id"].nunique(),
                                "evaluation_conditions": test_profile["condition_id"].nunique(),
                                "shared_conditions": shared_conditions,
                                "inner_best_macro_f1": inner_score,
                                "best_params": json.dumps(params, sort_keys=True),
                                "parameter_source": parameter_source,
                                "fit_seconds": fit_seconds,
                            }
                        )
                        for row in class_metric_rows(y.iloc[test_idx], prediction, component):
                            class_rows.append(
                                {
                                    "component": component,
                                    "scheme": scheme,
                                    "observation_seconds": seconds,
                                    "model": model_name,
                                    "fold": fold,
                                    **row,
                                }
                            )
                        matrix = confusion_matrix(y.iloc[test_idx], prediction, labels=spec["classes"])
                        key = (component, scheme, seconds, model_name)
                        pooled_matrices.setdefault(key, np.zeros_like(matrix))
                        pooled_matrices[key] += matrix
                        for actual_index, actual in enumerate(spec["classes"]):
                            for predicted_index, predicted in enumerate(spec["classes"]):
                                confusion_rows.append(
                                    {
                                        "component": component,
                                        "scheme": scheme,
                                        "observation_seconds": seconds,
                                        "model": model_name,
                                        "fold": fold,
                                        "actual": actual,
                                        "predicted": predicted,
                                        "count": int(matrix[actual_index, predicted_index]),
                                    }
                                )
                        for local_index, global_index in enumerate(test_idx):
                            probabilities = {
                                str(class_id): float(probability[local_index, class_index])
                                for class_index, class_id in enumerate(classes_from_model)
                            }
                            prediction_rows.append(
                                {
                                    "component": component,
                                    "scheme": scheme,
                                    "observation_seconds": seconds,
                                    "model": model_name,
                                    "fold": fold,
                                    "cycle_id": int(profile.iloc[global_index]["cycle_id"]),
                                    "actual": int(y.iloc[global_index]),
                                    "predicted": int(prediction[local_index]),
                                    "correct": int(y.iloc[global_index] == prediction[local_index]),
                                    "confidence": float(probability[local_index].max()),
                                    "probabilities": json.dumps(probabilities, sort_keys=True),
                                    "faulty_component_count": int(profile.iloc[global_index]["faulty_component_count"]),
                                    "fault_group": fault_group(int(profile.iloc[global_index]["faulty_component_count"])),
                                    "fault_pattern": profile.iloc[global_index]["fault_pattern"],
                                    "condition_id": int(profile.iloc[global_index]["condition_id"]),
                                    "inferred_block_id": int(profile.iloc[global_index]["inferred_block_id"]),
                                    "stable_flag": int(profile.iloc[global_index]["stable_flag"]),
                                }
                            )
                        logger.info(
                            "%s | %s | %ss | %s | fold %s | macro-F1 %.4f | severe recall %.4f",
                            component,
                            scheme,
                            seconds,
                            model_name,
                            fold,
                            metrics["macro_f1"],
                            metrics["severe_recall"],
                        )

    fold_results = pd.DataFrame(fold_rows)
    classwise = pd.DataFrame(class_rows)
    predictions = pd.DataFrame(prediction_rows)
    confusion = pd.DataFrame(confusion_rows)
    save_confusion_figures(pooled_matrices)
    return profile, fold_results, classwise, predictions, confusion


def fault_group(count: int) -> str:
    if count == 0:
        return "all_normal"
    if count == 1:
        return "single_fault"
    return "multiple_fault"


def save_confusion_figures(matrices: dict[tuple, np.ndarray]) -> None:
    output_dir = MULTI_RESULT_DIR / "confusion_matrices"
    output_dir.mkdir(parents=True, exist_ok=True)
    for (component, scheme, seconds, model_name), matrix in matrices.items():
        spec = COMPONENT_SPECS[component]
        labels = [spec["labels"][class_id] for class_id in spec["classes"]]
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{spec['display']} — {scheme.replace('_', ' ').title()}\n{model_name.replace('_', ' ').title()}, {seconds}s (pooled OOF)")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(output_dir / f"{component}_{scheme}_{model_name}_{seconds}s.png", dpi=170)
        plt.close(fig)


def aggregate_fold_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "macro_f1", "weighted_f1", "severe_recall"]
    summary = fold_results.groupby(["component", "scheme", "observation_seconds", "model"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def make_wide_benchmark(summary: pd.DataFrame) -> pd.DataFrame:
    macro = summary.pivot_table(
        index=["component", "observation_seconds", "model"],
        columns="scheme",
        values=["macro_f1_mean", "macro_f1_std"],
    )
    macro.columns = [f"{scheme}_{metric}" for metric, scheme in macro.columns]
    return macro.reset_index()


def build_fault_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouping_sets = [("fault_count", "faulty_component_count"), ("fault_group", "fault_group")]
    for analysis_level, group_column in grouping_sets:
        keys = ["component", "scheme", "observation_seconds", "model", group_column]
        for values, group in predictions.groupby(keys, sort=False):
            component, scheme, seconds, model_name, group_value = values
            spec = COMPONENT_SPECS[component]
            metrics = metric_values(group["actual"], group["predicted"], component)
            rows.append(
                {
                    "analysis_level": analysis_level,
                    "component": component,
                    "scheme": scheme,
                    "observation_seconds": seconds,
                    "model": model_name,
                    "group": group_value,
                    "cycles": len(group),
                    "observed_target_classes": group["actual"].nunique(),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def build_condition_analysis(predictions: pd.DataFrame, classwise: pd.DataFrame) -> pd.DataFrame:
    class_means = classwise.groupby(["component", "scheme", "observation_seconds", "model", "class_id", "class_label"])[["precision", "recall", "f1", "support"]].mean().reset_index()
    random = class_means[class_means["scheme"] == "random_cycle"].rename(columns={"precision": "random_precision", "recall": "random_recall", "f1": "random_f1"})
    holdout = class_means[class_means["scheme"] == "condition_holdout"].rename(columns={"precision": "holdout_precision", "recall": "holdout_recall", "f1": "holdout_f1"})
    class_compare = random.merge(
        holdout,
        on=["component", "observation_seconds", "model", "class_id", "class_label"],
        suffixes=("_random", "_holdout"),
    )
    class_compare["recall_change_holdout_minus_random"] = class_compare["holdout_recall"] - class_compare["random_recall"]
    class_compare["f1_change_holdout_minus_random"] = class_compare["holdout_f1"] - class_compare["random_f1"]
    class_compare.insert(0, "analysis_level", "class_comparison")

    condition_rows = []
    holdout_predictions = predictions[predictions["scheme"] == "condition_holdout"]
    for values, group in holdout_predictions.groupby(["component", "observation_seconds", "model", "condition_id", "fault_pattern"]):
        component, seconds, model_name, condition_id, pattern = values
        condition_rows.append(
            {
                "analysis_level": "condition_error",
                "component": component,
                "observation_seconds": seconds,
                "model": model_name,
                "condition_id": condition_id,
                "fault_pattern": pattern,
                "cycles": len(group),
                "accuracy": group["correct"].mean(),
                "errors": int((1 - group["correct"]).sum()),
            }
        )
    return pd.concat([class_compare, pd.DataFrame(condition_rows)], ignore_index=True, sort=False)


def make_figures(summary: pd.DataFrame, fold_results: pd.DataFrame, fault_analysis: pd.DataFrame, condition_analysis: pd.DataFrame) -> None:
    figure_dir = MULTI_RESULT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    components = list(COMPONENT_SPECS)
    colors = {"logistic_regression": "#2563eb", "random_forest": "#0f766e"}

    holdout = summary[summary["scheme"] == "condition_holdout"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, seconds in zip(axes, OBSERVATION_TIMES):
        subset = holdout[holdout["observation_seconds"] == seconds]
        x = np.arange(len(components))
        for index, model_name in enumerate(MODEL_NAMES):
            values = subset[subset["model"] == model_name].set_index("component").reindex(components)
            ax.bar(x + (index - 0.5) * 0.36, values["macro_f1_mean"], 0.36, yerr=values["macro_f1_std"], capsize=3, color=colors[model_name], label=model_name.replace("_", " ").title())
        ax.set(xticks=x, xticklabels=[COMPONENT_SPECS[c]["display"] for c in components], ylim=(0, 1.03), title=f"Condition holdout — {seconds}s", ylabel="Macro-F1")
        ax.grid(axis="y", alpha=0.2)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "component_macro_f1.png", dpi=180)
    plt.close(fig)

    best_by_component = summary.sort_values("macro_f1_mean", ascending=False).groupby(["component", "scheme", "observation_seconds"], as_index=False).first()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for ax, component in zip(axes.flat, components):
        subset = best_by_component[best_by_component["component"] == component]
        x = np.arange(len(SCHEME_ORDER))
        for index, seconds in enumerate(OBSERVATION_TIMES):
            values = subset[subset["observation_seconds"] == seconds].set_index("scheme").reindex(SCHEME_ORDER)
            ax.bar(x + (index - 0.5) * 0.36, values["macro_f1_mean"], 0.36, yerr=values["macro_f1_std"], capsize=3, label=f"{seconds}s")
        ax.set(xticks=x, xticklabels=[s.replace("_", " ") for s in SCHEME_ORDER], ylim=(0, 1.03), title=COMPONENT_SPECS[component]["display"], ylabel="Best-model Macro-F1")
        ax.tick_params(axis="x", rotation=12)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 1].legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "scheme_and_time_comparison.png", dpi=180)
    plt.close(fig)

    fault_subset = fault_analysis.query("analysis_level == 'fault_group' and scheme == 'condition_holdout' and observation_seconds == 60")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, model_name in zip(axes, MODEL_NAMES):
        data = fault_subset[fault_subset["model"] == model_name]
        for component in components:
            values = data[data["component"] == component].set_index("group").reindex(["single_fault", "multiple_fault"])
            ax.plot(["Single fault", "Multiple fault"], values["accuracy"], marker="o", label=COMPONENT_SPECS[component]["display"])
        ax.set(ylim=(0, 1.03), title=model_name.replace("_", " ").title(), ylabel="OOF accuracy")
        ax.grid(alpha=0.2)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "single_vs_multiple_fault.png", dpi=180)
    plt.close(fig)

    class_drop = condition_analysis[condition_analysis["analysis_level"] == "class_comparison"].copy()
    class_drop = class_drop.sort_values("recall_change_holdout_minus_random").head(20)
    class_drop["label"] = class_drop["component"] + " | " + class_drop["class_label"] + " | " + class_drop["observation_seconds"].astype(str) + "s | " + class_drop["model"]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(class_drop["label"][::-1], class_drop["recall_change_holdout_minus_random"][::-1], color="#dc2626")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title="Largest class-recall changes: condition holdout minus random CV", xlabel="Recall change")
    fig.tight_layout()
    fig.savefig(figure_dir / "condition_holdout_class_recall_drop.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, seconds in zip(axes, OBSERVATION_TIMES):
        data = fold_results[fold_results["observation_seconds"] == seconds]
        groups = []
        labels = []
        for component in components:
            for scheme in SCHEME_ORDER:
                groups.append(data[(data["component"] == component) & (data["scheme"] == scheme)]["macro_f1"].to_numpy())
                labels.append(f"{component[:3]}\n{scheme.split('_')[0]}")
        ax.boxplot(groups, tick_labels=labels, showmeans=True)
        ax.set(title=f"{seconds}s outer-fold Macro-F1", ylabel="Macro-F1", ylim=(0, 1.03))
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "fold_performance_distribution.png", dpi=180)
    plt.close(fig)


def write_report(profile: pd.DataFrame, summary: pd.DataFrame, fault_analysis: pd.DataFrame, condition_analysis: pd.DataFrame) -> None:
    best_holdout = (
        summary[summary["scheme"] == "condition_holdout"]
        .sort_values("macro_f1_mean", ascending=False)
        .groupby(["component", "observation_seconds"], as_index=False)
        .first()
    )
    hardest_row = best_holdout.sort_values("macro_f1_mean").iloc[0]
    class_compare = condition_analysis[condition_analysis["analysis_level"] == "class_comparison"].copy()
    weakest_class = class_compare.sort_values("holdout_recall").iloc[0]
    fault_group = fault_analysis.query("analysis_level == 'fault_group' and scheme == 'condition_holdout' and observation_seconds == 60")
    fault_lines = []
    for component in COMPONENT_SPECS:
        candidates = best_holdout[(best_holdout["component"] == component) & (best_holdout["observation_seconds"] == 60)]
        model_name = candidates.iloc[0]["model"]
        data = fault_group[(fault_group["component"] == component) & (fault_group["model"] == model_name)].set_index("group")
        single = data.loc["single_fault"] if "single_fault" in data.index else None
        multiple = data.loc["multiple_fault"] if "multiple_fault" in data.index else None
        if single is not None and multiple is not None:
            fault_lines.append(f"| {COMPONENT_SPECS[component]['display']} | {model_name} | {int(single.cycles)} | {single.accuracy:.4f} | {int(multiple.cycles)} | {multiple.accuracy:.4f} | {multiple.accuracy - single.accuracy:+.4f} |")

    table_lines = ["| Component | Time | Model | Random F1 | Block F1 | Holdout F1 |", "| --- | ---: | --- | ---: | ---: | ---: |"]
    for component in COMPONENT_SPECS:
        for seconds in OBSERVATION_TIMES:
            for model_name in MODEL_NAMES:
                rows = summary.query("component == @component and observation_seconds == @seconds and model == @model_name").set_index("scheme")
                table_lines.append(f"| {COMPONENT_SPECS[component]['display']} | {seconds}s | {model_name} | {rows.loc['random_cycle','macro_f1_mean']:.4f} | {rows.loc['block_aware','macro_f1_mean']:.4f} | {rows.loc['condition_holdout','macro_f1_mean']:.4f} |")

    distributions = []
    for component, spec in COMPONENT_SPECS.items():
        counts = profile[spec["column"]].value_counts().sort_index()
        distributions.append(f"- **{spec['display']}**: " + ", ".join(f"{code} ({spec['labels'][int(code)]}) = {count}" for code, count in counts.items()))

    report = f"""# Multi-Component Hydraulic Fault Diagnosis Report

## Dataset

The benchmark uses all 2,205 original UCI cycles and the same 153 causal sensor-statistic features as the pump PoC. Stable flag and all component labels are excluded from predictors.

{chr(10).join(distributions)}

Fault multiplicity was rechecked from the raw component labels: {(profile['faulty_component_count'] == 0).sum()} all-normal cycles, {(profile['faulty_component_count'] == 1).sum()} single-fault cycles, and {(profile['faulty_component_count'] >= 2).sum()} multiple-fault cycles.

## Methodology

- **Demo models:** four independent component classifiers for 5, 10, 20, 40, and 60 seconds, using the existing 70/15/15 cycle split. Dummy, Logistic Regression, and Random Forest are evaluated; validation macro-F1 selects each saved demo model.
- **Research benchmark:** 5- and 60-second features, five outer folds, three inner folds, Logistic Regression and Random Forest.
- **Random cycle:** stratified by pump class and shared across every component/time/model.
- **Block-aware:** contiguous runs with identical five-label profiles; these are inferred groups, not true session IDs.
- **Condition holdout:** complete four-component state tuples are disjoint between train and evaluation.
- Every fold contains every class for all four targets. Train/evaluation cycle overlap is zero. Pipeline preprocessing and tuning occur inside the training folds only.
- Pump hyperparameters are reused from the completed audit only where the folds, features, grids, target, and protocol are identical; outer models are refitted to generate OOF predictions.

## Results

{chr(10).join(table_lines)}

The hardest best-model condition-holdout setting was **{COMPONENT_SPECS[hardest_row.component]['display']} at {int(hardest_row.observation_seconds)} seconds** ({hardest_row.model}, macro-F1 {hardest_row.macro_f1_mean:.4f} ± {hardest_row.macro_f1_std:.4f}). The lowest class-level condition-holdout recall occurred for **{COMPONENT_SPECS[weakest_class.component]['display']} — {weakest_class.class_label}** at {int(weakest_class.observation_seconds)} seconds with {weakest_class.model} (recall {weakest_class.holdout_recall:.4f}).

## Multiple-Fault Analysis

The table uses each component/time's post-hoc best mean condition-holdout model. It is descriptive and is not a new independent model-selection result.

| Component | Model | Single cycles | Single accuracy | Multiple cycles | Multiple accuracy | Change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(fault_lines)}

Differences cannot be interpreted as the causal effect of simultaneous faults because target-class prevalence and condition severity differ between groups. The 21 all-normal cycles are too few for broad conclusions.

## Generalization Analysis

Random cycle CV is the interpolation baseline. Block-aware CV removes overlap of estimated contiguous runs, while condition holdout removes overlap of all four-component state combinations. `condition_holdout_analysis.csv` lists class-recall changes and the condition IDs where errors concentrate. Fold variability is retained rather than hidden by a single aggregate score.

## Discussion

1. **Hardest component:** the measured hardest setting is reported above; difficulty depends strongly on time and model rather than component alone.
2. **Largest split degradation:** see `condition_holdout_class_recall_drop.png`; condition holdout is the strictest test of unseen combinations.
3. **Weakest class:** {COMPONENT_SPECS[weakest_class.component]['display']} — {weakest_class.class_label}, under the setting stated above.
4. **Short observation:** compare 5s and 60s in the main table; the largest best-model gain identifies the component most dependent on longer dynamics.
5. **Multiple faults:** descriptive OOF differences are shown above, with sample counts. They are associations, not causal estimates.
6. **Physics-guided priority:** prioritize the component with the weakest condition-holdout class recall and largest 5s-to-60s gap. Start with pressure/flow/power/temperature consistency features and OOD detection, not a PINN claim.

## Physics-Guided ML Next Step

- **Pump:** pressure rise, flow loss, motor power, hydraulic power `Δp × Q`, and efficiency residuals; verify sensor locations across the pump and load path.
- **Valve:** upstream/downstream pressure transients, switching-time proxies, flow response, and pressure-flow hysteresis; valve command and spool position are missing.
- **Cooler:** inlet/outlet temperature difference, cooling power, flow-dependent heat transfer, and thermal time constants; ambient temperature and heat-exchanger parameters are missing.
- **Accumulator:** pressure ripple, charge/discharge slopes, effective compliance, and flow-pressure phase response; pre-charge pressure and circuit topology are required for stronger physics constraints.

Physics-guided feature engineering means adding interpretable conservation/residual features to ML. A PINN would additionally enforce governing differential equations during training. This dataset lacks several boundary conditions, control inputs, geometry, and component parameters, so physics-guided features should be tested first under the same condition-holdout folds.

## Limitations

- One experimental rig; no machine/site external validation.
- Inferred blocks are not published independent session IDs.
- Unseen combinations are not the same as unseen state values or new hardware.
- State labels describe configured experimental condition, not a confirmed maintenance root cause.
- No repair actions, failure progression, or maintenance outcomes are available.
- Some class/group subsets are small and imbalanced.
- Choosing future physics work from these CV results introduces research-selection bias and requires a new confirmation study.
"""
    (MULTI_RESULT_DIR / "MULTI_COMPONENT_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    logger = configure_logging()
    profile = load_profile()
    save_label_distribution(profile)
    logger.info("Starting four-component nested benchmark")
    profile, fold_results, classwise, predictions, confusion = run_nested_benchmark(logger)
    fold_results.to_csv(MULTI_RESULT_DIR / "nested_cv_fold_results.csv", index=False)
    classwise.to_csv(MULTI_RESULT_DIR / "classwise_metrics.csv", index=False)
    predictions.to_csv(MULTI_RESULT_DIR / "out_of_fold_predictions.csv", index=False)
    confusion.to_csv(MULTI_RESULT_DIR / "confusion_matrices.csv", index=False)
    summary = aggregate_fold_results(fold_results)
    summary.to_csv(MULTI_RESULT_DIR / "nested_cv_summary.csv", index=False)
    wide = make_wide_benchmark(summary)
    wide.to_csv(MULTI_RESULT_DIR / "multi_component_benchmark.csv", index=False)
    fault_analysis = build_fault_analysis(predictions)
    fault_analysis.to_csv(MULTI_RESULT_DIR / "single_vs_multi_fault.csv", index=False)
    condition_analysis = build_condition_analysis(predictions, classwise)
    condition_analysis.to_csv(MULTI_RESULT_DIR / "condition_holdout_analysis.csv", index=False)
    make_figures(summary, fold_results, fault_analysis, condition_analysis)
    write_report(profile, summary, fault_analysis, condition_analysis)
    logger.info("Benchmark complete: %s fold results, %s OOF predictions", len(fold_results), len(predictions))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
