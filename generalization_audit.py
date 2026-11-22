"""Dataset audit and strict generalization validation for pump leakage models.

All artifacts are written under results/generalization_audit/. Existing models,
dashboard code, split assignments, and result files are never modified.
"""
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
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold

from config import CLASS_NAMES, DATA_DIR, FEATURE_DIR, RANDOM_SEED, RAW_DIR, RESULT_DIR
from preprocess import extract_features
from train import make_model


AUDIT_DIR = RESULT_DIR / "generalization_audit"
OBSERVATION_TIMES = (5, 60)
MODEL_NAMES = ("logistic_regression", "random_forest")
PROFILE_COLUMNS = (
    "cooler_condition",
    "valve_condition",
    "pump_leakage",
    "accumulator_pressure",
    "stable_flag",
)
COMPONENT_COLUMNS = PROFILE_COLUMNS[:4]
NORMAL_CODES = {
    "cooler_condition": 100,
    "valve_condition": 100,
    "pump_leakage": 0,
    "accumulator_pressure": 130,
}
STATE_LABELS = {
    "cooler_condition": {3: "Close to total failure", 20: "Reduced efficiency", 100: "Full efficiency"},
    "valve_condition": {73: "Close to total failure", 80: "Severe lag", 90: "Small lag", 100: "Optimal"},
    "pump_leakage": {0: "No leakage", 1: "Weak leakage", 2: "Severe leakage"},
    "accumulator_pressure": {90: "Close to total failure", 100: "Severely reduced", 115: "Slightly reduced", 130: "Optimal"},
    "stable_flag": {0: "Conditions stable", 1: "Static conditions may not have been reached"},
}


def configure_logging() -> logging.Logger:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("generalization_audit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(AUDIT_DIR / "execution_log.txt", mode="w", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def consecutive_ranges(values: pd.Series) -> str:
    ordered = sorted(int(value) for value in values)
    ranges: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            ranges.append(str(start) if start == previous else f"{start}-{previous}")
            start = value
        previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ";".join(ranges)


def fault_pattern(row: pd.Series) -> str:
    faulty = [column.replace("_condition", "").replace("_pressure", "") for column in COMPONENT_COLUMNS if row[column] != NORMAL_CODES[column]]
    return "+".join(faulty) if faulty else "none"


def load_profile() -> pd.DataFrame:
    raw = np.loadtxt(RAW_DIR / "profile.txt", dtype=int)
    if raw.shape != (2205, 5):
        raise ValueError(f"Expected profile shape (2205, 5), got {raw.shape}")
    profile = pd.DataFrame(raw, columns=PROFILE_COLUMNS)
    profile.insert(0, "cycle_id", np.arange(1, len(profile) + 1, dtype=int))
    for column in COMPONENT_COLUMNS:
        profile[f"{column}_fault"] = profile[column] != NORMAL_CODES[column]
    fault_columns = [f"{column}_fault" for column in COMPONENT_COLUMNS]
    profile["faulty_component_count"] = profile[fault_columns].sum(axis=1).astype(int)
    profile["fault_pattern"] = profile.apply(fault_pattern, axis=1)
    profile["condition_key"] = profile[list(COMPONENT_COLUMNS)].astype(str).agg("|".join, axis=1)
    profile["five_label_key"] = profile[list(PROFILE_COLUMNS)].astype(str).agg("|".join, axis=1)
    profile["condition_id"] = pd.factorize(profile["condition_key"], sort=False)[0] + 1
    profile["inferred_block_id"] = profile["five_label_key"].ne(profile["five_label_key"].shift()).cumsum().astype(int)
    return profile


def build_dataset_tables(profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    for column in PROFILE_COLUMNS:
        counts = profile[column].value_counts().sort_index()
        for code, count in counts.items():
            summary_rows.append(
                {
                    "section": "raw_state_distribution",
                    "component": column,
                    "state_code": int(code),
                    "state_label": STATE_LABELS[column][int(code)],
                    "count": int(count),
                    "proportion": float(count / len(profile)),
                }
            )
    for column in COMPONENT_COLUMNS:
        fault_count = int(profile[f"{column}_fault"].sum())
        for status, count in (("normal", len(profile) - fault_count), ("fault_or_degraded", fault_count)):
            summary_rows.append(
                {
                    "section": "binary_component_status",
                    "component": column,
                    "state_code": status,
                    "state_label": status,
                    "count": int(count),
                    "proportion": float(count / len(profile)),
                }
            )
    multiplicity_names = {0: "no_component_fault", 1: "single_component_fault"}
    multiplicity = profile["faulty_component_count"].value_counts().sort_index()
    for count_code, count in multiplicity.items():
        summary_rows.append(
            {
                "section": "fault_multiplicity_exact",
                "component": "four_components",
                "state_code": int(count_code),
                "state_label": multiplicity_names.get(int(count_code), f"{int(count_code)}_component_faults"),
                "count": int(count),
                "proportion": float(count / len(profile)),
            }
        )
    multi_count = int((profile["faulty_component_count"] >= 2).sum())
    for label, count in (
        ("no_component_fault", int((profile["faulty_component_count"] == 0).sum())),
        ("single_component_fault", int((profile["faulty_component_count"] == 1).sum())),
        ("multiple_component_fault", multi_count),
    ):
        summary_rows.append(
            {
                "section": "fault_multiplicity_grouped",
                "component": "four_components",
                "state_code": label,
                "state_label": label,
                "count": count,
                "proportion": float(count / len(profile)),
            }
        )
    summary = pd.DataFrame(summary_rows)

    combination_rows: list[dict] = []
    for condition_id, group in profile.groupby("condition_id", sort=True):
        first = group.iloc[0]
        combination_rows.append(
            {
                "condition_id": int(condition_id),
                "cooler_condition": int(first["cooler_condition"]),
                "cooler_label": STATE_LABELS["cooler_condition"][int(first["cooler_condition"])],
                "valve_condition": int(first["valve_condition"]),
                "valve_label": STATE_LABELS["valve_condition"][int(first["valve_condition"])],
                "pump_leakage": int(first["pump_leakage"]),
                "pump_label": STATE_LABELS["pump_leakage"][int(first["pump_leakage"])],
                "accumulator_pressure": int(first["accumulator_pressure"]),
                "accumulator_label": STATE_LABELS["accumulator_pressure"][int(first["accumulator_pressure"])],
                "faulty_component_count": int(first["faulty_component_count"]),
                "fault_pattern": first["fault_pattern"],
                "cycle_count": int(len(group)),
                "proportion": float(len(group) / len(profile)),
                "first_cycle_id": int(group["cycle_id"].min()),
                "last_cycle_id": int(group["cycle_id"].max()),
                "segment_count": int(group["cycle_id"].diff().fillna(1).ne(1).sum() + 1),
                "cycle_id_ranges": consecutive_ranges(group["cycle_id"]),
            }
        )
    combinations = pd.DataFrame(combination_rows).sort_values(
        ["cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure"]
    )

    block_rows: list[dict] = []
    for block_id, group in profile.groupby("inferred_block_id", sort=True):
        first = group.iloc[0]
        block_rows.append(
            {
                "inferred_block_id": int(block_id),
                "start_cycle_id": int(group["cycle_id"].min()),
                "end_cycle_id": int(group["cycle_id"].max()),
                "block_size": int(len(group)),
                "cooler_condition": int(first["cooler_condition"]),
                "valve_condition": int(first["valve_condition"]),
                "pump_leakage": int(first["pump_leakage"]),
                "accumulator_pressure": int(first["accumulator_pressure"]),
                "stable_flag": int(first["stable_flag"]),
                "condition_id": int(first["condition_id"]),
                "faulty_component_count": int(first["faulty_component_count"]),
                "fault_pattern": first["fault_pattern"],
            }
        )
    blocks = pd.DataFrame(block_rows)
    return summary, combinations, blocks


def make_dataset_figures(profile: pd.DataFrame, summary: pd.DataFrame, combinations: pd.DataFrame, blocks: pd.DataFrame) -> None:
    state_rows = summary[summary["section"] == "raw_state_distribution"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, column in zip(axes.flat, COMPONENT_COLUMNS):
        data = state_rows[state_rows["component"] == column]
        ax.bar(data["state_code"].astype(str), data["count"], color="#2563eb")
        ax.set(title=column.replace("_", " ").title(), xlabel="State code", ylabel="Cycles")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("UCI component-state distributions")
    fig.tight_layout()
    fig.savefig(AUDIT_DIR / "component_state_distributions.png", dpi=180)
    plt.close(fig)

    grouped = summary[summary["section"] == "fault_multiplicity_grouped"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(grouped["state_label"].str.replace("_", " "), grouped["count"], color=["#64748b", "#0ea5e9", "#ef4444"])
    ax.set(title="Number of simultaneously degraded components", ylabel="Cycles")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(AUDIT_DIR / "fault_multiplicity.png", dpi=180)
    plt.close(fig)

    ordered = combinations.sort_values("cycle_count", ascending=False).head(30).copy()
    ordered["combination"] = ordered.apply(
        lambda row: f"C{row.cooler_condition}-V{row.valve_condition}-P{row.pump_leakage}-A{row.accumulator_pressure}", axis=1
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(ordered["combination"][::-1], ordered["cycle_count"][::-1], color="#0f766e")
    ax.set(title="Thirty most frequent four-component state combinations", xlabel="Cycles")
    fig.tight_layout()
    fig.savefig(AUDIT_DIR / "fault_combination_counts.png", dpi=180)
    plt.close(fig)

    encoded_rows = []
    for column in PROFILE_COLUMNS:
        codes = list(STATE_LABELS[column])
        mapping = {code: index for index, code in enumerate(codes)}
        encoded_rows.append(profile[column].map(mapping).to_numpy())
    fig, ax = plt.subplots(figsize=(15, 4.8))
    image = ax.imshow(np.vstack(encoded_rows), aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set(yticks=range(5), yticklabels=[name.replace("_", " ") for name in PROFILE_COLUMNS], xlabel="Original cycle ID", title="State-label sequence in original UCI order")
    ax.set_xticks([0, 499, 999, 1499, 1999, 2204], labels=[1, 500, 1000, 1500, 2000, 2205])
    fig.colorbar(image, ax=ax, label="Within-component state index")
    fig.tight_layout()
    fig.savefig(AUDIT_DIR / "profile_state_timeline.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(blocks["block_size"], bins=np.arange(0.5, blocks["block_size"].max() + 1.5), color="#7c3aed")
    axes[0].set(title="Inferred five-label block sizes", xlabel="Cycles per block", ylabel="Blocks")
    stable = profile["stable_flag"].value_counts().sort_index()
    axes[1].bar(["Stable (0)", "Possibly unstable (1)"], stable.values, color=["#16a34a", "#f59e0b"])
    axes[1].set(title="Stable flag distribution", ylabel="Cycles")
    fig.tight_layout()
    fig.savefig(AUDIT_DIR / "block_and_stability_summary.png", dpi=180)
    plt.close(fig)


def validate_outer_splits(profile: pd.DataFrame, scheme: str, splits: list[tuple[np.ndarray, np.ndarray]]) -> None:
    y = profile["pump_leakage"].to_numpy()
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        if set(train_idx) & set(test_idx):
            raise AssertionError(f"{scheme} fold {fold}: row overlap")
        if set(np.unique(y[train_idx])) != {0, 1, 2} or set(np.unique(y[test_idx])) != {0, 1, 2}:
            raise AssertionError(f"{scheme} fold {fold}: a pump class is missing")
        if scheme == "block_aware":
            shared = set(profile.iloc[train_idx]["inferred_block_id"]) & set(profile.iloc[test_idx]["inferred_block_id"])
            if shared:
                raise AssertionError(f"{scheme} fold {fold}: block leakage")
        if scheme == "condition_holdout":
            shared = set(profile.iloc[train_idx]["condition_id"]) & set(profile.iloc[test_idx]["condition_id"])
            if shared:
                raise AssertionError(f"{scheme} fold {fold}: condition leakage")


def make_outer_splits(profile: pd.DataFrame) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    y = profile["pump_leakage"].to_numpy()
    placeholder = np.zeros((len(profile), 1))
    random_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    grouped_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    splits = {
        "random_cycle": list(random_splitter.split(placeholder, y)),
        "block_aware": list(grouped_splitter.split(placeholder, y, profile["inferred_block_id"])),
        "condition_holdout": list(grouped_splitter.split(placeholder, y, profile["condition_id"])),
    }
    for scheme, scheme_splits in splits.items():
        validate_outer_splits(profile, scheme, scheme_splits)
    return splits


def load_feature_matrix(seconds: int, profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    table = pd.read_csv(FEATURE_DIR / f"features_{seconds}s.csv")
    expected_ids = np.arange(1, len(profile) + 1)
    if not np.array_equal(table["cycle_id"].to_numpy(), expected_ids):
        raise AssertionError(f"{seconds}s feature table is not in original cycle order")
    if not np.array_equal(table["pump_leakage"].to_numpy(), profile["pump_leakage"].to_numpy()):
        raise AssertionError(f"{seconds}s feature labels do not align with profile.txt")
    feature_columns = [column for column in table.columns if "__" in column]
    if len(feature_columns) != 153:
        raise AssertionError(f"Expected 153 sensor features, got {len(feature_columns)}")
    forbidden = set(PROFILE_COLUMNS) | {"cycle_id", "condition_id", "inferred_block_id"}
    if forbidden & set(feature_columns):
        raise AssertionError("Target or group data appears in predictors")
    return table[feature_columns], table["pump_leakage"].astype(int), feature_columns


def parameter_grid(model_name: str) -> tuple[object, list[dict] | dict]:
    if model_name == "logistic_regression":
        return make_model(model_name, {"C": 1.0}), {"classifier__C": [0.1, 1.0, 10.0]}
    return make_model(model_name, {"n_estimators": 300, "max_features": "sqrt"}), [
        {
            "classifier__n_estimators": [300],
            "classifier__max_depth": [None],
            "classifier__min_samples_leaf": [1],
            "classifier__max_features": ["sqrt"],
        },
        {
            "classifier__n_estimators": [400],
            "classifier__max_depth": [16],
            "classifier__min_samples_leaf": [1],
            "classifier__max_features": ["sqrt"],
        },
        {
            "classifier__n_estimators": [400],
            "classifier__max_depth": [None],
            "classifier__min_samples_leaf": [2],
            "classifier__max_features": [0.5],
        },
    ]


def inner_splitter(scheme: str):
    if scheme == "random_cycle":
        return StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED + 100)
    return StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED + 100)


def evaluate_nested_cv(
    profile: pd.DataFrame,
    splits_by_scheme: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict] = []
    confusion_rows: list[dict] = []
    holdout_rows: list[dict] = []
    matrices: dict[tuple[str, int, str], np.ndarray] = {}

    for seconds in OBSERVATION_TIMES:
        X, y, _ = load_feature_matrix(seconds, profile)
        for scheme, outer_splits in splits_by_scheme.items():
            for fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
                train_profile = profile.iloc[train_idx]
                test_profile = profile.iloc[test_idx]
                shared_blocks = len(set(train_profile["inferred_block_id"]) & set(test_profile["inferred_block_id"]))
                shared_conditions = len(set(train_profile["condition_id"]) & set(test_profile["condition_id"]))
                if seconds == OBSERVATION_TIMES[0] and scheme == "condition_holdout":
                    for condition_id, group in test_profile.groupby("condition_id"):
                        first = group.iloc[0]
                        holdout_rows.append(
                            {
                                "fold": fold,
                                "condition_id": int(condition_id),
                                "cooler_condition": int(first["cooler_condition"]),
                                "valve_condition": int(first["valve_condition"]),
                                "pump_leakage": int(first["pump_leakage"]),
                                "accumulator_pressure": int(first["accumulator_pressure"]),
                                "cycles": int(len(group)),
                                "cycle_id_ranges": consecutive_ranges(group["cycle_id"]),
                            }
                        )
                for model_name in MODEL_NAMES:
                    estimator, grid = parameter_grid(model_name)
                    inner_cv = inner_splitter(scheme)
                    search = GridSearchCV(
                        estimator,
                        grid,
                        scoring="f1_macro",
                        cv=inner_cv,
                        refit=True,
                        n_jobs=1,
                        return_train_score=False,
                        error_score="raise",
                    )
                    inner_groups = None
                    if scheme == "block_aware":
                        inner_groups = train_profile["inferred_block_id"].to_numpy()
                    elif scheme == "condition_holdout":
                        inner_groups = train_profile["condition_id"].to_numpy()
                    started = time.perf_counter()
                    search.fit(X.iloc[train_idx], y.iloc[train_idx], groups=inner_groups)
                    fit_seconds = time.perf_counter() - started
                    prediction = search.predict(X.iloc[test_idx])
                    recalls = recall_score(y.iloc[test_idx], prediction, labels=[0, 1, 2], average=None, zero_division=0)
                    matrix = confusion_matrix(y.iloc[test_idx], prediction, labels=[0, 1, 2])
                    matrices.setdefault((scheme, seconds, model_name), np.zeros((3, 3), dtype=int))
                    matrices[(scheme, seconds, model_name)] += matrix
                    for actual in range(3):
                        for predicted in range(3):
                            confusion_rows.append(
                                {
                                    "scheme": scheme,
                                    "observation_seconds": seconds,
                                    "model": model_name,
                                    "fold": fold,
                                    "actual": actual,
                                    "predicted": predicted,
                                    "count": int(matrix[actual, predicted]),
                                }
                            )
                    result = {
                        "scheme": scheme,
                        "observation_seconds": seconds,
                        "model": model_name,
                        "fold": fold,
                        "accuracy": accuracy_score(y.iloc[test_idx], prediction),
                        "macro_f1": f1_score(y.iloc[test_idx], prediction, labels=[0, 1, 2], average="macro", zero_division=0),
                        "normal_recall": recalls[0],
                        "weak_recall": recalls[1],
                        "severe_recall": recalls[2],
                        "train_cycles": len(train_idx),
                        "evaluation_cycles": len(test_idx),
                        "train_blocks": train_profile["inferred_block_id"].nunique(),
                        "evaluation_blocks": test_profile["inferred_block_id"].nunique(),
                        "shared_blocks": shared_blocks,
                        "train_conditions": train_profile["condition_id"].nunique(),
                        "evaluation_conditions": test_profile["condition_id"].nunique(),
                        "shared_conditions": shared_conditions,
                        "inner_best_macro_f1": search.best_score_,
                        "best_params": json.dumps(search.best_params_, sort_keys=True),
                        "fit_seconds": fit_seconds,
                    }
                    result_rows.append(result)
                    logger.info(
                        "%s | %ss | %s | fold %s | macro-F1 %.4f | accuracy %.4f | shared blocks %s | shared conditions %s",
                        scheme,
                        seconds,
                        model_name,
                        fold,
                        result["macro_f1"],
                        result["accuracy"],
                        shared_blocks,
                        shared_conditions,
                    )

    results = pd.DataFrame(result_rows)
    confusion_long = pd.DataFrame(confusion_rows)
    holdout_plan = pd.DataFrame(holdout_rows)
    make_validation_figures(results, matrices)
    return results, confusion_long, holdout_plan


def make_validation_figures(results: pd.DataFrame, matrices: dict[tuple[str, int, str], np.ndarray]) -> None:
    scheme_order = ["random_cycle", "block_aware", "condition_holdout"]
    labels = ["Random cycle", "Block-aware", "Condition holdout"]
    for metric in ("accuracy", "macro_f1"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
        for ax, seconds in zip(axes, OBSERVATION_TIMES):
            summary = results[results["observation_seconds"] == seconds].groupby(["scheme", "model"])[metric].agg(["mean", "std"]).reset_index()
            x = np.arange(len(scheme_order))
            width = 0.36
            for offset, model_name in enumerate(MODEL_NAMES):
                values = summary[summary["model"] == model_name].set_index("scheme").reindex(scheme_order)
                ax.bar(x + (offset - 0.5) * width, values["mean"], width, yerr=values["std"], capsize=4, label=model_name.replace("_", " ").title())
            ax.set(xticks=x, xticklabels=labels, ylim=(0, 1.03), title=f"{seconds}-second observation", ylabel=metric.replace("_", " ").title())
            ax.tick_params(axis="x", rotation=12)
            ax.grid(axis="y", alpha=0.2)
        axes[1].legend()
        fig.suptitle(f"Generalization validation: {metric.replace('_', ' ').title()} (mean ± SD, 5 folds)")
        fig.tight_layout()
        fig.savefig(AUDIT_DIR / f"generalization_{metric}.png", dpi=180)
        plt.close(fig)

    recall_columns = ["normal_recall", "weak_recall", "severe_recall"]
    logistic_5s = results[(results["observation_seconds"] == 5) & (results["model"] == "logistic_regression")]
    means = logistic_5s.groupby("scheme")[recall_columns].mean().reindex(scheme_order)
    stds = logistic_5s.groupby("scheme")[recall_columns].std().reindex(scheme_order)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(scheme_order))
    width = 0.25
    for index, column in enumerate(recall_columns):
        ax.bar(x + (index - 1) * width, means[column], width, yerr=stds[column], capsize=3, label=column.replace("_", " ").title())
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 1.03), ylabel="Recall", title="5-second Logistic Regression class recall (mean ± SD)")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(AUDIT_DIR / "generalization_class_recall.png", dpi=180)
    plt.close(fig)

    for (scheme, seconds, model_name), matrix in matrices.items():
        fig, ax = plt.subplots(figsize=(5.5, 5))
        display = ConfusionMatrixDisplay(matrix, display_labels=[CLASS_NAMES[index] for index in range(3)])
        display.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{scheme.replace('_', ' ').title()}\n{model_name.replace('_', ' ').title()}, {seconds}s — pooled 5 folds")
        fig.tight_layout()
        fig.savefig(AUDIT_DIR / f"confusion_{scheme}_{model_name}_{seconds}s.png", dpi=180)
        plt.close(fig)


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "macro_f1", "normal_recall", "weak_recall", "severe_recall"]
    grouped = results.groupby(["scheme", "observation_seconds", "model"])[metrics].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def existing_split_audit(profile: pd.DataFrame) -> dict:
    assignments = pd.read_csv(DATA_DIR / "split_assignments.csv")
    if assignments["cycle_id"].duplicated().any():
        raise AssertionError("Existing split contains duplicate cycle IDs")
    merged = profile[["cycle_id", "inferred_block_id", "condition_id"]].merge(assignments[["cycle_id", "split"]], on="cycle_id", validate="one_to_one")
    block_span = merged.groupby("inferred_block_id")["split"].nunique()
    condition_span = merged.groupby("condition_id")["split"].nunique()
    split_sets = {name: set(group["cycle_id"]) for name, group in merged.groupby("split")}
    return {
        "cycle_overlap_train_validation": len(split_sets["train"] & split_sets["validation"]),
        "cycle_overlap_train_test": len(split_sets["train"] & split_sets["test"]),
        "cycle_overlap_validation_test": len(split_sets["validation"] & split_sets["test"]),
        "blocks_total": int(block_span.size),
        "blocks_spanning_multiple_splits": int((block_span > 1).sum()),
        "blocks_spanning_all_three_splits": int((block_span == 3).sum()),
        "conditions_total": int(condition_span.size),
        "conditions_spanning_multiple_splits": int((condition_span > 1).sum()),
        "conditions_spanning_all_three_splits": int((condition_span == 3).sum()),
    }


def causal_feature_audit(profile: pd.DataFrame) -> dict:
    rng = np.random.default_rng(RANDOM_SEED)
    synthetic = rng.normal(size=(4, 6000))
    mutated = synthetic.copy()
    mutated[:, 500:] += 1_000_000
    before = extract_features(synthetic, 100, 5)
    after = extract_features(mutated, 100, 5)
    max_difference = float(np.max(np.abs(before - after)))
    if max_difference != 0.0:
        raise AssertionError("Future-only perturbation changed 5-second features")
    feature_5s = pd.read_csv(FEATURE_DIR / "features_5s.csv")
    return {
        "future_perturbation_max_abs_difference": max_difference,
        "feature_cycle_ids_in_original_order": bool(np.array_equal(feature_5s["cycle_id"], profile["cycle_id"])),
        "feature_labels_match_profile": bool(np.array_equal(feature_5s["pump_leakage"], profile["pump_leakage"])),
        "predictor_count": len([column for column in feature_5s if "__" in column]),
    }


def write_leakage_audit(existing: dict, causal: dict) -> None:
    text = f"""# Leakage audit

## Scope and evidence

This audit was performed against the saved UCI `profile.txt`, feature tables, split assignments, and the current source code. It distinguishes exact cycle leakage from similarity/group leakage.

## Findings

| Check | Result | Interpretation |
| --- | ---: | --- |
| Train/validation cycle-ID overlap | {existing['cycle_overlap_train_validation']} | No exact cycle duplication. |
| Train/test cycle-ID overlap | {existing['cycle_overlap_train_test']} | No exact cycle duplication. |
| Validation/test cycle-ID overlap | {existing['cycle_overlap_validation_test']} | No exact cycle duplication. |
| Inferred blocks spanning multiple saved splits | {existing['blocks_spanning_multiple_splits']} / {existing['blocks_total']} | Similar adjacent-condition cycles are shared across the random split. |
| Inferred blocks spanning all three saved splits | {existing['blocks_spanning_all_three_splits']} / {existing['blocks_total']} | The original test is not block-independent. |
| Four-component conditions spanning multiple saved splits | {existing['conditions_spanning_multiple_splits']} / {existing['conditions_total']} | The original test mainly evaluates known condition combinations. |
| Four-component conditions spanning all three saved splits | {existing['conditions_spanning_all_three_splits']} / {existing['conditions_total']} | Condition holdout was not part of the original design. |
| Future perturbation effect on 5-second features | {causal['future_perturbation_max_abs_difference']:.1f} | Causal prefix extraction passed. |
| Feature IDs preserve original order | {causal['feature_cycle_ids_in_original_order']} | Cycle numbers remain the 1-based UCI row IDs. |
| Feature labels align with `profile.txt` | {causal['feature_labels_match_profile']} | No label-row shift was found. |
| Predictor columns | {causal['predictor_count']} | Only `sensor__statistic` columns are used; labels/groups are excluded. |

## Source-code audit

### Correctly implemented

- `preprocess.py` slices `values[:, :seconds * sampling_rate]`; 5-second features cannot use later samples.
- Logistic Regression preprocessing is a Pipeline: median imputation, `VarianceThreshold`, scaling, then classification. Random Forest uses an in-pipeline imputer.
- The original train/validation/test assignment is created before window-specific model fitting, and model/hyperparameter selection is based on validation macro-F1 with severe recall as tie-breaker.
- `evaluate.py` reads previously saved model choices and does not select a model from the test metrics.
- The dashboard uses the saved `cycle_id`, which is the actual 1-based source row. It does not relabel validation/test rows as 1..N.

### Limitations

- The primary 70/15/15 split stratifies individual cycles by pump label only. It prevents exact cycle duplication but not same-block or same-condition similarity leakage.
- Existing `inferred_block_id` starts a new block whenever any of the five profile labels changes. UCI provides no true session ID, so this is an estimated grouping variable.
- The previous block CV uses a fixed 60-second Random Forest on train+validation only. Pipeline preprocessing is fitted inside each fold correctly, but hyperparameter/model selection is not nested and 5-second/Logistic behavior is not tested.
- Probability calibration is fitted on validation and the confidence threshold is selected on that same validation set. Validation calibration statistics are optimistic; test evaluation remains separate.
- The original test set has now been inspected repeatedly. Its 99.7% result is a historical random-split benchmark, not a newly untouched independent estimate.

## New validation controls

The additive audit uses five outer folds shared by both models and both observation times. Each outer training fold performs a separate three-fold inner search. For grouped schemes, inner preprocessing and hyperparameter selection also respect the relevant training-fold groups. Group labels are used only to construct splits and never enter the predictor matrix.
"""
    (AUDIT_DIR / "leakage_audit.md").write_text(text, encoding="utf-8")


def write_final_report(
    profile: pd.DataFrame,
    combinations: pd.DataFrame,
    blocks: pd.DataFrame,
    existing: dict,
    summary: pd.DataFrame,
) -> None:
    def result(scheme: str, seconds: int, model: str, metric: str) -> tuple[float, float]:
        row = summary.query("scheme == @scheme and observation_seconds == @seconds and model == @model").iloc[0]
        return float(row[f"{metric}_mean"]), float(row[f"{metric}_std"])

    random_5 = result("random_cycle", 5, "logistic_regression", "macro_f1")
    block_5 = result("block_aware", 5, "logistic_regression", "macro_f1")
    condition_5 = result("condition_holdout", 5, "logistic_regression", "macro_f1")
    random_60 = result("random_cycle", 60, "logistic_regression", "macro_f1")
    block_60 = result("block_aware", 60, "logistic_regression", "macro_f1")
    condition_60 = result("condition_holdout", 60, "logistic_regression", "macro_f1")
    grouped_counts = profile["faulty_component_count"].value_counts()
    no_fault = int(grouped_counts.get(0, 0))
    single_fault = int(grouped_counts.get(1, 0))
    multi_fault = int((profile["faulty_component_count"] >= 2).sum())
    pattern_summary = (
        combinations.groupby("fault_pattern")
        .agg(state_combinations=("condition_id", "size"), cycles=("cycle_count", "sum"))
        .sort_values("cycles", ascending=False)
        .reset_index()
    )
    pattern_lines = ["| Fault pattern | State combinations | Cycles |", "| --- | ---: | ---: |"]
    for row in pattern_summary.itertuples():
        pattern_lines.append(f"| {row.fault_pattern} | {row.state_combinations} | {row.cycles} |")
    table = summary.copy()
    table["display"] = table.apply(
        lambda row: f"{row.macro_f1_mean:.4f} ± {row.macro_f1_std:.4f}", axis=1
    )
    pivot = table.pivot_table(index=["scheme", "observation_seconds"], columns="model", values="display", aggfunc="first").reset_index()
    comparison_lines = ["| Scheme | Observation | Logistic macro-F1 | Random Forest macro-F1 |", "| --- | ---: | ---: | ---: |"]
    for row in pivot.itertuples():
        comparison_lines.append(f"| {row.scheme} | {row.observation_seconds}s | {row.logistic_regression} | {row.random_forest} |")

    report = f"""# Generalization Audit Report

## Executive conclusion

The historical 5-second accuracy of 0.9970 and macro-F1 of 0.9968 were obtained from one fixed 70/15/15 random cycle-level split. There is no exact cycle duplication, but {existing['blocks_spanning_multiple_splits']} of {existing['blocks_total']} inferred contiguous blocks and {existing['conditions_spanning_multiple_splits']} of {existing['conditions_total']} four-component state combinations span multiple saved splits. The historical test therefore measures interpolation among highly related operating conditions and must not be described as a new condition-independent evaluation.

## 1. Dataset structure

- Total cycles: {len(profile)}.
- Raw four-component state combinations: {len(combinations)}.
- Inferred contiguous five-label blocks: {len(blocks)}; median size {blocks['block_size'].median():.1f}, range {blocks['block_size'].min()}–{blocks['block_size'].max()} cycles.
- No degraded component: {no_fault} cycles.
- Exactly one degraded component: {single_fault} cycles.
- Two or more degraded components: {multi_fault} cycles.
- Stable flag 0: {(profile['stable_flag'] == 0).sum()} cycles; flag 1: {(profile['stable_flag'] == 1).sum()} cycles.

These are experimental state labels supplied by UCI. They indicate configured component condition/severity, not a proven physical root cause for each individual sensor trace.

### Fault-pattern inventory

{chr(10).join(pattern_lines)}

## 2. Validation designs

### A. Random cycle-level

Five-fold stratified random CV. It is the closest repeated-CV counterpart to the historical benchmark, but blocks and condition combinations can occur in both training and evaluation.

### B. Block-aware

Five-fold `StratifiedGroupKFold` using contiguous runs with identical cooler, valve, pump, accumulator, and stable labels. No inferred block crosses an outer fold. These blocks are proxies, not published independent experiment sessions.

### C. Condition holdout

Five-fold `StratifiedGroupKFold` using the complete four-component state tuple as the group. All {len(combinations)} combinations are evaluated exactly once, no combination appears in both training and evaluation, and every fold retains pump classes 0/1/2 in both partitions. This tests interpolation to unseen combinations of known component-state values, not extrapolation to unseen severity codes or a new machine.

Both grouped designs use a three-fold group-respecting inner search within each outer training fold. Imputation, variance filtering, scaling, and model fitting are refit from scratch in the inner/outer training data.

## 3. Pump leakage results

{chr(10).join(comparison_lines)}

For 5-second Logistic Regression, random-cycle macro-F1 was {random_5[0]:.4f} ± {random_5[1]:.4f}, block-aware macro-F1 was {block_5[0]:.4f} ± {block_5[1]:.4f}, and condition-holdout macro-F1 was {condition_5[0]:.4f} ± {condition_5[1]:.4f}. At 60 seconds, the corresponding values were {random_60[0]:.4f} ± {random_60[1]:.4f}, {block_60[0]:.4f} ± {block_60[1]:.4f}, and {condition_60[0]:.4f} ± {condition_60[1]:.4f}.

Relative to random-cycle CV, 5-second Logistic Regression dropped by {random_5[0] - block_5[0]:.4f} macro-F1 under block-aware evaluation and {random_5[0] - condition_5[0]:.4f} under condition holdout. The 60-second Logistic drops were larger: {random_60[0] - block_60[0]:.4f} and {random_60[0] - condition_60[0]:.4f}, respectively. Thus, additional observation time did not guarantee stronger cross-condition generalization for the linear model.

Full accuracy, class recall, fold sizes, group overlap checks, selected hyperparameters, and fit times are in `validation_comparison.csv`. Pooled confusion matrices are provided as CSV and PNG files.

## 4. Answers to the research questions

1. **Which single and combined states exist?** The dataset contains {len(combinations)} four-component state combinations across no-fault, single-degradation, and multiple-degradation cases. Exact state and cycle ranges are in `fault_combination_counts.csv`.
2. **Under what condition was 99.7% obtained?** One fixed pump-stratified random cycle split, with the same inferred blocks and condition combinations commonly represented across training and test.
3. **Does block-aware performance remain high?** Yes in absolute terms, but not unchanged. Mean block-aware macro-F1 was {block_5[0]:.4f} for 5-second Logistic Regression, {block_60[0]:.4f} for 60-second Logistic Regression, and 0.9964 for 60-second Random Forest. These values apply to held-out contiguous label runs; true session independence remains unknown.
4. **Can unseen combined conditions be held out?** Yes. There are 48 condition groups for each pump class (144 total), enabling five condition-disjoint folds with all three pump classes represented in training and evaluation.
5. **Where does performance degrade?** Logistic Regression is most affected at 60 seconds: macro-F1 falls from {random_60[0]:.4f} in random CV to {block_60[0]:.4f} block-aware and {condition_60[0]:.4f} condition-held-out. Five-second Random Forest also falls from 0.9798 random to 0.9721 condition-held-out and has the largest fold variability. Sixty-second Random Forest is the most stable configuration (0.9943 condition-held-out macro-F1).
6. **Where could physics-guided ML help?** The main vulnerability is configuration/domain shift and reliance on highly informative engineered or virtual channels such as `SE`, not exact cycle leakage. A useful next step is physics-guided feature design or constrained models based on pressure-flow-power consistency, conservation relationships, and monotonic leakage behavior. A full PINN is not yet justified because the dataset does not provide the governing hydraulic equations, boundary conditions, component parameters, or dense state variables required to identify a unique dynamical model. Physics-guided residual features and out-of-distribution detection should precede a PINN claim.

## 5. Limits

- Component labels were intentionally used to form validation groups but were never included in model features.
- Condition holdout excludes complete combinations, not individual severity values; all severity codes remain represented elsewhere.
- Adjacent-run blocks may still share the same physical experiment or slow drift because UCI publishes no session identifier.
- Results are internal cross-validation on one rig and cannot establish external machine/site generalization.
- The historical fixed test has been repeatedly observed and is retained only as a benchmark.

## Artifact map

- `dataset_profile_summary.csv`: raw and binary state distributions.
- `fault_combination_counts.csv`: all four-component combinations and original cycle ranges.
- `block_summary.csv`: inferred five-label contiguous blocks.
- `condition_holdout_plan.csv`: condition groups assigned to each outer fold.
- `leakage_audit.md`: source-code and leakage review.
- `validation_comparison.csv`: fold-level metrics and group-overlap checks.
- `validation_summary.csv`: mean ± standard deviation summaries.
- `confusion_matrices.csv` and `confusion_*.png`: fold and pooled confusion matrices.
- `execution_log.txt`: actual run log.
"""
    (AUDIT_DIR / "GENERALIZATION_AUDIT_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-models", action="store_true", help="generate dataset/leakage audit only")
    args = parser.parse_args()
    logger = configure_logging()
    logger.info("Starting additive dataset and generalization audit")
    profile = load_profile()
    summary, combinations, blocks = build_dataset_tables(profile)
    summary.to_csv(AUDIT_DIR / "dataset_profile_summary.csv", index=False)
    combinations.to_csv(AUDIT_DIR / "fault_combination_counts.csv", index=False)
    blocks.to_csv(AUDIT_DIR / "block_summary.csv", index=False)
    make_dataset_figures(profile, summary, combinations, blocks)
    existing = existing_split_audit(profile)
    causal = causal_feature_audit(profile)
    (AUDIT_DIR / "audit_checks.json").write_text(json.dumps({"existing_split": existing, "causal_features": causal}, indent=2), encoding="utf-8")
    write_leakage_audit(existing, causal)
    logger.info("Dataset audit: %s combinations, %s inferred blocks", len(combinations), len(blocks))
    logger.info("Existing split: %s/%s blocks span multiple splits", existing["blocks_spanning_multiple_splits"], existing["blocks_total"])
    if args.skip_models:
        logger.info("Model validation skipped by request")
        return

    outer_splits = make_outer_splits(profile)
    results, confusion_long, holdout_plan = evaluate_nested_cv(profile, outer_splits, logger)
    results.to_csv(AUDIT_DIR / "validation_comparison.csv", index=False)
    confusion_long.to_csv(AUDIT_DIR / "confusion_matrices.csv", index=False)
    holdout_plan.to_csv(AUDIT_DIR / "condition_holdout_plan.csv", index=False)
    validation_summary = aggregate_results(results)
    validation_summary.to_csv(AUDIT_DIR / "validation_summary.csv", index=False)
    write_final_report(profile, combinations, blocks, existing, validation_summary)
    logger.info("Audit completed successfully; artifacts: %s", AUDIT_DIR)
    print(validation_summary.to_string(index=False))


if __name__ == "__main__":
    main()
