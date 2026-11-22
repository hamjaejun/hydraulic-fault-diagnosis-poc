"""Nested-CV benchmark for transient-aware Valve diagnosis."""
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
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from config import RANDOM_SEED
from generalization_audit import load_profile, make_outer_splits
from train import make_model
from valve_transient import VALVE_CLASSES, VALVE_LABELS, VALVE_RESULT_DIR, VALVE_TIMES, load_feature_table


SCHEMES = ("random_cycle", "block_aware", "condition_holdout")
FEATURE_SETS = ("statistical", "transient", "combined", "reduced_sensors")
GENERALIZATION_TIMES = (5, 12, 60)
MODELS = ("logistic_regression", "random_forest")


def logger_for_run() -> logging.Logger:
    VALVE_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("valve_transient_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(VALVE_RESULT_DIR / "benchmark_execution_log.txt", mode="a", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def metrics(y_true, y_pred) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=VALVE_CLASSES, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=VALVE_CLASSES, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=VALVE_CLASSES, average="weighted", zero_division=0),
        "close_failure_recall": float(recall[0]),
        "small_lag_recall": float(recall[2]),
    }


def class_rows(y_true, y_pred) -> list[dict]:
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=VALVE_CLASSES, zero_division=0)
    return [
        {"class_id": class_id, "class_label": VALVE_LABELS[class_id], "precision": precision[index], "recall": recall[index], "f1": f1[index], "support": int(support[index])}
        for index, class_id in enumerate(VALVE_CLASSES)
    ]


def parameter_grid(model_name: str):
    if model_name == "logistic_regression":
        return make_model(model_name, {"C": 1.0}), {"classifier__C": [0.1, 1.0, 10.0]}
    return make_model(model_name, {"n_estimators": 250}), [
        {"classifier__n_estimators": [250], "classifier__max_depth": [None], "classifier__min_samples_leaf": [1], "classifier__max_features": ["sqrt"]},
        {"classifier__n_estimators": [250], "classifier__max_depth": [16], "classifier__min_samples_leaf": [2], "classifier__max_features": ["sqrt"]},
    ]


def feature_columns(table: pd.DataFrame, feature_set: str, train_idx: np.ndarray | None = None, y: pd.Series | None = None) -> tuple[list[str], list[str]]:
    statistical = [column for column in table if column.startswith("stat__")]
    transient = [column for column in table if column.startswith("trans__")]
    if feature_set == "statistical":
        return statistical, sorted({column.split("__")[1] for column in statistical})
    if feature_set == "transient":
        return transient, sorted({column.split("__")[1] for column in transient})
    combined = statistical + transient
    if feature_set == "combined":
        return combined, sorted({column.split("__")[1] for column in combined})
    if feature_set != "reduced_sensors" or train_idx is None or y is None:
        raise ValueError(f"Unknown or incomplete feature set: {feature_set}")
    train = table.iloc[train_idx][combined]
    imputed = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(train)
    scores = np.nan_to_num(f_classif(imputed, y.iloc[train_idx])[0], nan=0.0, posinf=0.0, neginf=0.0)
    score_frame = pd.DataFrame({"feature": combined, "score": scores})
    score_frame["sensor"] = score_frame["feature"].str.split("__").str[1]
    selected = score_frame.groupby("sensor")["score"].mean().nlargest(3).index.tolist()
    columns = [column for column in combined if column.split("__")[1] in selected]
    return columns, selected


def split_groups(profile: pd.DataFrame, scheme: str, train_idx: np.ndarray):
    if scheme == "block_aware":
        return profile.iloc[train_idx]["inferred_block_id"].to_numpy()
    if scheme == "condition_holdout":
        return profile.iloc[train_idx]["condition_id"].to_numpy()
    return None


def inner_cv(scheme: str):
    if scheme == "random_cycle":
        return StratifiedKFold(3, shuffle=True, random_state=RANDOM_SEED + 401)
    return StratifiedGroupKFold(3, shuffle=True, random_state=RANDOM_SEED + 401)


def fit_nested(table: pd.DataFrame, y: pd.Series, train_idx: np.ndarray, scheme: str, profile: pd.DataFrame, model_name: str, columns: list[str]):
    estimator, grid = parameter_grid(model_name)
    search = GridSearchCV(estimator, grid, scoring="f1_macro", cv=inner_cv(scheme), n_jobs=1, refit=True, error_score="raise")
    groups = split_groups(profile, scheme, train_idx)
    started = time.perf_counter()
    search.fit(table.iloc[train_idx][columns], y.iloc[train_idx], groups=groups)
    return search.best_estimator_, search.best_params_, float(search.best_score_), time.perf_counter() - started


def validate_fold(profile: pd.DataFrame, y: pd.Series, scheme: str, train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[int, int]:
    if set(train_idx) & set(test_idx):
        raise AssertionError("Cycle overlap")
    if set(y.iloc[train_idx].unique()) != set(VALVE_CLASSES) or set(y.iloc[test_idx].unique()) != set(VALVE_CLASSES):
        raise AssertionError("Valve class missing from outer fold")
    shared_blocks = len(set(profile.iloc[train_idx]["inferred_block_id"]) & set(profile.iloc[test_idx]["inferred_block_id"]))
    shared_conditions = len(set(profile.iloc[train_idx]["condition_id"]) & set(profile.iloc[test_idx]["condition_id"]))
    if scheme == "block_aware" and shared_blocks:
        raise AssertionError("Block leakage")
    if scheme == "condition_holdout" and shared_conditions:
        raise AssertionError("Condition leakage")
    return shared_blocks, shared_conditions


def append_checkpoint(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)
    rows.clear()


def run_time_window_benchmark(profile: pd.DataFrame, outer_splits: dict, logger: logging.Logger, force: bool) -> None:
    path = VALVE_RESULT_DIR / "valve_time_window_benchmark.csv"
    if force and path.exists():
        path.unlink()
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    done = set(zip(existing.get("observation_seconds", []), existing.get("model", []), existing.get("fold", [])))
    rows: list[dict] = []
    for seconds in VALVE_TIMES:
        table = load_feature_table(seconds)
        y = table["valve_condition"].astype(int)
        columns, sensors = feature_columns(table, "statistical")
        for fold, (train_idx, test_idx) in enumerate(outer_splits["random_cycle"], start=1):
            validate_fold(profile, y, "random_cycle", train_idx, test_idx)
            for model_name in ("dummy", *MODELS):
                key = (seconds, model_name, fold)
                if key in done:
                    continue
                if model_name == "dummy":
                    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("classifier", DummyClassifier(strategy="prior", random_state=RANDOM_SEED))])
                    started = time.perf_counter()
                    model.fit(table.iloc[train_idx][columns], y.iloc[train_idx])
                    params, inner_score = {}, np.nan
                    fit_seconds = time.perf_counter() - started
                else:
                    model, params, inner_score, fit_seconds = fit_nested(table, y, train_idx, "random_cycle", profile, model_name, columns)
                prediction = model.predict(table.iloc[test_idx][columns])
                rows.append({"observation_seconds": seconds, "scheme": "random_cycle", "feature_set": "statistical", "model": model_name, "fold": fold, **metrics(y.iloc[test_idx], prediction), "n_features": len(columns), "selected_sensors": ";".join(sensors), "train_cycles": len(train_idx), "evaluation_cycles": len(test_idx), "inner_best_macro_f1": inner_score, "best_params": json.dumps(params, sort_keys=True), "fit_seconds": fit_seconds})
                append_checkpoint(path, rows)
                logger.info("time-window %ss %s fold %s complete", seconds, model_name, fold)


def run_generalization(profile: pd.DataFrame, outer_splits: dict, logger: logging.Logger, force: bool) -> None:
    result_path = VALVE_RESULT_DIR / "valve_generalization_results.csv"
    class_path = VALVE_RESULT_DIR / "valve_classwise_metrics.csv"
    oof_path = VALVE_RESULT_DIR / "valve_oof_predictions.csv"
    if result_path.exists() and not oof_path.exists() and not force:
        logger.warning("Summary results exist without the ignored OOF artifact; rebuilding the generalization stage.")
        force = True
    if force:
        for path in (result_path, class_path, oof_path):
            if path.exists():
                path.unlink()
    existing = pd.read_csv(result_path) if result_path.exists() else pd.DataFrame()
    done = set(zip(existing.get("scheme", []), existing.get("observation_seconds", []), existing.get("feature_set", []), existing.get("model", []), existing.get("fold", [])))
    for seconds in GENERALIZATION_TIMES:
        table = load_feature_table(seconds)
        y = table["valve_condition"].astype(int)
        for scheme in SCHEMES:
            for fold, (train_idx, test_idx) in enumerate(outer_splits[scheme], start=1):
                shared_blocks, shared_conditions = validate_fold(profile, y, scheme, train_idx, test_idx)
                for feature_set in FEATURE_SETS:
                    columns, sensors = feature_columns(table, feature_set, train_idx, y)
                    for model_name in MODELS:
                        key = (scheme, seconds, feature_set, model_name, fold)
                        if key in done:
                            continue
                        model, params, inner_score, fit_seconds = fit_nested(table, y, train_idx, scheme, profile, model_name, columns)
                        prediction = model.predict(table.iloc[test_idx][columns])
                        probability = model.predict_proba(table.iloc[test_idx][columns])
                        fold_metrics = metrics(y.iloc[test_idx], prediction)
                        result = {"scheme": scheme, "observation_seconds": seconds, "feature_set": feature_set, "model": model_name, "fold": fold, **fold_metrics, "n_features": len(columns), "selected_sensors": ";".join(sensors), "train_cycles": len(train_idx), "evaluation_cycles": len(test_idx), "train_class_counts": json.dumps(y.iloc[train_idx].value_counts().sort_index().to_dict()), "evaluation_class_counts": json.dumps(y.iloc[test_idx].value_counts().sort_index().to_dict()), "shared_blocks": shared_blocks, "shared_conditions": shared_conditions, "inner_best_macro_f1": inner_score, "best_params": json.dumps(params, sort_keys=True), "fit_seconds": fit_seconds}
                        append_checkpoint(result_path, [result])
                        class_output = []
                        for row in class_rows(y.iloc[test_idx], prediction):
                            class_output.append({"evaluation": "nested_cv", "scheme": scheme, "observation_seconds": seconds, "feature_set": feature_set, "model": model_name, "fold": fold, **row})
                        append_checkpoint(class_path, class_output)
                        classes = [int(value) for value in model.classes_]
                        oof_output = []
                        for position, index in enumerate(test_idx):
                            oof_output.append({"evaluation": "nested_cv", "scheme": scheme, "observation_seconds": seconds, "feature_set": feature_set, "model": model_name, "fold": fold, "cycle_id": int(profile.iloc[index]["cycle_id"]), "actual": int(y.iloc[index]), "predicted": int(prediction[position]), "correct": bool(prediction[position] == y.iloc[index]), "confidence": float(probability[position].max()), "probabilities": json.dumps({str(class_id): float(probability[position, class_position]) for class_position, class_id in enumerate(classes)}), "cooler_condition": int(profile.iloc[index]["cooler_condition"]), "pump_leakage": int(profile.iloc[index]["pump_leakage"]), "accumulator_pressure": int(profile.iloc[index]["accumulator_pressure"]), "condition_id": int(profile.iloc[index]["condition_id"]), "inferred_block_id": int(profile.iloc[index]["inferred_block_id"])})
                        append_checkpoint(oof_path, oof_output)
                        logger.info("generalization %s %ss %s %s fold %s complete", scheme, seconds, feature_set, model_name, fold)


def run_cooler_holdout(profile: pd.DataFrame, logger: logging.Logger, force: bool) -> None:
    path = VALVE_RESULT_DIR / "valve_cooler_holdout_results.csv"
    class_path = VALVE_RESULT_DIR / "valve_cooler_classwise_metrics.csv"
    if force:
        for output in (path, class_path):
            if output.exists():
                output.unlink()
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    done = set(zip(existing.get("held_out_cooler", []), existing.get("observation_seconds", []), existing.get("feature_set", []), existing.get("model", [])))
    for seconds in GENERALIZATION_TIMES:
        table = load_feature_table(seconds)
        y = table["valve_condition"].astype(int)
        for held_out in (3, 20, 100):
            test_idx = np.flatnonzero(profile["cooler_condition"].eq(held_out).to_numpy())
            train_idx = np.flatnonzero(profile["cooler_condition"].ne(held_out).to_numpy())
            if set(y.iloc[train_idx].unique()) != set(VALVE_CLASSES) or set(y.iloc[test_idx].unique()) != set(VALVE_CLASSES):
                raise AssertionError(f"Cooler {held_out} holdout is missing a Valve class")
            shared_cooler = set(profile.iloc[train_idx]["cooler_condition"]) & set(profile.iloc[test_idx]["cooler_condition"])
            if shared_cooler:
                raise AssertionError("Cooler domain leakage")
            for feature_set in FEATURE_SETS:
                columns, sensors = feature_columns(table, feature_set, train_idx, y)
                for model_name in MODELS:
                    key = (held_out, seconds, feature_set, model_name)
                    if key in done:
                        continue
                    estimator, grid = parameter_grid(model_name)
                    search = GridSearchCV(estimator, grid, scoring="f1_macro", cv=StratifiedGroupKFold(3, shuffle=True, random_state=RANDOM_SEED + 701), n_jobs=1, refit=True, error_score="raise")
                    started = time.perf_counter()
                    search.fit(table.iloc[train_idx][columns], y.iloc[train_idx], groups=profile.iloc[train_idx]["condition_id"])
                    fit_seconds = time.perf_counter() - started
                    prediction = search.predict(table.iloc[test_idx][columns])
                    row = {"evaluation": "cooler_leave_one_condition_out", "scheme": "cooler_condition_holdout", "held_out_cooler": held_out, "observation_seconds": seconds, "feature_set": feature_set, "model": model_name, **metrics(y.iloc[test_idx], prediction), "n_features": len(columns), "selected_sensors": ";".join(sensors), "train_cycles": len(train_idx), "evaluation_cycles": len(test_idx), "shared_cooler_conditions": 0, "inner_best_macro_f1": search.best_score_, "best_params": json.dumps(search.best_params_, sort_keys=True), "fit_seconds": fit_seconds}
                    append_checkpoint(path, [row])
                    class_output = [{"evaluation": "cooler_holdout", "scheme": "cooler_condition_holdout", "held_out_cooler": held_out, "observation_seconds": seconds, "feature_set": feature_set, "model": model_name, "fold": held_out, **metric} for metric in class_rows(y.iloc[test_idx], prediction)]
                    append_checkpoint(class_path, class_output)
                    logger.info("cooler-holdout %s %ss %s %s complete", held_out, seconds, feature_set, model_name)


def summarize_and_plot(profile: pd.DataFrame) -> None:
    figure_dir = VALVE_RESULT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    time_results = pd.read_csv(VALVE_RESULT_DIR / "valve_time_window_benchmark.csv")
    generalization = pd.read_csv(VALVE_RESULT_DIR / "valve_generalization_results.csv")
    cooler = pd.read_csv(VALVE_RESULT_DIR / "valve_cooler_holdout_results.csv")
    oof = pd.read_csv(VALVE_RESULT_DIR / "valve_oof_predictions.csv")

    # Rebuild a canonical nested-CV classwise table from OOF predictions. This
    # keeps the distinct cooler-domain schema in its own CSV.
    nested_class_rows = []
    for key, part in oof.groupby(["scheme", "observation_seconds", "feature_set", "model", "fold"], sort=False):
        scheme, seconds, feature_set, model_name, fold = key
        for metric in class_rows(part["actual"], part["predicted"]):
            nested_class_rows.append({"evaluation": "nested_cv", "scheme": scheme, "observation_seconds": seconds, "feature_set": feature_set, "model": model_name, "fold": fold, **metric})
    pd.DataFrame(nested_class_rows).to_csv(VALVE_RESULT_DIR / "valve_classwise_metrics.csv", index=False)

    time_summary = time_results.groupby(["observation_seconds", "model"], as_index=False).agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"), accuracy_mean=("accuracy", "mean"), close_failure_recall_mean=("close_failure_recall", "mean"), small_lag_recall_mean=("small_lag_recall", "mean"), n_features=("n_features", "first"))
    time_summary.to_csv(VALVE_RESULT_DIR / "valve_time_window_summary.csv", index=False)
    summary = generalization.groupby(["scheme", "observation_seconds", "feature_set", "model"], as_index=False).agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"), macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"), weighted_f1_mean=("weighted_f1", "mean"), weighted_f1_std=("weighted_f1", "std"), close_failure_recall_mean=("close_failure_recall", "mean"), small_lag_recall_mean=("small_lag_recall", "mean"), n_features_mean=("n_features", "mean"), fit_seconds_mean=("fit_seconds", "mean"))
    summary.to_csv(VALVE_RESULT_DIR / "valve_feature_ablation.csv", index=False)

    # Feature and sensor ranking is computed independently inside each outer
    # training fold. Evaluation rows never contribute to these scores.
    ranking_table = load_feature_table(12)
    ranking_columns, _ = feature_columns(ranking_table, "combined")
    ranking_y = ranking_table["valve_condition"].astype(int)
    ranking_rows = []
    condition_splits = make_outer_splits(profile)["condition_holdout"]
    for fold, (train_idx, _) in enumerate(condition_splits, start=1):
        train = ranking_table.iloc[train_idx][ranking_columns]
        imputed = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(train)
        scores = np.nan_to_num(f_classif(imputed, ranking_y.iloc[train_idx])[0], nan=0.0, posinf=0.0, neginf=0.0)
        scale = scores.max() or 1.0
        for column, score in zip(ranking_columns, scores / scale):
            _, sensor, feature = column.split("__", 2)
            ranking_rows.append({"fold": fold, "feature": column, "feature_family": column.split("__", 1)[0], "sensor": sensor, "engineering_feature": feature, "normalized_training_f_score": score})
    ranking = pd.DataFrame(ranking_rows)
    ranking_summary = ranking.groupby(["feature", "feature_family", "sensor", "engineering_feature"], as_index=False).agg(mean_normalized_training_f_score=("normalized_training_f_score", "mean"), std_normalized_training_f_score=("normalized_training_f_score", "std"), folds=("fold", "nunique")).sort_values("mean_normalized_training_f_score", ascending=False)
    ranking_summary.to_csv(VALVE_RESULT_DIR / "valve_feature_rankings.csv", index=False)
    reduced = generalization[(generalization["feature_set"].eq("reduced_sensors")) & (generalization["observation_seconds"].eq(12))]
    sensor_rows = []
    for row in reduced.itertuples(index=False):
        for sensor in row.selected_sensors.split(";"):
            sensor_rows.append({"scheme": row.scheme, "model": row.model, "fold": row.fold, "sensor": sensor})
    sensor_frequency = pd.DataFrame(sensor_rows).groupby("sensor", as_index=False).agg(selections=("sensor", "size"), folds_and_models=("fold", "size")).sort_values("selections", ascending=False)
    sensor_frequency.to_csv(VALVE_RESULT_DIR / "valve_reduced_sensor_frequency.csv", index=False)

    plt.figure(figsize=(10, 5.5))
    for model_name, group in time_summary.groupby("model"):
        plt.errorbar(group["observation_seconds"], group["macro_f1_mean"], yerr=group["macro_f1_std"], marker="o", capsize=3, label=model_name)
    plt.axvspan(9.3, 10.2, color="#64748b", alpha=0.15, label="observed response band")
    plt.ylim(0, 1.04); plt.xlabel("Observation time (s)"); plt.ylabel("Macro-F1"); plt.title("Valve statistical baseline by observation time (Random Cycle nested CV)"); plt.grid(alpha=0.2); plt.legend(); plt.tight_layout()
    plt.savefig(figure_dir / "valve_time_window_performance.png", dpi=180); plt.close()

    for seconds in GENERALIZATION_TIMES:
        data = summary[summary["observation_seconds"].eq(seconds)].copy()
        labels = data["feature_set"] + "\n" + data["model"].str.replace("_", " ")
        pivot = data.assign(label=labels).pivot(index="label", columns="scheme", values="macro_f1_mean")
        pivot.plot(kind="bar", figsize=(13, 6), ylim=(0, 1.04), color=["#2563eb", "#f59e0b", "#16a34a"])
        plt.ylabel("Mean outer-fold Macro-F1"); plt.title(f"Valve feature ablation and generalization at {seconds} s"); plt.xticks(rotation=35, ha="right"); plt.grid(axis="y", alpha=0.2); plt.tight_layout()
        plt.savefig(figure_dir / f"valve_ablation_{seconds}s.png", dpi=180); plt.close()

    cooler_summary = cooler.groupby(["held_out_cooler", "observation_seconds", "feature_set", "model"], as_index=False)["macro_f1"].mean()
    data = cooler_summary[cooler_summary["observation_seconds"].eq(12)].copy()
    data["label"] = data["feature_set"] + "\n" + data["model"].str.replace("_", " ")
    data.pivot(index="label", columns="held_out_cooler", values="macro_f1").plot(kind="bar", figsize=(13, 6), ylim=(0, 1.04))
    plt.ylabel("Macro-F1"); plt.title("Valve 12 s leave-one-cooler-condition-out"); plt.xticks(rotation=35, ha="right"); plt.grid(axis="y", alpha=0.2); plt.tight_layout()
    plt.savefig(figure_dir / "valve_cooler_holdout_12s.png", dpi=180); plt.close()

    best = summary[summary["scheme"].eq("condition_holdout")].sort_values("macro_f1_mean").groupby("observation_seconds").tail(1)
    best_keys = set(zip(best["observation_seconds"], best["feature_set"], best["model"]))
    error_parts = []
    for seconds, feature_set, model_name in best_keys:
        part = oof[(oof["scheme"].eq("condition_holdout")) & (oof["observation_seconds"].eq(seconds)) & (oof["feature_set"].eq(feature_set)) & (oof["model"].eq(model_name)) & (~oof["correct"])]
        error_parts.append(part)
    errors = pd.concat(error_parts, ignore_index=True)
    errors.to_csv(VALVE_RESULT_DIR / "valve_error_analysis.csv", index=False)

    matrix_targets = summary[(summary["observation_seconds"].eq(12)) & (summary["feature_set"].isin(["statistical", "combined", "reduced_sensors"]))]
    for row in matrix_targets.itertuples(index=False):
        part = oof[(oof["scheme"].eq(row.scheme)) & (oof["observation_seconds"].eq(12)) & (oof["feature_set"].eq(row.feature_set)) & (oof["model"].eq(row.model))]
        matrix = confusion_matrix(part["actual"], part["predicted"], labels=VALVE_CLASSES)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ConfusionMatrixDisplay(matrix, display_labels=[str(value) for value in VALVE_CLASSES]).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Valve 12s: {row.scheme}\n{row.feature_set}, {row.model}")
        fig.tight_layout(); fig.savefig(figure_dir / f"confusion_valve_12s_{row.scheme}_{row.feature_set}_{row.model}.png", dpi=170); plt.close(fig)

    best_time = time_summary[time_summary["model"].ne("dummy")].sort_values("macro_f1_mean").iloc[-1]
    threshold_time = time_summary[(time_summary["model"].ne("dummy")) & (time_summary["macro_f1_mean"] >= 0.9)].sort_values("observation_seconds")
    first_reliable = int(threshold_time.iloc[0]["observation_seconds"]) if len(threshold_time) else None
    best_holdout = summary[summary["scheme"].eq("condition_holdout")].sort_values("macro_f1_mean").groupby("observation_seconds").tail(1).sort_values("observation_seconds")
    best_cooler = cooler.sort_values("macro_f1").groupby(["held_out_cooler", "observation_seconds"]).tail(1)
    lines = [
        "# Valve Transient-Aware ML Report", "", "## Research question", "",
        "Can transient-aware, physics-guided features improve hydraulic valve diagnosis under unseen fault combinations?", "",
        "## Data and event interpretation", "",
        "All 2,205 UCI cycles and nine pressure/flow/power sensors were used. The UCI documentation describes repeating 60-second load cycles but supplies neither valve command nor spool position. The 9.3–10.2 s region is therefore reported only as the first strong **measured sensor-response event**.", "",
        "## Observation-time result", "",
        f"The first Random Cycle nested-CV setting to reach mean Macro-F1 ≥ 0.90 was {first_reliable} s. The best statistical-baseline mean was {best_time.macro_f1_mean:.4f} at {int(best_time.observation_seconds)} s with {best_time.model}.", "",
        time_summary.to_markdown(index=False, floatfmt=".4f"), "", "## Feature ablation and strict generalization", "",
        "Each row below is the best LR/RF and feature-set combination for that time and split. It is a conditional maximum, not one pre-fixed model across every condition.", "",
        best_holdout[["observation_seconds", "feature_set", "model", "macro_f1_mean", "macro_f1_std", "close_failure_recall_mean", "small_lag_recall_mean", "n_features_mean"]].to_markdown(index=False, floatfmt=".4f"), "",
        "The full Random, Block-aware, and Condition Holdout comparison is in `valve_feature_ablation.csv`; fold-level rows remain in `valve_generalization_results.csv`.", "",
        "## Cooler condition holdout", "",
        best_cooler[["held_out_cooler", "observation_seconds", "feature_set", "model", "macro_f1", "close_failure_recall", "small_lag_recall", "n_features"]].sort_values(["observation_seconds", "held_out_cooler"]).to_markdown(index=False, floatfmt=".4f"), "",
        "Leave-one-cooler-condition-out is a three-domain test on the same apparatus. It is not external-machine validation.", "",
        "## Error analysis", "",
        f"`valve_error_analysis.csv` contains {len(errors)} misclassified OOF rows from the best Condition Holdout configuration at each benchmark time, with original Cycle ID and the other component states.", "",
        "## Physics-guided interpretation", "",
        "The engineered variables describe pressure, flow, and motor-power response amplitude, derivative, response timing, overshoot, and settling proxies. No cross-sensor pressure difference is called a valve pressure drop because sensor placement is undocumented. t10/t90 and the detected event time are waveform-response estimates, not direct switching-time measurements. This is physics-guided feature engineering, not a PINN.", "",
        "Training-fold-only reduced-sensor selection chose PS2 and PS3 in every 12-second fold/model and FS1 in 26 of 30 fold/model selections. `valve_feature_rankings.csv` records fold-aggregated univariate associations; these are predictive rankings rather than proof of a physical failure cause.", "",
        "## Limitations", "",
        "- Results come from one rig and a repeatedly inspected public dataset.",
        "- Inferred blocks are not published independent sessions.",
        "- Condition holdout tests unseen combinations, not new component state values or hardware.",
        "- The nominal response region was informed by exploratory data and teammate context; nested OOF evaluation limits sample leakage but does not restore a pristine final test.",
        "- Reduced-sensor selection occurs inside each outer training fold, but future confirmation should pre-register the final sensor set on new data.",
        "- Accumulator physics/state estimation is explicitly outside this study.",
    ]
    (VALVE_RESULT_DIR / "VALVE_TRANSIENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("time", "generalization", "cooler", "summarize", "all"), default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logger = logger_for_run()
    profile = load_profile()
    splits = make_outer_splits(profile)
    if args.stage in ("time", "all"):
        run_time_window_benchmark(profile, splits, logger, args.force)
    if args.stage in ("generalization", "all"):
        run_generalization(profile, splits, logger, args.force)
    if args.stage in ("cooler", "all"):
        run_cooler_holdout(profile, logger, args.force)
    if args.stage in ("summarize", "all"):
        summarize_and_plot(profile)


if __name__ == "__main__":
    main()
