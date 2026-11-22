"""Integrity, causality, split, and artifact checks for the Valve study."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import RAW_DIR, SENSORS
from valve_transient import (
    TRANSIENT_SENSORS,
    VALVE_FEATURE_DIR,
    VALVE_MODEL_DIR,
    VALVE_RESULT_DIR,
    VALVE_TIMES,
    load_feature_table,
    load_profile,
    predict_cycle,
    to_10_hz,
    transient_features,
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def read(name: str) -> pd.DataFrame:
    path = VALVE_RESULT_DIR / name
    check(path.exists() and path.stat().st_size > 0, f"{name} exists")
    return pd.read_csv(path)


def main() -> None:
    profile = load_profile()
    check(len(profile) == 2205 and profile["cycle_id"].tolist() == list(range(1, 2206)), "profile preserves all original Cycle IDs")

    for seconds in VALVE_TIMES:
        table = load_feature_table(seconds)
        check(len(table) == 2205 and table["cycle_id"].equals(profile["cycle_id"]), f"{seconds}s feature table aligns with profile")
        check(table["valve_condition"].equals(profile["valve_condition"]), f"{seconds}s Valve labels align")
        predictors = [column for column in table if column.startswith(("stat__", "trans__"))]
        check(len(predictors) >= 243 and not any("condition" in column or "stable" in column for column in predictors), f"{seconds}s predictors exclude labels and groups")
        check((VALVE_MODEL_DIR / f"valve_combined_{seconds}s.joblib").exists(), f"{seconds}s transient-aware demo model exists")

    raw = np.loadtxt(RAW_DIR / "PS2.txt", dtype=np.float32)
    values = to_10_hz(raw, SENSORS["PS2"][0])
    original = transient_features(values[:8], "PS2", 12)
    mutated = values[:8].copy()
    mutated[:, 120:] += 1_000_000.0
    changed = transient_features(mutated, "PS2", 12)
    check(np.allclose(original.to_numpy(), changed.to_numpy(), equal_nan=True), "12s transient features ignore every sample at and after 12s")
    check("trans__PS2__t90" in original and np.isfinite(original["trans__PS2__t90"]).all(), "PS2 response-timing proxy is defined without target labels")

    time_results = read("valve_time_window_benchmark.csv")
    check(len(time_results) == 10 * 3 * 5, "time-window benchmark has 10 windows × 3 models × 5 folds")
    check(set(time_results["observation_seconds"]) == set(VALVE_TIMES), "time-window benchmark covers all requested endpoints")

    generalization = read("valve_generalization_results.csv")
    check(len(generalization) == 3 * 3 * 4 * 2 * 5, "generalization benchmark has 360 outer-fold results")
    check((generalization.loc[generalization["scheme"] == "block_aware", "shared_blocks"] == 0).all(), "Block-aware folds have zero block overlap")
    check((generalization.loc[generalization["scheme"] == "condition_holdout", "shared_conditions"] == 0).all(), "Condition Holdout folds have zero condition overlap")
    all_classes_present = True
    for row in generalization.itertuples(index=False):
        train_counts = {int(key): value for key, value in json.loads(row.train_class_counts).items()}
        evaluation_counts = {int(key): value for key, value in json.loads(row.evaluation_class_counts).items()}
        all_classes_present &= set(train_counts) == {73, 80, 90, 100} and set(evaluation_counts) == {73, 80, 90, 100}
    check(all_classes_present, "all Valve classes occur in every outer train/evaluation fold")

    oof = read("valve_oof_predictions.csv")
    experiment = ["scheme", "observation_seconds", "feature_set", "model"]
    counts = oof.groupby(experiment).agg(rows=("cycle_id", "size"), cycles=("cycle_id", "nunique"), folds=("fold", "nunique"))
    check(len(oof) == 3 * 3 * 4 * 2 * 2205, "158,760 nested out-of-fold predictions are present")
    check(((counts["rows"] == 2205) & (counts["cycles"] == 2205) & (counts["folds"] == 5)).all(), "each experiment predicts every Cycle ID exactly once")

    cooler = read("valve_cooler_holdout_results.csv")
    check(len(cooler) == 3 * 3 * 4 * 2, "72 cooler-domain holdout results are present")
    check((cooler["shared_cooler_conditions"] == 0).all() and set(cooler["held_out_cooler"]) == {3, 20, 100}, "each cooler condition is held out without domain overlap")
    check((cooler["train_cycles"] + cooler["evaluation_cycles"] == 2205).all(), "cooler holdout partitions cover all cycles")

    classwise = read("valve_classwise_metrics.csv")
    cooler_classwise = read("valve_cooler_classwise_metrics.csv")
    check(len(classwise) == 360 * 4, "all Nested-CV classwise rows are present")
    check(len(cooler_classwise) == 72 * 4, "all cooler-holdout classwise rows are present")
    for frame, label in ((time_results, "time"), (generalization, "generalization"), (cooler, "cooler"), (classwise, "classwise"), (cooler_classwise, "cooler classwise")):
        metric_columns = [column for column in ("accuracy", "macro_f1", "weighted_f1", "close_failure_recall", "small_lag_recall", "precision", "recall", "f1") if column in frame]
        check(frame[metric_columns].apply(lambda column: column.between(0, 1).all()).all(), f"{label} metrics are bounded")

    ranking = read("valve_feature_rankings.csv")
    frequency = read("valve_reduced_sensor_frequency.csv")
    check((ranking["folds"] == 5).all(), "feature rankings aggregate five training folds only")
    check({"PS2", "PS3"} <= set(frequency.head(3)["sensor"]), "PS2 and PS3 are consistently selected reduced-model sensors")

    result = predict_cycle(12, 1)
    check(result["cycle_id"] == 1 and result["actual_code"] == int(profile.iloc[0]["valve_condition"]), "dashboard inference and truth use the same original Cycle ID")
    check(abs(sum(result["probabilities"].values()) - 1.0) < 1e-8, "Valve demo probabilities sum to one")

    required = ["VALVE_SIGNAL_ANALYSIS.md", "VALVE_TRANSIENT_REPORT.md", "valve_feature_ablation.csv", "valve_error_analysis.csv"]
    check(all((VALVE_RESULT_DIR / name).exists() for name in required), "required Valve reports and CSV artifacts exist")
    figures = list((VALVE_RESULT_DIR / "figures").glob("*.png"))
    check(len(figures) >= 10 and all(path.stat().st_size > 0 for path in figures), "signal, benchmark, cooler, and confusion figures exist")
    app_source = Path("app.py").read_text(encoding="utf-8")
    check("Valve transient analysis" in app_source and "observed response band" in app_source, "Streamlit includes the Valve UI and event interpretation")
    print("\nValve transient study verification completed successfully.")


if __name__ == "__main__":
    main()
