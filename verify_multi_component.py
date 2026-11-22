"""Integrity checks for the multi-component extension and its executed results."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import MODEL_DIR, OBSERVATION_TIMES
from multi_component import (
    COMPONENT_SPECS,
    MULTI_MODEL_DIR,
    MULTI_RESULT_DIR,
    ground_truth_for_cycle,
    load_cycle_metadata,
    predict_all_components,
    selected_bundle_path,
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def read_required(name: str) -> pd.DataFrame:
    path = MULTI_RESULT_DIR / name
    check(path.exists() and path.stat().st_size > 0, f"{name} exists and is non-empty")
    return pd.read_csv(path)


def main() -> None:
    root_models = list(MODEL_DIR.glob("*.joblib"))
    new_models = list(MULTI_MODEL_DIR.glob("*.joblib"))
    check(len(root_models) == 40, "all 40 original Pump joblib models remain in the original model directory")
    check(len(new_models) == 60, "60 isolated multi-component demo bundles exist")

    selected_path = MULTI_RESULT_DIR / "selected_demo_models.json"
    check(selected_path.exists(), "selected demo model manifest exists")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    check(set(selected) == set(COMPONENT_SPECS), "selection manifest covers exactly four components")
    for component in COMPONENT_SPECS:
        check(set(map(int, selected[component])) == set(OBSERVATION_TIMES), f"{component} supports all five observation windows")
        for seconds in OBSERVATION_TIMES:
            path = selected_bundle_path(component, seconds)
            bundle = joblib.load(path)
            check(bundle["component"] == component and bundle["seconds"] == seconds, f"{component} {seconds}s bundle metadata is consistent")
            check(all("__" in column for column in bundle["features"]), f"{component} {seconds}s uses sensor-derived features only")

    demo = read_required("demo_model_metrics.csv")
    check(len(demo) == 4 * 5 * 3 * 2, "demo metrics contain 4 components × 5 windows × 3 models × 2 splits")
    check(set(demo["split"]) == {"validation", "test"}, "demo and research evaluation results remain distinct")
    check(demo[["accuracy", "macro_f1", "weighted_f1", "severe_recall"]].apply(lambda column: column.between(0, 1).all()).all(), "all demo metrics are bounded")

    distribution = read_required("component_label_distribution.csv")
    check(len(distribution) == sum(len(spec["classes"]) for spec in COMPONENT_SPECS.values()), "label distribution has one row per component class")
    check((distribution.groupby("component")["count"].sum() == 2205).all(), "each component label distribution totals 2,205 cycles")

    folds = read_required("nested_cv_fold_results.csv")
    check(len(folds) == 4 * 3 * 2 * 2 * 5, "240 nested outer-fold results are present")
    check((folds.loc[folds["scheme"] == "block_aware", "shared_blocks"] == 0).all(), "block-aware outer folds have zero block overlap")
    check((folds.loc[folds["scheme"] == "condition_holdout", "shared_conditions"] == 0).all(), "condition-holdout outer folds have zero condition overlap")
    for row in folds.itertuples(index=False):
        required_classes = {str(value) for value in COMPONENT_SPECS[row.component]["classes"]}
        train_counts = json.loads(row.train_class_counts)
        evaluation_counts = json.loads(row.evaluation_class_counts)
        check(required_classes <= set(train_counts) and required_classes <= set(evaluation_counts), f"all classes occur in {row.component}/{row.scheme}/{row.observation_seconds}s/{row.model}/fold{row.fold}")

    summary = read_required("nested_cv_summary.csv")
    check(len(summary) == 4 * 3 * 2 * 2, "48 benchmark mean/standard-deviation rows are present")
    check(summary.filter(regex="_(mean|std)$").apply(lambda column: np.isfinite(column).all()).all(), "benchmark summary contains only finite results")

    classwise = read_required("classwise_metrics.csv")
    expected_classwise = sum(len(spec["classes"]) for spec in COMPONENT_SPECS.values()) * 3 * 2 * 2 * 5
    check(len(classwise) == expected_classwise, "all 840 fold-level classwise metric rows are present")
    check(classwise[["precision", "recall", "f1"]].apply(lambda column: column.between(0, 1).all()).all(), "all classwise metrics are bounded")

    oof = read_required("out_of_fold_predictions.csv")
    check(len(oof) == 4 * 3 * 2 * 2 * 2205, "105,840 out-of-fold predictions are present")
    experiment_columns = ["component", "scheme", "observation_seconds", "model"]
    oof_counts = oof.groupby(experiment_columns).agg(rows=("cycle_id", "size"), cycles=("cycle_id", "nunique"), folds=("fold", "nunique"))
    check(((oof_counts["rows"] == 2205) & (oof_counts["cycles"] == 2205) & (oof_counts["folds"] == 5)).all(), "every experiment predicts every original cycle exactly once across five folds")
    check(oof["cycle_id"].between(1, 2205).all(), "OOF cycle IDs preserve the original UCI range")

    matrices = read_required("confusion_matrices.csv")
    matrix_sums = matrices.groupby([*experiment_columns, "fold"], as_index=False)["count"].sum().rename(columns={"count": "matrix_total"})
    comparison = folds.merge(matrix_sums, on=[*experiment_columns, "fold"], validate="one_to_one")
    check((comparison["evaluation_cycles"] == comparison["matrix_total"]).all(), "every fold confusion matrix sums to its evaluation size")
    images = list((MULTI_RESULT_DIR / "confusion_matrices").glob("*.png"))
    check(len(images) == 48 and all(path.stat().st_size > 0 for path in images), "48 aggregated confusion-matrix images exist")
    figures = list((MULTI_RESULT_DIR / "figures").glob("*.png"))
    check(len(figures) >= 5 and all(path.stat().st_size > 0 for path in figures), "all required benchmark figures exist")

    fault = read_required("single_vs_multi_fault.csv")
    holdout = read_required("condition_holdout_analysis.csv")
    check(set(fault["analysis_level"]) == {"fault_count", "fault_group"}, "single/multiple-fault analysis contains both grouping views")
    check({"class_comparison", "condition_error"} <= set(holdout["analysis_level"]), "condition-holdout analysis includes class drops and difficult combinations")
    metadata = load_cycle_metadata()
    abnormal_count = (
        metadata["pump_leakage"].ne(0).astype(int)
        + metadata["valve_condition"].ne(100).astype(int)
        + metadata["cooler_condition"].ne(100).astype(int)
        + metadata["accumulator_pressure"].ne(130).astype(int)
    )
    check((abnormal_count == 0).sum() == 21, "actual data contains 21 all-normal cycles")
    check((abnormal_count == 1).sum() == 566, "actual data contains 566 single-component abnormal cycles")
    check((abnormal_count >= 2).sum() == 1618, "actual data contains 1,618 multiple-component abnormal cycles")

    predictions = predict_all_components(5, 1)
    truth = ground_truth_for_cycle(1)
    check(len(predictions) == 4 and {item["component"] for item in predictions} == set(COMPONENT_SPECS), "one call returns predictions for all four components")
    check(all(0 <= item["confidence"] <= 1 for item in predictions), "all-component inference returns valid probabilities")
    metadata_row = metadata.loc[metadata["cycle_id"] == 1].iloc[0]
    check(all(truth[key]["code"] == int(metadata_row[spec["column"]]) for key, spec in COMPONENT_SPECS.items()), "revealed truth matches the original metadata row for the same cycle ID")

    report = MULTI_RESULT_DIR / "MULTI_COMPONENT_REPORT.md"
    check(report.exists() and report.stat().st_size > 5000, "multi-component research report is substantive")
    app_source = Path("app.py").read_text(encoding="utf-8")
    check(all(label in app_source for label in ["All Components", "Nested generalization benchmark", "Multi-fault analysis"]), "Streamlit exposes simultaneous diagnosis, rigorous performance, and multi-fault views")
    print("\nMulti-component extension verification completed successfully.")


if __name__ == "__main__":
    main()
