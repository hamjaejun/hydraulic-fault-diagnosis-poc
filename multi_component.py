"""Component-specific demo models and unified four-component inference.

Artifacts are isolated under models/multi_component and
results/multi_component_benchmark so the original pump PoC remains intact.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from config import DATA_DIR, FEATURE_DIR, MODEL_DIR, OBSERVATION_TIMES, RANDOM_SEED, RESULT_DIR, ROOT
from preprocess import load_feature_table
from train import candidates, make_model


MULTI_MODEL_DIR = MODEL_DIR / "multi_component"
MULTI_RESULT_DIR = RESULT_DIR / "multi_component_benchmark"

COMPONENT_SPECS = {
    "pump": {
        "column": "pump_leakage",
        "display": "Pump",
        "classes": [0, 1, 2],
        "labels": {0: "Normal", 1: "Weak Leakage", 2: "Severe Leakage"},
        "severe_class": 2,
    },
    "valve": {
        "column": "valve_condition",
        "display": "Valve",
        "classes": [73, 80, 90, 100],
        "labels": {73: "Close to Failure", 80: "Severe Switching Lag", 90: "Small Switching Lag", 100: "Optimal Switching"},
        "severe_class": 73,
    },
    "cooler": {
        "column": "cooler_condition",
        "display": "Cooler",
        "classes": [3, 20, 100],
        "labels": {3: "Close to Failure", 20: "Reduced Efficiency", 100: "Full Efficiency"},
        "severe_class": 3,
    },
    "accumulator": {
        "column": "accumulator_pressure",
        "display": "Accumulator",
        "classes": [90, 100, 115, 130],
        "labels": {90: "Close to Failure", 100: "Severely Reduced Pressure", 115: "Slightly Reduced Pressure", 130: "Optimal Pressure"},
        "severe_class": 90,
    },
}


def ensure_multi_directories() -> None:
    MULTI_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MULTI_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (MULTI_RESULT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (MULTI_RESULT_DIR / "confusion_matrices").mkdir(parents=True, exist_ok=True)


def load_cycle_metadata() -> pd.DataFrame:
    metadata = pd.read_csv(FEATURE_DIR / "cycle_metadata.csv")
    expected = ["cycle_id", "cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure", "stable_flag"]
    if list(metadata.columns) != expected:
        raise ValueError(f"Unexpected cycle metadata columns: {list(metadata.columns)}")
    if len(metadata) != 2205 or not metadata["cycle_id"].is_unique:
        raise ValueError("Cycle metadata must contain unique cycle IDs 1..2205")
    if not np.array_equal(metadata["cycle_id"].to_numpy(), np.arange(1, 2206)):
        raise ValueError("Cycle metadata is not in original UCI order")
    return metadata


def load_component_table(seconds: int, component: str) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    if component not in COMPONENT_SPECS:
        raise ValueError(f"Unknown component: {component}")
    table = load_feature_table(seconds)
    metadata = load_cycle_metadata()
    feature_columns = [column for column in table.columns if "__" in column]
    merged = table[["cycle_id", *feature_columns]].merge(metadata, on="cycle_id", validate="one_to_one", sort=False)
    target = COMPONENT_SPECS[component]["column"]
    return merged[feature_columns], merged[target].astype(int), merged["cycle_id"].astype(int), feature_columns


def load_component_split(seconds: int, component: str, split_name: str):
    X, y, cycle_ids, feature_columns = load_component_table(seconds, component)
    assignments = pd.read_csv(DATA_DIR / "split_assignments.csv")[["cycle_id", "split"]]
    split_map = assignments.set_index("cycle_id")["split"]
    mask = cycle_ids.map(split_map).eq(split_name).to_numpy()
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True), cycle_ids.loc[mask].reset_index(drop=True), feature_columns


def summary_metrics(y_true, y_pred, classes: list[int]) -> dict:
    severe_class = classes[0] if classes[0] not in (0, 100, 130) else classes[-1]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0),
        "severe_recall": recall_score_for_class(y_true, y_pred, severe_class),
    }


def recall_score_for_class(y_true, y_pred, class_id: int) -> float:
    _, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=[class_id], zero_division=0)
    return float(recall[0])


def classwise_rows(y_true, y_pred, component: str) -> list[dict]:
    spec = COMPONENT_SPECS[component]
    classes = spec["classes"]
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
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
        for index, class_id in enumerate(classes)
    ]


def train_demo_models() -> pd.DataFrame:
    """Train 4 components × 5 windows × 3 model families using the saved split."""
    ensure_multi_directories()
    metric_rows: list[dict] = []
    class_rows: list[dict] = []
    confusion_rows: list[dict] = []
    selected: dict[str, dict] = {}

    for component, spec in COMPONENT_SPECS.items():
        selected[component] = {}
        classes = spec["classes"]
        for seconds in OBSERVATION_TIMES:
            X_train, y_train, _, feature_columns = load_component_split(seconds, component, "train")
            X_val, y_val, _, _ = load_component_split(seconds, component, "validation")
            X_test, y_test, test_ids, _ = load_component_split(seconds, component, "test")
            best_by_family: dict[str, dict] = {}

            for candidate in candidates():
                started = time.perf_counter()
                model = make_model(candidate.name, candidate.params)
                model.fit(X_train, y_train)
                fit_seconds = time.perf_counter() - started
                val_pred = model.predict(X_val)
                val_metrics = summary_metrics(y_val, val_pred, classes)
                key = (val_metrics["macro_f1"], val_metrics["severe_recall"])
                current = best_by_family.get(candidate.name)
                if current is None or key > current["key"]:
                    best_by_family[candidate.name] = {
                        "key": key,
                        "model": model,
                        "params": candidate.params,
                        "fit_seconds": fit_seconds,
                        "validation_metrics": val_metrics,
                    }

            for model_name, best in best_by_family.items():
                model = best["model"]
                bundle_path = MULTI_MODEL_DIR / f"{component}_{model_name}_{seconds}s.joblib"
                joblib.dump(
                    {
                        "model": model,
                        "component": component,
                        "target_column": spec["column"],
                        "features": feature_columns,
                        "seconds": seconds,
                        "model_name": model_name,
                        "params": best["params"],
                        "classes": classes,
                        "labels": spec["labels"],
                        "training_split": "train",
                        "selection_split": "validation",
                    },
                    bundle_path,
                )
                for split_name, X_eval, y_eval in (("validation", X_val, y_val), ("test", X_test, y_test)):
                    pred = model.predict(X_eval)
                    metrics = summary_metrics(y_eval, pred, classes)
                    metric_rows.append(
                        {
                            "component": component,
                            "observation_seconds": seconds,
                            "model": model_name,
                            "split": split_name,
                            **metrics,
                            "fit_seconds": best["fit_seconds"],
                            "params": json.dumps(best["params"], sort_keys=True),
                            "cycles": len(y_eval),
                        }
                    )
                    for row in classwise_rows(y_eval, pred, component):
                        class_rows.append(
                            {
                                "component": component,
                                "observation_seconds": seconds,
                                "model": model_name,
                                "split": split_name,
                                **row,
                            }
                        )
                    if split_name == "test":
                        matrix = confusion_matrix(y_eval, pred, labels=classes)
                        for actual_index, actual in enumerate(classes):
                            for predicted_index, predicted in enumerate(classes):
                                confusion_rows.append(
                                    {
                                        "component": component,
                                        "observation_seconds": seconds,
                                        "model": model_name,
                                        "actual": actual,
                                        "predicted": predicted,
                                        "count": int(matrix[actual_index, predicted_index]),
                                    }
                                )

            eligible = {name: value for name, value in best_by_family.items() if name != "dummy"}
            selected_name, chosen = max(eligible.items(), key=lambda item: item[1]["key"])
            selected[component][str(seconds)] = {
                "model": selected_name,
                "path": (MULTI_MODEL_DIR / f"{component}_{selected_name}_{seconds}s.joblib").relative_to(ROOT).as_posix(),
                "validation_macro_f1": chosen["validation_metrics"]["macro_f1"],
                "validation_severe_recall": chosen["validation_metrics"]["severe_recall"],
                "criterion": "validation_macro_f1_then_severe_recall",
            }
            print(f"Demo selected: {component} {seconds}s {selected_name} macro-F1={chosen['validation_metrics']['macro_f1']:.4f}")

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(MULTI_RESULT_DIR / "demo_model_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(MULTI_RESULT_DIR / "demo_classwise_metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(MULTI_RESULT_DIR / "demo_confusion_matrices.csv", index=False)
    (MULTI_RESULT_DIR / "selected_demo_models.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    return metrics


def selected_bundle_path(component: str, seconds: int) -> Path:
    selection_path = MULTI_RESULT_DIR / "selected_demo_models.json"
    if not selection_path.exists():
        raise FileNotFoundError("Multi-component demo models are missing. Run: python multi_component.py train")
    selected = json.loads(selection_path.read_text(encoding="utf-8"))
    model_name = selected[component][str(seconds)]["model"]
    return MULTI_MODEL_DIR / f"{component}_{model_name}_{seconds}s.joblib"


def predict_component(component: str, seconds: int, cycle_id: int) -> dict:
    bundle_path = selected_bundle_path(component, seconds)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing {bundle_path.name}. Run: python multi_component.py train")
    bundle = joblib.load(bundle_path)
    table = load_feature_table(seconds)
    row = table.loc[table["cycle_id"] == cycle_id]
    if len(row) != 1:
        raise ValueError(f"Cycle ID {cycle_id} was not found exactly once")
    probability = bundle["model"].predict_proba(row[bundle["features"]])[0]
    model_classes = [int(value) for value in bundle["model"].classes_]
    best_index = int(np.argmax(probability))
    predicted = model_classes[best_index]
    spec = COMPONENT_SPECS[component]
    return {
        "component": component,
        "component_display": spec["display"],
        "cycle_id": int(cycle_id),
        "observation_seconds": int(seconds),
        "model": bundle["model_name"],
        "predicted_code": predicted,
        "predicted_label": spec["labels"][predicted],
        "confidence": float(probability[best_index]),
        "probabilities": {spec["labels"][class_id]: float(probability[index]) for index, class_id in enumerate(model_classes)},
    }


def predict_all_components(seconds: int, cycle_id: int) -> list[dict]:
    """Run four independently trained classifiers for one original UCI cycle."""
    return [predict_component(component, seconds, cycle_id) for component in COMPONENT_SPECS]


def ground_truth_for_cycle(cycle_id: int) -> dict:
    metadata = load_cycle_metadata()
    row = metadata.loc[metadata["cycle_id"] == cycle_id]
    if len(row) != 1:
        raise ValueError(f"Cycle ID {cycle_id} was not found exactly once")
    record = row.iloc[0]
    output = {}
    for component, spec in COMPONENT_SPECS.items():
        code = int(record[spec["column"]])
        output[component] = {"code": code, "label": spec["labels"][code]}
    output["stable_flag"] = int(record["stable_flag"])
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", help="train all component/window demo models")
    predict_parser = subparsers.add_parser("predict", help="predict all components for one cycle")
    predict_parser.add_argument("--seconds", type=int, choices=OBSERVATION_TIMES, required=True)
    predict_parser.add_argument("--cycle-id", type=int, required=True)
    args = parser.parse_args()
    if args.command == "train":
        train_demo_models()
    else:
        print(json.dumps({"predictions": predict_all_components(args.seconds, args.cycle_id), "ground_truth": ground_truth_for_cycle(args.cycle_id)}, indent=2))


if __name__ == "__main__":
    main()
