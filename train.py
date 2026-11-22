"""Train and select full-sensor pump leakage classifiers using validation only."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import CLASS_NAMES, DATA_DIR, MODEL_DIR, OBSERVATION_TIMES, RANDOM_SEED, RESULT_DIR, ensure_directories
from preprocess import load_feature_table


@dataclass(frozen=True)
class Candidate:
    name: str
    params: dict


def make_model(model_name: str, params: dict | None = None):
    params = params or {}
    if model_name == "dummy":
        return Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("classifier", DummyClassifier(strategy="prior", random_state=RANDOM_SEED))]
        )
    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            C=float(params.get("C", 1.0)),
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_SEED,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("variance", VarianceThreshold()),
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )
    if model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 400)),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("classifier", classifier)])
    raise ValueError(f"Unknown model: {model_name}")


def candidates() -> list[Candidate]:
    return [
        Candidate("dummy", {}),
        Candidate("logistic_regression", {"C": 0.1}),
        Candidate("logistic_regression", {"C": 1.0}),
        Candidate("logistic_regression", {"C": 10.0}),
        Candidate("random_forest", {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"}),
        Candidate("random_forest", {"n_estimators": 400, "max_depth": 16, "min_samples_leaf": 1, "max_features": "sqrt"}),
        Candidate("random_forest", {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 2, "max_features": 0.5}),
    ]


def get_xy(seconds: int, split_name: str, selected_sensors: list[str] | None = None):
    table = load_feature_table(seconds)
    assignments = pd.read_csv(DATA_DIR / "split_assignments.csv")[["cycle_id", "split"]]
    merged = table.merge(assignments, on="cycle_id", validate="one_to_one")
    subset = merged.loc[merged["split"] == split_name].copy()
    feature_columns = [c for c in table.columns if "__" in c]
    if selected_sensors is not None:
        feature_columns = [c for c in feature_columns if c.split("__", 1)[0] in selected_sensors]
    return subset[feature_columns], subset["pump_leakage"].astype(int), subset["cycle_id"].astype(int), feature_columns


def metric_row(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "severe_recall": recall_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0],
    }


def train_all(force: bool = False) -> pd.DataFrame:
    ensure_directories()
    split_path = DATA_DIR / "split_assignments.csv"
    if not split_path.exists():
        raise FileNotFoundError("Run split_data.py before training")
    records: list[dict] = []
    selected: dict[str, dict] = {}

    for seconds in OBSERVATION_TIMES:
        X_train, y_train, _, columns = get_xy(seconds, "train")
        X_val, y_val, _, _ = get_xy(seconds, "validation")
        best_by_model: dict[str, tuple[float, float, dict, object, float, float]] = {}
        for candidate in candidates():
            started = time.perf_counter()
            model = make_model(candidate.name, candidate.params)
            model.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - started
            infer_started = time.perf_counter()
            pred = model.predict(X_val)
            inference_ms_per_cycle = (time.perf_counter() - infer_started) * 1000 / len(X_val)
            metrics = metric_row(y_val, pred)
            record = {
                "observation_seconds": seconds,
                "model": candidate.name,
                "params": json.dumps(candidate.params, sort_keys=True),
                "split": "validation",
                **metrics,
                "fit_seconds": fit_seconds,
                "inference_ms_per_cycle": inference_ms_per_cycle,
                "n_features": len(columns),
            }
            records.append(record)
            # Macro-F1 is primary; severe recall is the safety tie-breaker.
            key = (metrics["macro_f1"], metrics["severe_recall"])
            if candidate.name not in best_by_model or key > best_by_model[candidate.name][:2]:
                best_by_model[candidate.name] = (key[0], key[1], candidate.params, model, fit_seconds, inference_ms_per_cycle)

        for model_name, (_, _, params, model, _, _) in best_by_model.items():
            path = MODEL_DIR / f"pump_{model_name}_{seconds}s.joblib"
            joblib.dump({"model": model, "features": columns, "seconds": seconds, "model_name": model_name, "params": params}, path)

        eligible = {k: v for k, v in best_by_model.items() if k != "dummy"}
        chosen_name, chosen_data = max(eligible.items(), key=lambda item: item[1][:2])
        selected[str(seconds)] = {
            "model": chosen_name,
            "params": chosen_data[2],
            "selection_split": "validation",
            "criterion": "macro_f1_then_severe_recall",
            "macro_f1": chosen_data[0],
            "severe_recall": chosen_data[1],
        }
        print(f"{seconds}s selected: {chosen_name}, val macro-F1={chosen_data[0]:.4f}")

    output = pd.DataFrame(records)
    output.to_csv(RESULT_DIR / "validation_model_metrics.csv", index=False)
    (RESULT_DIR / "selected_models.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="reserved for reproducible reruns")
    args = parser.parse_args()
    train_all(force=args.force)


if __name__ == "__main__":
    main()

