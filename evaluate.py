"""One-shot final test evaluation plus block-aware robustness audit."""
from __future__ import annotations

import argparse
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from config import CLASS_NAMES, DATA_DIR, FIGURE_DIR, MODEL_DIR, OBSERVATION_TIMES, RANDOM_SEED, RESULT_DIR, ensure_directories
from train import get_xy, make_model


def detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict, pd.DataFrame]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    summary = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "severe_recall": recall[2],
    }
    per_class = pd.DataFrame(
        {"class_id": [0, 1, 2], "class_name": [CLASS_NAMES[i] for i in range(3)], "precision": precision, "recall": recall, "f1": f1, "support": support}
    )
    return summary, per_class


def evaluate_test() -> pd.DataFrame:
    selected = json.loads((RESULT_DIR / "selected_models.json").read_text(encoding="utf-8"))
    records, class_records, prediction_records, confusion_records = [], [], [], []
    model_names = ("dummy", "logistic_regression", "random_forest")

    for seconds in OBSERVATION_TIMES:
        X_test, y_test, cycle_ids, _ = get_xy(seconds, "test")
        for model_name in model_names:
            bundle = joblib.load(MODEL_DIR / f"pump_{model_name}_{seconds}s.joblib")
            model = bundle["model"]
            started = time.perf_counter()
            pred = model.predict(X_test[bundle["features"]])
            infer_ms = (time.perf_counter() - started) * 1000 / len(X_test)
            probability = model.predict_proba(X_test[bundle["features"]])
            summary, per_class = detailed_metrics(y_test.to_numpy(), pred)
            records.append(
                {
                    "observation_seconds": seconds,
                    "model": model_name,
                    "split": "test",
                    **summary,
                    "inference_ms_per_cycle": infer_ms,
                    "selected_on_validation": selected[str(seconds)]["model"] == model_name,
                }
            )
            per_class.insert(0, "model", model_name)
            per_class.insert(0, "observation_seconds", seconds)
            class_records.append(per_class)
            cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])
            for actual in range(3):
                for predicted in range(3):
                    confusion_records.append({"observation_seconds": seconds, "model": model_name, "actual": actual, "predicted": predicted, "count": int(cm[actual, predicted])})
            for idx, cycle_id in enumerate(cycle_ids):
                row = {"cycle_id": int(cycle_id), "observation_seconds": seconds, "model": model_name, "actual": int(y_test.iloc[idx]), "predicted": int(pred[idx]), "confidence": float(probability[idx].max())}
                row.update({f"probability_{CLASS_NAMES[c]}": float(probability[idx, c]) for c in range(3)})
                prediction_records.append(row)

    metrics = pd.DataFrame(records)
    metrics.to_csv(RESULT_DIR / "test_model_metrics.csv", index=False)
    pd.concat(class_records, ignore_index=True).to_csv(RESULT_DIR / "test_class_metrics.csv", index=False)
    pd.DataFrame(confusion_records).to_csv(RESULT_DIR / "confusion_matrices.csv", index=False)
    predictions = pd.DataFrame(prediction_records)
    predictions.to_csv(RESULT_DIR / "test_predictions.csv", index=False)
    make_figures(metrics, pd.DataFrame(confusion_records), predictions, selected)
    return metrics


def make_figures(metrics: pd.DataFrame, confusion_long: pd.DataFrame, predictions: pd.DataFrame, selected: dict) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for metric, ylabel in [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("severe_recall", "Severe leakage recall")]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name, group in metrics.groupby("model"):
            ax.plot(group["observation_seconds"], group[metric], marker="o", label=name.replace("_", " ").title())
        ax.set(xlabel="Observation time (s)", ylabel=ylabel, ylim=(0, 1.03), xticks=OBSERVATION_TIMES)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"observation_time_vs_{metric}.png", dpi=180)
        plt.close(fig)

    seconds = 60
    model_name = selected[str(seconds)]["model"]
    subset = confusion_long.query("observation_seconds == @seconds and model == @model_name")
    matrix = subset.pivot(index="actual", columns="predicted", values="count").reindex(index=[0, 1, 2], columns=[0, 1, 2]).to_numpy()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    display = ConfusionMatrixDisplay(matrix, display_labels=[CLASS_NAMES[i] for i in range(3)])
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Selected {model_name.replace('_', ' ').title()} — 60 s Test")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "confusion_matrix_selected_60s.png", dpi=180)
    plt.close(fig)

    comparison = metrics[metrics["observation_seconds"] == 5].sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(comparison["model"].str.replace("_", " "), comparison["macro_f1"], color=["#9ca3af", "#3b82f6", "#0f766e"])
    ax.set(ylabel="Test macro F1", ylim=(0, 1.03), title="Model comparison at 5 s")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_comparison_5s.png", dpi=180)
    plt.close(fig)

    chosen = selected["5"]["model"]
    pred = predictions.query("observation_seconds == 5 and model == @chosen").sort_values("cycle_id")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    positions = np.arange(len(pred))
    ax.scatter(positions, pred["actual"], s=12, label="Actual", alpha=0.65)
    ax.scatter(positions, pred["predicted"], s=11, marker="x", label="Predicted", alpha=0.8)
    ax.set(xlabel="Test cycle order", ylabel="Pump leakage class", yticks=[0, 1, 2], title="Actual vs predicted labels — selected 5 s model")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "actual_vs_predicted_5s.png", dpi=180)
    plt.close(fig)


def block_cv_audit() -> pd.DataFrame:
    """Run grouped CV only on train+validation, never on final test cycles."""
    assignments = pd.read_csv(DATA_DIR / "split_assignments.csv")
    development = assignments[assignments["split"].isin(["train", "validation"])]
    X_train, y_train, ids_train, _ = get_xy(60, "train")
    X_val, y_val, ids_val, _ = get_xy(60, "validation")
    X = pd.concat([X_train, X_val], ignore_index=True)
    y = pd.concat([y_train, y_val], ignore_index=True)
    ids = pd.concat([ids_train, ids_val], ignore_index=True)
    groups = ids.map(development.set_index("cycle_id")["inferred_block_id"]).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    records = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups), start=1):
        model = make_model("random_forest", {"n_estimators": 300, "min_samples_leaf": 2, "max_features": "sqrt"})
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[val_idx])
        summary, _ = detailed_metrics(y.iloc[val_idx].to_numpy(), pred)
        records.append({"fold": fold, **summary, "train_cycles": len(train_idx), "validation_cycles": len(val_idx), "validation_blocks": len(np.unique(groups[val_idx]))})
    output = pd.DataFrame(records)
    output.to_csv(RESULT_DIR / "block_group_cv_60s.csv", index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-block-cv", action="store_true")
    args = parser.parse_args()
    ensure_directories()
    metrics = evaluate_test()
    print(metrics.to_string(index=False))
    if not args.skip_block_cv:
        print("\nBlock-aware development-set audit:")
        print(block_cv_audit().to_string(index=False))


if __name__ == "__main__":
    main()

