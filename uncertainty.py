"""Probability calibration, selective prediction, and sequential decision policy."""
from __future__ import annotations

import argparse
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from calibration_model import OvRProbabilityCalibrator
from config import CLASS_NAMES, FIGURE_DIR, MODEL_DIR, OBSERVATION_TIMES, RANDOM_SEED, RESULT_DIR, ensure_directories
from train import get_xy


def multiclass_brier(y_true: np.ndarray, probability: np.ndarray) -> float:
    one_hot = np.eye(probability.shape[1])[np.asarray(y_true, dtype=int)]
    return float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))


def reliability_rows(y_true, probability, seconds, split_name, bins=10) -> list[dict]:
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for class_id in range(3):
        actual = (np.asarray(y_true) == class_id).astype(float)
        for bin_index in range(bins):
            left, right = edges[bin_index], edges[bin_index + 1]
            mask = (probability[:, class_id] >= left) & (probability[:, class_id] < right if bin_index < bins - 1 else probability[:, class_id] <= right)
            if mask.any():
                rows.append({"observation_seconds": seconds, "split": split_name, "class_id": class_id, "class_name": CLASS_NAMES[class_id], "bin": bin_index, "mean_predicted_probability": float(probability[mask, class_id].mean()), "observed_frequency": float(actual[mask].mean()), "count": int(mask.sum())})
    return rows


def coverage_curve(y_true, probability, seconds, split_name) -> pd.DataFrame:
    pred = probability.argmax(axis=1)
    confidence = probability.max(axis=1)
    rows = []
    for threshold in np.round(np.arange(0.34, 0.991, 0.01), 2):
        accepted = confidence >= threshold
        rows.append({"observation_seconds": seconds, "split": split_name, "threshold": threshold, "coverage": float(accepted.mean()), "confirmed_accuracy": float(accuracy_score(np.asarray(y_true)[accepted], pred[accepted])) if accepted.any() else np.nan, "accepted_count": int(accepted.sum())})
    return pd.DataFrame(rows)


def choose_threshold(curve: pd.DataFrame, target_accuracy=0.90, minimum_coverage=0.50) -> tuple[float, str]:
    feasible = curve[(curve["confirmed_accuracy"] >= target_accuracy) & (curve["coverage"] >= minimum_coverage)]
    if not feasible.empty:
        row = feasible.sort_values(["coverage", "threshold"], ascending=[False, True]).iloc[0]
        return float(row["threshold"]), "max coverage subject to validation accuracy >= 0.90 and coverage >= 0.50"
    eligible = curve[curve["coverage"] >= minimum_coverage].dropna(subset=["confirmed_accuracy"]).copy()
    eligible["utility"] = eligible["confirmed_accuracy"] - 0.10 * (1 - eligible["coverage"])
    row = eligible.sort_values(["utility", "coverage"], ascending=False).iloc[0]
    return float(row["threshold"]), "fallback: maximize validation accuracy - 0.10*(1-coverage), coverage >= 0.50"


def fit_and_evaluate() -> None:
    ensure_directories()
    selected = json.loads((RESULT_DIR / "selected_models.json").read_text(encoding="utf-8"))
    thresholds, summary_rows, curve_frames, reliability = {}, [], [], []
    for seconds in OBSERVATION_TIMES:
        model_name = selected[str(seconds)]["model"]
        bundle = joblib.load(MODEL_DIR / f"pump_{model_name}_{seconds}s.joblib")
        X_val, y_val, _, _ = get_xy(seconds, "validation")
        X_test, y_test, _, _ = get_xy(seconds, "test")
        calibrator = OvRProbabilityCalibrator(bundle["model"]).fit(X_val[bundle["features"]], y_val)
        calibrated_path = MODEL_DIR / f"pump_calibrated_{seconds}s.joblib"
        joblib.dump({**bundle, "model": calibrator, "calibration_split": "validation"}, calibrated_path)

        val_probability = calibrator.predict_proba(X_val[bundle["features"]])
        val_curve = coverage_curve(y_val.to_numpy(), val_probability, seconds, "validation")
        threshold, rule = choose_threshold(val_curve)
        thresholds[str(seconds)] = {"threshold": threshold, "rule": rule, "selected_on": "validation"}
        curve_frames.append(val_curve)
        reliability.extend(reliability_rows(y_val, val_probability, seconds, "validation"))

        for split_name, X, y in [("validation", X_val, y_val), ("test", X_test, y_test)]:
            probability = calibrator.predict_proba(X[bundle["features"]])
            pred = probability.argmax(axis=1)
            confidence = probability.max(axis=1)
            accepted = confidence >= threshold
            summary_rows.append({"observation_seconds": seconds, "split": split_name, "model": model_name, "threshold": threshold, "unconditional_accuracy": accuracy_score(y, pred), "multiclass_brier": multiclass_brier(y.to_numpy(), probability), "coverage": float(accepted.mean()), "confirmed_accuracy": accuracy_score(y.to_numpy()[accepted], pred[accepted]) if accepted.any() else np.nan, "accepted_count": int(accepted.sum()), "total_count": len(y)})
            if split_name == "test":
                curve_frames.append(coverage_curve(y.to_numpy(), probability, seconds, "test"))
                reliability.extend(reliability_rows(y, probability, seconds, "test"))

    (RESULT_DIR / "confidence_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULT_DIR / "uncertainty_metrics.csv", index=False)
    curves = pd.concat(curve_frames, ignore_index=True)
    curves.to_csv(RESULT_DIR / "coverage_accuracy_curves.csv", index=False)
    reliability_df = pd.DataFrame(reliability)
    reliability_df.to_csv(RESULT_DIR / "calibration_curves.csv", index=False)
    make_figures(curves, reliability_df)
    print(summary.to_string(index=False))


def make_figures(curves: pd.DataFrame, reliability: pd.DataFrame) -> None:
    test_curves = curves[curves["split"] == "test"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for seconds, group in test_curves.groupby("observation_seconds"):
        ax.plot(group["coverage"], group["confirmed_accuracy"], label=f"{seconds} s")
    ax.set(xlabel="Coverage", ylabel="Accuracy among confirmed diagnoses", xlim=(0, 1.02), ylim=(0, 1.02))
    ax.grid(alpha=0.25)
    ax.legend(title="Observation")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "coverage_vs_accuracy.png", dpi=180)
    plt.close(fig)

    subset = reliability[(reliability["split"] == "test") & (reliability["observation_seconds"] == 60)]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    for name, group in subset.groupby("class_name"):
        ax.plot(group["mean_predicted_probability"], group["observed_frequency"], marker="o", label=name)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency", xlim=(0, 1), ylim=(0, 1), title="60 s Test Reliability")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "calibration_60s.png", dpi=180)
    plt.close(fig)


def decision_status(seconds: int, confidence: float, threshold: float) -> str:
    if confidence >= threshold:
        return "Diagnosis Confirmed"
    return "Inspection Recommended" if seconds >= 60 else "Continue Monitoring"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    fit_and_evaluate()


if __name__ == "__main__":
    main()
