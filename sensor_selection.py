"""Validation-only grouped sensor importance and sensor-count trade-off study."""
from __future__ import annotations

import argparse
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from config import FIGURE_DIR, MODEL_DIR, OBSERVATION_TIMES, RANDOM_SEED, RESULT_DIR, SENSORS, ensure_directories
from train import get_xy, make_model


def grouped_permutation_importance(model, X: pd.DataFrame, y: pd.Series, repeats: int = 10) -> pd.DataFrame:
    """Jointly permute all derived columns from a sensor on validation cycles."""
    rng = np.random.default_rng(RANDOM_SEED)
    baseline = f1_score(y, model.predict(X), average="macro", zero_division=0)
    rows = []
    for sensor in SENSORS:
        columns = [c for c in X.columns if c.startswith(f"{sensor}__")]
        drops = []
        for _ in range(repeats):
            permuted = X.copy()
            order = rng.permutation(len(X))
            permuted.loc[:, columns] = X.iloc[order][columns].to_numpy()
            score = f1_score(y, model.predict(permuted), average="macro", zero_division=0)
            drops.append(baseline - score)
        rows.append({"sensor": sensor, "importance_mean_macro_f1_drop": float(np.mean(drops)), "importance_std": float(np.std(drops, ddof=1)), "repeats": repeats, "baseline_macro_f1": baseline})
    return pd.DataFrame(rows).sort_values("importance_mean_macro_f1_drop", ascending=False).reset_index(drop=True)


def rf_impurity_by_sensor(bundle: dict) -> pd.DataFrame:
    classifier = bundle["model"].named_steps["classifier"]
    raw = pd.Series(classifier.feature_importances_, index=bundle["features"])
    grouped = raw.groupby(raw.index.str.split("__").str[0]).sum().sort_values(ascending=False)
    return grouped.rename("rf_impurity_importance").rename_axis("sensor").reset_index()


def run_sensor_study(repeats: int = 10) -> None:
    ensure_directories()
    rf_bundle = joblib.load(MODEL_DIR / "pump_random_forest_60s.joblib")
    X_val, y_val, _, _ = get_xy(60, "validation")
    permutation = grouped_permutation_importance(rf_bundle["model"], X_val[rf_bundle["features"]], y_val, repeats)
    importance = permutation.merge(rf_impurity_by_sensor(rf_bundle), on="sensor", how="left")
    importance["rank"] = np.arange(1, len(importance) + 1)
    importance.to_csv(RESULT_DIR / "sensor_importance.csv", index=False)
    ranked_sensors = importance["sensor"].tolist()
    configurations = {"all": ranked_sensors, "top10": ranked_sensors[:10], "top5": ranked_sensors[:5], "top3": ranked_sensors[:3]}
    (RESULT_DIR / "sensor_configurations.json").write_text(json.dumps(configurations, indent=2), encoding="utf-8")

    results = []
    for seconds in OBSERVATION_TIMES:
        for config_name, sensors in configurations.items():
            X_train, y_train, _, columns = get_xy(seconds, "train", sensors)
            X_validation, y_validation, _, _ = get_xy(seconds, "validation", sensors)
            model = make_model("random_forest", {"n_estimators": 400, "min_samples_leaf": 2, "max_features": "sqrt"})
            model.fit(X_train, y_train)
            joblib.dump({"model": model, "features": columns, "seconds": seconds, "model_name": "random_forest", "sensor_configuration": config_name, "sensors": sensors}, MODEL_DIR / f"pump_rf_{config_name}_{seconds}s.joblib")
            for split_name, X, y in [("validation", X_validation, y_validation), ("test", *get_xy(seconds, "test", sensors)[:2])]:
                pred = model.predict(X[columns])
                results.append({"observation_seconds": seconds, "sensor_configuration": config_name, "sensor_count": len(sensors), "sensors": ",".join(sensors), "split": split_name, "accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro", zero_division=0), "severe_recall": recall_score(y, pred, labels=[2], average=None, zero_division=0)[0]})
    result_df = pd.DataFrame(results)
    result_df.to_csv(RESULT_DIR / "sensor_subset_metrics.csv", index=False)
    save_pareto_analysis(result_df)
    make_figure(importance, result_df)
    print(importance.to_string(index=False))


def make_figure(importance: pd.DataFrame, metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ordered = importance.sort_values("importance_mean_macro_f1_drop")
    axes[0].barh(ordered["sensor"], ordered["importance_mean_macro_f1_drop"], xerr=ordered["importance_std"], color="#2573b8")
    axes[0].set(xlabel="Validation macro-F1 drop", title="Grouped permutation importance (60 s)")
    test_60 = metrics.query("split == 'test' and observation_seconds == 60").sort_values("sensor_count")
    axes[1].plot(test_60["sensor_count"], test_60["macro_f1"], marker="o", label="Macro F1")
    axes[1].plot(test_60["sensor_count"], test_60["accuracy"], marker="s", label="Accuracy")
    axes[1].set(xlabel="Number of sensors", ylabel="Test score", ylim=(0, 1.03), title="Sensor-count trade-off (fixed validation ranking)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "sensor_optimization.png", dpi=180)
    plt.close(fig)


def save_pareto_analysis(metrics: pd.DataFrame) -> None:
    """Mark validation designs not dominated in time, sensor count, and macro-F1."""
    validation = metrics[metrics["split"] == "validation"].copy().reset_index(drop=True)
    pareto = []
    for index, row in validation.iterrows():
        dominated = (
            (validation["observation_seconds"] <= row["observation_seconds"])
            & (validation["sensor_count"] <= row["sensor_count"])
            & (validation["macro_f1"] >= row["macro_f1"])
            & (
                (validation["observation_seconds"] < row["observation_seconds"])
                | (validation["sensor_count"] < row["sensor_count"])
                | (validation["macro_f1"] > row["macro_f1"])
            )
        ).any()
        pareto.append(not dominated)
    validation["pareto_optimal"] = pareto
    validation.to_csv(RESULT_DIR / "pareto_validation.csv", index=False)

    baseline = float(validation.query("observation_seconds == 60 and sensor_configuration == 'all'")["macro_f1"].iloc[0])
    sensitivity_rows = []
    for allowed_drop in (0.0, 0.005, 0.01, 0.02, 0.05):
        eligible = validation[validation["macro_f1"] >= baseline - allowed_drop]
        if eligible.empty:
            continue
        chosen = eligible.sort_values(["observation_seconds", "sensor_count", "macro_f1"], ascending=[True, True, False]).iloc[0]
        sensitivity_rows.append({"allowed_macro_f1_drop": allowed_drop, "reference_60s_all_macro_f1": baseline, "selected_observation_seconds": int(chosen["observation_seconds"]), "selected_sensor_configuration": chosen["sensor_configuration"], "selected_sensor_count": int(chosen["sensor_count"]), "validation_macro_f1": float(chosen["macro_f1"])})
    pd.DataFrame(sensitivity_rows).to_csv(RESULT_DIR / "design_tolerance_sensitivity.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    run_sensor_study(repeats=args.repeats)


if __name__ == "__main__":
    main()

