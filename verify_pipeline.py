"""Fast integrity checks for artifacts, leakage controls, and inference."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from config import DATA_DIR, FEATURE_DIR, MODEL_DIR, OBSERVATION_TIMES, RESULT_DIR, SENSORS
from preprocess import extract_features
from split_data import audit_split
from train import get_xy


def main() -> None:
    manifest = json.loads((DATA_DIR / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["profile_shape"] == [2205, 5]
    assert len(manifest["sensor_shapes"]) == 17
    split = pd.read_csv(DATA_DIR / "split_assignments.csv")
    audit_split(split)

    for seconds in OBSERVATION_TIMES:
        table = pd.read_csv(FEATURE_DIR / f"features_{seconds}s.csv")
        assert table.shape == (2205, 155)
        assert table["cycle_id"].is_unique
        assert np.isfinite(table.iloc[:, 2:].to_numpy()).all()

    # A future-only perturbation must not alter a 5-second feature vector.
    rng = np.random.default_rng(42)
    signal = rng.normal(size=(4, 6000))
    mutated = signal.copy()
    mutated[:, 500:] += 1_000_000
    np.testing.assert_allclose(extract_features(signal, 100, 5), extract_features(mutated, 100, 5))

    for seconds in OBSERVATION_TIMES:
        bundle = joblib.load(MODEL_DIR / f"pump_calibrated_{seconds}s.joblib")
        X, _, _, _ = get_xy(seconds, "validation")
        probability = bundle["model"].predict_proba(X.iloc[:3][bundle["features"]])
        assert probability.shape == (3, 3)
        np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-8)

    required_results = [
        "validation_model_metrics.csv", "test_model_metrics.csv", "test_class_metrics.csv",
        "confusion_matrices.csv", "uncertainty_metrics.csv", "coverage_accuracy_curves.csv",
        "sensor_importance.csv", "sensor_subset_metrics.csv", "block_group_cv_60s.csv",
    ]
    missing = [name for name in required_results if not (RESULT_DIR / name).exists()]
    assert not missing, f"Missing results: {missing}"
    print("PASS: dataset, split, causal windows, model loading, probabilities, and result artifacts verified.")


if __name__ == "__main__":
    main()

