"""Leakage-safe, observation-window-specific feature extraction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import FEATURE_DIR, OBSERVATION_TIMES, RAW_DIR, SENSORS, ensure_directories


def feature_names(sensor: str) -> list[str]:
    return [f"{sensor}__{name}" for name in ("mean", "std", "min", "max", "rms", "initial", "final", "delta", "slope")]


def extract_features(values: np.ndarray, sampling_rate: int, seconds: int) -> np.ndarray:
    """Return nine row-wise features using only samples in [0, seconds)."""
    sample_count = seconds * sampling_rate
    if values.shape[1] < sample_count:
        raise ValueError(f"Need {sample_count} samples for {seconds}s, got {values.shape[1]}")
    x = np.asarray(values[:, :sample_count], dtype=np.float64)
    initial = x[:, 0]
    final = x[:, -1]
    # Closed-form least-squares slope in physical units per second.
    t = np.arange(sample_count, dtype=np.float64) / sampling_rate
    centered_t = t - t.mean()
    denominator = np.dot(centered_t, centered_t)
    slope = ((x - x.mean(axis=1, keepdims=True)) @ centered_t) / denominator if denominator else np.zeros(x.shape[0])
    return np.column_stack(
        [
            x.mean(axis=1),
            x.std(axis=1, ddof=0),
            x.min(axis=1),
            x.max(axis=1),
            np.sqrt(np.mean(np.square(x), axis=1)),
            initial,
            final,
            final - initial,
            slope,
        ]
    )


def build_features(force: bool = False) -> None:
    ensure_directories()
    outputs = {seconds: FEATURE_DIR / f"features_{seconds}s.csv" for seconds in OBSERVATION_TIMES}
    if not force and all(path.exists() for path in outputs.values()):
        print("All feature files already exist; use --force to rebuild.")
        return

    profile_path = RAW_DIR / "profile.txt"
    if not profile_path.exists():
        raise FileNotFoundError("Raw data missing. Run: python download_data.py")
    profile = np.loadtxt(profile_path, dtype=int)
    if profile.shape != (2205, 5):
        raise ValueError(f"Expected 2205 aligned labels, got {profile.shape}")

    frames: dict[int, list[pd.DataFrame]] = {seconds: [] for seconds in OBSERVATION_TIMES}
    for sensor, (rate, _, _) in SENSORS.items():
        path = RAW_DIR / f"{sensor}.txt"
        print(f"Reading {sensor} ({rate} Hz)...")
        values = np.loadtxt(path, dtype=np.float32)
        if values.shape != (2205, rate * 60):
            raise ValueError(f"{sensor}: expected (2205, {rate * 60}), got {values.shape}")
        for seconds in OBSERVATION_TIMES:
            matrix = extract_features(values, rate, seconds)
            frames[seconds].append(pd.DataFrame(matrix, columns=feature_names(sensor)))
        del values

    metadata = pd.DataFrame(
        {
            "cycle_id": np.arange(1, len(profile) + 1),
            "cooler_condition": profile[:, 0],
            "valve_condition": profile[:, 1],
            "pump_leakage": profile[:, 2],
            "accumulator_pressure": profile[:, 3],
            "stable_flag": profile[:, 4],
        }
    )
    metadata.to_csv(FEATURE_DIR / "cycle_metadata.csv", index=False)
    for seconds, parts in frames.items():
        output = pd.concat([metadata[["cycle_id", "pump_leakage"]], *parts], axis=1)
        if not np.isfinite(output.iloc[:, 2:].to_numpy()).all():
            raise ValueError(f"Non-finite feature found at {seconds}s")
        output.to_csv(outputs[seconds], index=False)
        print(f"Saved {outputs[seconds]}: {output.shape}")

    audit = {
        "cycles": len(profile),
        "sensors": len(SENSORS),
        "features_per_sensor": 9,
        "feature_count": len(SENSORS) * 9,
        "observation_times_seconds": list(OBSERVATION_TIMES),
        "causality_rule": "Each file uses values[:, :seconds * sensor_sampling_rate] only.",
    }
    (FEATURE_DIR / "feature_manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def load_feature_table(seconds: int) -> pd.DataFrame:
    if seconds not in OBSERVATION_TIMES:
        raise ValueError(f"seconds must be one of {OBSERVATION_TIMES}")
    return pd.read_csv(FEATURE_DIR / f"features_{seconds}s.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_features(force=args.force)


if __name__ == "__main__":
    main()

