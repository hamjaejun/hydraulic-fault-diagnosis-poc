"""Shared constants and paths for the hydraulic fault diagnosis PoC."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURE_DIR = DATA_DIR / "features"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
FIGURE_DIR = RESULT_DIR / "figures"

RANDOM_SEED = 42
OBSERVATION_TIMES = (5, 10, 20, 40, 60)
CLASS_NAMES = {0: "Normal", 1: "Weak Leakage", 2: "Severe Leakage"}

# Sampling rates are from the official UCI dataset documentation.
SENSORS = {
    "PS1": (100, "Pressure", "bar"),
    "PS2": (100, "Pressure", "bar"),
    "PS3": (100, "Pressure", "bar"),
    "PS4": (100, "Pressure", "bar"),
    "PS5": (100, "Pressure", "bar"),
    "PS6": (100, "Pressure", "bar"),
    "EPS1": (100, "Motor power", "W"),
    "FS1": (10, "Volume flow", "l/min"),
    "FS2": (10, "Volume flow", "l/min"),
    "TS1": (1, "Temperature", "degC"),
    "TS2": (1, "Temperature", "degC"),
    "TS3": (1, "Temperature", "degC"),
    "TS4": (1, "Temperature", "degC"),
    "VS1": (1, "Vibration", "mm/s"),
    "CE": (1, "Cooling efficiency", "%"),
    "CP": (1, "Cooling power", "kW"),
    "SE": (1, "Efficiency factor", "%"),
}

FEATURE_STATS = ("mean", "std", "min", "max", "rms", "initial", "final", "delta", "slope")


def ensure_directories() -> None:
    for path in (DATA_DIR, RAW_DIR, FEATURE_DIR, MODEL_DIR, RESULT_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)

