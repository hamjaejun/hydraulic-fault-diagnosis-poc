"""Causal Valve transient features and exploratory raw-signal analysis.

The approximately 10 s event is a measured response event, not a valve command
or spool-position measurement.  Every model feature uses samples strictly before
the requested observation endpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from config import DATA_DIR, FEATURE_DIR, MODEL_DIR, RANDOM_SEED, RAW_DIR, RESULT_DIR, ROOT, SENSORS
from multi_component import COMPONENT_SPECS
from preprocess import extract_features, feature_names
from train import make_model


VALVE_RESULT_DIR = RESULT_DIR / "valve_transient_analysis"
VALVE_FEATURE_DIR = FEATURE_DIR / "valve_transient"
VALVE_MODEL_DIR = MODEL_DIR / "valve_transient"
VALVE_TIMES = (2, 5, 8, 10, 12, 15, 20, 30, 40, 60)
TRANSIENT_SENSORS = ("PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "FS1", "FS2", "EPS1")
VALVE_CLASSES = (73, 80, 90, 100)
VALVE_LABELS = COMPONENT_SPECS["valve"]["labels"]
NOMINAL_RESPONSE_SECONDS = 10.0
EVENT_SEARCH_START_SECONDS = 8.0
EVENT_SEARCH_END_SECONDS = 12.0


def ensure_directories() -> None:
    for path in (VALVE_RESULT_DIR, VALVE_RESULT_DIR / "figures", VALVE_FEATURE_DIR, VALVE_MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_profile() -> pd.DataFrame:
    raw = np.loadtxt(RAW_DIR / "profile.txt", dtype=int)
    if raw.shape != (2205, 5):
        raise ValueError(f"Expected profile shape (2205, 5), got {raw.shape}")
    return pd.DataFrame(
        {
            "cycle_id": np.arange(1, 2206),
            "cooler_condition": raw[:, 0],
            "valve_condition": raw[:, 1],
            "pump_leakage": raw[:, 2],
            "accumulator_pressure": raw[:, 3],
            "stable_flag": raw[:, 4],
        }
    )


def load_sensor(sensor: str) -> np.ndarray:
    rate = SENSORS[sensor][0]
    values = np.loadtxt(RAW_DIR / f"{sensor}.txt", dtype=np.float32)
    expected = (2205, rate * 60)
    if values.shape != expected:
        raise ValueError(f"{sensor}: expected {expected}, got {values.shape}")
    return values


def to_10_hz(values: np.ndarray, rate: int) -> np.ndarray:
    if rate == 10:
        return values.astype(np.float32, copy=False)
    if rate == 100:
        return values.reshape(values.shape[0], 600, 10).mean(axis=2, dtype=np.float32)
    raise ValueError("Transient features are defined only for 10 Hz and 100 Hz sensors")


def _smooth_rows(x: np.ndarray, width: int = 5) -> np.ndarray:
    if x.shape[1] < width:
        return x.astype(np.float64)
    cumulative = np.cumsum(np.pad(x.astype(np.float64), ((0, 0), (1, 0))), axis=1)
    core = (cumulative[:, width:] - cumulative[:, :-width]) / width
    left = np.repeat(core[:, :1], width // 2, axis=1)
    right = np.repeat(core[:, -1:], x.shape[1] - core.shape[1] - left.shape[1], axis=1)
    return np.column_stack([left, core, right])


def _crossing_times(segment: np.ndarray, baseline: np.ndarray, endpoint: np.ndarray, start_seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    amplitude = endpoint - baseline
    direction = np.where(amplitude >= 0, 1.0, -1.0)
    directed = (segment - baseline[:, None]) * direction[:, None]
    magnitude = np.abs(amplitude)
    valid = magnitude > np.maximum(np.std(segment, axis=1) * 0.5, 1e-9)
    threshold10 = magnitude[:, None] * 0.1
    threshold90 = magnitude[:, None] * 0.9
    crossed10 = directed >= threshold10
    crossed90 = directed >= threshold90
    has10 = crossed10.any(axis=1) & valid
    has90 = crossed90.any(axis=1) & valid
    t10 = np.where(has10, start_seconds + crossed10.argmax(axis=1) / 10.0, np.nan)
    t90 = np.where(has90, start_seconds + crossed90.argmax(axis=1) / 10.0, np.nan)
    rise = np.where(has10 & has90 & (t90 >= t10), t90 - t10, np.nan)
    return t10, t90, rise


def transient_features(values_10hz: np.ndarray, sensor: str, seconds: int) -> pd.DataFrame:
    count = seconds * 10
    x = values_10hz[:, :count].astype(np.float64, copy=False)
    smooth = _smooth_rows(x)
    derivative = np.diff(smooth, axis=1) * 10.0 if count > 1 else np.zeros((len(x), 1))
    quarter = max(1, count // 4)
    half = max(1, count // 2)
    data: dict[str, np.ndarray] = {
        "range": np.ptp(x, axis=1),
        "diff_std": derivative.std(axis=1),
        "max_abs_derivative": np.max(np.abs(derivative), axis=1),
        "p95_abs_derivative": np.percentile(np.abs(derivative), 95, axis=1),
        "derivative_energy": np.mean(np.square(derivative), axis=1),
        "early_late_delta": x[:, -quarter:].mean(axis=1) - x[:, :quarter].mean(axis=1),
        "last_quarter_std": x[:, -quarter:].std(axis=1),
        "peak_time": np.argmax(x, axis=1) / 10.0,
        "trough_time": np.argmin(x, axis=1) / 10.0,
        "half_delta": x[:, half:].mean(axis=1) - x[:, :half].mean(axis=1),
    }
    if seconds >= 12:
        search_start = int(EVENT_SEARCH_START_SECONDS * 10)
        search_end = min(count, int(EVENT_SEARCH_END_SECONDS * 10))
        event_derivative = np.abs(derivative[:, search_start : search_end - 1])
        event_index = event_derivative.argmax(axis=1)
        data["response_event_time"] = EVENT_SEARCH_START_SECONDS + (event_index + 1) / 10.0
        data["event_max_abs_derivative"] = event_derivative[np.arange(len(x)), event_index]

        pre = x[:, int(8.0 * 10) : int(9.5 * 10)]
        post = x[:, int(10.5 * 10) : min(count, int(12.0 * 10))]
        pre_mean = pre.mean(axis=1)
        post_mean = post.mean(axis=1)
        data["pre_response_mean"] = pre_mean
        data["post_response_mean"] = post_mean
        data["response_amplitude"] = post_mean - pre_mean
        data["pre_response_std"] = pre.std(axis=1)
        data["post_response_std"] = post.std(axis=1)
        event_segment = x[:, int(8.0 * 10) : min(count, int(12.0 * 10))]
        data["response_variance"] = event_segment.var(axis=1)
        direction = np.where(post_mean >= pre_mean, 1.0, -1.0)
        directed_peak = (event_segment - post_mean[:, None]) * direction[:, None]
        data["response_overshoot"] = np.maximum(0.0, directed_peak.max(axis=1))
        settle = x[:, max(int(11.0 * 10), count - 10) : count]
        data["settling_deviation"] = np.mean(np.abs(settle - post_mean[:, None]), axis=1)
        t10, t90, rise = _crossing_times(event_segment, pre_mean, post_mean, 8.0)
        data["t10"] = t10
        data["t90"] = t90
        data["rise_time"] = rise
    return pd.DataFrame({f"trans__{sensor}__{name}": value for name, value in data.items()})


def build_feature_tables(force: bool = False) -> None:
    ensure_directories()
    outputs = {seconds: VALVE_FEATURE_DIR / f"valve_features_{seconds}s.csv" for seconds in VALVE_TIMES}
    if not force and all(path.exists() for path in outputs.values()):
        print("Valve feature tables already exist; use --force to rebuild.")
        return
    profile = load_profile()
    frames: dict[int, list[pd.DataFrame]] = {seconds: [] for seconds in VALVE_TIMES}
    for sensor, (rate, _, _) in SENSORS.items():
        print(f"Reading {sensor} ({rate} Hz)")
        raw = load_sensor(sensor)
        for seconds in VALVE_TIMES:
            stat = extract_features(raw, rate, seconds)
            stat_names = [f"stat__{name.replace('__', '__', 1)}" for name in feature_names(sensor)]
            frames[seconds].append(pd.DataFrame(stat, columns=stat_names))
        if sensor in TRANSIENT_SENSORS:
            values_10hz = to_10_hz(raw, rate)
            for seconds in VALVE_TIMES:
                frames[seconds].append(transient_features(values_10hz, sensor, seconds))
        del raw

    for seconds, parts in frames.items():
        output = pd.concat([profile[["cycle_id", "valve_condition"]], *parts], axis=1)
        output.to_csv(outputs[seconds], index=False)
        stat_count = sum(column.startswith("stat__") for column in output)
        transient_count = sum(column.startswith("trans__") for column in output)
        print(f"Saved {outputs[seconds]}: {len(output)} cycles, {stat_count} statistical + {transient_count} transient features")
    manifest = {
        "cycles": 2205,
        "observation_times_seconds": list(VALVE_TIMES),
        "statistical_features": "9 statistics for all 17 sensors",
        "transient_sensors": list(TRANSIENT_SENSORS),
        "transient_sampling_hz": 10,
        "nominal_response_seconds": NOMINAL_RESPONSE_SECONDS,
        "event_interpretation": "sensor response event; not valve command or spool position",
        "causality": "all features use samples strictly before the observation endpoint",
    }
    (VALVE_FEATURE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_feature_table(seconds: int) -> pd.DataFrame:
    if seconds not in VALVE_TIMES:
        raise ValueError(f"seconds must be one of {VALVE_TIMES}")
    return pd.read_csv(VALVE_FEATURE_DIR / f"valve_features_{seconds}s.csv")


def _plot_class_lines(ax, time: np.ndarray, values: np.ndarray, labels: np.ndarray, title: str) -> None:
    colors = {73: "#dc2626", 80: "#f97316", 90: "#eab308", 100: "#16a34a"}
    for class_id in VALVE_CLASSES:
        mean = values[labels == class_id].mean(axis=0)
        ax.plot(time, mean, color=colors[class_id], label=f"{class_id}: {VALVE_LABELS[class_id]}", linewidth=1.4)
    ax.set_title(title)
    ax.grid(alpha=0.2)


def run_signal_analysis() -> None:
    ensure_directories()
    profile = load_profile()
    labels = profile["valve_condition"].to_numpy()
    signals: dict[str, np.ndarray] = {}
    peak_rows: list[dict] = []
    separability_rows: list[dict] = []
    for sensor in TRANSIENT_SENSORS:
        values = to_10_hz(load_sensor(sensor), SENSORS[sensor][0])
        signals[sensor] = values
        for class_id in VALVE_CLASSES:
            mean = values[labels == class_id].mean(axis=0)
            smooth = _smooth_rows(mean[None, :])[0]
            derivative = np.abs(np.diff(smooth)) * 10
            lo, hi = 50, 150
            index = int(np.argmax(derivative[lo:hi]) + lo)
            peak_rows.append({"sensor": sensor, "valve_class": class_id, "response_peak_seconds": (index + 1) / 10, "peak_abs_derivative": derivative[index]})
        class_means = np.stack([values[labels == class_id].mean(axis=0) for class_id in VALVE_CLASSES])
        between = class_means.var(axis=0)
        within = np.mean([values[labels == class_id].var(axis=0) for class_id in VALVE_CLASSES], axis=0)
        ratio = between / (within + 1e-12)
        for index in np.argsort(ratio)[-20:][::-1]:
            separability_rows.append({"sensor": sensor, "time_seconds": index / 10, "between_within_variance_ratio": ratio[index]})
    pd.DataFrame(peak_rows).to_csv(VALVE_RESULT_DIR / "response_peak_summary.csv", index=False)
    pd.DataFrame(separability_rows).to_csv(VALVE_RESULT_DIR / "signal_separability_peaks.csv", index=False)

    time = np.arange(600) / 10
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True)
    for ax, sensor in zip(axes.flat, TRANSIENT_SENSORS):
        _plot_class_lines(ax, time, signals[sensor], labels, sensor)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Valve-class mean sensor signals over the 60 s cycle")
    fig.tight_layout()
    fig.savefig(VALVE_RESULT_DIR / "figures" / "valve_class_mean_signals.png", dpi=170)
    plt.close(fig)

    focus = ("PS1", "PS2", "PS3", "FS1", "EPS1")
    for name, limit, filename in (("Initial 0–5 s", 5, "initial_5s_class_means.png"), ("First measured response region", 13, "first_response_class_means.png")):
        fig, axes = plt.subplots(len(focus), 1, figsize=(12, 12), sharex=True)
        for ax, sensor in zip(axes, focus):
            _plot_class_lines(ax, time[: limit * 10], signals[sensor][:, : limit * 10], labels, sensor)
            if limit > 10:
                ax.axvspan(9.3, 10.2, color="#64748b", alpha=0.16, label="observed response band")
        axes[0].legend(fontsize=7, ncol=2)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(VALVE_RESULT_DIR / "figures" / filename, dpi=170)
        plt.close(fig)

    rng = np.random.default_rng(RANDOM_SEED)
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for ax, sensor in zip(axes, ("PS2", "FS1", "EPS1")):
        for class_id, color in zip(VALVE_CLASSES, ("#dc2626", "#f97316", "#eab308", "#16a34a")):
            members = np.flatnonzero(labels == class_id)
            chosen = rng.choice(members, size=5, replace=False)
            ax.plot(time[:150], signals[sensor][chosen, :150].T, color=color, alpha=0.16, linewidth=0.7)
            ax.plot(time[:150], signals[sensor][members, :150].mean(axis=0), color=color, linewidth=1.8, label=str(class_id))
        ax.set_title(sensor)
        ax.grid(alpha=0.2)
    axes[0].legend(title="Valve class", ncol=4)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Individual-cycle variability and class means (0–15 s)")
    fig.tight_layout()
    fig.savefig(VALVE_RESULT_DIR / "figures" / "individual_cycle_variability.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for ax, sensor in zip(axes, ("PS2", "FS1", "EPS1")):
        for cooler, color in zip((3, 20, 100), ("#dc2626", "#f59e0b", "#2563eb")):
            mask = (labels == 100) & profile["cooler_condition"].eq(cooler).to_numpy()
            ax.plot(time[:150], signals[sensor][mask, :150].mean(axis=0), color=color, label=f"Cooler {cooler}")
        ax.set_title(f"{sensor}, Valve 100 only")
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Other-component domain effect: cooler condition")
    fig.tight_layout()
    fig.savefig(VALVE_RESULT_DIR / "figures" / "cooler_stratified_response.png", dpi=170)
    plt.close(fig)

    peak_frame = pd.DataFrame(peak_rows)
    lines = [
        "# Valve Signal Analysis",
        "",
        "## Scope and interpretation",
        "",
        "The UCI documentation states that the rig repeats fixed 60-second load cycles, but it does not publish a valve command trace or spool position. Therefore the event below is a **measured sensor-response event**, not a directly measured switching command.",
        "",
        "## Observed first response",
        "",
        "Across the class-mean PS1–PS3, FS1, and EPS1 signals, the strongest early derivative occurs around 9.3–10.2 seconds. The class-specific EPS1 peaks range from "
        f"{peak_frame.loc[peak_frame.sensor.eq('EPS1'), 'response_peak_seconds'].min():.1f} to {peak_frame.loc[peak_frame.sensor.eq('EPS1'), 'response_peak_seconds'].max():.1f} seconds. "
        "This independently reproduces the approximate region reported by the teammate, without treating their accumulator work as part of this repository.",
        "",
        "The 0–5 second plots show far less separation because the dominant pressure/flow/power response has not yet occurred. This is evidence consistent with insufficient early-cycle switching information, not proof of the physical command timing.",
        "",
        "## Feature policy",
        "",
        "Transient features are computed from each cycle independently. Event-specific pre/post, t10, t90, rise-time, overshoot, and settling proxies are created only for observation windows of at least 12 seconds. No future sample is used, and no pressure-pair feature is called a valve pressure drop because sensor placement is not available in the dataset documentation.",
        "",
        "## Figures",
        "",
        "- `valve_class_mean_signals.png`: full-cycle class means for nine pressure/flow/power sensors.",
        "- `initial_5s_class_means.png`: initial region before the dominant measured response.",
        "- `first_response_class_means.png`: first measured response region with the observed 9.3–10.2 s band.",
        "- `individual_cycle_variability.png`: individual cycles plus class means.",
        "- `cooler_stratified_response.png`: same valve class under different cooler domains.",
    ]
    (VALVE_RESULT_DIR / "VALVE_SIGNAL_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


def train_demo_models() -> None:
    ensure_directories()
    assignments = pd.read_csv(DATA_DIR / "split_assignments.csv").set_index("cycle_id")["split"]
    selected: dict[str, dict] = {}
    metric_rows = []
    for seconds in VALVE_TIMES:
        table = load_feature_table(seconds)
        columns = [column for column in table if column.startswith(("stat__", "trans__"))]
        split = table["cycle_id"].map(assignments)
        train_mask, validation_mask = split.eq("train"), split.eq("validation")
        best = None
        for model_name, candidates in (
            ("logistic_regression", ({"C": 0.1}, {"C": 1.0}, {"C": 10.0})),
            ("random_forest", ({"n_estimators": 250, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"}, {"n_estimators": 250, "max_depth": 16, "min_samples_leaf": 2, "max_features": "sqrt"})),
        ):
            for params in candidates:
                model = make_model(model_name, params)
                model.fit(table.loc[train_mask, columns], table.loc[train_mask, "valve_condition"])
                prediction = model.predict(table.loc[validation_mask, columns])
                score = f1_score(table.loc[validation_mask, "valve_condition"], prediction, average="macro")
                metric_rows.append({"observation_seconds": seconds, "model": model_name, "params": json.dumps(params), "validation_macro_f1": score, "n_features": len(columns)})
                if best is None or score > best[0]:
                    best = (score, model_name, params, model)
        score, model_name, params, model = best
        path = VALVE_MODEL_DIR / f"valve_combined_{seconds}s.joblib"
        joblib.dump({"model": model, "features": columns, "seconds": seconds, "model_name": model_name, "params": params, "feature_set": "combined", "selection_split": "validation"}, path)
        selected[str(seconds)] = {"path": path.relative_to(ROOT).as_posix(), "model": model_name, "params": params, "validation_macro_f1": score}
        print(f"Valve transient demo {seconds}s: {model_name}, validation macro-F1={score:.4f}")
    pd.DataFrame(metric_rows).to_csv(VALVE_RESULT_DIR / "demo_model_selection.csv", index=False)
    (VALVE_RESULT_DIR / "selected_valve_demo_models.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")


def predict_cycle(seconds: int, cycle_id: int) -> dict:
    selection = json.loads((VALVE_RESULT_DIR / "selected_valve_demo_models.json").read_text(encoding="utf-8"))
    bundle_path = Path(selection[str(seconds)]["path"])
    if not bundle_path.is_absolute():
        bundle_path = ROOT / bundle_path
    bundle = joblib.load(bundle_path)
    table = load_feature_table(seconds)
    row = table.loc[table["cycle_id"] == cycle_id]
    probability = bundle["model"].predict_proba(row[bundle["features"]])[0]
    classes = [int(value) for value in bundle["model"].classes_]
    index = int(np.argmax(probability))
    actual = int(row.iloc[0]["valve_condition"])
    return {
        "cycle_id": cycle_id,
        "observation_seconds": seconds,
        "predicted_code": classes[index],
        "predicted_label": VALVE_LABELS[classes[index]],
        "actual_code": actual,
        "actual_label": VALVE_LABELS[actual],
        "confidence": float(probability[index]),
        "probabilities": {VALVE_LABELS[class_id]: float(probability[position]) for position, class_id in enumerate(classes)},
        "model": bundle["model_name"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-features")
    build.add_argument("--force", action="store_true")
    subparsers.add_parser("analyze-signals")
    subparsers.add_parser("train-demo")
    predict = subparsers.add_parser("predict")
    predict.add_argument("--seconds", type=int, choices=VALVE_TIMES, required=True)
    predict.add_argument("--cycle-id", type=int, required=True)
    args = parser.parse_args()
    if args.command == "build-features":
        build_feature_tables(force=args.force)
    elif args.command == "analyze-signals":
        run_signal_analysis()
    elif args.command == "train-demo":
        train_demo_models()
    else:
        print(json.dumps(predict_cycle(args.seconds, args.cycle_id), indent=2))


if __name__ == "__main__":
    main()
