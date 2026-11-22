"""Streamlit dashboard for reliability-oriented hydraulic fault diagnosis."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import CLASS_NAMES, DATA_DIR, MODEL_DIR, OBSERVATION_TIMES, RAW_DIR, RESULT_DIR, SENSORS
from multi_component import (
    COMPONENT_SPECS,
    MULTI_RESULT_DIR,
    ground_truth_for_cycle,
    predict_all_components,
    predict_component,
    selected_bundle_path,
)
from train import get_xy
from uncertainty import decision_status

st.set_page_config(page_title="Hydraulic Fault Diagnosis", page_icon="⚙️", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem;}
    [data-testid="stMetric"] {background:#f5f8fb;border:1px solid #dce6ef;padding:12px;border-radius:8px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def required(path: Path) -> Path:
    if not path.exists():
        st.error(f"Required base artifact is missing: {path.name}. Follow the README pipeline first.")
        st.stop()
    return path


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_bundle(seconds: int):
    return joblib.load(required(MODEL_DIR / f"pump_calibrated_{seconds}s.joblib"))


@st.cache_data
def load_signal(sensor: str, cycle_id: int) -> pd.DataFrame:
    rate, quantity, unit = SENSORS[sensor]
    values = np.loadtxt(RAW_DIR / f"{sensor}.txt", skiprows=cycle_id - 1, max_rows=1)
    return pd.DataFrame({"time_seconds": np.arange(values.size) / rate, "value": values, "sensor": sensor, "unit": unit})


def multi_models_ready(seconds: int | None = None) -> bool:
    selection = MULTI_RESULT_DIR / "selected_demo_models.json"
    if not selection.exists():
        return False
    times = [seconds] if seconds is not None else list(OBSERVATION_TIMES)
    try:
        return all(selected_bundle_path(component, time).exists() for component in COMPONENT_SPECS for time in times)
    except (KeyError, FileNotFoundError, json.JSONDecodeError):
        return False


@st.cache_data(show_spinner=False)
def cached_predictions(seconds: int, cycle_id: int) -> list[dict]:
    return predict_all_components(seconds, cycle_id)


def render_sensor_signals(cycle_id: int, observation: int, sensors_to_plot: list[str]) -> None:
    if not sensors_to_plot:
        return
    signal_parts = []
    for sensor in sensors_to_plot:
        signal = load_signal(sensor, cycle_id)
        signal_parts.append(signal[signal["time_seconds"] < observation])
    signals = pd.concat(signal_parts, ignore_index=True)
    fig = px.line(signals, x="time_seconds", y="value", facet_row="sensor", color="sensor", title=f"Observed signals only: 0–{observation} s")
    fig.update_yaxes(matches=None)
    fig.update_layout(showlegend=False, height=220 * len(sensors_to_plot))
    st.plotly_chart(fig, width="stretch")


def render_legacy_pump_diagnosis(cycle_id: int, observation: int, demo_split: str, reveal: bool) -> None:
    """Preserve the original calibrated pump path as a safe fallback."""
    bundle = load_bundle(observation)
    X, y, ids, _ = get_xy(observation, demo_split)
    row_index = int(np.flatnonzero(ids.to_numpy() == cycle_id)[0])
    probabilities = bundle["model"].predict_proba(X.iloc[[row_index]][bundle["features"]])[0]
    predicted = int(probabilities.argmax())
    confidence = float(probabilities.max())
    threshold = float(thresholds[str(observation)]["threshold"])
    status = decision_status(observation, confidence, threshold)
    cols = st.columns(5)
    cols[0].metric("Observation", f"{observation} s")
    cols[1].metric("Pump prediction", CLASS_NAMES[predicted])
    cols[2].metric("Calibrated confidence", f"{confidence:.1%}")
    cols[3].metric("Validation threshold", f"{threshold:.0%}")
    cols[4].metric("Decision", status)
    probability_df = pd.DataFrame({"Condition": list(CLASS_NAMES.values()), "Probability": probabilities})
    st.plotly_chart(px.bar(probability_df, x="Condition", y="Probability", range_y=[0, 1], color="Condition", title="Calibrated pump class probabilities"), width="stretch")
    if reveal:
        actual = int(y.iloc[row_index])
        st.info(f"Pump ground truth: **{CLASS_NAMES[actual]}** ({'correct' if actual == predicted else 'incorrect'} prediction)")


manifest = json.loads(required(DATA_DIR / "dataset_manifest.json").read_text(encoding="utf-8"))
thresholds = json.loads(required(RESULT_DIR / "confidence_thresholds.json").read_text(encoding="utf-8"))
split = load_csv(str(required(DATA_DIR / "split_assignments.csv")))
test_metrics = load_csv(str(required(RESULT_DIR / "test_model_metrics.csv")))

st.title("Hydraulic System Multi-Component Fault Diagnosis")
st.caption("Pump · Valve · Cooler · Accumulator | causal observation windows | generalization-aware evaluation")

with st.sidebar:
    st.header("Demo controls")
    demo_split = st.radio("Cycle source", ["validation", "test"], index=0, help="Use validation during development; the saved test split is historical, not a new independent evaluation.")
    available = split.loc[split["split"] == demo_split, "cycle_id"].tolist()
    cycle_id = st.selectbox("Operating cycle", available)
    diagnosis_target = st.selectbox("Diagnosis target", ["All Components", "Pump", "Valve", "Cooler", "Accumulator"])
    observation = st.select_slider("Observation time", options=list(OBSERVATION_TIMES), value=10)
    sensors_to_plot = st.multiselect("Signals", list(SENSORS), default=["PS1", "FS1", "EPS1"], max_selections=5)
    reveal = st.checkbox("Reveal ground truth", value=False)

tabs = st.tabs(["Live diagnosis", "Prediction timeline", "Model performance", "Sensor optimization", "Multi-fault analysis", "Dataset & methodology"])

with tabs[0]:
    if multi_models_ready(observation):
        truth = ground_truth_for_cycle(cycle_id) if reveal else None
        if diagnosis_target == "All Components":
            predictions = cached_predictions(observation, cycle_id)
            rows = []
            for result in predictions:
                component = result["component"]
                row = {
                    "Component": result["component_display"],
                    "Predicted condition": result["predicted_label"],
                    "Confidence": result["confidence"],
                    "Selected model": result["model"].replace("_", " ").title(),
                }
                if reveal:
                    row["Ground truth"] = truth[component]["label"]
                    row["Match"] = result["predicted_code"] == truth[component]["code"]
                rows.append(row)
            result_df = pd.DataFrame(rows)
            st.dataframe(
                result_df,
                width="stretch",
                hide_index=True,
                column_config={"Confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="percent")},
            )
            st.caption("Confidence is the selected demo model's class probability; unlike the legacy pump panel, these probabilities are not separately calibrated.")
        else:
            component = diagnosis_target.lower()
            result = predict_component(component, observation, cycle_id)
            cols = st.columns(4)
            cols[0].metric("Observation", f"{observation} s")
            cols[1].metric("Prediction", result["predicted_label"])
            cols[2].metric("Model probability", f"{result['confidence']:.1%}")
            cols[3].metric("Selected model", result["model"].replace("_", " ").title())
            probability_df = pd.DataFrame({"Condition": result["probabilities"].keys(), "Probability": result["probabilities"].values()})
            st.plotly_chart(px.bar(probability_df, x="Condition", y="Probability", range_y=[0, 1], color="Condition", title=f"{diagnosis_target} class probabilities"), width="stretch")
            if reveal:
                actual = truth[component]
                st.info(f"Ground truth: **{actual['label']}** ({'correct' if actual['code'] == result['predicted_code'] else 'incorrect'} prediction)")
    else:
        st.warning("Multi-component demo artifacts are incomplete. Run `python multi_component.py train` to restore them. Existing calibrated Pump diagnosis remains available below.")
        if diagnosis_target in ("All Components", "Pump"):
            render_legacy_pump_diagnosis(cycle_id, observation, demo_split, reveal)
    render_sensor_signals(cycle_id, observation, sensors_to_plot)

with tabs[1]:
    if multi_models_ready():
        timeline_rows = []
        target_components = list(COMPONENT_SPECS) if diagnosis_target == "All Components" else [diagnosis_target.lower()]
        for seconds in OBSERVATION_TIMES:
            for component in target_components:
                result = predict_component(component, seconds, cycle_id)
                timeline_rows.append({
                    "Observation (s)": seconds,
                    "Component": result["component_display"],
                    "Predicted condition": result["predicted_label"],
                    "Confidence": result["confidence"],
                    "Model": result["model"].replace("_", " ").title(),
                })
        timeline_df = pd.DataFrame(timeline_rows)
        st.plotly_chart(px.line(timeline_df, x="Observation (s)", y="Confidence", color="Component", markers=True, range_y=[0, 1.03], title="Selected demo-model confidence over time"), width="stretch")
        display = timeline_df.copy()
        display["Confidence"] = display["Confidence"].map(lambda value: f"{value:.1%}")
        st.dataframe(display, width="stretch", hide_index=True)
        if reveal:
            truth = ground_truth_for_cycle(cycle_id)
            truth_rows = [{"Component": COMPONENT_SPECS[key]["display"], "Ground truth": truth[key]["label"]} for key in target_components]
            st.dataframe(pd.DataFrame(truth_rows), width="stretch", hide_index=True)
    else:
        st.warning("Prediction timelines for all components require `python multi_component.py train`.")

with tabs[2]:
    performance_view = st.radio("Evaluation result", ["Nested generalization benchmark", "Historical fixed test"], horizontal=True)
    if performance_view == "Nested generalization benchmark" and (MULTI_RESULT_DIR / "nested_cv_summary.csv").exists():
        st.info("Five outer-fold out-of-fold evaluation. Inner folds alone select hyperparameters. Block and condition groups never cross an outer train/evaluation boundary.")
        summary = load_csv(str(MULTI_RESULT_DIR / "nested_cv_summary.csv"))
        c1, c2, c3 = st.columns(3)
        component_choice = c1.selectbox("Component", [spec["display"] for spec in COMPONENT_SPECS.values()], key="benchmark_component")
        benchmark_component = component_choice.lower()
        benchmark_model = c2.selectbox("Model", ["logistic_regression", "random_forest"], format_func=lambda value: value.replace("_", " ").title())
        metric_choice = c3.selectbox("Metric", ["macro_f1", "accuracy", "weighted_f1", "severe_recall"], format_func=lambda value: value.replace("_", " ").title())
        chart = summary[(summary["component"] == benchmark_component) & (summary["model"] == benchmark_model)].copy()
        chart["Scheme"] = chart["scheme"].map({"random_cycle": "Random cycle", "block_aware": "Block-aware", "condition_holdout": "Condition holdout"})
        mean_col, std_col = f"{metric_choice}_mean", f"{metric_choice}_std"
        st.plotly_chart(px.bar(chart, x="observation_seconds", y=mean_col, error_y=std_col, color="Scheme", barmode="group", range_y=[0, 1.03], labels={"observation_seconds": "Observation (s)", mean_col: metric_choice.replace("_", " ").title()}, title=f"Nested OOF {metric_choice.replace('_', ' ').title()} (mean ± fold SD)"), width="stretch")
        display_cols = ["scheme", "observation_seconds", "model", mean_col, std_col]
        st.dataframe(chart[display_cols].sort_values(["observation_seconds", "scheme"]), width="stretch", hide_index=True)
        st.subheader("Confusion matrix")
        q1, q2 = st.columns(2)
        matrix_scheme = q1.selectbox("Validation scheme", ["random_cycle", "block_aware", "condition_holdout"], format_func=lambda value: value.replace("_", " ").title())
        matrix_time = q2.selectbox("Observation", [5, 60], format_func=lambda value: f"{value} s")
        matrix_path = MULTI_RESULT_DIR / "confusion_matrices" / f"{benchmark_component}_{matrix_scheme}_{matrix_time}s_{benchmark_model}.png"
        if matrix_path.exists():
            st.image(str(matrix_path), caption="Aggregated out-of-fold confusion matrix")
    elif performance_view == "Historical fixed test" and (MULTI_RESULT_DIR / "demo_model_metrics.csv").exists():
        st.warning("This is the repeatedly inspected saved test split for demo-model reporting. It is not a new independent final evaluation.")
        demo_metrics = load_csv(str(MULTI_RESULT_DIR / "demo_model_metrics.csv"))
        d1, d2, d3 = st.columns(3)
        component_choice = d1.selectbox("Component", [spec["display"] for spec in COMPONENT_SPECS.values()], key="demo_component")
        demo_model = d2.selectbox("Model", sorted(demo_metrics["model"].unique()), format_func=lambda value: value.replace("_", " ").title(), key="demo_model")
        demo_metric = d3.selectbox("Metric", ["accuracy", "macro_f1", "weighted_f1", "severe_recall"], format_func=lambda value: value.replace("_", " ").title(), key="demo_metric")
        chart = demo_metrics[(demo_metrics["component"] == component_choice.lower()) & (demo_metrics["model"] == demo_model) & (demo_metrics["split"] == "test")]
        st.plotly_chart(px.line(chart, x="observation_seconds", y=demo_metric, markers=True, range_y=[0, 1.03], title=f"Historical fixed-test {demo_metric.replace('_', ' ').title()}"), width="stretch")
        st.dataframe(chart[["observation_seconds", "model", "accuracy", "macro_f1", "weighted_f1", "severe_recall", "cycles"]], width="stretch", hide_index=True)
    else:
        st.warning("Multi-component performance artifacts are missing. Run the training and benchmark commands shown in the README.")

    with st.expander("Original calibrated Pump uncertainty analysis"):
        chosen_model = st.selectbox("Legacy Pump model", sorted(test_metrics["model"].unique()))
        metric_name = st.radio("Legacy metric", ["accuracy", "macro_f1", "severe_recall", "balanced_accuracy"], horizontal=True)
        chart_data = test_metrics[test_metrics["model"] == chosen_model]
        st.plotly_chart(px.line(chart_data, x="observation_seconds", y=metric_name, markers=True, range_y=[0, 1.03], title=f"Historical Pump test {metric_name.replace('_', ' ').title()}"), width="stretch")
        uncertainty = load_csv(str(required(RESULT_DIR / "uncertainty_metrics.csv")))
        st.dataframe(uncertainty[uncertainty["split"] == "test"], width="stretch", hide_index=True)
        curves = load_csv(str(required(RESULT_DIR / "coverage_accuracy_curves.csv")))
        curves = curves[curves["split"] == "test"]
        st.plotly_chart(px.line(curves, x="coverage", y="confirmed_accuracy", color="observation_seconds", range_x=[0, 1], range_y=[0, 1.03], title="Coverage–accuracy trade-off"), width="stretch")

with tabs[3]:
    importance = load_csv(str(required(RESULT_DIR / "sensor_importance.csv")))
    subsets = load_csv(str(required(RESULT_DIR / "sensor_subset_metrics.csv")))
    st.plotly_chart(px.bar(importance.sort_values("importance_mean_macro_f1_drop"), x="importance_mean_macro_f1_drop", y="sensor", orientation="h", error_x="importance_std", title="Pump validation grouped permutation importance (60 s)"), width="stretch")
    test_subsets = subsets[subsets["split"] == "test"]
    st.plotly_chart(px.line(test_subsets, x="sensor_count", y="macro_f1", color="observation_seconds", markers=True, title="Pre-trained Pump sensor subsets: historical test macro-F1"), width="stretch")
    st.caption("Rankings are predictive associations, not causal effects. Each point is a separately trained model using a subset fixed from validation data.")

with tabs[4]:
    fault_path = MULTI_RESULT_DIR / "single_vs_multi_fault.csv"
    holdout_path = MULTI_RESULT_DIR / "condition_holdout_analysis.csv"
    if fault_path.exists() and holdout_path.exists():
        fault = load_csv(str(fault_path))
        distribution = fault[(fault["analysis_level"] == "fault_count") & (fault["component"] == "pump") & (fault["scheme"] == "random_cycle") & (fault["observation_seconds"] == 5) & (fault["model"] == "logistic_regression")][["group", "cycles"]].drop_duplicates().sort_values("group")
        distribution["group"] = distribution["group"].astype(str)
        st.plotly_chart(px.bar(distribution, x="group", y="cycles", labels={"group": "Abnormal components", "cycles": "Cycles"}, title="Distribution by number of abnormal components"), width="stretch")
        f1 = fault[(fault["analysis_level"] == "fault_count") & (fault["observation_seconds"] == 60) & (fault["scheme"] == "condition_holdout")].copy()
        st.plotly_chart(px.line(f1, x="group", y="macro_f1", color="component", line_dash="model", markers=True, range_y=[0, 1.03], labels={"group": "Abnormal components"}, title="Condition-holdout 60 s performance by fault count"), width="stretch")
        st.caption("These post-hoc groups have different target-class mixtures. Their performance difference is descriptive and must not be interpreted as a causal effect of multiple faults.")
        holdout = load_csv(str(holdout_path))
        vulnerable = holdout[holdout["analysis_level"] == "class_comparison"].copy().nsmallest(12, "recall_change_holdout_minus_random")
        st.subheader("Classes most affected by condition holdout")
        st.dataframe(vulnerable[["component", "observation_seconds", "model", "class_label", "random_recall", "holdout_recall", "recall_change_holdout_minus_random", "support_holdout"]], width="stretch", hide_index=True)
        condition_errors = holdout[holdout["analysis_level"] == "condition_error"].copy()
        if not condition_errors.empty:
            st.subheader("Highest-error held-out conditions")
            st.dataframe(condition_errors.sort_values("errors", ascending=False).head(15)[["component", "observation_seconds", "model", "fault_pattern", "cycles", "accuracy", "errors"]], width="stretch", hide_index=True)
    else:
        st.warning("Multi-fault analysis artifacts are missing. Run `python multi_component_benchmark.py`.")

with tabs[5]:
    counts = split.groupby("split").size()
    cols = st.columns(5)
    cols[0].metric("Cycles", manifest["profile_shape"][0])
    cols[1].metric("Train", int(counts["train"]))
    cols[2].metric("Validation", int(counts["validation"]))
    cols[3].metric("Test", int(counts["test"]))
    cols[4].metric("Sensors", len(SENSORS))
    sensor_info = pd.DataFrame([{"Sensor": name, "Sampling rate (Hz)": rate, "Quantity": quantity, "Unit": unit} for name, (rate, quantity, unit) in SENSORS.items()])
    st.dataframe(sensor_info, width="stretch", hide_index=True)
    if (MULTI_RESULT_DIR / "component_label_distribution.csv").exists():
        label_distribution = load_csv(str(MULTI_RESULT_DIR / "component_label_distribution.csv"))
        st.plotly_chart(px.bar(label_distribution, x="component", y="count", color="class_label", barmode="group", title="Four-component label distribution"), width="stretch")
    st.markdown("""
    **Leakage controls.** Original cycle IDs align sensor rows and all four target labels. Target labels, Stable Flag, block IDs, and condition IDs are never predictors. Every early feature uses only the prefix ending at its named observation time. Imputation, scaling, variance filtering, model fitting, and hyperparameter selection are fit inside the relevant training fold.

    **Evaluation distinction.** Demo models use the saved train/validation/test split for interactive diagnosis. The research benchmark uses five outer folds and three inner folds; its OOF results are the appropriate evidence for split sensitivity. The saved test set has already been inspected and is not called a new independent test.

    **Group limitations.** Contiguous identical five-label runs are inferred blocks, not known acquisition sessions. Condition holdout tests unseen four-component state combinations within this apparatus; it does not demonstrate transfer to a new machine.
    """)

