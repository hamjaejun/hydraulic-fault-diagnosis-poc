# Hydraulic Fault Diagnosis PoC

Reliability-oriented machine learning for early, multi-component fault diagnosis on the [UCI Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) dataset.

The project diagnoses Pump leakage, Valve switching condition, Cooler efficiency, and Hydraulic Accumulator pressure from causal prefixes of each 60-second operating cycle. It includes uncertainty analysis, sensor optimization, strict group-aware validation, multiple-fault analysis, transient-aware Valve features, and an interactive Streamlit dashboard.

## Project overview

The central research question is not simply whether a random split produces high accuracy. It is whether performance remains credible when contiguous operating blocks or complete four-component state combinations are withheld.

The Valve extension asks:

> Can transient-aware, physics-guided features improve hydraulic Valve diagnosis under unseen fault combinations?

This repository implements physics-guided feature engineering. It does **not** implement a physics-informed neural network (PINN).

## Dataset

The UCI dataset contains 2,205 operating cycles, each 60 seconds long, with 17 process channels sampled at 1, 10, or 100 Hz. `profile.txt` supplies five cycle-level state values:

- Cooler: 3, 20, 100
- Valve: 73, 80, 90, 100
- Pump leakage: 0, 1, 2
- Accumulator pressure: 90, 100, 115, 130
- Stable Flag: 0 or 1

Stable Flag and all component labels are excluded from predictors. Original raw sensor files are downloaded locally and are not committed.

## Implemented methods

- Component-specific Pump, Valve, Cooler, and Accumulator classification
- Dummy, Logistic Regression, and Random Forest baselines
- Causal 5/10/20/40/60-second multi-component features
- Valve-specific 2/5/8/10/12/15/20/30/40/60-second study
- Statistical and transient pressure/flow/motor-power features
- Per-cycle response amplitude, derivative, t10/t90, rise-time, overshoot, and settling proxies
- Five outer × three inner fold Nested Cross-Validation
- Random Cycle, inferred Block-aware, and four-component Condition Holdout evaluation
- Leave-One-Cooler-Condition-Out evaluation
- Train-fold-only reduced-sensor selection
- OOF predictions, classwise metrics, confusion matrices, and Cycle-ID error tracing

The event near 9.3–10.2 seconds is reported as a measured sensor-response region. The dataset does not provide Valve command or spool-position measurements, so it is not called the physical command time.

## Multi-component baseline results

The values below are the higher Macro-F1 of Logistic Regression and Random Forest under each condition. They do not represent one pre-fixed model across every column.

| Component | Time | Random | Block-aware | Condition Holdout |
|---|---:|---:|---:|---:|
| Pump | 5 s | 0.9965 | 0.9901 | 0.9881 |
| Pump | 60 s | 0.9972 | 0.9964 | 0.9943 |
| Valve | 5 s | 0.6424 | 0.1735 | 0.1450 |
| Valve | 60 s | 0.9829 | 0.8225 | 0.8682 |
| Cooler | 5 s | 0.9982 | 0.9787 | 0.9909 |
| Cooler | 60 s | 0.9995 | 0.9925 | 0.9943 |
| Accumulator | 5 s | 0.9776 | 0.9028 | 0.8990 |
| Accumulator | 60 s | 0.9804 | 0.8749 | 0.8439 |

See `results/multi_component_benchmark/MULTI_COMPONENT_REPORT.md` for both models, fold variability, class recalls, and limitations.

## Valve transient research results

### Observation time

Random Cycle Nested-CV statistical baseline Macro-F1:

| Time | Logistic Regression | Random Forest |
|---:|---:|---:|
| 2 s | 0.3973 | 0.6237 |
| 5 s | 0.3906 | 0.6285 |
| 8 s | 0.3949 | 0.6430 |
| 10 s | 0.9970 | 0.9987 |
| 12 s | 0.9984 | 0.9973 |
| 60 s | 0.9813 | 0.9815 |

The first tested endpoint exceeding mean Macro-F1 0.90 was 10 seconds, immediately after the first strong measured response. This supports the conclusion that additional observation is more important than model complexity before that event.

### Feature ablation and generalization

At 12 seconds, the Combined Random Forest achieved Macro-F1 1.0000 in Random, Block-aware, and Condition Holdout OOF evaluation. The Statistical-only Condition Holdout results were 0.9841 for Logistic Regression and 0.9791 for Random Forest. Transient-only Random Forest also reached 1.0000.

The result is explained largely by PS2 response timing: its per-cycle t90 proxy takes 9.4, 9.5, 9.6, and 9.7 seconds for Valve classes 100, 90, 80, and 73 respectively. The feature uses only measured PS2 samples before 12 seconds and no target label, but it remains a waveform proxy rather than a direct spool measurement.

Train-fold-only reduced models selected PS2 and PS3 in every 12-second fold/model and FS1 in 26 of 30 selections. These three-sensor models used about 96 features instead of 360 and also reached 1.0000 Condition Holdout Macro-F1 with Random Forest.

### Cooler domain holdout

At 12 seconds, Transient Random Forest Macro-F1 was:

| Held-out Cooler condition | Macro-F1 |
|---:|---:|
| 3 | 0.9986 |
| 20 | 1.0000 |
| 100 | 1.0000 |

These are domain-holdout results on the same test rig, not external-machine validation. Full results are in `results/valve_transient_analysis/VALVE_TRANSIENT_REPORT.md`.

## Installation

Windows PowerShell:

```powershell
git clone https://github.com/hamjaejun/hydraulic-fault-diagnosis-poc.git
cd hydraulic-fault-diagnosis-poc
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If script activation is blocked, run `\.venv\Scripts\python.exe` explicitly as shown below.

## Data preparation

```powershell
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe preprocess.py
.\.venv\Scripts\python.exe split_data.py
```

The download script obtains the official UCI archive and verifies shapes and alignment. If direct download fails, place the official ZIP at `data\condition_monitoring_of_hydraulic_systems.zip` and rerun the command.

## Model training and evaluation

Original Pump PoC:

```powershell
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe evaluate.py
.\.venv\Scripts\python.exe uncertainty.py
.\.venv\Scripts\python.exe sensor_selection.py --repeats 10
```

Multi-component models and benchmark:

```powershell
.\.venv\Scripts\python.exe multi_component.py train
.\.venv\Scripts\python.exe multi_component_benchmark.py
```

Valve study:

```powershell
.\.venv\Scripts\python.exe valve_transient.py build-features
.\.venv\Scripts\python.exe valve_transient.py analyze-signals
.\.venv\Scripts\python.exe valve_transient.py train-demo
.\.venv\Scripts\python.exe valve_transient_benchmark.py --stage all
```

The Valve benchmark writes checkpoints after completed experiments. The complete Nested-CV run is CPU intensive; use `--stage time`, `generalization`, `cooler`, or `summarize` for staged execution.

## Verification

```powershell
.\.venv\Scripts\python.exe verify_pipeline.py
.\.venv\Scripts\python.exe verify_generalization_audit.py
.\.venv\Scripts\python.exe verify_multi_component.py
.\.venv\Scripts\python.exe verify_valve_transient.py
```

The checks cover causal windows, Cycle-ID alignment, class presence, group isolation, model loading, probability validity, expected result counts, and OOF completeness.

## Streamlit dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The dashboard includes simultaneous four-component diagnosis, timelines, fixed-test and Nested-CV results, sensor optimization, multiple-fault analysis, and a Valve transient tab. Showing a full 60-second waveform does not change the highlighted prefix used by the selected prediction model.

## Reproducibility and repository policy

- Random seed: 42
- Raw data, generated feature tables, trained `.joblib` models, OOF-scale CSVs, and logs are ignored by Git.
- Small result summaries, reports, and figures are committed.
- Generated models are rebuilt using the commands above.
- Existing historical test results are not presented as a new independent test after repeated inspection.

## Limitations

- One hydraulic test rig and no independent circuit or field dataset
- Inferred label blocks rather than published session IDs
- Unseen combinations are not the same as unseen hardware
- No Valve command trace or spool-position signal
- State settings are not maintenance-confirmed root-cause labels
- Transient timing proxies can encode the configured Valve lag extremely directly
- No PINN and insufficient circuit parameters/boundary conditions for a defensible PINN claim
- Subsequent research choices are informed by these results and require new confirmation data

## Future research

- Accumulator physics-guided state estimation — teammate research, not merged here
- Hydraulic circuit and sensor-location modeling
- PINN feasibility after governing equations and boundary conditions are available
- Cross-machine and cross-site validation
- Maintenance decision and selective-prediction policies

See `METHODOLOGY.md` and `CONTRIBUTING.md` for research boundaries and collaboration rules.
