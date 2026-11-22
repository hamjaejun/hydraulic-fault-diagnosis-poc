# Hydraulic Fault Diagnosis ML PoC

A reproducible multi-component hydraulic fault diagnosis baseline using the UCI Condition Monitoring of Hydraulic Systems dataset.

The project provides component-specific Pump, Valve, Cooler, and Accumulator classifiers, causal 5/10/20/40/60-second observation windows, uncertainty analysis, sensor optimization, Random Cycle, inferred Block-aware, and four-component Condition Holdout evaluation with five outer and three inner folds.

## Reproduction

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python download_data.py
python preprocess.py
python split_data.py
python train.py
python evaluate.py
python uncertainty.py
python sensor_selection.py --repeats 10
python multi_component.py train
python multi_component_benchmark.py
python verify_pipeline.py
python verify_generalization_audit.py
python verify_multi_component.py
python -m streamlit run app.py
```

Raw UCI data, generated feature matrices, and trained model binaries are reproducible and intentionally excluded from Git. Research summaries and figures are stored under `results/`.

See `METHODOLOGY.md` and `results/multi_component_benchmark/MULTI_COMPONENT_REPORT.md` for split definitions, leakage controls, actual results, and interpretation boundaries.
