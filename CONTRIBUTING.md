# Contributing

This repository supports parallel research while keeping the validated baseline reproducible.

## Branches

- `main`: reviewed, reproducible, and verified work only
- `feature/valve-transient`: Valve transient-aware machine learning
- `research/accumulator-physics`: teammate-owned Accumulator state-estimation work
- `research/hydraulic-circuit`: circuit and sensor-location modeling
- `feature/pinn`: future work only after the required equations and boundary conditions exist

Create a focused branch and open a Pull Request. Do not merge another researcher's code, experiments, or documents without confirming ownership and intended sharing scope.

## Experiment outputs

Store a new study under `results/<study_name>/`. Commit concise summaries, reports, and interpretable figures. Do not commit raw UCI files, generated feature matrices, OOF-scale prediction files, trained model binaries, environment folders, caches, logs, credentials, or machine-specific paths.

Every reported metric must identify:

- dataset and original Cycle IDs
- target and class mapping
- observation endpoint
- feature family and sensor set
- outer and inner split strategy
- random seed
- model and search space
- mean, standard deviation, and classwise metrics where applicable
- whether the result is a fixed historical test, OOF estimate, domain holdout, or external test

Never fabricate or manually edit experiment scores.

## Leakage prevention

- Split original cycles before fitting learned transformations.
- Keep a Cycle ID in only one outer train/evaluation partition.
- Fit imputation, scaling, variance filtering, calibration, feature selection, and hyperparameter selection using training data only.
- Keep labels, Stable Flag, condition IDs, and inferred block IDs out of predictors.
- For early diagnosis, never read samples at or after the stated observation endpoint.
- Group complete conditions or inferred blocks when that is the declared evaluation unit.
- Do not call a repeatedly inspected public test set a new independent test.

## Baseline changes

Changes affecting feature extraction, splitting, labels, metrics, or saved artifacts must run:

```powershell
.\.venv\Scripts\python.exe verify_pipeline.py
.\.venv\Scripts\python.exe verify_generalization_audit.py
.\.venv\Scripts\python.exe verify_multi_component.py
.\.venv\Scripts\python.exe verify_valve_transient.py
```

Also run the Streamlit app or its automated smoke test. Explain intentional result changes in the Pull Request and preserve old result folders unless the change explicitly supersedes them.

## Data and models

Use `download_data.py` to obtain the official dataset. Share large generated artifacts outside Git using an agreed team storage location and record a checksum and generation command. Never place secrets, GitHub tokens, or personal paths in source or results.

Model files are reproducible artifacts and are ignored by default. If a deployment model must be released, use a reviewed GitHub Release asset or model registry with provenance and checksum rather than adding it to normal Git history.

## Pull Request checklist

- [ ] Scope and research question are clear.
- [ ] New code does not modify teammate-owned research without approval.
- [ ] Commands and seeds are documented.
- [ ] Train/evaluation and group-overlap checks pass.
- [ ] Early-window causality is tested.
- [ ] Metrics come from actual execution.
- [ ] Large files and secrets are excluded.
- [ ] README and methodology boundaries are updated.
- [ ] Existing baseline verification still passes.

Request review before merging to `main`. Collaborators are added only after the repository owner supplies their GitHub usernames.
