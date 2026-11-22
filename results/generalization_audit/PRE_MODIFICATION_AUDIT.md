# Pre-modification validation audit

Recorded before adding the generalization-audit implementation. Existing source files, fitted models, dashboard code, and prior result files were not changed.

## Evidence reviewed

- `data/raw/profile.txt` has shape `(2205, 5)`.
- `data/features/features_5s.csv` and `features_60s.csv` each have shape `(2205, 155)`: cycle ID, pump label, and 153 sensor-derived features.
- `data/split_assignments.csv`, `preprocess.py`, `split_data.py`, `train.py`, `evaluate.py`, `uncertainty.py`, `app.py`, and `verify_pipeline.py` were inspected directly.

## Existing implementation: correct controls

- Cycle IDs are unique and the saved train/validation/test assignment has no cycle-ID overlap (`split_data.py:55-73`).
- Feature extraction uses only `values[:, :seconds * sampling_rate]`, so short-window features exclude future samples (`preprocess.py:18-43`).
- Imputation, variance filtering, and scaling are inside scikit-learn pipelines and are fitted with the training data passed to `fit` (`train.py:32-63`).
- Model/hyperparameter selection uses the validation split, not the test split (`train.py:107-149`).
- The saved app cycle selector uses the actual `cycle_id` column; the displayed operating-cycle number is the original 1-based UCI row ID (`app.py`, `train.py:79-87`).
- Feature tables and metadata are created in original UCI row order. The merge in `get_xy` uses the feature table as its left input and does not explicitly sort.

## Existing implementation: limitations requiring stricter validation

- The primary 70/15/15 split stratifies only by pump class and randomly distributes individual cycles (`split_data.py:35-49`). It does not keep inferred contiguous experimental blocks intact.
- `inferred_block_id` is defined as a new run whenever any of the five profile labels changes (`split_data.py:14-23`). It is an estimated group, not a published experiment/session ID.
- The existing block audit uses only train+validation cycles from the already randomized split, a fixed 60-second Random Forest, and five-fold `StratifiedGroupKFold` (`evaluate.py:142-162`). It does not audit the 5-second model or Logistic Regression.
- Preprocessing in that block audit is correctly refitted inside each outer-fold pipeline, but hyperparameters are fixed rather than selected within each training fold.
- The original validation set is used both to fit probability calibrators and to choose confidence thresholds; reported validation calibration/threshold performance is therefore optimistic. The test calculation remains separate, but the saved test set has since been inspected repeatedly and should no longer be described as a new untouched evaluation set.
- Condition combinations are not held out by the original split, so the reported 99.7% test accuracy measures interpolation across randomly sampled cycles, not performance on unseen combined operating/fault conditions.

## Planned additive change

Add a separate `generalization_audit.py` workflow and write all new artifacts only under `results/generalization_audit/`. Existing models, dashboard, split files, and prior results remain unchanged.
