# Leakage audit

## Scope and evidence

This audit was performed against the saved UCI `profile.txt`, feature tables, split assignments, and the current source code. It distinguishes exact cycle leakage from similarity/group leakage.

## Findings

| Check | Result | Interpretation |
| --- | ---: | --- |
| Train/validation cycle-ID overlap | 0 | No exact cycle duplication. |
| Train/test cycle-ID overlap | 0 | No exact cycle duplication. |
| Validation/test cycle-ID overlap | 0 | No exact cycle duplication. |
| Inferred blocks spanning multiple saved splits | 155 / 194 | Similar adjacent-condition cycles are shared across the random split. |
| Inferred blocks spanning all three saved splits | 102 / 194 | The original test is not block-independent. |
| Four-component conditions spanning multiple saved splits | 142 / 144 | The original test mainly evaluates known condition combinations. |
| Four-component conditions spanning all three saved splits | 101 / 144 | Condition holdout was not part of the original design. |
| Future perturbation effect on 5-second features | 0.0 | Causal prefix extraction passed. |
| Feature IDs preserve original order | True | Cycle numbers remain the 1-based UCI row IDs. |
| Feature labels align with `profile.txt` | True | No label-row shift was found. |
| Predictor columns | 153 | Only `sensor__statistic` columns are used; labels/groups are excluded. |

## Source-code audit

### Correctly implemented

- `preprocess.py` slices `values[:, :seconds * sampling_rate]`; 5-second features cannot use later samples.
- Logistic Regression preprocessing is a Pipeline: median imputation, `VarianceThreshold`, scaling, then classification. Random Forest uses an in-pipeline imputer.
- The original train/validation/test assignment is created before window-specific model fitting, and model/hyperparameter selection is based on validation macro-F1 with severe recall as tie-breaker.
- `evaluate.py` reads previously saved model choices and does not select a model from the test metrics.
- The dashboard uses the saved `cycle_id`, which is the actual 1-based source row. It does not relabel validation/test rows as 1..N.

### Limitations

- The primary 70/15/15 split stratifies individual cycles by pump label only. It prevents exact cycle duplication but not same-block or same-condition similarity leakage.
- Existing `inferred_block_id` starts a new block whenever any of the five profile labels changes. UCI provides no true session ID, so this is an estimated grouping variable.
- The previous block CV uses a fixed 60-second Random Forest on train+validation only. Pipeline preprocessing is fitted inside each fold correctly, but hyperparameter/model selection is not nested and 5-second/Logistic behavior is not tested.
- Probability calibration is fitted on validation and the confidence threshold is selected on that same validation set. Validation calibration statistics are optimistic; test evaluation remains separate.
- The original test set has now been inspected repeatedly. Its 99.7% result is a historical random-split benchmark, not a newly untouched independent estimate.

## New validation controls

The additive audit uses five outer folds shared by both models and both observation times. Each outer training fold performs a separate three-fold inner search. For grouped schemes, inner preprocessing and hyperparameter selection also respect the relevant training-fold groups. Group labels are used only to construct splits and never enter the predictor matrix.
