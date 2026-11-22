# Research methodology and limitations

## Study scope

The original PoC diagnoses the **current internal pump leakage condition** (normal / weak / severe). The extension adds separate current-state classifiers for valve switching condition, cooler efficiency, and hydraulic accumulator pressure. It does not forecast remaining useful life or a future failure. Each of the 2,205 sixty-second cycles is one prediction unit; the cycles are not claimed to be statistically independent experimental sessions.

## Causal feature construction

For every sensor and observation endpoint (5, 10, 20, 40, 60 s), the code slices exactly `seconds × sampling_rate` samples from the beginning of the cycle. It then calculates mean, population standard deviation, minimum, maximum, RMS, initial value, final observed value, final-minus-initial, and least-squares slope per physical second. A regression test mutates every sample after 5 seconds and verifies that the 5-second features do not change.

## Split and leakage controls

- Cycle-level stratified split with seed 42: train 1,543, validation 331, test 331.
- Class counts are present in every split. Cycle overlap is asserted to be zero.
- Imputation, variance removal, scaling, classification, probability calibration, confidence thresholds, and sensor ranking never fit on test data.
- Logistic Regression and Random Forest hyperparameters are selected by validation macro-F1, with severe-leakage recall as tie-breaker. Dummy is a baseline, not a selectable final model.
- The final test is evaluated once by `evaluate.py`. Its outcomes are not fed back into model or threshold selection.

UCI supplies no explicit acquisition session/block ID. `split_data.py` therefore infers a conservative block whenever any of the five profile columns changes in the ordered file. These 194 contiguous identical-profile runs are used only in a five-fold `StratifiedGroupKFold` robustness audit on train+validation. They may not correspond to genuinely independent experimental sessions, which limits the strength of the generalization claim.

## Uncertainty

The selected estimator is frozen and its validation probabilities receive one-vs-rest Platt calibration. The three calibrated scores are normalized to sum to one. Multiclass Brier score and one-vs-rest reliability bins are reported. The confidence threshold is chosen on validation by maximizing coverage subject to confirmed accuracy ≥0.90 and coverage ≥0.50; a documented utility fallback is used if no threshold is feasible. These target values are engineering design criteria, not universal statistical constants.

On this run, the 5-second validation predictions already met the design constraint at threshold 0.34 with full coverage. Consequently, selective prediction did not improve test accuracy by deferral. This negative result is retained rather than forcing an arbitrary 0.80 threshold.

## Sensor optimization

All nine derived features belonging to one sensor are permuted together on validation data. Ranking uses the macro-F1 drop of the 60-second Random Forest. Fixed top-10, top-5, and top-3 sets are then separately retrained for every observation time and only afterwards evaluated on test.

`SE` (a virtual efficiency factor) dominates the importance ranking. This can be physically informative but also means the “three-sensor” result is not equivalent to three raw transducers; deploying `SE` requires access to the signals or computation used to produce it. Importance measures predictive association, not causality. Correlated sensors can receive near-zero conditional permutation importance even when physically useful.

## Interpretation boundaries

The very high scores reflect one controlled test rig and the published operating-condition schedule. Random cycle splitting can share operating regimes across partitions; the inferred-block audit partly probes this but cannot substitute for a new machine, a new day, or a site-level external test. Calibration is estimated from 331 validation cycles and should be revalidated after domain shift. No claim of industrial safety certification or field performance is made.

## Multi-component research benchmark

The multi-component benchmark uses the same original Cycle IDs, causal features, and outer-fold assignments for every component so cross-component comparisons are paired at the cycle level. Random Cycle is the permissive baseline. Block-aware grouping uses 194 contiguous runs with the same five-label profile as a proxy only; UCI provides no session identifier. Condition Holdout separates all 144 observed four-component state combinations while ensuring every target class remains represented in outer training and evaluation folds.

For each component, observation window, split scheme, and estimator, five outer folds estimate performance and three inner folds select the small hyperparameter set. Imputation, variance filtering, scaling, and estimation live inside the pipeline and are fit only to the relevant training fold. Labels, Stable Flag, inferred block IDs, and condition IDs are never predictive features. The saved fixed test is reported as a historical Demo result, not as a fresh independent evaluation after repeated inspection.

The actual benchmark contains 240 outer-fold rows and 105,840 out-of-fold predictions. All Block-aware folds have zero shared inferred blocks, all Condition Holdout folds have zero shared conditions, and all target classes occur in every train/evaluation fold. Confusion matrices and classwise metrics are computed from those predictions. Post-hoc single-versus-multiple-fault comparisons are descriptive because the groups have different target-class mixtures; they do not isolate a causal effect of simultaneous faults.

Valve diagnosis at five seconds is the clearest generalization weakness, and Accumulator Random Forest performance is also sensitive to strict grouping. These findings motivate pressure/flow transient, switching-time, hysteresis, ripple, and compliance features before considering a PINN. The present UCI release does not provide the valve command/spool position, accumulator precharge and topology, complete physical parameters, boundary conditions, or governing-equation residual targets needed to call ordinary engineered-feature models a PINN. Detailed numbers and research priorities are recorded in `results/multi_component_benchmark/MULTI_COMPONENT_REPORT.md`.

