# Generalization Audit Report

## Executive conclusion

The historical 5-second accuracy of 0.9970 and macro-F1 of 0.9968 were obtained from one fixed 70/15/15 random cycle-level split. There is no exact cycle duplication, but 155 of 194 inferred contiguous blocks and 142 of 144 four-component state combinations span multiple saved splits. The historical test therefore measures interpolation among highly related operating conditions and must not be described as a new condition-independent evaluation.

## 1. Dataset structure

- Total cycles: 2205.
- Raw four-component state combinations: 144.
- Inferred contiguous five-label blocks: 194; median size 10.0, range 1–210 cycles.
- No degraded component: 21 cycles.
- Exactly one degraded component: 566 cycles.
- Two or more degraded components: 1618 cycles.
- Stable flag 0: 1449 cycles; flag 1: 756 cycles.

These are experimental state labels supplied by UCI. They indicate configured component condition/severity, not a proven physical root cause for each individual sensor trace.

### Fault-pattern inventory

| Fault pattern | State combinations | Cycles |
| --- | ---: | ---: |
| cooler+valve+pump_leakage+accumulator | 36 | 360 |
| cooler+accumulator | 6 | 326 |
| accumulator | 3 | 272 |
| cooler | 2 | 242 |
| cooler+valve+accumulator | 18 | 180 |
| valve+pump_leakage+accumulator | 18 | 180 |
| cooler+pump_leakage+accumulator | 12 | 132 |
| cooler+valve+pump_leakage | 12 | 120 |
| valve+accumulator | 9 | 90 |
| pump_leakage+accumulator | 6 | 66 |
| cooler+valve | 6 | 60 |
| valve+pump_leakage | 6 | 60 |
| cooler+pump_leakage | 4 | 44 |
| valve | 3 | 30 |
| pump_leakage | 2 | 22 |
| none | 1 | 21 |

## 2. Validation designs

### A. Random cycle-level

Five-fold stratified random CV. It is the closest repeated-CV counterpart to the historical benchmark, but blocks and condition combinations can occur in both training and evaluation.

### B. Block-aware

Five-fold `StratifiedGroupKFold` using contiguous runs with identical cooler, valve, pump, accumulator, and stable labels. No inferred block crosses an outer fold. These blocks are proxies, not published independent experiment sessions.

### C. Condition holdout

Five-fold `StratifiedGroupKFold` using the complete four-component state tuple as the group. All 144 combinations are evaluated exactly once, no combination appears in both training and evaluation, and every fold retains pump classes 0/1/2 in both partitions. This tests interpolation to unseen combinations of known component-state values, not extrapolation to unseen severity codes or a new machine.

Both grouped designs use a three-fold group-respecting inner search within each outer training fold. Imputation, variance filtering, scaling, and model fitting are refit from scratch in the inner/outer training data.

## 3. Pump leakage results

| Scheme | Observation | Logistic macro-F1 | Random Forest macro-F1 |
| --- | ---: | ---: | ---: |
| block_aware | 5s | 0.9901 ± 0.0081 | 0.9740 ± 0.0135 |
| block_aware | 60s | 0.9832 ± 0.0157 | 0.9964 ± 0.0038 |
| condition_holdout | 5s | 0.9881 ± 0.0070 | 0.9721 ± 0.0166 |
| condition_holdout | 60s | 0.9788 ± 0.0139 | 0.9943 ± 0.0058 |
| random_cycle | 5s | 0.9965 ± 0.0012 | 0.9798 ± 0.0096 |
| random_cycle | 60s | 0.9972 ± 0.0027 | 0.9952 ± 0.0031 |

For 5-second Logistic Regression, random-cycle macro-F1 was 0.9965 ± 0.0012, block-aware macro-F1 was 0.9901 ± 0.0081, and condition-holdout macro-F1 was 0.9881 ± 0.0070. At 60 seconds, the corresponding values were 0.9972 ± 0.0027, 0.9832 ± 0.0157, and 0.9788 ± 0.0139.

Relative to random-cycle CV, 5-second Logistic Regression dropped by 0.0064 macro-F1 under block-aware evaluation and 0.0084 under condition holdout. The 60-second Logistic drops were larger: 0.0140 and 0.0184, respectively. Thus, additional observation time did not guarantee stronger cross-condition generalization for the linear model.

Full accuracy, class recall, fold sizes, group overlap checks, selected hyperparameters, and fit times are in `validation_comparison.csv`. Pooled confusion matrices are provided as CSV and PNG files.

## 4. Answers to the research questions

1. **Which single and combined states exist?** The dataset contains 144 four-component state combinations across no-fault, single-degradation, and multiple-degradation cases. Exact state and cycle ranges are in `fault_combination_counts.csv`.
2. **Under what condition was 99.7% obtained?** One fixed pump-stratified random cycle split, with the same inferred blocks and condition combinations commonly represented across training and test.
3. **Does block-aware performance remain high?** Yes in absolute terms, but not unchanged. Mean block-aware macro-F1 was 0.9901 for 5-second Logistic Regression, 0.9832 for 60-second Logistic Regression, and 0.9964 for 60-second Random Forest. These values apply to held-out contiguous label runs; true session independence remains unknown.
4. **Can unseen combined conditions be held out?** Yes. There are 48 condition groups for each pump class (144 total), enabling five condition-disjoint folds with all three pump classes represented in training and evaluation.
5. **Where does performance degrade?** Logistic Regression is most affected at 60 seconds: macro-F1 falls from 0.9972 in random CV to 0.9832 block-aware and 0.9788 condition-held-out. Five-second Random Forest also falls from 0.9798 random to 0.9721 condition-held-out and has the largest fold variability. Sixty-second Random Forest is the most stable configuration (0.9943 condition-held-out macro-F1).
6. **Where could physics-guided ML help?** The main vulnerability is configuration/domain shift and reliance on highly informative engineered or virtual channels such as `SE`, not exact cycle leakage. A useful next step is physics-guided feature design or constrained models based on pressure-flow-power consistency, conservation relationships, and monotonic leakage behavior. A full PINN is not yet justified because the dataset does not provide the governing hydraulic equations, boundary conditions, component parameters, or dense state variables required to identify a unique dynamical model. Physics-guided residual features and out-of-distribution detection should precede a PINN claim.

## 5. Limits

- Component labels were intentionally used to form validation groups but were never included in model features.
- Condition holdout excludes complete combinations, not individual severity values; all severity codes remain represented elsewhere.
- Adjacent-run blocks may still share the same physical experiment or slow drift because UCI publishes no session identifier.
- Results are internal cross-validation on one rig and cannot establish external machine/site generalization.
- The historical fixed test has been repeatedly observed and is retained only as a benchmark.

## Artifact map

- `dataset_profile_summary.csv`: raw and binary state distributions.
- `fault_combination_counts.csv`: all four-component combinations and original cycle ranges.
- `block_summary.csv`: inferred five-label contiguous blocks.
- `condition_holdout_plan.csv`: condition groups assigned to each outer fold.
- `leakage_audit.md`: source-code and leakage review.
- `validation_comparison.csv`: fold-level metrics and group-overlap checks.
- `validation_summary.csv`: mean ± standard deviation summaries.
- `confusion_matrices.csv` and `confusion_*.png`: fold and pooled confusion matrices.
- `execution_log.txt`: actual run log.
