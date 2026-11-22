# Multi-Component Hydraulic Fault Diagnosis Report

## Dataset

The benchmark uses all 2,205 original UCI cycles and the same 153 causal sensor-statistic features as the pump PoC. Stable flag and all component labels are excluded from predictors.

- **Pump**: 0 (Normal) = 1221, 1 (Weak Leakage) = 492, 2 (Severe Leakage) = 492
- **Valve**: 73 (Close to Failure) = 360, 80 (Severe Switching Lag) = 360, 90 (Small Switching Lag) = 360, 100 (Optimal Switching) = 1125
- **Cooler**: 3 (Close to Failure) = 732, 20 (Reduced Efficiency) = 732, 100 (Full Efficiency) = 741
- **Accumulator**: 90 (Close to Failure) = 808, 100 (Severely Reduced Pressure) = 399, 115 (Slightly Reduced Pressure) = 399, 130 (Optimal Pressure) = 599

Fault multiplicity was rechecked from the raw component labels: 21 all-normal cycles, 566 single-fault cycles, and 1618 multiple-fault cycles.

## Methodology

- **Demo models:** four independent component classifiers for 5, 10, 20, 40, and 60 seconds, using the existing 70/15/15 cycle split. Dummy, Logistic Regression, and Random Forest are evaluated; validation macro-F1 selects each saved demo model.
- **Research benchmark:** 5- and 60-second features, five outer folds, three inner folds, Logistic Regression and Random Forest.
- **Random cycle:** stratified by pump class and shared across every component/time/model.
- **Block-aware:** contiguous runs with identical five-label profiles; these are inferred groups, not true session IDs.
- **Condition holdout:** complete four-component state tuples are disjoint between train and evaluation.
- Every fold contains every class for all four targets. Train/evaluation cycle overlap is zero. Pipeline preprocessing and tuning occur inside the training folds only.
- Pump hyperparameters are reused from the completed audit only where the folds, features, grids, target, and protocol are identical; outer models are refitted to generate OOF predictions.

## Results

| Component | Time | Model | Random F1 | Block F1 | Holdout F1 |
| --- | ---: | --- | ---: | ---: | ---: |
| Pump | 5s | logistic_regression | 0.9965 | 0.9901 | 0.9881 |
| Pump | 5s | random_forest | 0.9798 | 0.9740 | 0.9721 |
| Pump | 60s | logistic_regression | 0.9972 | 0.9832 | 0.9788 |
| Pump | 60s | random_forest | 0.9952 | 0.9964 | 0.9943 |
| Valve | 5s | logistic_regression | 0.3822 | 0.1735 | 0.1450 |
| Valve | 5s | random_forest | 0.6424 | 0.1268 | 0.1139 |
| Valve | 60s | logistic_regression | 0.9813 | 0.8225 | 0.8682 |
| Valve | 60s | random_forest | 0.9829 | 0.7860 | 0.8189 |
| Cooler | 5s | logistic_regression | 0.9982 | 0.9787 | 0.9743 |
| Cooler | 5s | random_forest | 0.9977 | 0.9618 | 0.9909 |
| Cooler | 60s | logistic_regression | 0.9995 | 0.9869 | 0.9868 |
| Cooler | 60s | random_forest | 0.9986 | 0.9925 | 0.9943 |
| Accumulator | 5s | logistic_regression | 0.9530 | 0.9028 | 0.8990 |
| Accumulator | 5s | random_forest | 0.9776 | 0.8299 | 0.7928 |
| Accumulator | 60s | logistic_regression | 0.9519 | 0.8749 | 0.8439 |
| Accumulator | 60s | random_forest | 0.9804 | 0.7629 | 0.6935 |

The hardest best-model condition-holdout setting was **Valve at 5 seconds** (logistic_regression, macro-F1 0.1450 ± 0.0617). The lowest class-level condition-holdout recall occurred for **Valve — Small Switching Lag** at 5 seconds with random_forest (recall 0.0162).

## Multiple-Fault Analysis

The table uses each component/time's post-hoc best mean condition-holdout model. It is descriptive and is not a new independent model-selection result.

| Component | Model | Single cycles | Single accuracy | Multiple cycles | Multiple accuracy | Change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Pump | random_forest | 566 | 1.0000 | 1618 | 0.9944 | -0.0056 |
| Valve | logistic_regression | 566 | 0.6078 | 1618 | 0.9302 | +0.3224 |
| Cooler | random_forest | 566 | 0.9788 | 1618 | 0.9994 | +0.0206 |
| Accumulator | logistic_regression | 566 | 0.8746 | 1618 | 0.8622 | -0.0124 |

Differences cannot be interpreted as the causal effect of simultaneous faults because target-class prevalence and condition severity differ between groups. The 21 all-normal cycles are too few for broad conclusions.

## Generalization Analysis

Random cycle CV is the interpolation baseline. Block-aware CV removes overlap of estimated contiguous runs, while condition holdout removes overlap of all four-component state combinations. `condition_holdout_analysis.csv` lists class-recall changes and the condition IDs where errors concentrate. Fold variability is retained rather than hidden by a single aggregate score.

## Discussion

1. **Hardest component:** the measured hardest setting is reported above; difficulty depends strongly on time and model rather than component alone.
2. **Largest split degradation:** see `condition_holdout_class_recall_drop.png`; condition holdout is the strictest test of unseen combinations.
3. **Weakest class:** Valve — Small Switching Lag, under the setting stated above.
4. **Short observation:** compare 5s and 60s in the main table; the largest best-model gain identifies the component most dependent on longer dynamics.
5. **Multiple faults:** descriptive OOF differences are shown above, with sample counts. They are associations, not causal estimates.
6. **Physics-guided priority:** prioritize the component with the weakest condition-holdout class recall and largest 5s-to-60s gap. Start with pressure/flow/power/temperature consistency features and OOD detection, not a PINN claim.

## Physics-Guided ML Next Step

- **Pump:** pressure rise, flow loss, motor power, hydraulic power `Δp × Q`, and efficiency residuals; verify sensor locations across the pump and load path.
- **Valve:** upstream/downstream pressure transients, switching-time proxies, flow response, and pressure-flow hysteresis; valve command and spool position are missing.
- **Cooler:** inlet/outlet temperature difference, cooling power, flow-dependent heat transfer, and thermal time constants; ambient temperature and heat-exchanger parameters are missing.
- **Accumulator:** pressure ripple, charge/discharge slopes, effective compliance, and flow-pressure phase response; pre-charge pressure and circuit topology are required for stronger physics constraints.

Physics-guided feature engineering means adding interpretable conservation/residual features to ML. A PINN would additionally enforce governing differential equations during training. This dataset lacks several boundary conditions, control inputs, geometry, and component parameters, so physics-guided features should be tested first under the same condition-holdout folds.

## Limitations

- One experimental rig; no machine/site external validation.
- Inferred blocks are not published independent session IDs.
- Unseen combinations are not the same as unseen state values or new hardware.
- State labels describe configured experimental condition, not a confirmed maintenance root cause.
- No repair actions, failure progression, or maintenance outcomes are available.
- Some class/group subsets are small and imbalanced.
- Choosing future physics work from these CV results introduces research-selection bias and requires a new confirmation study.
