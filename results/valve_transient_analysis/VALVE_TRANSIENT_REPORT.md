# Valve Transient-Aware ML Report

## Research question

Can transient-aware, physics-guided features improve hydraulic valve diagnosis under unseen fault combinations?

## Data and event interpretation

All 2,205 UCI cycles and nine pressure/flow/power sensors were used. The UCI documentation describes repeating 60-second load cycles but supplies neither valve command nor spool position. The 9.3–10.2 s region is therefore reported only as the first strong **measured sensor-response event**.

## Observation-time result

The first Random Cycle nested-CV setting to reach mean Macro-F1 ≥ 0.90 was 10 s. The best statistical-baseline mean was 0.9987 at 10 s with random_forest.

|   observation_seconds | model               |   macro_f1_mean |   macro_f1_std |   accuracy_mean |   close_failure_recall_mean |   small_lag_recall_mean |   n_features |
|----------------------:|:--------------------|----------------:|---------------:|----------------:|----------------------------:|------------------------:|-------------:|
|                     2 | dummy               |          0.1689 |         0.0034 |          0.5102 |                      0.0000 |                  0.0000 |          153 |
|                     2 | logistic_regression |          0.3973 |         0.0304 |          0.4871 |                      0.3831 |                  0.3022 |          153 |
|                     2 | random_forest       |          0.6237 |         0.0091 |          0.6948 |                      0.6591 |                  0.5077 |          153 |
|                     5 | dummy               |          0.1689 |         0.0034 |          0.5102 |                      0.0000 |                  0.0000 |          153 |
|                     5 | logistic_regression |          0.3906 |         0.0164 |          0.4844 |                      0.3706 |                  0.3231 |          153 |
|                     5 | random_forest       |          0.6285 |         0.0164 |          0.7007 |                      0.6563 |                  0.5380 |          153 |
|                     8 | dummy               |          0.1689 |         0.0034 |          0.5102 |                      0.0000 |                  0.0000 |          153 |
|                     8 | logistic_regression |          0.3949 |         0.0234 |          0.4830 |                      0.3502 |                  0.3451 |          153 |
|                     8 | random_forest       |          0.6430 |         0.0221 |          0.7125 |                      0.6822 |                  0.5386 |          153 |
|                    10 | dummy               |          0.1689 |         0.0034 |          0.5102 |                      0.0000 |                  0.0000 |          153 |
|                    10 | logistic_regression |          0.9970 |         0.0037 |          0.9977 |                      1.0000 |                  0.9943 |          153 |
|                    10 | random_forest       |          0.9987 |         0.0030 |          0.9991 |                      1.0000 |                  1.0000 |          153 |
|                    12 | dummy               |          0.1689 |         0.0034 |          0.5102 |                      0.0000 |                  0.0000 |          153 |
|                    12 | logistic_regression |          0.9984 |         0.0025 |          0.9986 |                      1.0000 |                  0.9970 |          153 |
|                    12 | random_forest       |          0.9973 |         0.0037 |          0.9982 |                      1.0000 |                  0.9949 |          153 |
|                    15 | dummy               |          0.1689 |         0.0034 |          0.5102 |                      0.0000 |                  0.0000 |          153 |
|                    15 | logistic_regression |          0.9954 |         0.0046 |          0.9964 |                      0.9946 |                  0.9914 |          153 |
|                    15 | random_forest       |          0.9945 |         0.0013 |          0.9959 |                      1.0000 |                  0.9838 |          153 |
|                    20 | dummy               |          0.1689 |         0.0034 |          0.5102 |                      0.0000 |                  0.0000 |          153 |
|                    20 | logistic_regression |          0.9952 |         0.0051 |          0.9964 |                      0.9973 |                  0.9914 |          153 |
|                    20 | random_forest       |          0.9904 |         0.0038 |          0.9923 |                      1.0000 |                  0.9646 |          153 |
|                    30 | dummy               |          0.1689 |         0.0034 |          0.5102 |                      0.0000 |                  0.0000 |          153 |
|                    30 | logistic_regression |          0.9720 |         0.0128 |          0.9773 |                      0.9833 |                  0.9521 |          153 |
|                    30 | random_forest       |          0.9899 |         0.0037 |          0.9923 |                      1.0000 |                  0.9753 |          153 |
|                    40 | dummy               |          0.1689 |         0.0034 |          0.5102 |                      0.0000 |                  0.0000 |          153 |
|                    40 | logistic_regression |          0.9709 |         0.0140 |          0.9764 |                      0.9693 |                  0.9585 |          153 |
|                    40 | random_forest       |          0.9893 |         0.0023 |          0.9914 |                      1.0000 |                  0.9752 |          153 |
|                    60 | dummy               |          0.1689 |         0.0034 |          0.5102 |                      0.0000 |                  0.0000 |          153 |
|                    60 | logistic_regression |          0.9813 |         0.0075 |          0.9846 |                      0.9864 |                  0.9606 |          153 |
|                    60 | random_forest       |          0.9815 |         0.0036 |          0.9850 |                      1.0000 |                  0.9590 |          153 |

## Feature ablation and strict generalization

Each row below is the best LR/RF and feature-set combination for that time and split. It is a conditional maximum, not one pre-fixed model across every condition.

|   observation_seconds | feature_set     | model               |   macro_f1_mean |   macro_f1_std |   close_failure_recall_mean |   small_lag_recall_mean |   n_features_mean |
|----------------------:|:----------------|:--------------------|----------------:|---------------:|----------------------------:|------------------------:|------------------:|
|                     5 | transient       | logistic_regression |          0.1734 |         0.0573 |                      0.2111 |                  0.2140 |           90.0000 |
|                    12 | reduced_sensors | random_forest       |          1.0000 |         0.0000 |                      1.0000 |                  1.0000 |           96.0000 |
|                    60 | transient       | random_forest       |          1.0000 |         0.0000 |                      1.0000 |                  1.0000 |          207.0000 |

The full Random, Block-aware, and Condition Holdout comparison is in `valve_feature_ablation.csv`; fold-level rows remain in `valve_generalization_results.csv`.

## Cooler condition holdout

|   held_out_cooler |   observation_seconds | feature_set     | model         |   macro_f1 |   close_failure_recall |   small_lag_recall |   n_features |
|------------------:|----------------------:|:----------------|:--------------|-----------:|-----------------------:|-------------------:|-------------:|
|                 3 |                     5 | reduced_sensors | random_forest |     0.2211 |                 0.0000 |             0.1750 |           47 |
|                20 |                     5 | reduced_sensors | random_forest |     0.3375 |                 0.2250 |             0.1250 |           47 |
|               100 |                     5 | reduced_sensors | random_forest |     0.2852 |                 0.6500 |             0.0083 |           37 |
|                 3 |                    12 | combined        | random_forest |     0.9986 |                 1.0000 |             0.9917 |          360 |
|                20 |                    12 | reduced_sensors | random_forest |     1.0000 |                 1.0000 |             1.0000 |           96 |
|               100 |                    12 | transient       | random_forest |     1.0000 |                 1.0000 |             1.0000 |          207 |
|                 3 |                    60 | transient       | random_forest |     1.0000 |                 1.0000 |             1.0000 |          207 |
|                20 |                    60 | transient       | random_forest |     1.0000 |                 1.0000 |             1.0000 |          207 |
|               100 |                    60 | reduced_sensors | random_forest |     1.0000 |                 1.0000 |             1.0000 |           96 |

Leave-one-cooler-condition-out is a three-domain test on the same apparatus. It is not external-machine validation.

## Error analysis

`valve_error_analysis.csv` contains 1788 misclassified OOF rows from the best Condition Holdout configuration at each benchmark time, with original Cycle ID and the other component states.

## Physics-guided interpretation

The engineered variables describe pressure, flow, and motor-power response amplitude, derivative, response timing, overshoot, and settling proxies. No cross-sensor pressure difference is called a valve pressure drop because sensor placement is undocumented. t10/t90 and the detected event time are waveform-response estimates, not direct switching-time measurements. This is physics-guided feature engineering, not a PINN.

Training-fold-only reduced-sensor selection chose PS2 and PS3 in every 12-second fold/model and FS1 in 26 of 30 fold/model selections. `valve_feature_rankings.csv` records fold-aggregated univariate associations; these are predictive rankings rather than proof of a physical failure cause.

## Limitations

- Results come from one rig and a repeatedly inspected public dataset.
- Inferred blocks are not published independent sessions.
- Condition holdout tests unseen combinations, not new component state values or hardware.
- The nominal response region was informed by exploratory data and teammate context; nested OOF evaluation limits sample leakage but does not restore a pristine final test.
- Reduced-sensor selection occurs inside each outer training fold, but future confirmation should pre-register the final sensor set on new data.
- Accumulator physics/state estimation is explicitly outside this study.