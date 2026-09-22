# UFC Predictor V2.1 Accuracy Experiment — Final Report

Status: evaluation complete; website integration intentionally not performed.

## Result

The locked V2.1 ensemble scored **62.69% accuracy on the untouched 2025–2026 lockbox**. This is **+1.87 percentage points** above the published V1.0 accuracy of 60.82%, but the V2 lockbox 95% accuracy interval is **59.20%–66.06%**. Because that interval includes 60.82%, the improvement over V1.0 is **not statistically established**.

V2.1 does meaningfully beat the strongest simple lockbox benchmark: pre-fight UFC win percentage scored 58.03%, a **+4.66 point** V2 advantage with a paired 95% interval of **+0.78 to +8.55 points**.

## Primary metrics

These rows use different evaluation periods and are labeled to prevent a false direct comparison.

| Model / evaluation | Accuracy | ROC AUC | Log Loss | Brier | Calibration Error |
|---|---:|---:|---:|---:|---:|
| Frozen V1.0 — published historical test | 60.82% | 64.70% | 0.6566 | 0.2325 | 1.03% |
| V2.1 ensemble — 2019–2024 walk-forward development | 61.31% | 65.48% | 0.6549 | 0.2315 | 1.75% |
| **V2.1 ensemble — untouched 2025–2026 lockbox** | **62.69%** | **66.79%** | **0.6486** | **0.2287** | **2.65%** |

V2.1 is a validation-selected probability ensemble:

- 25% regularized logistic regression
- 75% XGBoost
- 18-month / 548-day recent-performance half-life
- 107 leakage-safe pre-fight features
- no betting odds and no post-hoc calibration

Ensemble weights were chosen using only 2019–2022 out-of-fold predictions. The ensemble then had to improve accuracy, log loss, and Brier score on a separate 2023–2024 confirmation period. Architecture, features, hyperparameters, and weights were written to `ARCHITECTURE_LOCK.json` before the holdout was opened.

## Evaluation periods and fight counts

| Purpose | Period | Fights |
|---|---|---:|
| Walk-forward development predictions | 2019–2024 | 2,970 |
| Final model training before lockbox | 1994-03-11 to 2024-12-14 | 7,809 |
| Untouched lockbox | 2025-01-11 to 2026-06-27 | 772 |
| Production refit after evaluation | 1994-03-11 to 2026-06-27 | 8,581 |

The lockbox was evaluated once. A persisted sentinel prevents a second evaluation.

## Standalone walk-forward candidates

| Model | Accuracy | ROC AUC | Log Loss | Brier | Calibration Error |
|---|---:|---:|---:|---:|---:|
| Rich 18-month logistic, C=1 | 61.21% | 64.60% | 0.6641 | 0.2349 | 2.46% |
| Rich 12-month logistic, C=0.1 | 61.11% | 64.77% | 0.6622 | 0.2342 | 2.52% |
| XGBoost, 18-month | 61.04% | 65.03% | 0.6569 | 0.2325 | 2.27% |
| Elastic-net logistic, l1 ratio 0.75 | 61.01% | 65.20% | 0.6583 | 0.2327 | 1.10% |
| Gradient Boosting, 18-month | 60.77% | 64.48% | 0.6610 | 0.2343 | 2.67% |
| Random Forest, 18-month | 59.97% | 64.65% | 0.6590 | 0.2334 | 2.03% |
| Extra Trees, 18-month | 60.07% | 63.95% | 0.6649 | 0.2358 | 1.10% |
| HistGradientBoosting, 18-month | 60.30% | 64.53% | 0.6610 | 0.2343 | 2.30% |

LightGBM and CatBoost were not installed. Adding them was not justified merely to expand the model list: the inputs are numeric, and the existing tree families already supplied the intended nonlinear comparison.

## Lockbox benchmarks

| Benchmark | Accuracy |
|---|---:|
| Majority | 49.87% |
| Better recent five-fight record | 57.90% |
| Standard Elo | 55.96% |
| **Better pre-fight UFC win percentage** | **58.03%** |
| **V2.1 ensemble** | **62.69%** |

No verified point-in-time historical odds were available in the immutable dataset, so no betting-favorite benchmark was reported.

## Confidence buckets on the untouched lockbox

| Predicted confidence | Fights | Coverage | Mean prediction | Actual win rate / accuracy |
|---|---:|---:|---:|---:|
| 50–55% | 220 | 28.50% | 52.35% | 57.27% |
| 55–60% | 206 | 26.68% | 57.32% | 56.80% |
| 60–65% | 146 | 18.91% | 62.34% | 65.07% |
| 65–70% | 105 | 13.60% | 67.33% | 67.62% |
| 70–75% | 57 | 7.38% | 72.10% | 71.93% |
| 75–80% | 28 | 3.63% | 77.29% | 89.29% |
| 80%+ | 10 | 1.30% | 81.76% | 90.00% |

The predefined buckets that achieved at least 75% lockbox accuracy cover exactly **38 of 772 fights, or 4.92%**. Combined, those 38 fights were correct 34 times (**89.47%**). This is a small sample and must not be presented as a guaranteed future 89% subset.

## Top 15 features

Importance was measured by permutation on unseen 2023–2024 confirmation fights, using the final ensemble. It is predictive importance, not causation.

| Rank | Feature | Plain-language interpretation |
|---:|---|---|
| 1 | `takedowns_per15_diff` | Takedown-volume advantage |
| 2 | `elo_k16_diff` | Conservative Elo rating advantage |
| 3 | `reach_diff` | Reach advantage |
| 4 | `elo_k24_diff` | Medium-update Elo advantage |
| 5 | `elo_k48_diff` | Fast-update Elo advantage |
| 6 | `age_x_experience_diff` | Age and UFC-experience interaction |
| 7 | `takedown_accuracy_diff` | Takedown accuracy advantage |
| 8 | `opponent_adjusted_accuracy_diff` | Striking accuracy adjusted for prior opponent defense |
| 9 | `striking_differential_pm_diff` | Significant-strike differential per minute |
| 10 | `reversals_per15_diff` | Grappling reversal activity |
| 11 | `takedown_defense_decay_548d_diff` | Recent takedown-defense advantage |
| 12 | `sig_landed_pm_decay_548d_diff` | Recent significant-strike output |
| 13 | `age_squared_diff` | Nonlinear age-curve difference |
| 14 | `opponent_adjusted_striking_differential_decay_548d_diff` | Recent opponent-adjusted striking differential |
| 15 | `decision_rate_diff` | Historical decision-win tendency difference |

## Feature ablation

Removing every tested group reduced the accuracy of the strongest standalone logistic component. The largest accuracy losses were:

| Removed group | Accuracy loss |
|---|---:|
| Age / nonlinear age context | 1.62 points |
| Striking | 1.28 points |
| Opponent quality | 1.04 points |
| Elo / ratings | 1.04 points |
| Physical attributes | 0.67 points |
| Activity / inactivity | 0.57 points |
| Fight context | 0.44 points |
| Time decay | 0.44 points |
| Opponent-adjusted performance | 0.24 points |
| Matchup interactions | 0.20 points |
| Recent form | 0.17 points |
| Career history | 0.13 points |
| Grappling | 0.10 points |

No complete group consistently hurt validation accuracy enough to remove. Grappling and time-decay removal slightly improved logistic-component log loss while reducing accuracy, so they were retained for the accuracy-focused ensemble. No unrestricted polynomial expansion was used.

## Leakage audit

All acceptance tests passed after the final feature changes:

- Future mutation: deleting every 2023–2026 fight did not change the selected 2022 feature vector.
- Current fight: removing the selected fight's result and generated statistics did not change its pre-fight vector.
- Future status: changing current active/retired status did not change historical features.
- Future career: future changes did not alter record, Elo, form, striking, grappling, opponent-quality, or experience features at time T.
- Timestamp rule: every row satisfies `feature_timestamp < fight_timestamp`.
- Same-date cards are snapshotted before any result from that date is applied, including early tournament cards.

Current roster status remains a product filter and never enters historical training rows. Stance matchup was rejected because the source does not timestamp stance changes. Late-round performance was rejected because this dataset does not provide reliable point-in-time per-round histories.

## Calibration and statistical conclusion

The untouched-lockbox expected calibration error is **2.65%**, with log loss 0.6486 and Brier score 0.2287. Calibration is reasonable overall, but individual division and high-confidence estimates have small samples.

- Improvement over V1.0: **+1.87 points**, not statistically proven because the V2 95% interval includes 60.82%. The comparison is unpaired because V1.0 used a different test period.
- Improvement over the strongest simple lockbox baseline: **+4.66 points**, statistically meaningful in the paired fight-level comparison.
- 2025 accuracy: 63.69% on 515 fights.
- Partial 2026 accuracy: 60.70% on 257 fights.

The honest conclusion is that V2.1 is promising and beat all simple lockbox benchmarks, but the evidence does **not** prove that it is better than frozen V1.0. More genuinely future fights are needed.

## Known limitations

- The immutable source was retrieved on 2026-09-01 but its newest completed fight is 2026-06-27. It is not fully current as of this report.
- Reach is missing for about 50.44% of fighter profiles; DOB is missing for 19.51%, stance for 21.18%, height for 6.94%, and weight for 1.96%. Imputation is fitted inside each training fold.
- Historical career fields are reconstructable UFC history, not complete timestamped pre-UFC career records.
- The active-fighter list is a conservative activity proxy plus reviewed overrides, not an authoritative roster feed.
- Project-stable fighter IDs are canonical-name UUIDs with an explicit alias table; unreviewed name changes can still require manual alias entries.
- Injuries, camps, travel, short-notice replacements, and other non-statistical context are unavailable.
- The model does not use market odds. It is an independent statistical model, not evidence of betting edge.
- The lockbox includes only 772 fights and 2026 is partial.

## Frozen artifacts

- V1.0 remains unchanged: SHA-256 `E592F2FFF8EE03B5A860BBF3FC5F8C82C1ECFE9668379C2AE219AC965719F825`.
- V2.1 production artifact: `models/v2.1-accuracy/ufc_predictor_v2_1_accuracy.joblib`.
- V2.1 artifact SHA-256: `5C0725ED3BEC1FF9B82C9DB548963BC27279CB9DCB6852A462A7F443CB059416`.
- Architecture lock, feature schema, one-time holdout report, and manifest are stored beside the artifact.
- The production artifact was refit on all 8,581 decisive fights only after the one-time evaluation.
- V2.1 has not been connected to the website.
