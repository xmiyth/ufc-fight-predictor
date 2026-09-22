# UFC Predictor Model V1.0

Status: **Frozen**  
Model: **Pruned Rich Logistic Regression**  
Historical fights used: **8,581**

## Chronological evaluation

- Training period: **March 11, 1994 – October 29, 2022**
- Test period: **November 5, 2022 – June 27, 2026**
- Test fights: **1,863**

| Metric | V1.0 result |
|---|---:|
| Accuracy | **60.82%** |
| ROC AUC | **0.6470** |
| Log loss | **0.6566** |
| Brier score | **0.2325** |

## Calibration

- Expected calibration error: **1.03 percentage points**
- For predictions in the 65%–75% band, the average prediction was **69.0%**.
- Those predicted fighters actually won **70.2%** of the time across 369 fights.
- The observed 95% interval was **65.3%–74.6%**.

## Important features

The most influential standardized V1.0 features were:

1. Age difference
2. Pre-fight Elo difference
3. Quality-adjusted win score difference
4. Average opponent Elo difference
5. Pre-fight UFC win-percentage difference
6. Significant-strike accuracy difference
7. Prior UFC wins difference
8. Striking-defense difference
9. Recent three-fight win-percentage difference
10. Significant strikes absorbed per minute difference

The model also retains recent five-fight form, win/loss streaks, method-of-win
percentages, physical attributes, detailed striking distributions, grappling
efficiency, layoff, experience, and leakage-safe matchup interactions.

## Benchmark performance

| Benchmark | Accuracy |
|---|---:|
| Random coin flip | 49.76% |
| Training-majority baseline | 51.10% |
| Better pre-fight UFC win percentage | 59.58% |
| Better recent five-fight record | 57.97% |
| Elo rating | 55.50% |

V1.0 beat the pre-fight UFC win-percentage benchmark by **1.24 percentage
points** on the common test fights. This difference was not statistically clear
(paired 95% interval: -1.29 to +3.76 percentage points; exact McNemar p=0.353).

## Known limitations

- Accuracy is modest; predictions are probabilities, not guarantees.
- Results only cover UFCStats-derived history through **June 27, 2026**.
- Historical betting odds were unavailable and are not used.
- Historical non-UFC career records were unavailable without leakage and are not used.
- Reach is missing for 25.4% of rows, age for 14.6%, and height for 12.6%; the
  pipeline imputes missing values using training-period medians.
- Debut/layoff information is unavailable for 25.1% of rows, mainly because a
  fighter has no previous recorded UFC fight.
- Injuries, training camps, short-notice replacements, weight cuts, travel, and
  other real-world context are not represented.
- Aggregate calibration does not guarantee that every individual 70% forecast
  has exactly a 70% chance of winning.

## Frozen artifact

The exact model is stored at:

`models/v1.0/ufc_predictor_v1_0.joblib`

SHA-256:

`E592F2FFF8EE03B5A860BBF3FC5F8C82C1ECFE9668379C2AE219AC965719F825`

The version directory also contains its evaluation report, dependency pins,
exact feature list, manifest, and prediction-source snapshot. No model was
retrained or reevaluated while creating this frozen release.

