# UFC winner predictor V1.1 advanced research report

## Decision

The advanced V1.1 candidate was **rejected and not promoted**. V1.0 remains the
benchmark because the candidate was worse on the locked chronological test.
The test period was not moved or used to choose features, models, ensemble
weights, or calibration.

V1.0 artifact SHA-256:
`E592F2FFF8EE03B5A860BBF3FC5F8C82C1ECFE9668379C2AE219AC965719F825`.

## Fixed-test result

Period: 2022-11-05 through 2026-06-27. Fights: 1,863.

| Model | Accuracy | ROC AUC | Log loss | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| V1.0 Full Rich Logistic | 61.19% | 0.6467 | 0.6577 | 0.2330 | 0.0209 |
| Advanced V1.1 candidate | 59.26% | 0.6358 | 0.6628 | 0.2354 | 0.0195 |

The candidate changed 312 winner calls. Of those, 138 repaired a V1.0 error and
174 changed a correct V1.0 call into an error, a net loss of 36 correct calls.
The exact paired McNemar/binomial p-value is 0.0474. The difference is
statistically meaningful at 0.05, but in the wrong direction.

## V1.0 error analysis

All 723 incorrect predictions are in `v1_1_error_rows.csv`. Categories overlap;
the most important counts were:

| Category | Errors | Percent of all V1.0 errors |
|---|---:|---:|
| Debutant / fighter with at most 2 prior UFC fights | 351 | 48.55% |
| Close prediction (under 55% confidence) | 229 | 31.67% |
| Recent decline | 203 | 28.08% |
| Fighter moving divisions | 113 | 15.63% |
| Recent improvement | 103 | 14.25% |
| Wrestler-versus-striker proxy | 90 | 12.45% |
| Major upset (70%+ wrong favorite) | 70 | 9.68% |
| Long layoff (540+ days) | 57 | 7.88% |
| Power-versus-poor-defense proxy | 45 | 6.22% |
| Age gap of at least 8 years | 43 | 5.95% |
| Reach gap of at least 6 inches | 31 | 4.29% |
| Grappler-versus-submission-vulnerability proxy | 29 | 4.01% |
| Size gap of at least 2 historical divisions | 3 | 0.41% |

Short-notice status was not present in the supplied data and was not inferred.

## What was implemented and evaluated

The point-in-time feature builder contains career and recent windows, 365/730
day and exponential-decay statistics, opponent-quality and opponent-adjusted
performance, multiple pre-fight Elo variants, activity/layoff and decline
signals, weight-class/physical context, and style matchup interactions. Missing
statistics remain missing until a median imputer fitted only on the training
partition handles them. Current profile totals, future opponent results, future
ratings, and current-fight statistics are excluded.

The supplied data does not contain reliable point-in-time professional records,
promotion history, regional Elo, or short-notice flags. Current profile career
totals were deliberately excluded because applying them to old fights would
leak future information.

Model families evaluated chronologically were logistic regression, random
forest, extra trees, gradient boosting, histogram gradient boosting, XGBoost,
Elo, and probability ensembles. Selection used 2019-2020 OOF predictions for
tuning and 2021 through 2022-10-29 for confirmation. The selected candidate was
20% logistic + 40% histogram gradient boosting + 40% Elo, with a separate
debutant-safe logistic route for matchups involving a fighter with fewer than
three prior UFC fights.

Platt scaling improved confirmation log loss and Brier but reduced confirmation
accuracy; isotonic regression also failed the promotion rule. The locked
candidate therefore used its uncalibrated probabilities.

## Feature-group confirmation ablation

Training ended in 2020 and confirmation covered 2021-01-16 through 2022-10-29.

| Feature set | Accuracy | Log loss | Change in accuracy |
|---|---:|---:|---:|
| Career-average baseline | 60.60% | 0.6600 | — |
| + recent form | 59.85% | 0.6656 | -0.75 pp |
| + opponent strength | 60.50% | 0.6600 | -0.11 pp |
| + Elo | 61.14% | 0.6571 | +0.54 pp |
| + style interactions | 60.50% | 0.6606 | -0.11 pp |
| + age/decline | 59.31% | 0.6631 | -1.29 pp |
| + debutant experience | 60.60% | 0.6600 | 0.00 pp |
| + size/weight context | 60.50% | 0.6594 | -0.11 pp |
| Full rich 548-day set | 60.71% | 0.6648 | +0.11 pp |

Elo was the clearest helpful incremental group. Size/context modestly improved
log loss but not accuracy. The expanded recent, style, and age/decline groups
hurt this confirmation slice. Despite better late-validation results from tree
models, that apparent gain did not generalize to the fixed test.

## Fixed-test subgroups

| Subgroup | Fights | V1.0 accuracy | Candidate accuracy |
|---|---:|---:|---:|
| Involves UFC debutant | 349 | 54.44% | 51.00% |
| Involves fighter with fewer than 3 UFC fights | 821 | 57.25% | 56.76% |
| Both fighters have 3+ UFC fights | 1,042 | 64.30% | 61.23% |
| Age gap 8+ years | 139 | 69.06% | 66.91% |
| Reach gap 6+ inches | 96 | 67.71% | 66.67% |

Full weight-class results are stored in `v1_1_advanced_evaluation.json`.

## Confidence buckets on the fixed test

| Confidence | V1.0 fights / accuracy | Candidate fights / accuracy |
|---|---:|---:|
| 50-55% | 523 / 56.21% | 588 / 50.85% |
| 55-60% | 465 / 56.34% | 533 / 57.79% |
| 60-65% | 342 / 60.53% | 398 / 63.57% |
| 65-70% | 266 / 67.67% | 224 / 65.18% |
| 70-80% | 232 / 71.55% | 105 / 80.00% |
| 80%+ | 35 / 88.57% | 15 / 93.33% |

## Production safeguards and tests

V1.0 was not overwritten. The rejected advanced candidate was not deployed.
The existing API/UI continues to support debutants and low-sample fighters and
reports data-quality warnings separately from winner probability. Winner
inference now scores both fighter orientations so swapping A and B exactly
inverts the displayed probability. A separate, auditable log-odds size prior is
used only for hypothetical gaps of three or more divisions; it cannot alter
normal same- or adjacent-division fights.

The fast regression suite passes 39 tests, including debutant, one-fight,
two-fight, extreme-size, direct-probability, and new symmetry cases. The existing
real-history temporal tests verify current-bout removal, future-bout removal,
and strict `feature_timestamp < fight_timestamp`; their feature-building code
was not changed by the inference symmetry patch.

## Recommendation for V1.2

Do not add more UFC-only rolling features indiscriminately. The most promising
next step is acquiring timestamped non-UFC professional records and promotion
history for debutants, then evaluating them with a new untouched future
lockbox. Rework ensemble selection to reward stability across multiple annual
folds rather than peak performance on the latest confirmation interval. Keep
Elo as a feature, but do not give standalone Elo a large ensemble weight unless
that stability criterion is met.
