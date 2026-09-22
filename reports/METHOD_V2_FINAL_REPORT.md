# Method of Victory V2.0 — final report

## Outcome

The method system remains separate from frozen Winner Model V1.0. It now models
`P(method | fighter wins, matchup)` in both directions and exposes a normalized
six-outcome distribution (A/B × KO/TKO/Submission/Decision).

The old unrealistic Decision results were caused by three related limitations:

1. The method model received one bout-level weight-class flag, not each fighter's
   pre-fight historical division, so cross-division fantasy matchups looked like
   same-division bouts.
2. Historical method rates and the dominant Decision prior could overwhelm the
   limited matchup interactions.
3. The selected tree model was unable to extrapolate beyond the very sparse
   extreme-size training region.

## Leakage controls and features

Every historical row is snapshotted before the current fight updates either
fighter. Historical division is the fighter's last observed UFC division before
the bout; a UFC debut falls back to the already-known booked division. Static
profile weight, future division changes, final career totals, the current fight's
statistics, and future fights are not used.

Men's ordered ladder: Flyweight, Bantamweight, Featherweight, Lightweight,
Welterweight, Middleweight, Light Heavyweight, Heavyweight. Women's divisions
use a separate Strawweight-through-Featherweight ladder.

Added inputs include fighter-specific division one-hots and ranks, absolute
division distance, same/one/two-plus flags, NORMAL/MODERATE/LARGE/EXTREME flags,
reference weights, signed estimated weight gap, relative weight gap, height,
reach, age, stance, a division-reference BMI proxy, KO/submission/decision loss
rates, recent method/finish rates, and style/matchup statistics.

Reference limits are 125/135/145/155/170/185/205/265 lb for men's divisions and
115/125/135/145 lb for women's divisions. These are neutral size proxies—not
claimed fight-night weights. `estimated_weight_gap = winner reference - opponent
reference`; `relative_weight_gap = estimated gap / smaller reference weight`.

Interactions: size × KO rate, size × submission rate, size × takedowns/15,
size × opponent KO-loss rate, size × opponent submission-loss rate, reach ×
striking accuracy, division distance × finish rate, and division distance ×
decision rate.

## Model comparison and chronological test

The untouched chronological test contains 1,860 fights from 2022-11-05 through
2026-06-27; training ends 2022-10-29.

| Model | Accuracy | Macro F1 | Log loss |
|---|---:|---:|---:|
| Multinomial logistic (deployed) | 53.49% | 0.4200 | 0.9555 |
| Gradient boosting | 53.76% | — | 0.9543 |
| XGBoost | 54.09% | 0.4001 | 0.9473 |

XGBoost had the best in-distribution accuracy/log loss. Regularized multinomial
logistic regression is deployed because it is simpler, has better macro F1,
Submission recall, and calibration ECE, and—critically—changes continuously for
OOD division gaps rather than saturating at the largest observed tree split.

Deployed per-class results:

| Class | Precision | Recall |
|---|---:|---:|
| KO/TKO | 50.10% | 41.28% |
| Submission | 36.96% | 10.09% |
| Decision | 55.99% | 77.13% |

No post-fit probability calibration is applied. Held-out top-label ECE is
0.0156; no calibration transform had evidence of improving selection.

## Weight-feature ablation

The same logistic architecture was held constant.

| Features | Accuracy | Macro F1 | Log loss | KO recall | SUB recall | DEC recall |
|---|---:|---:|---:|---:|---:|---:|
| A: style, no weight | 54.62% | 0.4283 | 0.9526 | 41.95% | 10.09% | 78.96% |
| B: + division/weight | 53.49% | 0.4133 | 0.9538 | 39.93% | 8.90% | 78.43% |
| C: + physical/interactions | 53.49% | 0.4200 | 0.9555 | 41.28% | 10.09% | 77.13% |

Weight features do **not** improve aggregate in-distribution chronology metrics.
This is expected but important: 7,337 eligible rows are same-division, 1,032 are
one division apart, 53 are two apart, and only 11 are three or more apart. They
are retained to make explicit fantasy-matchup extrapolation responsive and are
paired with LOW reliability—not represented as validated accuracy gains.

## Sanity diagnostics

Methods below are conditional on the displayed predicted winner winning.

| Matchup | Winner (win %) | KO | SUB | DEC | Mismatch/reliability |
|---|---|---:|---:|---:|---|
| Jon Jones vs Demetrious Johnson | Jones (94.6%) | 64.8% | 11.5% | 23.7% | EXTREME / LOW |
| Jon Jones vs Brandon Moreno | Jones (92.1%) | 55.7% | 17.8% | 26.5% | EXTREME / LOW |
| Tom Aspinall vs Alexandre Pantoja | Aspinall (77.4%) | 86.2% | 2.4% | 11.4% | EXTREME / LOW |
| Islam Makhachev vs Charles Oliveira | Makhachev (86.6%) | 16.3% | 28.0% | 55.7% | MODERATE / MEDIUM |
| Alex Pereira vs Israel Adesanya | Pereira (58.3%) | 33.4% | 3.0% | 63.6% | LARGE / LOW |
| Merab Dvalishvili vs Sean O'Malley | O'Malley (62.0%) | 27.0% | 4.8% | 68.1% | NORMAL / HIGH |
| Max Holloway vs Dustin Poirier | Holloway (61.7%) | 24.2% | 8.7% | 67.1% | NORMAL / HIGH |

The Demetrious Johnson progression diagnostic spans division distances 0, 1, 3,
4, and 7. His conditional KO probability changes from 5.9% against Brandon
Moreno to 87.1% against Tom Aspinall, demonstrating that the pipeline responds
materially as size mismatch grows (without asserting that Johnson would win).

## Remaining limitations

Extreme fantasy fights are out of distribution: only 11 training examples have
3+ prior-division distance. Their direction is model-based but not empirically
calibrated. Reference limits cannot represent cuts, rehydration, body composition,
or fight-night mass. Submission recall remains low. Historical division means
last observed UFC division, which can lag an athlete's current intended class.
Winner V1.0 itself was deliberately left unchanged, so its cross-division win
probabilities retain the limitations of that frozen model.

