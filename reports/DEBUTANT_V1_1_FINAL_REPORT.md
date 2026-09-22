# UFC Winner Model V1.1 — debutant and low-sample support

## Implementation

V1.1 keeps V1.0's model family and pipeline structure: training-set median
imputation, standard scaling, and regularized logistic regression (`C=0.05`).
V1.0 remains preserved at `models/v1.0/ufc_predictor_v1_0.joblib`; V1.1 is a
separate, checksum-verified artifact.

The selection restriction is now zero fights. Profile-only fighters are seeded
with empty point-in-time state, making 0-, 1-, and 2-fight athletes selectable.
The API returns per-fighter prior UFC fight counts, DEBUTANT / LOW_SAMPLE /
NORMAL modes, and a separate warning. No probability is shrunk toward 50%.

For a fighter with no prior UFC history, performance rates are missing—not zero.
This includes UFC win percentage, SLpM, SApM, striking accuracy/defense,
takedown accuracy/defense, submission activity, finish/method rates, opponent
quality, and recent form. The pipeline imputes these from training-set medians.
Zero remains valid only for actual counts such as prior UFC fights and wins.

Added active inputs are UFC debutant flags, per-side UFC fight counts, pre-fight
KO/TKO/submission/decision win counts, stance flags, and booked weight-class
flags. Existing age, height, reach, UFC experience/history, matchup, and rating
features remain. The raw dataset does not contain point-in-time professional MMA
records or previous promotion. Current UFCStats profile summaries cannot be used
historically because they contain later career information, so they are excluded.

## Chronological holdout

Training: 6,718 fights through 2022-10-29. Untouched test: 1,863 fights from
2022-11-05 through 2026-06-27.

| Metric | Frozen V1.0 on missing-safe inputs | V1.1 |
|---|---:|---:|
| Accuracy | 58.56% | **59.85%** |
| ROC AUC | 0.6226 | **0.6342** |
| Log loss | 0.66961 | **0.66315** |
| Brier score | 0.23852 | **0.23558** |

Because V1.1 improves ROC AUC, log loss, Brier score, and accuracy with the new
input semantics, it satisfies the replacement gate. The original artifact is
still retained for reproducibility.

## Required subgroups (V1.1)

| Subgroup | Fights | Accuracy |
|---|---:|---:|
| Both fighters had 3+ prior UFC fights | 1,042 | 62.28% |
| At least one fighter had 0 prior UFC fights | 349 | 51.00% |
| At least one had 1–2, with no debutant | 472 | 61.02% |
| Both fighters were UFC debutants (diagnostic) | 50 | 50.00% |

These subgroup results describe uncertainty; the API does not post-process or
artificially flatten the model probabilities.

## Leakage controls and limitations

Historical rows are snapshotted before their fight updates fighter state. Booked
weight class is known pre-fight. Final UFCStats performance summaries, future
division changes, future fights/results, and post-fight statistics are excluded.

Professional MMA record and prior-promotion data would be especially useful for
two-debutant fights, but no leakage-safe historical source is supplied. Such
matchups therefore rely on available age/height/reach/stance, booked division,
neutral rating priors, and explicit zero-UFC-experience indicators. The observed
50% both-debutant holdout accuracy confirms that these predictions should be
treated as high uncertainty.

Machine-readable details are in `reports/debutant_evaluation.json`.
