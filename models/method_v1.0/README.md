# UFC Winning Method Model V2.0

This remains separate from—and does not alter—the frozen UFC Winner Model V1.0.

It estimates KO/TKO, Submission, and Decision conditional on either fighter
winning. The API combines those two conditional distributions with frozen winner
probabilities to produce six joint outcomes.

The deployed regularized multinomial logistic model uses strict pre-fight
snapshots plus each athlete's last observed UFC division, reference-weight gaps,
physical matchup features, style/durability statistics, and compact interactions.
Extreme cross-division predictions are explicitly marked LOW reliability.

Chronological evaluation (1,860 fights, 2022-11-05 through 2026-06-27):

- Accuracy: 53.49%
- Macro F1: 0.4200
- Multiclass log loss: 0.9555
- KO/TKO precision/recall: 50.10% / 41.28%
- Submission precision/recall: 36.96% / 10.09%
- Decision precision/recall: 55.99% / 77.13%
- Top-label ECE: 1.56%; no post-fit calibration applied

Exact metrics and ablations are in `evaluation.json`; the narrative report is
`reports/METHOD_V2_FINAL_REPORT.md`. Deployment artifact SHA-256:
`1877349434ED89329B3DDD8B8644085E1E5DEDAA2F9E492CF6B7072647CF8D3C`.
