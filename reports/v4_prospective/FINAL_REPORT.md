# UFC Predictor V4 prospective research report

## Decision

The verified production champion remains V2.1 at **62.69% (484/772)** on the
January 11, 2025 through June 27, 2026 chronological holdout. That period was
already inspected, so it was not used to select or score V4.

V4 is **not promoted**. Its development-selected mixture improved a separate
2023-2024 confirmation block by only one fight. It is frozen as a prospective
research candidate and needs an untouched test consisting only of fights after
September 11, 2026.

## Temporal results

| Model | Period | Fights | Correct | Accuracy | ROC AUC | Log loss | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen V2 architecture | 2019-2022 selection | 1,953 | 1,183 | 60.57% | 0.6566 | 0.6541 | 0.2311 | 0.0238 |
| Development-selected V4 gate | 2019-2022 selection | 1,953 | 1,195 | 61.19% | 0.6580 | 0.6533 | 0.2308 | 0.0108 |
| Frozen V2 architecture | 2023-2024 confirmation | 1,017 | 638 | 62.73% | 0.6512 | 0.6562 | 0.2322 | 0.0427 |
| Development-selected V4 gate | 2023-2024 confirmation | 1,017 | 639 | 62.83% | 0.6580 | 0.6540 | 0.2311 | 0.0411 |

The V4 gate changed 102 development predictions: 57 became correct and 45
became incorrect. On confirmation it changed 71: 36 became correct and 35
became incorrect. The confirmation gain is therefore not strong evidence of a
real accuracy improvement.

## Architecture and features

The existing 25% regularized-logistic / 75% XGBoost structure and rich-548 V2
feature set were retained. V4 adds:

- explicit DEBUTANT, LOW_SAMPLE, and NORMAL matchup features;
- per-side UFC-history reliability and minimum-history features;
- training-fold-only global and weight-division median priors;
- empirical-Bayes shrinkage with a four-fight prior, selected on 2019-2022;
- missing UFC performance for debutants, handled by training-fold median
  imputation plus missing indicators instead of interpreting absence as zero;
- a development-selected gate that uses the V2 branch for debutant matchups and
  the V4 branch otherwise.

The standalone V4 branch improved 2019-2022 accuracy from 60.57% to 60.98%, but
fell to 61.95% versus 62.73% on 2023-2024. It was rejected. It improved
established-fighter accuracy on both blocks, but materially harmed debutants.

## Prior research and failed groups

The existing V3 work already tested regularized logistic regression,
HistGradientBoosting, Extra Trees, XGBoost, LightGBM, CatBoost, Glicko,
Bradley-Terry, division ratings, recency decays, opponent-adjusted statistics,
trends, division movement, and MMA-specific matchup interactions. None produced
a stable temporal-validation accuracy gain over V2. The selected V3 ensemble
was also not promoted. Expanded opponent adjustment, extra recency features,
Glicko/Bradley-Terry, and the larger interaction set therefore remain rejected
for production; the useful versions already present in V2 remain intact.

No historical pre-fight odds source was present in the primary immutable raw
dataset, so no market model was mixed into the stats-only predictor.

## Leakage and integrity

- All 8,581 decisive feature rows satisfy `feature_timestamp < fight_timestamp`.
- Division priors are fitted independently inside each expanding training fold.
- Same-date source fights retain the existing batch-update protection.
- No 2025 or 2026 outcome was scored or used for V4 model/feature selection.
- V4's final prospective artifact is trained through June 27, 2026 only after
  its architecture was selected; its next valid test begins after September 11,
  2026.
- Frozen V1/V2 code, data, and model hashes still match the V2.1 lock manifest.
- The 42-test fast regression suite passes; the live API health endpoint returns
  HTTP 200. The website remains on the existing V1.1 winner model and Method
  V2.0 because V4 has not earned promotion.

## Artifacts

- `models/v4-prospective/ufc_predictor_v4_prospective.joblib`
- `models/v4-prospective/ARCHITECTURE_LOCK.json`
- `models/v4-prospective/ARCHITECTURE_LOCK.sha256`
- `reports/v4_prospective/development_report.json`
- `reports/v4_prospective/model_comparison.csv`
- `reports/v4_prospective/walk_forward_by_year.csv`
- `reports/v4_prospective/subgroup_comparison.json`

The chief remaining weakness is debutant prediction. UFC-only data cannot infer
missing pre-UFC style and competition quality; a meaningful improvement likely
requires a chronologically versioned professional-record and prior-promotion
source rather than more transformations of absent UFC statistics.
