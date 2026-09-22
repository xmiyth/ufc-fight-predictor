# UFC Fight Predictor — machine-learning baseline

This project trains a logistic-regression model to estimate each fighter's win
probability. It contains **only the ML system**; no website has been built.

## What the model learns from

For every historical bout, the pipeline creates a snapshot using only fights
that happened earlier. It includes differences between Fighter A and Fighter B
for:

- age, height, and reach
- prior UFC wins, losses, win percentage, and experience
- significant-strike accuracy, landed per minute, and absorbed per minute
- takedown accuracy and takedown defense
- submission attempts per 15 minutes
- recent form (win percentage over the previous five fights)

The model does not use betting odds, winner-dependent columns, finish method,
or statistics from the target fight. Half of bouts are deterministically
swapped so "Fighter A" is not just another name for the red corner.

## Dataset

The checked-in raw files are sourced from the public
[UFC DataLab repository](https://github.com/komaksym/UFC-DataLab), which states
that it scrapes official [UFCStats](http://ufcstats.com/) data and refreshes the
datasets every three months.

- `data/raw/fights.csv`: `data/stats/stats_raw.csv` (raw bout results/stats)
- `data/raw/fighters.csv`: `data/external_data/raw_fighter_details.csv`

Refresh both files at any time with:

```powershell
.venv\Scripts\python -m ufc_predictor.download_data
```

## Setup (Windows PowerShell)

Python 3.12 and the `.venv` environment have already been set up. To recreate
the environment later:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Train and evaluate

```powershell
.venv\Scripts\python -m ufc_predictor.train
```

For the rigorous multi-model comparison (logistic regression, random forest,
gradient boosting, and XGBoost), use:

```powershell
.venv\Scripts\python -m ufc_predictor.evaluate
```

This writes `reports/model_evaluation.json`, calibration bins, missing-data
diagnostics, common-test-period predictions, an untouched-test model at
`models/best_evaluated_model.joblib`, and a deployment copy refit on all known
fights at `models/best_model.joblib`.

The split is chronological: the oldest 80% of event dates train the model and
the newest 20% test it. Outputs are:

- `models/logistic_regression.joblib` — model plus latest fighter histories
- `reports/metrics.json` — accuracy, dates, fight counts, feature importance
- `reports/test_predictions.csv` — every held-out probability
- `data/processed/prefight_features.csv` — auditable pre-fight feature table

Current baseline result (with the included June 2026 snapshot): **60.5% test
accuracy** on 1,863 newer fights. Training used 6,718 older fights, with
2022-10-29 as the cutoff. Re-running training after a data refresh may change
these numbers.

## Predict two fighters

```powershell
.venv\Scripts\python -m ufc_predictor.predict "Jon Jones" "Stipe Miocic"
```

Example output format:

```text
Fighter A (Jon Jones): 65.0%
Fighter B (Stipe Miocic): 35.0%
```

Probabilities are model estimates, not guarantees or betting advice. A fighter's
historical state is only as current as the downloaded fight file; refresh and
retrain before using the predictions.

## Local V2.1 web interface

Start the minimal local interface with:

```powershell
.venv\Scripts\python -m ufc_predictor.web
```

Then open `http://127.0.0.1:8000/`. The interface loads the checksum-verified
`models/v2.1-accuracy/ufc_predictor_v2_1_accuracy.joblib` ensemble (25% logistic
regression, 75% XGBoost). Its saved historical holdout accuracy is 62.69% on
772 fights; fighter data runs through June 27, 2026. No training or holdout
rescoring occurs when serving predictions. V2 histories are reconstructed from
the raw data at that cutoff and cached with source checksums.

The interface retains V1.1 profile/display histories and the separate method
model. Winner predictions use V2 features and both V2.1 ensemble components.
The existing order-symmetry and extreme-size adjustments remain in place;
62.69% is the frozen ensemble's historical evaluation, not a separately measured
score for these website adjustments. Explanations show local ensemble
sensitivity to feature replacement, not additive logistic coefficients.


## Upcoming UFC events

The Events page reads the public ESPN UFC schedule for the next 120 days.
It refreshes on page requests at most once per hour and caches only scheduled
matchups, dates, venue, weight class, and round count in `data/upcoming_espn.json`.
Contender Series, live/completed/cancelled bouts, and unannounced opponents are
excluded. The feed does not supply model features, records, odds, or results.

Opening a card generates and timestamps predictions for fighters found in the
frozen model histories. Missing identities stay unavailable. New picks are
blocked on the event date, when the schedule cannot be freshly verified, or
when an event is not after every model data cutoff. Raw fighter data is checked
against the V2.1 architecture-lock checksums at startup. There is no training,
re-evaluation, or automatic result scoring in this schedule integration.

Opponent, date, weight-class or round changes create a new matchup identity;
previous predictions remain in `data/prediction_history.json`. Future feeds
may change format or be unavailable: cached cards remain visible with a warning,
but new predictions wait for a successful refresh. Set `UFC_EVENT_SOURCE=local`
to use `data/upcoming_events.json` instead.
