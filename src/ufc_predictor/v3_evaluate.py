"""Advanced leakage-safe V3 walk-forward model and rating comparison."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .v2_data import file_sha256
from .v2_evaluate import feature_names as v2_feature_names, metrics
from .v3_features import V3_EXTRA_FEATURE_NAMES, V3_FEATURE_NAMES, build_v3_dataset


RANDOM_STATE = 20260902
FOLD_YEARS = tuple(range(2019, 2027))
TUNING_YEARS = tuple(range(2019, 2025))
CONFIRMATION_YEARS = (2025, 2026)


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    feature_set: str
    params: dict[str, Any]


def candidate_list() -> list[Candidate]:
    return [
        Candidate("v2_logistic_reference", "logistic", "v2", {"C": 1.0}),
        Candidate("v3_logistic_c0.03", "logistic", "v3", {"C": 0.03}),
        Candidate("v3_logistic_c0.1", "logistic", "v3", {"C": 0.1}),
        Candidate("v3_elastic_l1r0.5", "elastic", "v3", {"C": 0.1, "l1_ratio": 0.5}),
        Candidate("v3_extra_trees", "extra_trees", "v3", {"n_estimators": 300, "min_samples_leaf": 7, "max_features": 0.6}),
        Candidate("v3_hist_gradient", "hist_gradient", "v3", {"max_iter": 200, "learning_rate": 0.04, "max_leaf_nodes": 15, "l2_regularization": 3.0}),
        Candidate("v3_xgboost", "xgboost", "v3", {"n_estimators": 400, "learning_rate": 0.02, "max_depth": 2, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.45, "reg_alpha": 0.5, "reg_lambda": 4.0}),
        Candidate("v3_lightgbm", "lightgbm", "v3", {"n_estimators": 300, "learning_rate": 0.025, "num_leaves": 12, "max_depth": 4, "min_child_samples": 35, "subsample": 0.8, "colsample_bytree": 0.5, "reg_alpha": 0.5, "reg_lambda": 4.0}),
        Candidate("v3_catboost", "catboost", "v3", {"iterations": 350, "learning_rate": 0.025, "depth": 4, "l2_leaf_reg": 5.0}),
    ]


def selected_features(key: str) -> list[str]:
    if key == "v2":
        return v2_feature_names("rich_548")
    if key == "v3":
        return list(V3_FEATURE_NAMES)
    raise KeyError(key)


def make_model(candidate: Candidate) -> Pipeline:
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    if candidate.family in {"logistic", "elastic"}:
        model = LogisticRegression(
            **candidate.params,
            penalty="elasticnet" if candidate.family == "elastic" else "l2",
            solver="saga" if candidate.family == "elastic" else "liblinear",
            max_iter=6000,
            random_state=RANDOM_STATE,
        )
        return Pipeline([("imputer", imputer), ("scale", StandardScaler()), ("model", model)])
    if candidate.family == "extra_trees":
        model = ExtraTreesClassifier(**candidate.params, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced")
    elif candidate.family == "hist_gradient":
        model = HistGradientBoostingClassifier(**candidate.params, random_state=RANDOM_STATE)
    elif candidate.family == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(**candidate.params, objective="binary:logistic", eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE)
    elif candidate.family == "lightgbm":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(**candidate.params, n_jobs=-1, random_state=RANDOM_STATE, verbosity=-1)
    elif candidate.family == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(**candidate.params, random_seed=RANDOM_STATE, verbose=False, thread_count=-1, allow_writing_files=False)
    else:
        raise KeyError(candidate.family)
    return Pipeline([("imputer", imputer), ("model", model)])


def load_or_build(root: Path) -> pd.DataFrame:
    processed = root / "data" / "processed"
    cache = processed / "v3_point_in_time.joblib"
    metadata_path = processed / "v3_point_in_time.metadata.json"
    fights = root / "data" / "raw" / "fights.csv"
    fighters = root / "data" / "raw" / "fighters.csv"
    code = root / "src" / "ufc_predictor" / "v3_features.py"
    expected = {
        "fights_sha256": file_sha256(fights),
        "fighters_sha256": file_sha256(fighters),
        "feature_code_sha256": file_sha256(code),
        "feature_names": V3_FEATURE_NAMES,
    }
    if cache.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in expected.items()):
            return joblib.load(cache)
    frame = build_v3_dataset(fights, fighters)
    processed.mkdir(parents=True, exist_ok=True)
    joblib.dump(frame, cache, compress=3)
    metadata_path.write_text(json.dumps({
        **expected,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(frame),
        "features": len(V3_FEATURE_NAMES),
        "timestamp_contract": "feature_timestamp < fight_timestamp",
    }, indent=2), encoding="utf-8")
    return frame


def decisive(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[frame.fighter_a_won.notna()].copy()
    result["fighter_a_won"] = result.fighter_a_won.astype(int)
    result["fight_date"] = pd.to_datetime(result.fight_date)
    return result.sort_values("fight_date", kind="stable").reset_index(drop=True)


def walk_forward(frame: pd.DataFrame, candidate: Candidate) -> tuple[pd.DataFrame, list[dict]]:
    features = selected_features(candidate.feature_set)
    predictions, folds = [], []
    for year in FOLD_YEARS:
        train = frame.loc[frame.fight_date.dt.year < year]
        test = frame.loc[frame.fight_date.dt.year == year]
        if test.empty:
            continue
        model = make_model(candidate)
        model.fit(train[features], train.fighter_a_won)
        probability = model.predict_proba(test[features])[:, 1]
        values = metrics(test.fighter_a_won.to_numpy(), probability)
        folds.append({"year": year, "train_fights": len(train), "test_fights": len(test), **values})
        part = test[["fight_id", "fight_date", "fighter_a_won", "weight_class"]].copy()
        part["probability"] = probability
        predictions.append(part)
    return pd.concat(predictions, ignore_index=True), folds


def _rating_predictions(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "standard_elo": 1.0 / (1.0 + 10.0 ** (-frame.elo_diff.fillna(0).to_numpy() / 400.0)),
        "elo_k16": 1.0 / (1.0 + 10.0 ** (-frame.elo_k16_diff.fillna(0).to_numpy() / 400.0)),
        "elo_k24": 1.0 / (1.0 + 10.0 ** (-frame.elo_k24_diff.fillna(0).to_numpy() / 400.0)),
        "elo_k48": 1.0 / (1.0 + 10.0 ** (-frame.elo_k48_diff.fillna(0).to_numpy() / 400.0)),
        "recency_elo": 1.0 / (1.0 + 10.0 ** (-frame.recency_adjusted_elo_diff.fillna(0).to_numpy() / 400.0)),
        "division_elo": 1.0 / (1.0 + 10.0 ** (-frame.weight_class_elo_diff.fillna(0).to_numpy() / 400.0)),
        "glicko2": 1.0 / (1.0 + 10.0 ** (-frame.v3_glicko_rating_diff.fillna(0).to_numpy() / 400.0)),
        "bradley_terry": 1.0 / (1.0 + np.exp(-frame.v3_bradley_terry_strength_diff.fillna(0).to_numpy())),
        "division_bradley_terry": 1.0 / (1.0 + np.exp(-frame.v3_division_strength_diff.fillna(0).to_numpy())),
    }


def run(root: Path) -> dict:
    output = root / "reports" / "v3_research"
    output.mkdir(parents=True, exist_ok=True)
    frame = decisive(load_or_build(root))
    aggregate, all_folds = [], []
    for index, candidate in enumerate(candidate_list(), 1):
        prediction_path = output / f"oof_{candidate.name}.csv"
        fold_path = output / f"folds_{candidate.name}.json"
        if prediction_path.exists() and fold_path.exists():
            print(f"[{index}/{len(candidate_list())}] reuse {candidate.name}", flush=True)
            predictions = pd.read_csv(prediction_path, parse_dates=["fight_date"])
            folds = json.loads(fold_path.read_text(encoding="utf-8"))
        else:
            print(f"[{index}/{len(candidate_list())}] {candidate.name}", flush=True)
            predictions, folds = walk_forward(frame, candidate)
            predictions.to_csv(prediction_path, index=False)
            fold_path.write_text(json.dumps(folds, indent=2), encoding="utf-8")
        tuning = predictions.loc[predictions.fight_date.dt.year.isin(TUNING_YEARS)]
        confirmation = predictions.loc[predictions.fight_date.dt.year.isin(CONFIRMATION_YEARS)]
        tune_metrics = metrics(tuning.fighter_a_won.to_numpy(), tuning.probability.to_numpy())
        confirmation_metrics = metrics(confirmation.fighter_a_won.to_numpy(), confirmation.probability.to_numpy())
        aggregate.append({
            **asdict(candidate), "feature_count": len(selected_features(candidate.feature_set)),
            **{f"tuning_{key}": value for key, value in tune_metrics.items()},
            **{f"confirmation_{key}": value for key, value in confirmation_metrics.items()},
            "tuning_accuracy_std": float(np.std([row["accuracy"] for row in folds if row["year"] in TUNING_YEARS])),
        })
        all_folds.extend([{"candidate": candidate.name, **row} for row in folds])

    candidate_table = pd.DataFrame(aggregate).sort_values(
        ["tuning_accuracy", "tuning_log_loss"], ascending=[False, True]
    )
    candidate_table.to_csv(output / "model_comparison.csv", index=False)
    pd.DataFrame(all_folds).to_csv(output / "walk_forward_by_year.csv", index=False)

    rating_rows = []
    evaluation_rows = frame.loc[frame.fight_date.dt.year.isin(FOLD_YEARS)].copy()
    for name, probability in _rating_predictions(evaluation_rows).items():
        for period_name, years in (("tuning", TUNING_YEARS), ("confirmation", CONFIRMATION_YEARS)):
            mask = evaluation_rows.fight_date.dt.year.isin(years).to_numpy()
            values = metrics(evaluation_rows.fighter_a_won.to_numpy()[mask], probability[mask])
            rating_rows.append({"rating": name, "period": period_name, "fights": int(mask.sum()), **values})
    rating_table = pd.DataFrame(rating_rows)
    rating_table.to_csv(output / "rating_comparison.csv", index=False)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "development and confirmation only; final V3 lockbox not opened",
        "tuning_years": list(TUNING_YEARS),
        "confirmation_years": list(CONFIRMATION_YEARS),
        "fight_range": [str(frame.fight_date.min().date()), str(frame.fight_date.max().date())],
        "decisive_fights": len(frame),
        "v3_features": len(V3_FEATURE_NAMES),
        "v3_new_features": len(V3_EXTRA_FEATURE_NAMES),
        "best_tuning_candidate": candidate_table.iloc[0].to_dict(),
        "best_confirmation_candidate_by_accuracy": candidate_table.sort_values(
            ["confirmation_accuracy", "confirmation_log_loss"], ascending=[False, True]
        ).iloc[0].to_dict(),
        "rating_results": rating_rows,
        "final_lockbox": "must be post-2026-06-27; no defensible untouched outcomes exist in the current immutable dataset",
    }
    (output / "development_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(candidate_table[["name", "tuning_accuracy", "tuning_roc_auc", "tuning_log_loss", "tuning_brier", "tuning_ece", "confirmation_accuracy"]].to_string(index=False), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    run(Path(args.project_root).resolve())


if __name__ == "__main__":
    main()
