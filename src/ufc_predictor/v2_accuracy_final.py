"""One-time V2.1 accuracy lockbox evaluation and production artifact build."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .v2_accuracy_development import (
    LOGISTIC,
    WEIGHTS,
    XGBOOST,
    accuracy_interval,
    confidence_table,
    paired_accuracy_interval,
)
from .v2_data import file_sha256
from .v2_evaluate import decisive, feature_names, load_or_build_dataset, make_model, metrics


def _verify_lock(root: Path, lock: dict) -> None:
    checks = lock["checksums"]
    expected = {
        "development_final_sha256": root / "reports" / "v2_accuracy" / "development_final.json",
        "ensemble_analysis_sha256": root / "reports" / "v2_accuracy" / "ensemble_analysis.json",
        "raw_fights_sha256": root / "data" / "raw" / "fights.csv",
        "raw_fighters_sha256": root / "data" / "raw" / "fighters.csv",
        "v2_features_sha256": root / "src" / "ufc_predictor" / "v2_features.py",
        "v2_evaluate_sha256": root / "src" / "ufc_predictor" / "v2_evaluate.py",
        "v1_model_sha256": root / "models" / "v1.0" / "ufc_predictor_v1_0.joblib",
    }
    mismatches = {
        name: {"expected": checks[name], "actual": file_sha256(path)}
        for name, path in expected.items()
        if file_sha256(path) != checks[name]
    }
    if mismatches:
        raise RuntimeError(f"Architecture-lock checksum mismatch: {mismatches}")


def _ensemble_probability(logistic, xgboost, values: pd.DataFrame) -> np.ndarray:
    return (
        WEIGHTS["logistic"] * logistic.predict_proba(values)[:, 1]
        + WEIGHTS["xgboost"] * xgboost.predict_proba(values)[:, 1]
    )


def run(root: Path) -> dict:
    model_dir = root / "models" / "v2.1-accuracy"
    lock_path = model_dir / "ARCHITECTURE_LOCK.json"
    holdout_path = model_dir / "HOLDOUT_EVALUATION.json"
    if holdout_path.exists():
        raise RuntimeError(
            "The final holdout has already been evaluated. Refusing to score it again."
        )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    _verify_lock(root, lock)

    frame = decisive(load_or_build_dataset(root))
    features = feature_names("rich_548")
    train = frame.loc[frame.fight_date < "2025-01-01"].copy()
    holdout = frame.loc[frame.fight_date >= "2025-01-01"].copy()
    logistic = make_model(LOGISTIC).fit(train[features], train.fighter_a_won)
    xgboost = make_model(XGBOOST).fit(train[features], train.fighter_a_won)
    probability = _ensemble_probability(logistic, xgboost, holdout[features])
    y = holdout.fighter_a_won.to_numpy()
    correct = ((probability >= 0.5).astype(int) == y)
    confidence, high_accuracy_coverage = confidence_table(y, probability)

    majority_value = int(train.fighter_a_won.mean() >= 0.5)
    benchmarks = {
        "majority": np.full(len(holdout), majority_value),
        "better_ufc_win_pct": (holdout.ufc_win_pct_diff.fillna(0).to_numpy() >= 0).astype(int),
        "better_recent_record": (holdout.recent_5_win_pct_diff.fillna(0).to_numpy() >= 0).astype(int),
        "elo": (holdout.elo_diff.fillna(0).to_numpy() >= 0).astype(int),
    }
    benchmark_results = {
        name: float((prediction == y).mean()) for name, prediction in benchmarks.items()
    }
    strongest_name = max(benchmark_results, key=benchmark_results.get)
    strongest_correct = benchmarks[strongest_name] == y

    by_year = []
    for year, indexes in holdout.groupby(holdout.fight_date.dt.year).groups.items():
        index = holdout.index.get_indexer(indexes)
        by_year.append({"year": int(year), "fights": len(index), **metrics(y[index], probability[index])})
    by_weight_class = []
    for weight_class, indexes in holdout.groupby("weight_class").groups.items():
        index = holdout.index.get_indexer(indexes)
        if len(index) >= 20:
            by_weight_class.append({"weight_class": weight_class, "fights": len(index), **metrics(y[index], probability[index])})

    accuracy_ci = accuracy_interval(correct)
    evaluation = {
        "model_version": "UFC Predictor V2.1 Accuracy Experiment",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_count": 1,
        "lockbox_period": [str(holdout.fight_date.min().date()), str(holdout.fight_date.max().date())],
        "training_period": [str(train.fight_date.min().date()), str(train.fight_date.max().date())],
        "training_fights": int(len(train)),
        "holdout_fights": int(len(holdout)),
        "metrics": metrics(y, probability),
        "accuracy_95pct_ci": accuracy_ci,
        "confidence_buckets": confidence,
        "coverage_of_buckets_with_at_least_75pct_accuracy": float(high_accuracy_coverage),
        "benchmarks": benchmark_results,
        "strongest_simple_baseline": strongest_name,
        "paired_vs_strongest_baseline": paired_accuracy_interval(correct, strongest_correct),
        "comparison_to_v1": {
            "published_v1_accuracy": 0.6082,
            "accuracy_difference": float(correct.mean() - 0.6082),
            "statistically_meaningful_by_v2_ci": bool(
                0.6082 < accuracy_ci[0] or 0.6082 > accuracy_ci[1]
            ),
            "note": "unpaired comparison because V1 used a different historical test split",
        },
        "by_year": by_year,
        "by_weight_class_minimum_20_fights": by_weight_class,
        "architecture_lock_sha256": file_sha256(lock_path),
        "data_sha256": lock["checksums"]["raw_fights_sha256"],
        "methodology": "one-time score after architecture/weights/features were locked",
    }
    # Persist the one-time result before any production refit.
    holdout_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    # Production refit uses every completed decisive fight only after evaluation.
    production_logistic = make_model(LOGISTIC).fit(frame[features], frame.fighter_a_won)
    production_xgboost = make_model(XGBOOST).fit(frame[features], frame.fighter_a_won)
    artifact = {
        "model_version": "V2.1-accuracy",
        "architecture": "0.25 logistic + 0.75 XGBoost",
        "components": {"logistic": production_logistic, "xgboost": production_xgboost},
        "weights": WEIGHTS,
        "feature_names": features,
        "trained_through": str(frame.fight_date.max().date()),
        "training_fights": int(len(frame)),
        "data_sha256": lock["checksums"]["raw_fights_sha256"],
        "evaluation_file": "HOLDOUT_EVALUATION.json",
    }
    artifact_path = model_dir / "ufc_predictor_v2_1_accuracy.joblib"
    joblib.dump(artifact, artifact_path, compress=3)
    (model_dir / "FEATURES.json").write_text(
        json.dumps({"feature_set": "rich_548", "features": features}, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "model_version": "V2.1-accuracy",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact": artifact_path.name,
        "artifact_sha256": file_sha256(artifact_path),
        "features_sha256": file_sha256(model_dir / "FEATURES.json"),
        "architecture_lock_sha256": file_sha256(lock_path),
        "holdout_evaluation_sha256": file_sha256(holdout_path),
        "production_training_range": [str(frame.fight_date.min().date()), str(frame.fight_date.max().date())],
        "production_training_fights": int(len(frame)),
        "website_integration": False,
    }
    (model_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(evaluation, indent=2))
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    run(Path(args.project_root).resolve())


if __name__ == "__main__":
    main()
