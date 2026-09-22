"""Final development-only diagnostics before the V2 accuracy architecture lock."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from .v2_evaluate import (
    Experiment,
    RANDOM_STATE,
    decisive,
    feature_names,
    load_or_build_dataset,
    make_model,
    metrics,
)


LOGISTIC = Experiment("logistic_rich_548_c1", "logistic", "rich_548", {"C": 1.0})
XGBOOST = Experiment(
    "xgboost_rich548", "xgboost", "rich_548",
    {"n_estimators": 350, "learning_rate": 0.025, "max_depth": 2,
     "min_child_weight": 8, "subsample": 0.8, "colsample_bytree": 0.8},
)
WEIGHTS = {"logistic": 0.25, "xgboost": 0.75}


def confidence_table(y: np.ndarray, p: np.ndarray) -> tuple[list[dict], float]:
    confidence = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    edges = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.01]
    rows = []
    covered = 0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lower) & (confidence < upper)
        if not mask.any():
            continue
        accuracy = float(correct[mask].mean())
        rows.append({
            "bucket": f"{lower:.0%}-{min(upper, 1):.0%}" if upper <= 1 else "80%+",
            "fights": int(mask.sum()),
            "coverage": float(mask.mean()),
            "average_predicted_probability": float(confidence[mask].mean()),
            "actual_win_rate": accuracy,
            "accuracy": accuracy,
        })
        if accuracy >= 0.75:
            covered += int(mask.sum())
    return rows, covered / len(y)


def accuracy_interval(correct: np.ndarray, repetitions: int = 10000) -> list[float]:
    rng = np.random.default_rng(RANDOM_STATE + 20)
    values = np.empty(repetitions)
    for index in range(repetitions):
        values[index] = correct[rng.integers(0, len(correct), len(correct))].mean()
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def paired_accuracy_interval(first: np.ndarray, second: np.ndarray) -> dict:
    difference = first.astype(float) - second.astype(float)
    rng = np.random.default_rng(RANDOM_STATE + 21)
    values = np.empty(10000)
    for index in range(len(values)):
        sample = rng.integers(0, len(difference), len(difference))
        values[index] = difference[sample].mean()
    return {
        "difference": float(difference.mean()),
        "95pct_ci": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "statistically_meaningful": bool(np.quantile(values, 0.025) > 0 or np.quantile(values, 0.975) < 0),
    }


def run(root: Path) -> dict:
    report_dir = root / "reports" / "v2_accuracy"
    logistic = pd.read_csv(report_dir / "oof_logistic_rich_548_c1.csv", parse_dates=["fight_date"])
    xgboost = pd.read_csv(report_dir / "oof_xgboost_rich548.csv", parse_dates=["fight_date"])
    predictions = logistic.merge(
        xgboost[["fight_id", "probability"]], on="fight_id",
        suffixes=("_logistic", "_xgboost"), validate="one_to_one",
    )
    predictions["probability"] = (
        WEIGHTS["logistic"] * predictions.probability_logistic
        + WEIGHTS["xgboost"] * predictions.probability_xgboost
    )
    y = predictions.fighter_a_won.to_numpy()
    p = predictions.probability.to_numpy()
    correct = ((p >= 0.5).astype(int) == y)
    buckets, high_accuracy_coverage = confidence_table(y, p)

    point_in_time = decisive(load_or_build_dataset(root))
    predictions = predictions.merge(
        point_in_time[["fight_id", "ufc_win_pct_diff"]],
        on="fight_id", validate="one_to_one",
    )
    baseline_correct = (
        (predictions.ufc_win_pct_diff.fillna(0).to_numpy() >= 0).astype(int) == y
    )
    logistic_correct = (
        (predictions.probability_logistic.to_numpy() >= 0.5).astype(int) == y
    )

    yearly = []
    for year, group in predictions.groupby(predictions.fight_date.dt.year):
        yearly.append({
            "year": int(year), "fights": len(group),
            **metrics(group.fighter_a_won.to_numpy(), group.probability.to_numpy()),
        })

    # Unseen permutation importance: train through 2022, inspect 2023-2024 only.
    features = feature_names("rich_548")
    development = point_in_time.loc[point_in_time.fight_date < "2025-01-01"]
    train = development.loc[development.fight_date.dt.year <= 2022]
    confirmation = development.loc[development.fight_date.dt.year >= 2023]
    logistic_model = make_model(LOGISTIC).fit(train[features], train.fighter_a_won)
    xgboost_model = make_model(XGBOOST).fit(train[features], train.fighter_a_won)

    def ensemble_probability(values: pd.DataFrame) -> np.ndarray:
        return (
            WEIGHTS["logistic"] * logistic_model.predict_proba(values)[:, 1]
            + WEIGHTS["xgboost"] * xgboost_model.predict_proba(values)[:, 1]
        )

    base_probability = ensemble_probability(confirmation[features])
    base_accuracy = accuracy_score(confirmation.fighter_a_won, base_probability >= 0.5)
    base_log_loss = metrics(confirmation.fighter_a_won.to_numpy(), base_probability)["log_loss"]
    rng = np.random.default_rng(RANDOM_STATE + 22)
    importances = []
    for name in features:
        permuted = confirmation[features].copy()
        permuted[name] = rng.permutation(permuted[name].to_numpy())
        probability = ensemble_probability(permuted)
        values = metrics(confirmation.fighter_a_won.to_numpy(), probability)
        importances.append({
            "feature": name,
            "accuracy_decrease": float(base_accuracy - values["accuracy"]),
            "log_loss_increase": float(values["log_loss"] - base_log_loss),
        })
    importances.sort(
        key=lambda row: (row["accuracy_decrease"], row["log_loss_increase"]), reverse=True
    )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "development complete; lockbox not opened",
        "architecture": {
            "type": "probability ensemble",
            "components": [LOGISTIC.name, XGBOOST.name],
            "weights": WEIGHTS,
            "feature_set": "rich_548",
            "features": features,
        },
        "walk_forward_metrics": metrics(y, p),
        "walk_forward_accuracy_95pct_ci": accuracy_interval(correct),
        "walk_forward_by_year": yearly,
        "confidence_buckets": buckets,
        "coverage_of_buckets_with_at_least_75pct_accuracy": float(high_accuracy_coverage),
        "paired_accuracy": {
            "ensemble_vs_logistic": paired_accuracy_interval(correct, logistic_correct),
            "ensemble_vs_ufc_win_pct": paired_accuracy_interval(correct, baseline_correct),
        },
        "comparison_to_published_v1_accuracy": {
            "v1_accuracy": 0.6082,
            "point_difference": float(correct.mean() - 0.6082),
            "meaningful_by_v2_accuracy_ci": bool(
                0.6082 < accuracy_interval(correct)[0] or 0.6082 > accuracy_interval(correct)[1]
            ),
            "note": "not paired because the frozen V1 evaluation used a different historical split",
        },
        "top_15_unseen_permutation_features": importances[:15],
        "all_permutation_features": importances,
        "unsafe_or_unavailable_features_rejected": {
            "stance_matchup": "profile stance is not timestamped historically",
            "late_round_performance": "source has bout totals, not reliable per-round history",
            "betting_odds": "no verified point-in-time odds in the immutable source",
            "current_rankings": "would leak future status into historical rows",
        },
    }
    (report_dir / "development_final.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "metrics": report["walk_forward_metrics"],
        "accuracy_ci": report["walk_forward_accuracy_95pct_ci"],
        "high_accuracy_coverage": high_accuracy_coverage,
        "top_features": importances[:15],
    }, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    run(Path(args.project_root).resolve())


if __name__ == "__main__":
    main()
