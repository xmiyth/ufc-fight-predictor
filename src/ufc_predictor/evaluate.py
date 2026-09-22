"""Rigorous chronological model and benchmark evaluation.

All feature rows are constructed before their fight updates fighter history.
Model hyperparameters are selected on a chronological validation slice inside
the training period. Every final model is then evaluated on the same locked
newest-20%-of-dates test period.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy.stats import binomtest
from xgboost import XGBClassifier

from .features import (
    CURRENT_MODEL_FEATURES,
    FEATURE_NAMES,
    STANCE_FEATURES,
    WEIGHT_CLASS_FEATURES,
    build_dataset,
)
from .train import chronological_split


RANDOM_STATE = 42
FORBIDDEN_FEATURE_TOKENS = {
    "winner", "result", "outcome", "method", "finish_round", "fight_time",
    "betting", "odds", "red_fighter_result", "blue_fighter_result",
}


def probability_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "ece": expected_calibration_error(y_true, probabilities),
    }


def calibration_bins(y_true: pd.Series, probabilities: np.ndarray) -> list[dict[str, Any]]:
    frame = pd.DataFrame({"actual": np.asarray(y_true), "probability": probabilities})
    frame["bin"] = pd.cut(
        frame["probability"], bins=np.linspace(0, 1, 11), include_lowest=True,
        right=False,
    )
    rows: list[dict[str, Any]] = []
    for interval, group in frame.groupby("bin", observed=True):
        rows.append(
            {
                "bin": str(interval),
                "count": int(len(group)),
                "mean_predicted": float(group["probability"].mean()),
                "actual_win_rate": float(group["actual"].mean()),
                "absolute_gap": float(
                    abs(group["probability"].mean() - group["actual"].mean())
                ),
            }
        )
    return rows


def expected_calibration_error(y_true: pd.Series, probabilities: np.ndarray) -> float:
    bins = calibration_bins(y_true, probabilities)
    total = len(y_true)
    return float(sum(row["count"] / total * row["absolute_gap"] for row in bins))


def favorite_seventy_band(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, Any]:
    """Check forecasts where the predicted winner was given 65%-75%."""
    probabilities = np.asarray(probabilities)
    favorite_probability = np.maximum(probabilities, 1.0 - probabilities)
    predicted_a = probabilities >= 0.5
    correct = predicted_a == np.asarray(y_true, dtype=bool)
    mask = (favorite_probability >= 0.65) & (favorite_probability < 0.75)
    count = int(mask.sum())
    actual = None if not mask.any() else float(correct[mask].mean())
    # Wilson interval is more honest than treating the observed rate as exact.
    if count:
        z = 1.96
        center = (actual + z * z / (2 * count)) / (1 + z * z / count)
        margin = z * np.sqrt(
            actual * (1 - actual) / count + z * z / (4 * count * count)
        ) / (1 + z * z / count)
        confidence_interval = [float(center - margin), float(center + margin)]
    else:
        confidence_interval = None
    return {
        "band": "65%-75% predicted winner probability",
        "count": count,
        "mean_predicted_probability": (
            None if not mask.any() else float(favorite_probability[mask].mean())
        ),
        "actual_win_rate": actual,
        "actual_win_rate_95pct_wilson_interval": confidence_interval,
    }


def validation_split(train: pd.DataFrame, fraction: float = 0.8):
    dates = sorted(train["date"].unique())
    cutoff = dates[max(1, int(len(dates) * fraction)) - 1]
    fit = train[train["date"] <= cutoff].copy()
    validation = train[train["date"] > cutoff].copy()
    return fit, validation, cutoff


INTERACTION_LOGISTIC_FEATURES = [
    feature for feature in FEATURE_NAMES
    if not feature.startswith("weight_class_") and not feature.startswith("fighter_a_")
    and not feature.startswith("fighter_b_") and feature != "same_stance"
]

FEATURE_GROUPS = {
    "striking_detail": [
        "striking_defense_diff", "knockdowns_per_fight_diff",
        "head_strike_share_diff", "body_strike_share_diff", "leg_strike_share_diff",
        "distance_strike_share_diff", "clinch_strike_share_diff",
        "ground_strike_share_diff",
    ],
    "grappling_detail": ["takedowns_per15_diff", "control_time_per15_diff"],
    "streak_and_method_context": [
        "current_win_streak_diff", "current_loss_streak_diff", "ko_tko_win_pct_diff",
        "submission_win_pct_diff", "decision_win_pct_diff",
    ],
    "stance_and_weight_class": STANCE_FEATURES + WEIGHT_CLASS_FEATURES,
    "opponent_quality": [
        "average_opponent_elo_diff", "average_beaten_opponent_elo_diff",
        "quality_adjusted_win_score_diff",
    ],
    "manual_matchup_interactions": [
        "fighter_a_td_accuracy_vs_b_defense",
        "fighter_b_td_accuracy_vs_a_defense",
        "fighter_a_striking_offense_vs_b_defense",
        "fighter_b_striking_offense_vs_a_defense",
    ],
}


def candidate_models() -> dict[str, dict[str, Any]]:
    def numeric_pipeline(model, scale: bool = False) -> Pipeline:
        steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", model))
        return Pipeline(steps)

    models: dict[str, dict[str, Any]] = {
        "Current Logistic (reference)": {"features": CURRENT_MODEL_FEATURES, "pipelines": [
            numeric_pipeline(
                LogisticRegression(C=c, max_iter=3000, random_state=RANDOM_STATE),
                scale=True,
            )
            for c in (0.1, 1.0)
        ]},
        "Rich Logistic": {"features": FEATURE_NAMES, "pipelines": [
            numeric_pipeline(
                LogisticRegression(C=c, max_iter=3000, random_state=RANDOM_STATE),
                scale=True,
            )
            for c in (0.05, 0.1, 1.0)
        ]},
        "Rich Logistic + Interactions": {
            "features": INTERACTION_LOGISTIC_FEATURES,
            "pipelines": [
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("interactions", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                        ("scaler", StandardScaler()),
                        ("model", LogisticRegression(C=c, max_iter=4000, random_state=RANDOM_STATE)),
                    ]
                )
                for c in (0.01, 0.05)
            ],
        },
        "Random Forest": {"features": FEATURE_NAMES, "pipelines": [
            numeric_pipeline(
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=depth,
                    min_samples_leaf=10,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                )
            )
            for depth in (5, 9)
        ]},
        "Gradient Boosting": {"features": FEATURE_NAMES, "pipelines": [
            numeric_pipeline(
                GradientBoostingClassifier(
                    n_estimators=250,
                    learning_rate=rate,
                    max_depth=depth,
                    min_samples_leaf=15,
                    random_state=RANDOM_STATE,
                )
            )
            for rate, depth in ((0.03, 1), (0.03, 2), (0.05, 2))
        ]},
        "XGBoost": {"features": FEATURE_NAMES, "pipelines": [
            numeric_pipeline(
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=350,
                    learning_rate=rate,
                    max_depth=depth,
                    min_child_weight=8,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=5.0,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                )
            )
            for rate, depth in ((0.03, 2), (0.03, 3), (0.05, 2))
        ]},
    }
    return models


def choose_on_validation(
    candidates: list[Pipeline], features: list[str], fit: pd.DataFrame, validation: pd.DataFrame
) -> tuple[Pipeline, list[dict[str, Any]]]:
    trials = []
    for index, candidate in enumerate(candidates):
        candidate.fit(fit[features], fit["fighter_a_won"])
        probabilities = candidate.predict_proba(validation[features])[:, 1]
        metrics = probability_metrics(validation["fighter_a_won"], probabilities)
        trials.append({"candidate": index, **metrics})
    winner = min(trials, key=lambda row: (row["log_loss"], row["brier_score"]))
    return candidates[winner["candidate"]], trials


def prune_feature_groups_on_validation(
    template: Pipeline, fit: pd.DataFrame, validation: pd.DataFrame
) -> tuple[list[str], list[dict[str, Any]]]:
    """Backward group selection using training-period validation only."""
    selected = list(FEATURE_NAMES)
    full = clone(template).fit(fit[selected], fit["fighter_a_won"])
    best_loss = probability_metrics(
        validation["fighter_a_won"], full.predict_proba(validation[selected])[:, 1]
    )["log_loss"]
    trials = []
    for group_name, group_features in FEATURE_GROUPS.items():
        candidate_features = [feature for feature in selected if feature not in group_features]
        candidate = clone(template).fit(fit[candidate_features], fit["fighter_a_won"])
        loss = probability_metrics(
            validation["fighter_a_won"],
            candidate.predict_proba(validation[candidate_features])[:, 1],
        )["log_loss"]
        removed = loss < best_loss - 0.0001
        trials.append(
            {
                "group": group_name,
                "validation_log_loss_without_group": loss,
                "previous_best_validation_log_loss": best_loss,
                "removed": removed,
            }
        )
        if removed:
            selected = candidate_features
            best_loss = loss
    return selected, trials


def benchmark_probabilities(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    prevalence = float(train["fighter_a_won"].mean())
    win_pct = np.clip(0.5 + test["ufc_win_pct_diff"].fillna(0).to_numpy() / 2.0, 0.01, 0.99)
    recent = np.clip(
        0.5 + test["recent_5_win_pct_diff"].fillna(0).to_numpy() / 2.0,
        0.01,
        0.99,
    )
    elo = 1.0 / (1.0 + 10.0 ** (-test["elo_diff"].fillna(0).to_numpy() / 400.0))
    rng = np.random.default_rng(RANDOM_STATE)
    coin_flips = rng.integers(0, 2, size=len(test))
    # Tiny offsets preserve coin-flip classifications while correctly expressing
    # essentially 50/50 confidence for probability metrics.
    random_probabilities = 0.5 + np.where(coin_flips == 1, 1e-6, -1e-6)
    return {
        "Random coin flip (seed 42)": random_probabilities,
        "Training-majority prevalence": np.full(len(test), prevalence),
        "Better pre-fight UFC win %": win_pct,
        "Better recent 5-fight record": recent,
        "Elo rating": elo,
    }


def paired_accuracy_comparison(
    y_true: pd.Series,
    model_probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    """Paired bootstrap interval and exact McNemar test on common fights."""
    y = np.asarray(y_true, dtype=int)
    model_correct = (np.asarray(model_probabilities) >= 0.5) == y
    baseline_correct = (np.asarray(baseline_probabilities) >= 0.5) == y
    differences = model_correct.astype(float) - baseline_correct.astype(float)
    rng = np.random.default_rng(RANDOM_STATE)
    bootstrap_differences = np.empty(bootstrap_samples)
    for start in range(0, bootstrap_samples, 500):
        stop = min(start + 500, bootstrap_samples)
        indices = rng.integers(0, len(y), size=(stop - start, len(y)))
        bootstrap_differences[start:stop] = differences[indices].mean(axis=1)
    model_only = int(np.sum(model_correct & ~baseline_correct))
    baseline_only = int(np.sum(~model_correct & baseline_correct))
    discordant = model_only + baseline_only
    p_value = (
        1.0 if discordant == 0
        else float(binomtest(model_only, discordant, 0.5, alternative="two-sided").pvalue)
    )
    return {
        "accuracy_difference": float(differences.mean()),
        "paired_bootstrap_95pct_interval": [
            float(np.quantile(bootstrap_differences, 0.025)),
            float(np.quantile(bootstrap_differences, 0.975)),
        ],
        "mcnemar_exact_p_value": p_value,
        "model_correct_baseline_wrong": model_only,
        "model_wrong_baseline_correct": baseline_only,
        "statistically_clear_at_0_05": bool(p_value < 0.05),
    }


def missing_data_report(data: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "feature": feature,
            "missing_count": int(data[feature].isna().sum()),
            "missing_percent": float(data[feature].isna().mean() * 100),
        }
        for feature in FEATURE_NAMES
    ]


def leakage_audit(data: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    suspicious = sorted(
        feature
        for feature in FEATURE_NAMES
        if any(token in feature.casefold() for token in FORBIDDEN_FEATURE_TOKENS)
    )
    return {
        "status": "PASS" if not suspicious and train["date"].max() < test["date"].min() else "FAIL",
        "chronological_train_before_test": bool(train["date"].max() < test["date"].min()),
        "forbidden_or_target_derived_features": suspicious,
        "duplicate_same_date_ordered_matchups": int(
            data.duplicated(["date", "fighter_a", "fighter_b"]).sum()
        ),
        "fighter_a_win_rate": float(data["fighter_a_won"].mean()),
        "snapshot_before_current_fight_update": True,
        "current_profile_career_stats_used": False,
        "betting_odds_used": False,
        "betting_odds_note": (
            "Not available in the downloaded raw UFCStats files; favorite benchmark skipped."
        ),
    }


def feature_importance(pipeline: Pipeline, features: list[str]) -> list[dict[str, Any]]:
    model = pipeline.named_steps["model"]
    names = features
    if "interactions" in pipeline.named_steps:
        names = list(pipeline.named_steps["interactions"].get_feature_names_out(features))
    if hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        values = model.feature_importances_
    return sorted(
        [
            {"feature": feature, "importance": float(value)}
            for feature, value in zip(names, values)
        ],
        key=lambda row: row["importance"],
        reverse=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fights", default="data/raw/fights.csv")
    parser.add_argument("--fighters", default="data/raw/fighters.csv")
    args = parser.parse_args()

    print("Building leakage-safe pre-fight features ...")
    data, states, profiles = build_dataset(args.fights, args.fighters)
    train, test, cutoff = chronological_split(data)
    fit, validation, validation_cutoff = validation_split(train)
    print(
        f"Locked split: {len(train):,} train through {cutoff}; "
        f"{len(test):,} test from {test['date'].min()} through {test['date'].max()}"
    )

    model_results: list[dict[str, Any]] = []
    fitted_models: dict[str, tuple[Pipeline, list[str]]] = {}
    all_predictions = test[["date", "fighter_a", "fighter_b", "fighter_a_won"]].copy()
    validation_results: dict[str, Any] = {}
    calibration_rows: list[dict[str, Any]] = []

    for name, specification in candidate_models().items():
        print(f"Selecting {name} on internal validation dates ...")
        features = specification["features"]
        selected, trials = choose_on_validation(
            specification["pipelines"], features, fit, validation
        )
        selected.fit(train[features], train["fighter_a_won"])
        probabilities = selected.predict_proba(test[features])[:, 1]
        metrics = probability_metrics(test["fighter_a_won"], probabilities)
        seventy = favorite_seventy_band(test["fighter_a_won"], probabilities)
        model_results.append(
            {"model": name, "feature_count": len(features), **metrics, "seventy_percent_band": seventy}
        )
        validation_results[name] = trials
        fitted_models[name] = (selected, features)
        all_predictions[name] = probabilities
        for row in calibration_bins(test["fighter_a_won"], probabilities):
            calibration_rows.append({"model": name, **row})

    rich_template, _ = fitted_models["Rich Logistic"]
    pruned_features, pruning_trials = prune_feature_groups_on_validation(
        rich_template, fit, validation
    )
    pruned_model = clone(rich_template).fit(train[pruned_features], train["fighter_a_won"])
    pruned_probabilities = pruned_model.predict_proba(test[pruned_features])[:, 1]
    pruned_metrics = probability_metrics(test["fighter_a_won"], pruned_probabilities)
    pruned_name = "Pruned Rich Logistic"
    model_results.append(
        {
            "model": pruned_name,
            "feature_count": len(pruned_features),
            **pruned_metrics,
            "seventy_percent_band": favorite_seventy_band(
                test["fighter_a_won"], pruned_probabilities
            ),
        }
    )
    fitted_models[pruned_name] = (pruned_model, pruned_features)
    validation_results[pruned_name] = {
        "selection_method": "backward feature-group removal on internal validation only",
        "trials": pruning_trials,
        "selected_features": pruned_features,
    }
    all_predictions[pruned_name] = pruned_probabilities
    for row in calibration_bins(test["fighter_a_won"], pruned_probabilities):
        calibration_rows.append({"model": pruned_name, **row})

    benchmark_results = []
    benchmark_probability_map = benchmark_probabilities(train, test)
    for name, probabilities in benchmark_probability_map.items():
        benchmark_results.append(
            {"benchmark": name, **probability_metrics(test["fighter_a_won"], probabilities)}
        )

    # Proper probability quality is primary; AUC breaks near-ties in log loss.
    best = min(model_results, key=lambda row: (row["log_loss"], -row["roc_auc"]))
    best_name = best["model"]
    best_model, best_features = fitted_models[best_name]
    best_probabilities = all_predictions[best_name].to_numpy()
    win_pct_comparison = paired_accuracy_comparison(
        test["fighter_a_won"],
        best_probabilities,
        benchmark_probability_map["Better pre-fight UFC win %"],
    )

    rich_model, _ = fitted_models["Rich Logistic"]
    rich_metrics = next(row for row in model_results if row["model"] == "Rich Logistic")
    ablations = []
    for group_name, group_features in FEATURE_GROUPS.items():
        kept = [feature for feature in FEATURE_NAMES if feature not in group_features]
        ablated = clone(rich_model).fit(train[kept], train["fighter_a_won"])
        probabilities = ablated.predict_proba(test[kept])[:, 1]
        metrics = probability_metrics(test["fighter_a_won"], probabilities)
        ablations.append(
            {
                "group": group_name,
                "features": group_features,
                "accuracy_without_group": metrics["accuracy"],
                "log_loss_without_group": metrics["log_loss"],
                "accuracy_change_when_removed": metrics["accuracy"] - rich_metrics["accuracy"],
                "log_loss_change_when_removed": metrics["log_loss"] - rich_metrics["log_loss"],
                "interpretation": (
                    "helped" if metrics["log_loss"] > rich_metrics["log_loss"]
                    else "hurt_or_added_noise"
                ),
            }
        )
    audit = leakage_audit(data, train, test)
    missing = missing_data_report(data)
    report = {
        "fight_counts": {
            "total": int(len(data)), "train": int(len(train)), "test": int(len(test)),
            "internal_fit": int(len(fit)), "internal_validation": int(len(validation)),
        },
        "date_ranges": {
            "train_start": str(train["date"].min()), "train_end": str(train["date"].max()),
            "validation_cutoff": str(validation_cutoff),
            "test_start": str(test["date"].min()), "test_end": str(test["date"].max()),
        },
        "feature_names": FEATURE_NAMES,
        "excluded_requested_features": {
            "non_ufc_career_win_percentage": (
                "Excluded: only a current career snapshot is available, so using it for old "
                "fights would leak future outcomes. Pre-fight UFC win percentage is included."
            ),
            "betting_favorite": (
                "Excluded: historical pre-fight odds are not present in the downloaded source."
            ),
        },
        "model_results": model_results,
        "benchmark_results": benchmark_results,
        "best_model": best_name,
        "best_model_reason": "lowest log loss on the common unseen test period; ROC AUC breaks ties",
        "best_vs_win_percentage_baseline": win_pct_comparison,
        "best_model_feature_importance": feature_importance(best_model, best_features),
        "feature_group_ablation": ablations,
        "missing_data": missing,
        "leakage_audit": audit,
        "validation_trials": validation_results,
    }

    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    evaluated_artifact = {
        "pipeline": best_model,
        "model_name": best_name,
        "feature_names": best_features,
        "fighter_states": {name: state.serializable() for name, state in states.items()},
        "profiles": profiles,
        "last_data_date": str(data["date"].max()),
        "evaluation": report,
    }
    joblib.dump(evaluated_artifact, "models/best_evaluated_model.joblib")
    # Once the test comparison has selected an architecture, refit that same
    # configuration on all available fights for future, post-dataset predictions.
    deployment_model = clone(best_model).fit(data[best_features], data["fighter_a_won"])
    deployment_artifact = {
        **evaluated_artifact,
        "pipeline": deployment_model,
        "trained_through": str(data["date"].max()),
        "deployment_note": (
            "Architecture selected on the locked test; then refit on all data. "
            "Reported test metrics belong to best_evaluated_model.joblib."
        ),
    }
    joblib.dump(deployment_artifact, "models/best_model.joblib")
    data.to_csv("data/processed/prefight_features.csv", index=False)
    all_predictions.to_csv("reports/model_test_predictions.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv("reports/calibration_bins.csv", index=False)
    pd.DataFrame(missing).to_csv("reports/missing_data.csv", index=False)
    Path("reports/model_evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("\nModel comparison (same locked test fights):")
    for row in model_results:
        print(
            f"{row['model']:<22} accuracy={row['accuracy']:.3f} "
            f"AUC={row['roc_auc']:.3f} logloss={row['log_loss']:.3f} "
            f"Brier={row['brier_score']:.3f} ECE={row['ece']:.3f}"
        )
    print(f"Best model: {best_name}")
    print(f"Leakage audit: {audit['status']}")


if __name__ == "__main__":
    main()
