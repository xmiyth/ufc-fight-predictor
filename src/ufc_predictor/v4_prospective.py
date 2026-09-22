"""Prospective low-history experiments layered over the frozen V2 dataset.

This module deliberately never scores the already-inspected 2025-2026 block.
Candidate selection uses 2019-2022 walk-forward predictions and the selected
candidate is checked once on 2023-2024.  V1/V2/V3 artifacts are read-only.
"""

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

from .v2_evaluate import Experiment, feature_names, make_model, metrics
from .v3_evaluate import decisive, load_or_build


RANDOM_STATE = 20260911
DEVELOPMENT_YEARS = (2019, 2020, 2021, 2022)
CONFIRMATION_YEARS = (2023, 2024)
ALL_YEARS = DEVELOPMENT_YEARS + CONFIRMATION_YEARS

# These fields have both a matchup difference and a matchup mean in the cached
# point-in-time dataset, so the two pre-fight fighter values can be recovered.
SHRINKABLE_STATS = (
    "finish_rate", "ko_tko_rate", "submission_rate", "decision_rate",
    "average_fight_duration", "recent_3_win_pct", "quality_adjusted_win_rate",
    "sig_landed_pm", "sig_absorbed_pm", "sig_accuracy", "striking_defense",
    "knockdowns_per15", "takedowns_per15", "takedown_accuracy",
    "takedown_defense", "submissions_per15", "control_pct",
    "head_strike_share", "body_strike_share", "leg_strike_share",
    "distance_strike_share", "clinch_strike_share", "ground_strike_share",
)
PERFORMANCE_TOKENS = (
    "finish_rate", "ko_tko_rate", "submission_rate", "decision_rate",
    "average_fight_duration", "sig_", "striking_", "strike_volume",
    "knockdown", "takedown", "submissions", "reversals", "control_pct",
    "head_strike", "body_strike", "leg_strike", "distance_strike",
    "clinch_strike", "ground_strike", "opponent_adjusted",
)


@dataclass(frozen=True)
class V4Candidate:
    name: str
    prior_fights: float


def _history(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Recover oriented per-side UFC fight counts from mean and difference."""
    a = frame["ufc_fights_mean"] + frame["ufc_fights_diff"] / 2.0
    b = frame["ufc_fights_mean"] - frame["ufc_fights_diff"] / 2.0
    return a.clip(lower=0).round(6), b.clip(lower=0).round(6)


def fit_division_priors(frame: pd.DataFrame) -> dict[str, Any]:
    """Estimate performance priors using training rows only.

    A fighter snapshot with no prior UFC fights is excluded rather than treated
    as a zero observation.  Medians make this robust to duplicated snapshots
    across a fighter's subsequent bouts.
    """
    a_count, b_count = _history(frame)
    result: dict[str, Any] = {"global": {}, "division": {}}
    for stat in SHRINKABLE_STATS:
        mean_name, diff_name = f"{stat}_mean", f"{stat}_diff"
        if mean_name not in frame or diff_name not in frame:
            continue
        a_value = frame[mean_name] + frame[diff_name] / 2.0
        b_value = frame[mean_name] - frame[diff_name] / 2.0
        values = pd.concat([
            pd.DataFrame({"division": frame.weight_class, "value": a_value, "count": a_count}),
            pd.DataFrame({"division": frame.weight_class, "value": b_value, "count": b_count}),
        ], ignore_index=True)
        observed = values.loc[(values["count"] > 0) & values["value"].notna()]
        global_prior = float(observed["value"].median()) if not observed.empty else 0.0
        result["global"][stat] = global_prior
        result["division"][stat] = {
            str(key): float(value)
            for key, value in observed.groupby("division")["value"].median().items()
        }
    return result


def augment_low_history(
    frame: pd.DataFrame,
    priors: dict[str, Any],
    *,
    prior_fights: float,
) -> pd.DataFrame:
    """Add low-history features and remove zero-as-missing UFC performance."""
    result = frame.copy()
    a_count, b_count = _history(result)
    a_debut, b_debut = a_count.eq(0), b_count.eq(0)
    a_low, b_low = a_count.between(1, 2), b_count.between(1, 2)
    a_normal, b_normal = a_count.ge(3), b_count.ge(3)
    result["v4_debutant_diff"] = a_debut.astype(float) - b_debut.astype(float)
    result["v4_low_sample_diff"] = a_low.astype(float) - b_low.astype(float)
    result["v4_normal_diff"] = a_normal.astype(float) - b_normal.astype(float)
    result["v4_either_debutant"] = (a_debut | b_debut).astype(float)
    result["v4_both_debutants"] = (a_debut & b_debut).astype(float)
    result["v4_either_low_sample"] = (a_low | b_low).astype(float)
    result["v4_both_normal"] = (a_normal & b_normal).astype(float)
    result["v4_min_ufc_fights"] = np.minimum(a_count, b_count)
    result["v4_history_reliability_diff"] = (
        a_count / (a_count + prior_fights) - b_count / (b_count + prior_fights)
    )
    result["v4_history_reliability_mean"] = (
        a_count / (a_count + prior_fights) + b_count / (b_count + prior_fights)
    ) / 2.0

    # Raw performance difference fields are invalid whenever either side has no
    # UFC observations. Leave them missing for fold-fitted median imputation.
    no_complete_matchup = a_debut | b_debut
    for name in feature_names("rich_548"):
        if name.endswith("_diff") and any(token in name for token in PERFORMANCE_TOKENS):
            result.loc[no_complete_matchup, name] = np.nan

    # Empirical-Bayes shrinkage: each observed fighter is pulled toward a
    # division prior, while a debutant lands exactly on that prior.
    for stat in SHRINKABLE_STATS:
        mean_name, diff_name = f"{stat}_mean", f"{stat}_diff"
        if mean_name not in result or diff_name not in result or stat not in priors["global"]:
            continue
        a_raw = result[mean_name] + frame[diff_name] / 2.0
        b_raw = result[mean_name] - frame[diff_name] / 2.0
        division_priors = result.weight_class.map(priors["division"][stat]).fillna(priors["global"][stat])
        a_observed = a_raw.mask(a_debut)
        b_observed = b_raw.mask(b_debut)
        a_shrunk = (a_count * a_observed.fillna(division_priors) + prior_fights * division_priors) / (a_count + prior_fights)
        b_shrunk = (b_count * b_observed.fillna(division_priors) + prior_fights * division_priors) / (b_count + prior_fights)
        result[f"v4_{stat}_shrunk_diff"] = a_shrunk - b_shrunk
    return result


def v4_feature_names(frame: pd.DataFrame) -> list[str]:
    return feature_names("rich_548") + sorted(name for name in frame if name.startswith("v4_"))


def _ensemble(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, list[Any]]:
    specs = (
        (0.25, Experiment("v4_logistic", "logistic", "rich_548", {"C": 1.0})),
        (0.75, Experiment("v4_xgboost", "xgboost", "rich_548", {
            "n_estimators": 350, "learning_rate": 0.025, "max_depth": 2,
            "min_child_weight": 8, "subsample": 0.8, "colsample_bytree": 0.8,
        })),
    )
    probability = np.zeros(len(test), dtype=float)
    models = []
    for weight, spec in specs:
        model = make_model(spec)
        model.fit(train[features], train.fighter_a_won)
        probability += weight * model.predict_proba(test[features])[:, 1]
        models.append(model)
    return probability, models


def walk_forward(frame: pd.DataFrame, candidate: V4Candidate) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    predictions, folds = [], []
    for year in ALL_YEARS:
        train = frame.loc[frame.fight_date.dt.year < year]
        test = frame.loc[frame.fight_date.dt.year == year]
        if test.empty:
            continue
        priors = fit_division_priors(train)
        train_v4 = augment_low_history(train, priors, prior_fights=candidate.prior_fights)
        test_v4 = augment_low_history(test, priors, prior_fights=candidate.prior_fights)
        features = v4_feature_names(train_v4)
        probability, _ = _ensemble(train_v4, test_v4, features)
        values = metrics(test.fighter_a_won.to_numpy(), probability)
        folds.append({"candidate": candidate.name, "year": year, "train_fights": len(train), "test_fights": len(test), **values})
        part = test[["fight_id", "fight_date", "fighter_a_won", "weight_class"]].copy()
        part["probability"] = probability
        predictions.append(part)
    return pd.concat(predictions, ignore_index=True), folds


def _period_metrics(predictions: pd.DataFrame, years: tuple[int, ...]) -> dict[str, float]:
    part = predictions.loc[predictions.fight_date.dt.year.isin(years)]
    return metrics(part.fighter_a_won.to_numpy(), part.probability.to_numpy())


def _baseline_predictions(root: Path) -> pd.DataFrame:
    logistic = pd.read_csv(root / "reports/v2_accuracy/oof_logistic_rich_548_c1.csv", parse_dates=["fight_date"])
    xgboost = pd.read_csv(root / "reports/v2_accuracy/oof_xgboost_rich548.csv", parse_dates=["fight_date"])
    keys = ["fight_id", "fight_date", "fighter_a_won"]
    merged = logistic.merge(xgboost[keys + ["probability"]], on=keys, suffixes=("_logistic", "_xgboost"), validate="one_to_one")
    merged["probability"] = 0.25 * merged.probability_logistic + 0.75 * merged.probability_xgboost
    return merged


def run(root: Path) -> dict[str, Any]:
    output = root / "reports/v4_prospective"
    output.mkdir(parents=True, exist_ok=True)
    frame = decisive(load_or_build(root))
    frame = frame.loc[frame.fight_date.dt.year <= 2024].copy()
    candidates = [V4Candidate(f"low_history_bayes_k{k:g}", k) for k in (2.0, 4.0, 8.0)]
    rows, all_folds, prediction_map = [], [], {}
    for candidate in candidates:
        predictions, folds = walk_forward(frame, candidate)
        prediction_map[candidate.name] = predictions
        development = _period_metrics(predictions, DEVELOPMENT_YEARS)
        confirmation = _period_metrics(predictions, CONFIRMATION_YEARS)
        rows.append({**asdict(candidate), **{f"development_{k}": v for k, v in development.items()}, **{f"confirmation_{k}": v for k, v in confirmation.items()}, "fold_accuracy_std": float(np.std([f["accuracy"] for f in folds if f["year"] in DEVELOPMENT_YEARS]))})
        all_folds.extend(folds)

    baseline = _baseline_predictions(root)
    baseline = baseline.loc[baseline.fight_date.dt.year.isin(ALL_YEARS)]
    baseline_development = _period_metrics(baseline, DEVELOPMENT_YEARS)
    baseline_confirmation = _period_metrics(baseline, CONFIRMATION_YEARS)
    table = pd.DataFrame(rows).sort_values(["development_accuracy", "development_log_loss"], ascending=[False, True])
    selected = table.iloc[0].to_dict()
    selected_confirmation = {key.removeprefix("confirmation_"): value for key, value in selected.items() if key.startswith("confirmation_")}
    accepted = (
        selected["development_accuracy"] > baseline_development["accuracy"]
        and selected["development_log_loss"] <= baseline_development["log_loss"]
        and selected["development_brier"] <= baseline_development["brier"]
        and selected["confirmation_accuracy"] > baseline_confirmation["accuracy"]
        and selected["confirmation_log_loss"] <= baseline_confirmation["log_loss"]
        and selected["confirmation_brier"] <= baseline_confirmation["brier"]
    )
    table.to_csv(output / "model_comparison.csv", index=False)
    pd.DataFrame(all_folds).to_csv(output / "walk_forward_by_year.csv", index=False)
    prediction_map[str(selected["name"])].to_csv(output / "selected_oof_predictions.csv", index=False)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "prospective development; inspected 2025-2026 outcomes were not scored",
        "selection_period": ["2019-01-01", "2022-12-31"],
        "confirmation_period": ["2023-01-01", "2024-12-31"],
        "future_final_holdout": "must begin after 2026-09-11",
        "baseline": {"development": baseline_development, "confirmation": baseline_confirmation},
        "selected_by_development_only": selected,
        "selected_confirmation": selected_confirmation,
        "promotion_rule_passed": bool(accepted),
        "promotion_decision": "eligible for prospective freeze" if accepted else "rejected; keep V2.1 as verified champion",
        "method": "training-fold division medians, empirical-Bayes shrinkage, explicit low-history modes, median imputation with indicators",
        "locked_artifacts_modified": False,
    }
    (output / "development_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # A research artifact is safe to retain even when rejected; it is explicitly
    # not the production champion and contains no 2025-2026 label feedback.
    selected_candidate = next(c for c in candidates if c.name == selected["name"])
    priors = fit_division_priors(frame)
    frame_v4 = augment_low_history(frame, priors, prior_fights=selected_candidate.prior_fights)
    features = v4_feature_names(frame_v4)
    fitted_models = []
    for _, spec in (
        (0.25, Experiment("v4_logistic", "logistic", "rich_548", {"C": 1.0})),
        (0.75, Experiment("v4_xgboost", "xgboost", "rich_548", {"n_estimators": 350, "learning_rate": 0.025, "max_depth": 2, "min_child_weight": 8, "subsample": 0.8, "colsample_bytree": 0.8})),
    ):
        model = make_model(spec)
        model.fit(frame_v4[features], frame_v4.fighter_a_won)
        fitted_models.append(model)
    artifact_dir = root / "models/v4-prospective"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"status": "research_not_production", "candidate": asdict(selected_candidate), "features": features, "priors": priors, "weights": [0.25, 0.75], "models": fitted_models, "trained_through": "2024-12-31", "never_scored_years": [2025, 2026]}, artifact_dir / "development_candidate.joblib", compress=3)
    (artifact_dir / "README.md").write_text("# V4 prospective research candidate\n\nThis artifact is not the production champion. It was selected on 2019-2022 and confirmed once on 2023-2024. The already-inspected 2025-2026 labels were deliberately not scored. See `reports/v4_prospective/development_report.json`.\n", encoding="utf-8")
    print(table.to_string(index=False), flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    run(Path(args.project_root).resolve())


if __name__ == "__main__":
    main()
