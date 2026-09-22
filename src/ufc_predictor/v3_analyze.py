"""Finalize V3 research analyses without opening a future lockbox.

Model/ensemble/calibration choices use 2019-2024 walk-forward predictions only.
The previously inspected 2025-2026 period is reported as confirmation, never as
an untouched V3 lockbox.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import binomtest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .v2_evaluate import expected_calibration_error, metrics
from .v3_evaluate import Candidate, decisive, load_or_build, make_model
from .v3_features import V3_FEATURE_GROUPS, V3_FEATURE_NAMES


DEV_YEARS = tuple(range(2019, 2025))
CONFIRM_YEARS = (2025, 2026)
COMPONENTS = ("v2_logistic_reference", "v3_xgboost", "v3_catboost", "v3_lightgbm")
FINAL_CANDIDATE = Candidate(
    "v3_xgboost", "xgboost", "v3",
    {"n_estimators": 400, "learning_rate": 0.02, "max_depth": 2,
     "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.45,
     "reg_alpha": 0.5, "reg_lambda": 4.0},
)


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _aligned_predictions(report_dir: Path) -> pd.DataFrame:
    result = None
    keys = ["fight_id", "fight_date", "fighter_a_won", "weight_class"]
    for name in COMPONENTS:
        part = pd.read_csv(report_dir / f"oof_{name}.csv", parse_dates=["fight_date"])
        part["occurrence"] = part.groupby(["fight_id", "fight_date"], sort=False).cumcount()
        if result is None:
            result = part[keys + ["occurrence"]].copy()
        aligned = part[keys + ["occurrence", "probability"]].rename(columns={"probability": name})
        result = result.merge(aligned, on=keys + ["occurrence"], how="inner", validate="one_to_one")
    assert result is not None
    return result


def _fit_weights(y: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    def objective(weights: np.ndarray) -> float:
        p = np.clip(matrix @ weights, 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    start = np.repeat(1.0 / matrix.shape[1], matrix.shape[1])
    fit = minimize(objective, start, method="SLSQP", bounds=[(0, 1)] * len(start),
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0})
    if not fit.success:
        raise RuntimeError(f"Ensemble optimization failed: {fit.message}")
    weights = np.maximum(fit.x, 0)
    return weights / weights.sum()


def _platt_fit(p: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    model = LogisticRegression(C=1e6, solver="lbfgs")
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    model.fit(logit.reshape(-1, 1), y)
    return lambda values: model.predict_proba(
        np.log(np.clip(values, 1e-6, 1 - 1e-6) / np.clip(1 - values, 1e-6, 1)).reshape(-1, 1)
    )[:, 1]


def _isotonic_fit(p: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    model = IsotonicRegression(out_of_bounds="clip").fit(p, y)
    return lambda values: model.predict(values)


def _calibration_choice(predictions: pd.DataFrame, base_probability: np.ndarray) -> tuple[str, Callable]:
    early = predictions.fight_date.dt.year <= 2022
    late = predictions.fight_date.dt.year.isin((2023, 2024))
    y_train = predictions.loc[early, "fighter_a_won"].to_numpy()
    y_test = predictions.loc[late, "fighter_a_won"].to_numpy()
    candidates: dict[str, Callable] = {"uncalibrated": lambda p: p}
    candidates["platt"] = _platt_fit(base_probability[early], y_train)
    candidates["isotonic"] = _isotonic_fit(base_probability[early], y_train)
    scores = {}
    for name, transform in candidates.items():
        value = metrics(y_test, np.clip(transform(base_probability[late]), 1e-6, 1 - 1e-6))
        scores[name] = value
    base = scores["uncalibrated"]
    eligible = [name for name, value in scores.items()
                if value["log_loss"] < base["log_loss"] and value["brier"] < base["brier"]]
    choice = min(eligible, key=lambda name: scores[name]["log_loss"]) if eligible else "uncalibrated"
    all_dev = predictions.fight_date.dt.year.isin(DEV_YEARS)
    if choice == "platt":
        final = _platt_fit(base_probability[all_dev], predictions.loc[all_dev, "fighter_a_won"].to_numpy())
    elif choice == "isotonic":
        final = _isotonic_fit(base_probability[all_dev], predictions.loc[all_dev, "fighter_a_won"].to_numpy())
    else:
        final = lambda p: p
    return choice, final, scores


def _confidence_rows(y: np.ndarray, p: np.ndarray) -> list[dict]:
    confidence = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(int)
    edges = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 1.001)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence < high)
        rows.append({"bucket": f"{low:.0%}-{min(high, 1):.0%}", "fights": int(mask.sum()),
                     "coverage": float(mask.mean()),
                     "mean_confidence": float(confidence[mask].mean()) if mask.any() else None,
                     "accuracy": float(correct[mask].mean()) if mask.any() else None})
    return rows


def _thresholds(y: np.ndarray, p: np.ndarray) -> list[dict]:
    confidence = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(int)
    rows = []
    for target in (0.70, 0.75, 0.80):
        options = []
        for threshold in np.arange(0.50, 0.901, 0.005):
            mask = confidence >= threshold
            if mask.sum() >= 30 and correct[mask].mean() >= target:
                options.append((threshold, mask))
        if options:
            threshold, mask = min(options, key=lambda item: item[0])
            rows.append({"target": target, "threshold": float(threshold), "fights": int(mask.sum()),
                         "coverage": float(mask.mean()), "accuracy": float(correct[mask].mean())})
        else:
            rows.append({"target": target, "threshold": None, "fights": 0, "coverage": 0.0, "accuracy": None})
    return rows


def _apply_thresholds(rows: list[dict], y: np.ndarray, p: np.ndarray) -> list[dict]:
    confidence = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(int)
    output = []
    for row in rows:
        threshold = row["threshold"]
        mask = confidence >= threshold if threshold is not None else np.zeros(len(y), dtype=bool)
        output.append({"target": row["target"], "threshold": threshold, "fights": int(mask.sum()),
                       "coverage": float(mask.mean()), "accuracy": float(correct[mask].mean()) if mask.any() else None})
    return output


def _wilson_interval(correct: int, total: int) -> list[float]:
    interval = binomtest(correct, total).proportion_ci(confidence_level=0.95, method="wilson")
    return [float(interval.low), float(interval.high)]


def _mcnemar(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> dict:
    a = (p_a >= 0.5).astype(int) == y
    b = (p_b >= 0.5).astype(int) == y
    a_only, b_only = int((a & ~b).sum()), int((b & ~a).sum())
    test = binomtest(min(a_only, b_only), a_only + b_only, 0.5) if a_only + b_only else None
    return {"candidate_only_correct": a_only, "reference_only_correct": b_only,
            "exact_p_value": float(test.pvalue) if test else 1.0}


def _benchmarks(frame: pd.DataFrame, years: tuple[int, ...]) -> list[dict]:
    subset = frame.loc[frame.fight_date.dt.year.isin(years)]
    y = subset.fighter_a_won.to_numpy()
    values = {
        "majority": np.repeat(int(y.mean() >= 0.5), len(y)),
        "better_pre_fight_ufc_win_percentage": (subset.ufc_win_pct_diff.fillna(0).to_numpy() >= 0).astype(int),
        "better_recent_five_fight_record": (subset.recent_5_win_pct_diff.fillna(0).to_numpy() >= 0).astype(int),
        "standard_elo": (subset.elo_diff.fillna(0).to_numpy() >= 0).astype(int),
        "division_bradley_terry": (subset.v3_division_strength_diff.fillna(0).to_numpy() >= 0).astype(int),
    }
    return [{"benchmark": name, "fights": len(y), "accuracy": float((prediction == y).mean())}
            for name, prediction in values.items()]


def _subgroups(frame: pd.DataFrame, p: np.ndarray) -> dict[str, list[dict]]:
    data = frame.copy()
    data["probability"] = p
    data["correct"] = ((p >= 0.5).astype(int) == data.fighter_a_won.to_numpy()).astype(int)
    minimum_history = data.ufc_fights_mean - data.ufc_fights_abs_diff / 2
    data["experience_group"] = pd.cut(minimum_history, [-1, 0, 2, 5, np.inf], labels=["debutant", "1-2", "3-5", "6+"])
    data["sex_group"] = np.where(data.weight_class.str.startswith("womens_"), "women", "men")
    data["round_group"] = np.where(data.five_round_fight == 1, "five_round", "three_round")
    data["division_change_group"] = np.where(data.v3_any_division_change == 1, "division_change", "no_change")
    data["age_group"] = np.where(data.age_mean >= 35, "average_age_35_plus", "average_age_under_35")
    data["method_group"] = data.method_label.fillna("unknown")
    data["submission_activity_group"] = np.where(
        data.submissions_per15_mean >= data.submissions_per15_mean.quantile(.75),
        "top_quartile_submission_activity", "lower_submission_activity",
    )
    data["confidence_group"] = np.where(np.maximum(p, 1-p) >= .75, "75_plus", "under_75")
    result = {}
    for key in ("weight_class", "sex_group", "round_group", "experience_group", "division_change_group",
                "age_group", "method_group", "submission_activity_group", "confidence_group"):
        rows = []
        for value, group in data.groupby(key, observed=True):
            rows.append({"group": str(value), "fights": len(group), "accuracy": float(group.correct.mean()),
                         "mean_confidence": float(np.maximum(group.probability, 1-group.probability).mean())})
        result[key] = rows
    return result


def _ablation(frame: pd.DataFrame, report_dir: Path) -> list[dict]:
    output = report_dir / "ablation_xgboost.csv"
    if output.exists():
        return pd.read_csv(output).to_dict("records")
    baseline = pd.read_csv(report_dir / "oof_v3_xgboost.csv", parse_dates=["fight_date"])
    baseline = baseline.loc[baseline.fight_date.dt.year.isin(DEV_YEARS)]
    baseline_metrics = metrics(baseline.fighter_a_won.to_numpy(), baseline.probability.to_numpy())
    rows = []
    for index, (group_name, group_features) in enumerate(V3_FEATURE_GROUPS.items(), 1):
        features = [name for name in V3_FEATURE_NAMES if name not in set(group_features)]
        predictions = []
        for year in DEV_YEARS:
            train = frame.loc[frame.fight_date.dt.year < year]
            test = frame.loc[frame.fight_date.dt.year == year]
            model = make_model(FINAL_CANDIDATE)
            model.fit(train[features], train.fighter_a_won)
            predictions.append(pd.DataFrame({"y": test.fighter_a_won, "p": model.predict_proba(test[features])[:, 1]}))
        joined = pd.concat(predictions, ignore_index=True)
        value = metrics(joined.y.to_numpy(), joined.p.to_numpy())
        rows.append({"removed_group": group_name, "removed_features": len(group_features), **value,
                     "accuracy_change": value["accuracy"] - baseline_metrics["accuracy"],
                     "log_loss_change": value["log_loss"] - baseline_metrics["log_loss"]})
        print(f"[ablation {index}/{len(V3_FEATURE_GROUPS)}] {group_name}", flush=True)
        pd.DataFrame(rows).to_csv(output, index=False)
    return rows


def _top_features(frame: pd.DataFrame, root: Path) -> list[dict]:
    train = frame.loc[frame.fight_date.dt.year <= 2024]
    model = make_model(FINAL_CANDIDATE)
    model.fit(train[V3_FEATURE_NAMES], train.fighter_a_won)
    importance = model.named_steps["model"].feature_importances_[:len(V3_FEATURE_NAMES)]
    order = np.argsort(importance)[::-1][:20]
    rows = [{"rank": rank, "feature": V3_FEATURE_NAMES[index], "importance": float(importance[index])}
            for rank, index in enumerate(order, 1)]
    artifact_dir = root / "models" / "v3-research"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"status": "research candidate; not website model", "model": model,
                 "feature_names": V3_FEATURE_NAMES, "trained_through": "2024-12-31"},
                artifact_dir / "development_candidate.joblib", compress=3)
    return rows


def run(root: Path) -> dict:
    report_dir = root / "reports" / "v3_research"
    frame = decisive(load_or_build(root))
    aligned = _aligned_predictions(report_dir)
    dev = aligned.fight_date.dt.year.isin(DEV_YEARS)
    confirm = aligned.fight_date.dt.year.isin(CONFIRM_YEARS)
    weights = _fit_weights(aligned.loc[dev, "fighter_a_won"].to_numpy(), aligned.loc[dev, COMPONENTS].to_numpy())
    ensemble = aligned.loc[:, COMPONENTS].to_numpy() @ weights
    calibration, transform, calibration_validation = _calibration_choice(aligned, ensemble)
    final_probability = np.clip(transform(ensemble), 1e-6, 1 - 1e-6)
    dev_values = metrics(aligned.loc[dev, "fighter_a_won"].to_numpy(), final_probability[dev])
    confirm_y = aligned.loc[confirm, "fighter_a_won"].to_numpy()
    confirm_p = final_probability[confirm]
    confirm_values = metrics(confirm_y, confirm_p)
    confirmation_frame = frame.loc[frame.fight_date.dt.year.isin(CONFIRM_YEARS)].copy().reset_index(drop=True)
    if len(confirmation_frame) != confirm.sum():
        raise RuntimeError("Confirmation alignment failed")
    thresholds = _thresholds(aligned.loc[dev, "fighter_a_won"].to_numpy(), final_probability[dev])
    ablation = _ablation(frame, report_dir)
    top_features = _top_features(frame, root)
    reference_p = aligned.loc[confirm, "v2_logistic_reference"].to_numpy()
    correct = int(((confirm_p >= .5).astype(int) == confirm_y).sum())
    ensemble_years = []
    for year in range(2019, 2027):
        mask = aligned.fight_date.dt.year == year
        ensemble_years.append({"year": year, "fights": int(mask.sum()),
                               **metrics(aligned.loc[mask, "fighter_a_won"].to_numpy(), final_probability[mask])})
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "architecture frozen prospectively; no untouched post-freeze outcomes available",
        "development_period": ["2019-01-01", "2024-12-31"],
        "confirmation_period": ["2025-01-11", "2026-06-27"],
        "final_lockbox": {"status": "NOT OPENED / NOT AVAILABLE", "required_period": "after 2026-09-02",
                          "reason": "all outcomes through 2026-06-27 were inspected before V3; later official results were viewed during source discovery"},
        "selected_individual_model": FINAL_CANDIDATE.name,
        "selection_rule": "best V3 development ROC AUC, log loss, and ECE; confirmation did not select the model",
        "ensemble_components": dict(zip(COMPONENTS, map(float, weights))),
        "calibration": calibration,
        "calibration_selection_2019_2024": calibration_validation,
        "ensemble_development_metrics_note": "optimistic for weights because the combiner was fitted on these OOF predictions",
        "ensemble_development_metrics": dev_values,
        "ensemble_confirmation_metrics": confirm_values,
        "ensemble_performance_by_year": ensemble_years,
        "confirmation_accuracy_95pct_wilson": _wilson_interval(correct, len(confirm_y)),
        "paired_vs_v2_reference_confirmation": _mcnemar(confirm_y, confirm_p, reference_p),
        "benchmarks_development": _benchmarks(frame, DEV_YEARS),
        "benchmarks_confirmation": _benchmarks(frame, CONFIRM_YEARS),
        "confidence_buckets_development": _confidence_rows(aligned.loc[dev, "fighter_a_won"].to_numpy(), final_probability[dev]),
        "confidence_buckets_confirmation": _confidence_rows(confirm_y, confirm_p),
        "thresholds_selected_on_development": thresholds,
        "thresholds_applied_to_confirmation": _apply_thresholds(thresholds, confirm_y, confirm_p),
        "subgroups_confirmation": _subgroups(confirmation_frame, confirm_p),
        "ablation_development": ablation,
        "top_20_xgboost_gain_features": top_features,
        "market_benchmark": {"status": "not accepted as leakage-verified", "reason": "odds values exist, but no collection timestamp is present"},
    }
    path = report_dir / "final_analysis.json"
    path.write_text(json.dumps(report, indent=2, default=_jsonable), encoding="utf-8")
    architecture = {
        "frozen_at": report["created_at"], "status": "prospective V3 research architecture; not production",
        "features": V3_FEATURE_NAMES, "individual_candidate": FINAL_CANDIDATE.__dict__,
        "ensemble_components": report["ensemble_components"], "calibration": calibration,
        "final_lockbox_policy": "first meaningful completed-fight block strictly after 2026-09-02; evaluate once",
    }
    architecture_path = root / "models" / "v3-research" / "ARCHITECTURE_LOCK.json"
    architecture_path.write_text(json.dumps(architecture, indent=2), encoding="utf-8")
    digest = hashlib.sha256(architecture_path.read_bytes()).hexdigest().upper()
    (architecture_path.parent / "ARCHITECTURE_LOCK.sha256").write_text(f"{digest}  ARCHITECTURE_LOCK.json\n", encoding="utf-8")
    print(json.dumps({"weights": report["ensemble_components"], "calibration": calibration,
                      "confirmation": confirm_values, "lockbox": report["final_lockbox"]}, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    run(Path(args.project_root).resolve())


if __name__ == "__main__":
    main()
