"""Validation-only selection and fixed-test evaluation for advanced V1.1."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .features import build_dataset, normalize_name
from .v2_accuracy_development import confidence_table
from .v2_evaluate import (
    Experiment,
    decisive,
    feature_names,
    load_or_build_dataset,
    make_model,
    metrics,
)


TRAIN_END = pd.Timestamp("2022-10-29")
TEST_START = pd.Timestamp("2022-11-05")
FEATURE_SET = "rich_548"
FEATURES = feature_names(FEATURE_SET)
COMPONENTS = {
    "logistic": ("logistic_rich_548_c1", Experiment("logistic_rich_548_c1", "logistic", FEATURE_SET, {"C": 1.0})),
    "xgboost": ("xgboost_rich548", Experiment("xgboost_rich548", "xgboost", FEATURE_SET, {"n_estimators": 350, "learning_rate": .025, "max_depth": 2, "min_child_weight": 8, "subsample": .8, "colsample_bytree": .8})),
    "gradient_boosting": ("gradient_boosting_rich548", Experiment("gradient_boosting_rich548", "gradient_boosting", FEATURE_SET, {"n_estimators": 100, "learning_rate": .04, "max_depth": 2})),
    "random_forest": ("random_forest_rich548", Experiment("random_forest_rich548", "random_forest", FEATURE_SET, {"n_estimators": 300, "min_samples_leaf": 8, "max_features": "sqrt"})),
    "extra_trees": ("extra_trees_rich548", Experiment("extra_trees_rich548", "extra_trees", FEATURE_SET, {"n_estimators": 300, "min_samples_leaf": 6, "max_features": .7})),
    "hist_gradient": ("hist_gradient_rich548", Experiment("hist_gradient_rich548", "hist_gradient", FEATURE_SET, {"max_iter": 180, "learning_rate": .05, "max_leaf_nodes": 15, "l2_regularization": 2.0})),
}


def pair_key(first: str, second: str) -> str:
    return "||".join(sorted((normalize_name(first), normalize_name(second))))


def evaluate_table(frame: pd.DataFrame, probability: np.ndarray) -> dict:
    return metrics(frame.fighter_a_won.to_numpy(dtype=int), probability)


def load_oof(root: Path, frame: pd.DataFrame) -> pd.DataFrame:
    merged = None
    report_dir = root / "reports" / "v2_accuracy"
    for label, (file_stem, _) in COMPONENTS.items():
        part = pd.read_csv(report_dir / f"oof_{file_stem}.csv", parse_dates=["fight_date"])
        part = part[["fight_id", "fight_date", "fighter_a_won", "probability"]].rename(columns={"probability": label})
        merged = part if merged is None else merged.merge(part[["fight_id", label]], on="fight_id", validate="one_to_one")
    assert merged is not None
    merged = merged.merge(frame[["fight_id", "elo_diff"]], on="fight_id", validate="one_to_one")
    merged["elo"] = 1.0 / (1.0 + 10.0 ** (-merged.elo_diff.fillna(0.0) / 400.0))
    return merged.loc[merged.fight_date <= TRAIN_END].copy()


def choose_ensemble(oof: pd.DataFrame) -> tuple[dict, list[dict]]:
    tune = oof.loc[oof.fight_date.dt.year <= 2020].copy()
    confirm = oof.loc[oof.fight_date.dt.year >= 2021].copy()
    specs = [((name,), (1.0,)) for name in [*COMPONENTS, "elo"]]
    for partner in [name for name in COMPONENTS if name != "logistic"] + ["elo"]:
        for logistic_weight in (.25, .5, .75):
            specs.append((("logistic", partner), (logistic_weight, 1 - logistic_weight)))
    for tree in ("xgboost", "gradient_boosting", "extra_trees", "hist_gradient"):
        for weights in ((.2, .6, .2), (.4, .4, .2), (.2, .4, .4)):
            specs.append((("logistic", tree, "elo"), weights))
    baseline_tune = evaluate_table(tune, tune.logistic.to_numpy())
    baseline_confirm = evaluate_table(confirm, confirm.logistic.to_numpy())
    trials = []
    for names, weights in specs:
        tune_p = sum(weight * tune[name].to_numpy() for name, weight in zip(names, weights))
        confirm_p = sum(weight * confirm[name].to_numpy() for name, weight in zip(names, weights))
        tune_metrics, confirm_metrics = evaluate_table(tune, tune_p), evaluate_table(confirm, confirm_p)
        trials.append({
            "components": list(names), "weights": list(weights),
            "tuning": tune_metrics, "confirmation": confirm_metrics,
            "eligible": (
                tune_metrics["accuracy"] >= baseline_tune["accuracy"]
                and tune_metrics["log_loss"] <= baseline_tune["log_loss"]
                and confirm_metrics["accuracy"] >= baseline_confirm["accuracy"]
                and confirm_metrics["log_loss"] <= baseline_confirm["log_loss"]
            ),
        })
    eligible = [row for row in trials if row["eligible"]]
    chosen = min(
        eligible,
        key=lambda row: (-row["confirmation"]["accuracy"], row["confirmation"]["log_loss"], row["confirmation"]["brier"]),
    ) if eligible else next(row for row in trials if row["components"] == ["logistic"])
    return chosen, trials


def calibration_selection(oof: pd.DataFrame, architecture: dict) -> dict:
    names, weights = architecture["components"], architecture["weights"]
    tune = oof.loc[oof.fight_date.dt.year <= 2020].copy()
    confirm = oof.loc[oof.fight_date.dt.year >= 2021].copy()
    tune_p = sum(w * tune[n].to_numpy() for n, w in zip(names, weights))
    confirm_p = sum(w * confirm[n].to_numpy() for n, w in zip(names, weights))
    eps = 1e-6
    platt = LogisticRegression(C=1.0).fit(
        np.log(np.clip(tune_p, eps, 1-eps) / np.clip(1-tune_p, eps, 1-eps)).reshape(-1, 1),
        tune.fighter_a_won,
    )
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(tune_p, tune.fighter_a_won)
    candidates = {
        "uncalibrated": confirm_p,
        "platt": platt.predict_proba(np.log(np.clip(confirm_p, eps, 1-eps) / np.clip(1-confirm_p, eps, 1-eps)).reshape(-1, 1))[:, 1],
        "isotonic": isotonic.predict(confirm_p),
    }
    scores = {name: evaluate_table(confirm, values) for name, values in candidates.items()}
    base = scores["uncalibrated"]
    eligible = [name for name in ("platt", "isotonic") if scores[name]["log_loss"] < base["log_loss"] and scores[name]["brier"] < base["brier"] and scores[name]["accuracy"] >= base["accuracy"]]
    selected = min(eligible, key=lambda name: scores[name]["log_loss"]) if eligible else "uncalibrated"
    return {"selected": selected, "confirmation_scores": scores, "platt": platt, "isotonic": isotonic}


def orient_to_basic(v2_test: pd.DataFrame, probability: np.ndarray, basic: pd.DataFrame) -> np.ndarray:
    rich = v2_test[["fight_date", "fighter_a", "fighter_b"]].copy()
    rich["pair_key"] = [pair_key(a, b) for a, b in zip(rich.fighter_a, rich.fighter_b)]
    rich["rich_a"] = rich.fighter_a.map(normalize_name)
    rich["rich_probability"] = probability
    basic_keys = basic[["date", "fighter_a", "fighter_b"]].copy()
    basic_keys["fight_date"] = pd.to_datetime(basic_keys.date)
    basic_keys["pair_key"] = [pair_key(a, b) for a, b in zip(basic_keys.fighter_a, basic_keys.fighter_b)]
    basic_keys["basic_a"] = basic_keys.fighter_a.map(normalize_name)
    aligned = basic_keys.merge(rich[["fight_date", "pair_key", "rich_a", "rich_probability"]], on=["fight_date", "pair_key"], validate="one_to_one")
    return np.where(aligned.basic_a == aligned.rich_a, aligned.rich_probability, 1.0 - aligned.rich_probability)


def detailed_evaluation(frame: pd.DataFrame, probability: np.ndarray) -> dict:
    y = frame.fighter_a_won.to_numpy(dtype=int)
    predicted = probability >= .5
    confidence = np.maximum(probability, 1-probability)
    result = metrics(y, probability)
    buckets, _ = confidence_table(y, probability)
    result["confidence_buckets"] = buckets
    return result


def run(root: Path) -> dict:
    rich = decisive(load_or_build_dataset(root)).copy()
    rich["fight_date"] = pd.to_datetime(rich.fight_date)
    oof = load_oof(root, rich)
    architecture, trials = choose_ensemble(oof)
    calibration = calibration_selection(oof, architecture)
    train = rich.loc[rich.fight_date <= TRAIN_END].copy()
    test = rich.loc[rich.fight_date >= TEST_START].copy()
    models = {}
    for name in architecture["components"]:
        if name == "elo":
            continue
        models[name] = make_model(COMPONENTS[name][1]).fit(train[FEATURES], train.fighter_a_won)
    rich_probability = np.zeros(len(test))
    for name, weight in zip(architecture["components"], architecture["weights"]):
        part = (1.0 / (1.0 + 10.0 ** (-test.elo_diff.fillna(0).to_numpy() / 400.0))) if name == "elo" else models[name].predict_proba(test[FEATURES])[:, 1]
        rich_probability += weight * part
    if calibration["selected"] == "platt":
        p = np.clip(rich_probability, 1e-6, 1-1e-6)
        rich_probability = calibration["platt"].predict_proba(np.log(p/(1-p)).reshape(-1, 1))[:, 1]
    elif calibration["selected"] == "isotonic":
        rich_probability = calibration["isotonic"].predict(rich_probability)

    basic, basic_states, basic_profiles = build_dataset(
        str(root / "data" / "raw" / "fights.csv"), str(root / "data" / "raw" / "fighters.csv")
    )
    basic = basic.loc[basic.date >= TEST_START.date().isoformat()].reset_index(drop=True)
    aligned_rich = orient_to_basic(test, rich_probability, basic)
    low_artifact = joblib.load(root / "models" / "debutant_safe_candidate.joblib")
    low_probability = low_artifact["pipeline"].predict_proba(basic[low_artifact["feature_names"]])[:, 1]
    low_mask = (basic.fighter_a_prior_ufc_fights < 3) | (basic.fighter_b_prior_ufc_fights < 3)
    hybrid_probability = np.where(low_mask, low_probability, aligned_rich)

    saved = pd.read_csv(root / "reports" / "model_test_predictions.csv")
    baseline_probability = saved["Rich Logistic"].to_numpy()
    y = basic.fighter_a_won.to_numpy(dtype=int)
    baseline_correct = (baseline_probability >= .5) == y
    candidate_correct = (hybrid_probability >= .5) == y
    changed = (baseline_probability >= .5) != (hybrid_probability >= .5)
    discordant = int((baseline_correct & ~candidate_correct).sum() + (~baseline_correct & candidate_correct).sum())
    improved_count = int((~baseline_correct & candidate_correct).sum())
    worsened_count = int((baseline_correct & ~candidate_correct).sum())
    significance = binomtest(improved_count, discordant, .5).pvalue if discordant else 1.0
    report = {
        "version": "V1.1 advanced candidate",
        "fixed_test_period": [str(TEST_START.date()), str(test.fight_date.max().date())],
        "test_fights": int(len(test)),
        "validation_policy": "OOF 2019-2020 tuning; OOF 2021 through 2022-10-29 confirmation; no final-test selection",
        "selected_architecture": {k: architecture[k] for k in ("components", "weights", "tuning", "confirmation")},
        "calibration": {"selected": calibration["selected"], "confirmation_scores": calibration["confirmation_scores"]},
        "v1_0_full_rich": detailed_evaluation(basic, baseline_probability),
        "v1_1_hybrid": detailed_evaluation(basic, hybrid_probability),
        "prediction_changes": {
            "changed": int(changed.sum()),
            "changed_became_correct": improved_count,
            "previously_correct_became_wrong": worsened_count,
            "mcnemar_exact_p_value": float(significance),
            "statistically_meaningful_at_0_05": bool(significance < .05),
        },
    }
    masks = {
        "ufc_debutant": (basic.fighter_a_prior_ufc_fights == 0) | (basic.fighter_b_prior_ufc_fights == 0),
        "under_3_ufc_fights": low_mask,
        "both_3_plus": ~low_mask,
        "large_age_gap_8_plus": basic.age_diff.abs() >= 8,
        "large_reach_gap_6_plus": basic.reach_diff.abs() >= 6,
    }
    report["subgroups"] = {
        name: {
            "fights": int(mask.sum()),
            "v1_accuracy": float(baseline_correct[mask].mean()) if mask.any() else None,
            "v1_1_accuracy": float(candidate_correct[mask].mean()) if mask.any() else None,
        }
        for name, mask in masks.items()
    }
    report["by_weight_class"] = []
    booked = basic[[name for name in basic.columns if name.startswith("weight_class_")]].idxmax(axis=1).str.removeprefix("weight_class_")
    for name, indexes in booked.groupby(booked).groups.items():
        idx = np.asarray(list(indexes), dtype=int)
        report["by_weight_class"].append({"weight_class": name, "fights": len(idx), "v1_accuracy": float(baseline_correct[idx].mean()), "v1_1_accuracy": float(candidate_correct[idx].mean())})
    output = root / "reports" / "v1_1_advanced_evaluation.json"
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    prediction_frame = basic[["date", "fighter_a", "fighter_b", "fighter_a_won"]].copy()
    prediction_frame["v1_probability"] = baseline_probability
    prediction_frame["v1_1_probability"] = hybrid_probability
    prediction_frame["low_data_route"] = low_mask
    prediction_frame.to_csv(root / "reports" / "v1_1_test_predictions.csv", index=False)
    print(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    run(Path(".").resolve())
