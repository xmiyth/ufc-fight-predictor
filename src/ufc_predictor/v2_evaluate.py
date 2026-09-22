"""Leakage-safe walk-forward development and one-time V2 lockbox evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .v2_data import file_sha256
from .v2_features import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    INTERACTION_FEATURES,
    build_v2_dataset,
)


DEV_YEARS = tuple(range(2019, 2025))
LOCKBOX_START = pd.Timestamp("2025-01-01")
RANDOM_STATE = 20260901


@dataclass(frozen=True)
class Experiment:
    name: str
    family: str
    feature_set: str
    params: dict[str, Any]


def _base_feature_names() -> list[str]:
    return [name for name in FEATURE_NAMES if "_decay_" not in name]


def feature_names(key: str) -> list[str]:
    if key == "core":
        wanted = [
            "age_diff", "height_diff", "reach_diff", "ufc_fights_diff",
            "ufc_win_pct_diff", "recent_3_win_pct_diff", "recent_5_win_pct_diff",
            "current_win_streak_diff", "current_loss_streak_diff",
            "days_since_last_fight_diff", "finish_rate_diff", "elo_diff",
            "sig_landed_pm_diff", "sig_absorbed_pm_diff", "sig_accuracy_diff",
            "striking_defense_diff", "takedowns_per15_diff",
            "takedown_accuracy_diff", "takedown_defense_diff",
            "submissions_per15_diff", "scheduled_rounds", "five_round_fight",
        ]
        return [name for name in wanted if name in FEATURE_NAMES]
    if key.startswith("rich_"):
        half_life = key.split("_", 1)[1]
        return _base_feature_names() + [
            name for name in FEATURE_NAMES if f"_decay_{half_life}d" in name
        ]
    if key == "all_decay":
        return list(FEATURE_NAMES)
    raise KeyError(key)


def experiments() -> list[Experiment]:
    result = []
    for feature_set in ("core", "rich_180", "rich_365", "rich_548", "rich_730", "rich_1095"):
        result.append(Experiment(f"logistic_{feature_set}_c0.1", "logistic", feature_set, {"C": 0.1}))
    for feature_set in ("core", "rich_548", "rich_730"):
        result.append(Experiment(f"logistic_{feature_set}_c1", "logistic", feature_set, {"C": 1.0}))
    for ratio in (0.25, 0.5, 0.75):
        result.append(Experiment(f"elastic_net_rich548_l1r{ratio:g}", "elastic_net", "rich_548", {"C": 0.1, "l1_ratio": ratio}))
    result.extend([
        Experiment("random_forest_rich548", "random_forest", "rich_548", {"n_estimators": 300, "min_samples_leaf": 8, "max_features": "sqrt"}),
        Experiment("extra_trees_rich548", "extra_trees", "rich_548", {"n_estimators": 300, "min_samples_leaf": 6, "max_features": 0.7}),
        Experiment("gradient_boosting_rich548", "gradient_boosting", "rich_548", {"n_estimators": 100, "learning_rate": 0.04, "max_depth": 2}),
        Experiment("hist_gradient_rich548", "hist_gradient", "rich_548", {"max_iter": 180, "learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 2.0}),
        Experiment("xgboost_rich548", "xgboost", "rich_548", {"n_estimators": 350, "learning_rate": 0.025, "max_depth": 2, "min_child_weight": 8, "subsample": 0.8, "colsample_bytree": 0.8}),
    ])
    return result


def make_model(experiment: Experiment) -> Pipeline:
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    if experiment.family in {"logistic", "elastic_net"}:
        model = LogisticRegression(
            **experiment.params,
            penalty="elasticnet" if experiment.family == "elastic_net" else "l2",
            solver="saga" if experiment.family == "elastic_net" else "liblinear",
            max_iter=5000, random_state=RANDOM_STATE,
        )
        return Pipeline([("imputer", imputer), ("scale", StandardScaler()), ("model", model)])
    if experiment.family == "random_forest":
        model = RandomForestClassifier(**experiment.params, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced")
    elif experiment.family == "extra_trees":
        model = ExtraTreesClassifier(**experiment.params, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced")
    elif experiment.family == "gradient_boosting":
        model = GradientBoostingClassifier(**experiment.params, random_state=RANDOM_STATE)
    elif experiment.family == "hist_gradient":
        model = HistGradientBoostingClassifier(**experiment.params, random_state=RANDOM_STATE)
    elif experiment.family == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            **experiment.params, objective="binary:logistic", eval_metric="logloss",
            n_jobs=-1, random_state=RANDOM_STATE,
        )
    else:
        raise KeyError(experiment.family)
    return Pipeline([("imputer", imputer), ("model", model)])


def expected_calibration_error(y: np.ndarray, p: np.ndarray) -> float:
    confidence = np.maximum(p, 1.0 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    edges = np.arange(0.5, 1.0001, 0.05)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if mask.any():
            error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "accuracy": float(accuracy_score(y, clipped >= 0.5)),
        "roc_auc": float(roc_auc_score(y, clipped)),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
        "brier": float(brier_score_loss(y, clipped)),
        "ece": expected_calibration_error(y, clipped),
    }


def load_or_build_dataset(root: Path) -> pd.DataFrame:
    cache = root / "data" / "processed" / "v2_accuracy_point_in_time.joblib"
    metadata = root / "data" / "processed" / "v2_accuracy_point_in_time.metadata.json"
    fights = root / "data" / "raw" / "fights.csv"
    fighters = root / "data" / "raw" / "fighters.csv"
    hashes = {"fights": file_sha256(fights), "fighters": file_sha256(fighters)}
    if cache.exists() and metadata.exists():
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        if saved.get("source_sha256") == hashes and saved.get("feature_names") == FEATURE_NAMES:
            return joblib.load(cache)
    frame, _, _ = build_v2_dataset(fights, fighters)
    cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(frame, cache, compress=3)
    metadata.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashes,
        "rows": len(frame),
        "columns": len(frame.columns),
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "strict_rule": "feature_timestamp < fight_timestamp; prior bouts only; same-date batch",
    }, indent=2), encoding="utf-8")
    return frame


def decisive(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[frame["fighter_a_won"].notna()].copy()
    result["fight_date"] = pd.to_datetime(result["fight_date"])
    result["fighter_a_won"] = result["fighter_a_won"].astype(int)
    return result.sort_values("fight_date", kind="stable").reset_index(drop=True)


def walk_forward(frame: pd.DataFrame, experiment: Experiment) -> tuple[pd.DataFrame, list[dict]]:
    names = feature_names(experiment.feature_set)
    predictions = []
    fold_rows = []
    for year in DEV_YEARS:
        train = frame.loc[frame.fight_date.dt.year < year]
        test = frame.loc[frame.fight_date.dt.year == year]
        pipeline = make_model(experiment)
        pipeline.fit(train[names], train.fighter_a_won)
        probability = pipeline.predict_proba(test[names])[:, 1]
        fold_metrics = metrics(test.fighter_a_won.to_numpy(), probability)
        fold_rows.append({"year": year, "train_fights": len(train), "test_fights": len(test), **fold_metrics})
        part = test[["fight_id", "fight_date", "weight_class", "fighter_a_won"]].copy()
        part["probability"] = probability
        part["year"] = year
        predictions.append(part)
    return pd.concat(predictions, ignore_index=True), fold_rows


def benchmark_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    rows, summaries = [], []
    for year in DEV_YEARS:
        train = frame.loc[frame.fight_date.dt.year < year]
        test = frame.loc[frame.fight_date.dt.year == year].copy()
        y = test.fighter_a_won.to_numpy()
        majority_value = int(train.fighter_a_won.mean() >= 0.5)
        candidates = {
            "random": np.fromiter(
                (
                    hashlib.sha256(f"random-benchmark|{value}".encode("utf-8")).digest()[0] % 2
                    for value in test.fight_id
                ),
                dtype=int,
            ),
            "majority": np.full(len(test), majority_value),
            "better_ufc_win_pct": (test.ufc_win_pct_diff.fillna(0).to_numpy() >= 0).astype(int),
            "better_recent_form": (test.recent_5_win_pct_diff.fillna(0).to_numpy() >= 0).astype(int),
            "elo": (test.elo_diff.fillna(0).to_numpy() >= 0).astype(int),
            "elo_k16": (test.elo_k16_diff.fillna(0).to_numpy() >= 0).astype(int),
            "elo_k24": (test.elo_k24_diff.fillna(0).to_numpy() >= 0).astype(int),
            "elo_k48": (test.elo_k48_diff.fillna(0).to_numpy() >= 0).astype(int),
            "recency_adjusted_elo": (test.recency_adjusted_elo_diff.fillna(0).to_numpy() >= 0).astype(int),
            "weight_class_elo": (test.weight_class_elo_diff.fillna(0).to_numpy() >= 0).astype(int),
        }
        for name, prediction in candidates.items():
            rows.append({"benchmark": name, "year": year, "accuracy": float(accuracy_score(y, prediction)), "fights": len(test)})
    detail = pd.DataFrame(rows)
    for name, group in detail.groupby("benchmark"):
        summaries.append({"benchmark": name, "accuracy": float(np.average(group.accuracy, weights=group.fights)), "fights": int(group.fights.sum())})
    return detail, summaries


def calibration_table(predictions: pd.DataFrame) -> list[dict]:
    p = predictions.probability.to_numpy()
    y = predictions.fighter_a_won.to_numpy()
    confidence = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(int)
    edges = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.01]
    result = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lower) & (confidence < upper)
        if mask.any():
            result.append({"bucket": f"{lower:.0%}-{min(upper, 1):.0%}", "fights": int(mask.sum()), "mean_prediction": float(confidence[mask].mean()), "observed_win_rate": float(correct[mask].mean())})
    return result


def bootstrap_ci(y: np.ndarray, p: np.ndarray, repetitions: int = 2000) -> dict[str, list[float]]:
    rng = np.random.default_rng(RANDOM_STATE)
    samples = {key: [] for key in ("accuracy", "roc_auc", "log_loss", "brier", "ece")}
    for _ in range(repetitions):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) < 2:
            continue
        values = metrics(y[index], p[index])
        for key in samples:
            samples[key].append(values[key])
    return {key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] for key, values in samples.items()}


def paired_bootstrap(
    y: np.ndarray, selected: np.ndarray, comparison: np.ndarray, repetitions: int = 3000
) -> dict[str, Any]:
    rng = np.random.default_rng(RANDOM_STATE + 1)
    selected = np.clip(selected, 1e-7, 1 - 1e-7)
    comparison = np.clip(comparison, 1e-7, 1 - 1e-7)
    per_fight = {
        "accuracy": ((selected >= 0.5).astype(int) == y).astype(float)
        - ((comparison >= 0.5).astype(int) == y).astype(float),
        "log_loss": -(y * np.log(selected) + (1 - y) * np.log(1 - selected))
        + (y * np.log(comparison) + (1 - y) * np.log(1 - comparison)),
        "brier": (selected - y) ** 2 - (comparison - y) ** 2,
    }
    differences = {key: [] for key in per_fight}
    for _ in range(repetitions):
        index = rng.integers(0, len(y), len(y))
        for key, values in per_fight.items():
            differences[key].append(float(values[index].mean()))
    return {
        key: {
            "mean_difference": float(np.mean(values)),
            "95pct_ci": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        }
        for key, values in differences.items()
    }


def run_development(root: Path) -> dict:
    output = root / "reports" / "v2_accuracy"
    output.mkdir(parents=True, exist_ok=True)
    frame = decisive(load_or_build_dataset(root))
    development = frame.loc[frame.fight_date < LOCKBOX_START].copy()
    aggregate_rows, fold_rows = [], []
    prediction_cache: dict[str, pd.DataFrame] = {}
    experiment_path = output / "experiments.csv"
    walk_forward_path = output / "walk_forward.csv"
    expected_names = {item.name for item in experiments()}
    if experiment_path.exists() and walk_forward_path.exists():
        cached_table = pd.read_csv(experiment_path)
        if set(cached_table["name"]) == expected_names:
            table = cached_table.sort_values(
                ["accuracy", "log_loss", "brier"], ascending=[False, True, True]
            )
        else:
            table = pd.DataFrame()
    else:
        table = pd.DataFrame()
    if table.empty:
        for index, experiment in enumerate(experiments(), 1):
            checkpoint = output / f"oof_{experiment.name}.csv"
            fold_checkpoint = output / f"folds_{experiment.name}.json"
            if checkpoint.exists() and fold_checkpoint.exists():
                print(f"[{index}/{len(experiments())}] reuse {experiment.name}", flush=True)
                predictions = pd.read_csv(checkpoint, parse_dates=["fight_date"])
                folds = json.loads(fold_checkpoint.read_text(encoding="utf-8"))
            else:
                print(f"[{index}/{len(experiments())}] {experiment.name}", flush=True)
                predictions, folds = walk_forward(development, experiment)
                predictions.to_csv(checkpoint, index=False)
                fold_checkpoint.write_text(json.dumps(folds, indent=2), encoding="utf-8")
            prediction_cache[experiment.name] = predictions
            aggregate = metrics(predictions.fighter_a_won.to_numpy(), predictions.probability.to_numpy())
            aggregate_rows.append({**asdict(experiment), "features": len(feature_names(experiment.feature_set)), "validation_fights": len(predictions), **aggregate, "accuracy_std_by_year": float(np.std([row["accuracy"] for row in folds]))})
            fold_rows.extend([{"experiment": experiment.name, **row} for row in folds])
        table = pd.DataFrame(aggregate_rows).sort_values(["accuracy", "log_loss", "brier"], ascending=[False, True, True])
        table.to_csv(experiment_path, index=False)
        pd.DataFrame(fold_rows).to_csv(walk_forward_path, index=False)
    else:
        print("Reusing completed candidate metrics.", flush=True)

    best_name = str(table.iloc[0]["name"])
    best_experiment = next(item for item in experiments() if item.name == best_name)
    selected_prediction_path = output / "selected_oof_predictions.csv"
    if best_name not in prediction_cache and selected_prediction_path.exists():
        prediction_cache[best_name] = pd.read_csv(
            selected_prediction_path, parse_dates=["fight_date"]
        )
    if best_name not in prediction_cache:
        print(f"Replaying selected candidate: {best_name}", flush=True)
        prediction_cache[best_name], _ = walk_forward(development, best_experiment)
    best_predictions = prediction_cache[best_name]
    best_predictions.to_csv(selected_prediction_path, index=False)
    benchmark_detail, benchmarks = benchmark_predictions(development)
    benchmark_detail.to_csv(output / "benchmarks_by_year.csv", index=False)

    # Leakage-safe group ablation on the chosen family/configuration only.
    selected_names = feature_names(best_experiment.feature_set)
    ablation_path = output / "ablation.csv"
    if ablation_path.exists():
        ablations = pd.read_csv(ablation_path).to_dict("records")
    else:
        ablations = []
    ablation_groups = {
        **FEATURE_GROUPS,
        "matchup_interactions": INTERACTION_FEATURES,
        "time_decay": [name for name in selected_names if "_decay_" in name],
    }
    completed_ablation_names = {row["removed_group"] for row in ablations}
    for group_name, group_features in ablation_groups.items():
        if group_name in completed_ablation_names:
            continue
        kept = [name for name in selected_names if name not in set(group_features)]
        if len(kept) == len(selected_names) or not kept:
            continue
        print(f"Ablating: {group_name}", flush=True)
        custom = Experiment(f"ablate_{group_name}", best_experiment.family, best_experiment.feature_set, best_experiment.params)
        yearly_predictions = []
        for year in DEV_YEARS:
            train = development.loc[development.fight_date.dt.year < year]
            test = development.loc[development.fight_date.dt.year == year]
            model = make_model(custom)
            model.fit(train[kept], train.fighter_a_won)
            probability = model.predict_proba(test[kept])[:, 1]
            yearly_predictions.append(pd.DataFrame({"fighter_a_won": test.fighter_a_won.to_numpy(), "probability": probability}))
        combined = pd.concat(yearly_predictions, ignore_index=True)
        values = metrics(combined.fighter_a_won.to_numpy(), combined.probability.to_numpy())
        ablations.append({"removed_group": group_name, "remaining_features": len(kept), **values})
        pd.DataFrame(ablations).to_csv(ablation_path, index=False)

    logistic_experiment = next(
        item for item in experiments() if item.name == "logistic_core_c0.1"
    )
    logistic_path = output / "logistic_core_oof_predictions.csv"
    if logistic_path.exists():
        logistic_predictions = pd.read_csv(logistic_path, parse_dates=["fight_date"])
    else:
        print("Replaying logistic comparator.", flush=True)
        logistic_predictions, _ = walk_forward(development, logistic_experiment)
        logistic_predictions.to_csv(logistic_path, index=False)
    aligned = best_predictions.merge(
        logistic_predictions[["fight_id", "probability"]],
        on="fight_id", suffixes=("_selected", "_logistic"), validate="one_to_one",
    )
    ensemble_probability = (
        aligned["probability_selected"].to_numpy()
        + aligned["probability_logistic"].to_numpy()
    ) / 2.0
    ensemble = metrics(aligned.fighter_a_won.to_numpy(), ensemble_probability)
    ensemble["definition"] = "fixed 50% selected model + 50% core logistic; no lockbox tuning"
    paired = {
        "selected_vs_logistic": paired_bootstrap(
            aligned.fighter_a_won.to_numpy(),
            aligned.probability_selected.to_numpy(),
            aligned.probability_logistic.to_numpy(),
        ),
        "ensemble_vs_selected": paired_bootstrap(
            aligned.fighter_a_won.to_numpy(),
            ensemble_probability,
            aligned.probability_selected.to_numpy(),
        ),
    }

    y = best_predictions.fighter_a_won.to_numpy()
    p = best_predictions.probability.to_numpy()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "development_period": [str(development.fight_date.min().date()), "2024-12-31"],
        "walk_forward_years": list(DEV_YEARS),
        "lockbox_policy": "2025-01-01 onward untouched during development",
        "selected_experiment": asdict(best_experiment),
        "selected_features": selected_names,
        "selected_metrics": metrics(y, p),
        "selected_95pct_bootstrap_ci": bootstrap_ci(y, p),
        "calibration": calibration_table(best_predictions),
        "benchmarks": benchmarks,
        "ablation": ablations,
        "fixed_ensemble": ensemble,
        "paired_bootstrap_comparisons": paired,
        "model_families_unavailable": {
            "LightGBM": "not installed; omitted to avoid adding a redundant dependency before evidence of need",
            "CatBoost": "not installed; omitted because the current inputs are numeric and do not exploit its categorical specialization",
            "betting_favorite": "no verified point-in-time historical odds in the immutable source",
        },
    }
    (output / "development_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["selected_metrics"], indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--phase", choices=["develop"], default="develop")
    args = parser.parse_args()
    run_development(Path(args.project_root).resolve())


if __name__ == "__main__":
    main()
