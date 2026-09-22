"""Separate leakage-safe V2 method-of-victory model development."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .v2_evaluate import DEV_YEARS, RANDOM_STATE, load_or_build_dataset
from .v2_features import METHOD_FEATURE_NAMES


CLASSES = ["Decision", "KO/TKO", "Submission"]


@dataclass(frozen=True)
class MethodExperiment:
    name: str
    family: str
    params: dict[str, Any]


def candidates() -> list[MethodExperiment]:
    return [
        MethodExperiment("method_logistic_c0.1", "logistic", {"C": 0.1}),
        MethodExperiment("method_logistic_c1", "logistic", {"C": 1.0}),
        MethodExperiment("method_random_forest", "random_forest", {"n_estimators": 250, "min_samples_leaf": 10, "max_features": "sqrt"}),
        MethodExperiment("method_extra_trees", "extra_trees", {"n_estimators": 250, "min_samples_leaf": 8, "max_features": 0.7}),
        MethodExperiment("method_gradient_boosting", "gradient_boosting", {"n_estimators": 80, "learning_rate": 0.04, "max_depth": 2}),
        MethodExperiment("method_hist_gradient", "hist_gradient", {"max_iter": 180, "learning_rate": 0.04, "max_leaf_nodes": 15, "l2_regularization": 2.0}),
        MethodExperiment("method_xgboost", "xgboost", {"n_estimators": 300, "learning_rate": 0.025, "max_depth": 2, "min_child_weight": 8, "subsample": 0.8, "colsample_bytree": 0.8}),
    ]


def make_method_model(candidate: MethodExperiment) -> Pipeline:
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    if candidate.family == "logistic":
        model = LogisticRegression(
            **candidate.params, solver="lbfgs", max_iter=3000,
            random_state=RANDOM_STATE,
        )
        return Pipeline([("imputer", imputer), ("scale", StandardScaler()), ("model", model)])
    if candidate.family == "random_forest":
        model = RandomForestClassifier(**candidate.params, n_jobs=-1, random_state=RANDOM_STATE)
    elif candidate.family == "extra_trees":
        model = ExtraTreesClassifier(**candidate.params, n_jobs=-1, random_state=RANDOM_STATE)
    elif candidate.family == "gradient_boosting":
        model = GradientBoostingClassifier(**candidate.params, random_state=RANDOM_STATE)
    elif candidate.family == "hist_gradient":
        model = HistGradientBoostingClassifier(**candidate.params, random_state=RANDOM_STATE)
    elif candidate.family == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            **candidate.params, objective="multi:softprob", num_class=len(CLASSES),
            eval_metric="mlogloss", n_jobs=-1, random_state=RANDOM_STATE,
        )
    else:
        raise KeyError(candidate.family)
    return Pipeline([("imputer", imputer), ("model", model)])


def method_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(CLASSES)[probability.argmax(axis=1)]
    encoded = np.column_stack([(y == label).astype(float) for label in CLASSES])
    confidence = probability.max(axis=1)
    correct = (prediction == y).astype(float)
    ece = 0.0
    for lower in np.arange(0.30, 1.0, 0.10):
        upper = lower + 0.10
        mask = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= 1)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    recalls = recall_score(y, prediction, labels=CLASSES, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "multiclass_log_loss": float(log_loss(y, probability, labels=CLASSES)),
        "multiclass_brier": float(np.mean(np.sum((probability - encoded) ** 2, axis=1))),
        "ece": float(ece),
        "per_class_recall": {label: float(value) for label, value in zip(CLASSES, recalls)},
        "confusion_matrix": confusion_matrix(y, prediction, labels=CLASSES).tolist(),
    }


def _ordered_probability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(features)
    model_classes = list(model.named_steps["model"].classes_)
    return raw[:, [model_classes.index(label) for label in CLASSES]]


def walk_forward(frame: pd.DataFrame, candidate: MethodExperiment) -> tuple[pd.DataFrame, list[dict]]:
    predictions, folds = [], []
    for year in DEV_YEARS:
        train = frame.loc[frame.fight_date.dt.year < year]
        test = frame.loc[frame.fight_date.dt.year == year]
        model = make_method_model(candidate)
        if candidate.family == "xgboost":
            encoded = train.method_label.map({label: index for index, label in enumerate(CLASSES)})
            model.fit(train[METHOD_FEATURE_NAMES], encoded)
            probability = model.predict_proba(test[METHOD_FEATURE_NAMES])
        else:
            model.fit(train[METHOD_FEATURE_NAMES], train.method_label)
            probability = _ordered_probability(model, test[METHOD_FEATURE_NAMES])
        values = method_metrics(test.method_label.to_numpy(), probability)
        folds.append({"year": year, "train_fights": len(train), "test_fights": len(test), "accuracy": values["accuracy"], "multiclass_log_loss": values["multiclass_log_loss"], "multiclass_brier": values["multiclass_brier"], "ece": values["ece"]})
        part = test[["fight_id", "fight_date", "weight_class", "method_label"]].copy()
        for index, label in enumerate(CLASSES):
            part[f"p_{label}"] = probability[:, index]
        predictions.append(part)
    return pd.concat(predictions, ignore_index=True), folds


def run_method_development(root: Path) -> dict:
    output = root / "reports" / "v2"
    output.mkdir(parents=True, exist_ok=True)
    frame = load_or_build_dataset(root)
    frame["fight_date"] = pd.to_datetime(frame["fight_date"])
    frame = frame.loc[frame.method_label.notna() & (frame.fight_date < "2025-01-01")].copy()
    aggregate, fold_rows = [], []
    predictions_by_name = {}
    for index, candidate in enumerate(candidates(), 1):
        checkpoint = output / f"{candidate.name}_oof.csv"
        fold_checkpoint = output / f"{candidate.name}_folds.json"
        if checkpoint.exists() and fold_checkpoint.exists():
            print(f"[method {index}/{len(candidates())}] reuse {candidate.name}", flush=True)
            predictions = pd.read_csv(checkpoint, parse_dates=["fight_date"])
            folds = json.loads(fold_checkpoint.read_text(encoding="utf-8"))
        else:
            print(f"[method {index}/{len(candidates())}] {candidate.name}", flush=True)
            predictions, folds = walk_forward(frame, candidate)
            predictions.to_csv(checkpoint, index=False)
            fold_checkpoint.write_text(json.dumps(folds, indent=2), encoding="utf-8")
        predictions_by_name[candidate.name] = predictions
        probability = predictions[[f"p_{label}" for label in CLASSES]].to_numpy()
        values = method_metrics(predictions.method_label.to_numpy(), probability)
        aggregate.append({**asdict(candidate), "features": len(METHOD_FEATURE_NAMES), "validation_fights": len(predictions), "accuracy": values["accuracy"], "multiclass_log_loss": values["multiclass_log_loss"], "multiclass_brier": values["multiclass_brier"], "ece": values["ece"]})
        fold_rows.extend([{"experiment": candidate.name, **row} for row in folds])
    table = pd.DataFrame(aggregate).sort_values(["multiclass_log_loss", "multiclass_brier", "accuracy"], ascending=[True, True, False])
    table.to_csv(output / "method_experiments.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "method_walk_forward.csv", index=False)
    selected_name = str(table.iloc[0]["name"])
    selected = next(item for item in candidates() if item.name == selected_name)
    predictions = predictions_by_name[selected_name]
    predictions.to_csv(output / "method_selected_oof_predictions.csv", index=False)
    probability = predictions[[f"p_{label}" for label in CLASSES]].to_numpy()
    values = method_metrics(predictions.method_label.to_numpy(), probability)
    majority = frame.loc[frame.fight_date.dt.year.isin(DEV_YEARS), "method_label"].value_counts(normalize=True).max()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "formulation": "unconditional P(method | matchup); separate from winner model",
        "selected_experiment": asdict(selected),
        "features": METHOD_FEATURE_NAMES,
        "metrics": values,
        "majority_method_accuracy": float(majority),
        "class_order": CLASSES,
        "calibration_note": "top-class expected calibration error",
        "reliability_note": "method probabilities are inherently less reliable than winner probabilities and must be shown as a distribution",
    }
    (output / "method_development_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(values, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    run_method_development(Path(args.project_root).resolve())


if __name__ == "__main__":
    main()
