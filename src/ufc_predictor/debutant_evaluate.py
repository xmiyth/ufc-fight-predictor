"""Chronological evaluation of debutant-safe winner inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import (
    LOW_DATA_FEATURES,
    STANCE_FEATURES,
    WEIGHT_CLASS_FEATURES,
    build_dataset,
)


TRAIN_END = "2022-10-29"
TEST_START = "2022-11-05"
ADDED_FEATURES = [
    "ufc_debutant_diff", "ko_tko_wins_diff", "submission_wins_diff",
    "decision_wins_diff", *LOW_DATA_FEATURES, *STANCE_FEATURES,
    *WEIGHT_CLASS_FEATURES,
]


def metrics(labels: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, probabilities >= 0.5)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def subgroup_results(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, Any]:
    a = frame["fighter_a_ufc_fights"]
    b = frame["fighter_b_ufc_fights"]
    masks = {
        "both_3_plus": (a >= 3) & (b >= 3),
        "involves_debutant": (a == 0) | (b == 0),
        "involves_1_or_2_no_debutant": ((a.between(1, 2)) | (b.between(1, 2)))
        & (a > 0) & (b > 0),
        "both_debutants": (a == 0) & (b == 0),
    }
    rows = {}
    labels = frame["fighter_a_won"]
    for name, mask in masks.items():
        rows[name] = {
            "fights": int(mask.sum()),
            "accuracy": (
                None if not mask.any()
                else float(accuracy_score(labels[mask], probabilities[mask] >= 0.5))
            ),
        }
    return rows


def run(project_root: Path) -> dict[str, Any]:
    frozen_path = project_root / "models" / "v1.0" / "ufc_predictor_v1_0.joblib"
    frozen = joblib.load(frozen_path)
    data, states, profiles = build_dataset(
        str(project_root / "data" / "raw" / "fights.csv"),
        str(project_root / "data" / "raw" / "fighters.csv"),
    )
    train = data[data["date"] <= TRAIN_END].copy()
    test = data[data["date"] >= TEST_START].copy()
    if train["date"].max() >= test["date"].min():
        raise RuntimeError("Chronological split overlaps")

    frozen_features = list(frozen["feature_names"])
    frozen_probability = frozen["pipeline"].predict_proba(test[frozen_features])[:, 1]
    candidate_features = frozen_features + [
        name for name in ADDED_FEATURES if name not in frozen_features
    ]
    candidate = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=0.05, max_iter=3000, random_state=42)),
        ]
    )
    candidate.fit(train[candidate_features], train["fighter_a_won"])
    candidate_probability = candidate.predict_proba(test[candidate_features])[:, 1]
    frozen_metrics = metrics(test["fighter_a_won"], frozen_probability)
    candidate_metrics = metrics(test["fighter_a_won"], candidate_probability)
    improved = (
        candidate_metrics["log_loss"] < frozen_metrics["log_loss"]
        and candidate_metrics["brier_score"] < frozen_metrics["brier_score"]
        and candidate_metrics["roc_auc"] >= frozen_metrics["roc_auc"]
    )
    report = {
        "task": "debutant-safe winner prediction",
        "split": {
            "train_end": TRAIN_END,
            "test_start": TEST_START,
            "train_fights": int(len(train)),
            "test_fights": int(len(test)),
        },
        "frozen_model": {
            "sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest().upper(),
            "metrics": frozen_metrics,
            "subgroups": subgroup_results(test, frozen_probability),
        },
        "same_structure_candidate": {
            "features": candidate_features,
            "added_features": ADDED_FEATURES,
            "metrics": candidate_metrics,
            "subgroups": subgroup_results(test, candidate_probability),
        },
        "candidate_demonstrably_improved": improved,
        "deployment_decision": (
            "deployed_as_v1.1" if improved else "keep_frozen_v1"
        ),
        "unavailable_features": {
            "professional_mma_record": (
                "Not present as point-in-time history. Current UFCStats profile summaries "
                "would leak future results into old fights and are excluded."
            ),
            "previous_promotion": "Not present in the supplied datasets.",
        },
        "leakage_audit": {
            "snapshot_before_current_fight_update": True,
            "debutant_ufc_performance_is_missing": True,
            "training_set_median_imputation": True,
            "future_profile_performance_summaries_used": False,
        },
    }
    candidate_path = project_root / "models" / "debutant_safe_candidate.joblib"
    joblib.dump(
        {
            "pipeline": candidate,
            "model_name": "Pruned Rich Logistic + Low Data",
            "feature_names": candidate_features,
            "fighter_states": {key: value.serializable() for key, value in states.items()},
            "profiles": profiles,
            "last_data_date": str(data["date"].max()),
            "evaluation": report,
            "trained_through": TRAIN_END,
        },
        candidate_path,
    )
    if improved:
        deployment = clone(candidate).fit(
            data[candidate_features], data["fighter_a_won"]
        )
        deployment_path = (
            project_root / "models" / "v1.1" / "ufc_predictor_v1_1.joblib"
        )
        deployment_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": deployment,
                "model_name": "Pruned Rich Logistic + Low Data",
                "model_version": "V1.1",
                "feature_names": candidate_features,
                "fighter_states": {
                    key: value.serializable() for key, value in states.items()
                },
                "profiles": profiles,
                "last_data_date": str(data["date"].max()),
                "evaluation": report,
                "trained_through": str(data["date"].max()),
                "deployment_note": (
                    "Same regularized logistic structure as V1.0; refit on all "
                    "eligible fights after chronological low-data validation."
                ),
            },
            deployment_path,
        )
        report["deployment_artifact"] = str(deployment_path.relative_to(project_root))
        report["deployment_sha256"] = hashlib.sha256(
            deployment_path.read_bytes()
        ).hexdigest().upper()
    output = project_root / "reports" / "debutant_evaluation.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run(Path(".").resolve())
