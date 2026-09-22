"""Freeze the development-selected V4 gate for a future untouched test."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib

from .v2_data import file_sha256
from .v2_evaluate import Experiment, feature_names, make_model
from .v3_evaluate import decisive, load_or_build
from .v4_prospective import augment_low_history, fit_division_priors, v4_feature_names


WEIGHTS = (0.25, 0.75)
SPECS = (
    Experiment("logistic", "logistic", "rich_548", {"C": 1.0}),
    Experiment("xgboost", "xgboost", "rich_548", {
        "n_estimators": 350, "learning_rate": 0.025, "max_depth": 2,
        "min_child_weight": 8, "subsample": 0.8, "colsample_bytree": 0.8,
    }),
)


def _fit(frame, features):
    models = []
    for spec in SPECS:
        model = make_model(spec)
        model.fit(frame[features], frame.fighter_a_won)
        models.append(model)
    return models


def run(root: Path) -> dict:
    model_dir = root / "models/v4-prospective"
    report_path = root / "reports/v4_prospective/subgroup_comparison.json"
    development_path = root / "reports/v4_prospective/development_report.json"
    analysis = json.loads(report_path.read_text(encoding="utf-8"))
    development = json.loads(development_path.read_text(encoding="utf-8"))
    frame = decisive(load_or_build(root))
    priors = fit_division_priors(frame)
    candidate_frame = augment_low_history(frame, priors, prior_fights=4.0)
    baseline_features = feature_names("rich_548")
    candidate_features = v4_feature_names(candidate_frame)

    artifact = {
        "status": "prospective_candidate_not_production_champion",
        "version": "V4 prospective 2026-09-11",
        "training_end": str(frame.fight_date.max().date()),
        "future_holdout_start": "strictly after 2026-09-11",
        "gate": "use baseline branch if either fighter has zero prior UFC fights; V4 branch otherwise",
        "prior_fights": 4.0,
        "priors": priors,
        "baseline_features": baseline_features,
        "candidate_features": candidate_features,
        "weights": WEIGHTS,
        "baseline_models": _fit(frame, baseline_features),
        "candidate_models": _fit(candidate_frame, candidate_features),
    }
    artifact_path = model_dir / "ufc_predictor_v4_prospective.joblib"
    joblib.dump(artifact, artifact_path, compress=3)
    lock = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": artifact["status"],
        "architecture": {"gate": artifact["gate"], "prior_fights": 4.0, "weights": WEIGHTS, "components": [spec.name for spec in SPECS]},
        "selection_period": development["selection_period"],
        "confirmation_period": development["confirmation_period"],
        "confirmation_metrics": analysis["periods"]["confirmation"]["gated_candidate"],
        "verified_v2_1_champion_accuracy": 0.6269430051813472,
        "promotion": "not promoted; requires untouched fights after 2026-09-11",
        "training_rows": len(frame),
        "training_range": [str(frame.fight_date.min().date()), str(frame.fight_date.max().date())],
        "checksums": {
            "artifact_sha256": file_sha256(artifact_path),
            "v4_prospective_code_sha256": file_sha256(root / "src/ufc_predictor/v4_prospective.py"),
            "v4_analysis_sha256": file_sha256(report_path),
            "raw_fights_sha256": file_sha256(root / "data/raw/fights.csv"),
            "raw_fighters_sha256": file_sha256(root / "data/raw/fighters.csv"),
        },
    }
    lock_path = model_dir / "ARCHITECTURE_LOCK.json"
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    digest = hashlib.sha256(lock_path.read_bytes()).hexdigest().upper()
    (model_dir / "ARCHITECTURE_LOCK.sha256").write_text(f"{digest}  ARCHITECTURE_LOCK.json\n", encoding="utf-8")
    print(json.dumps(lock, indent=2))
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    run(Path(args.project_root).resolve())


if __name__ == "__main__":
    main()
