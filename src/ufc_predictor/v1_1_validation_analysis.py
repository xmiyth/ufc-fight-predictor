"""Pre-lockbox feature-group and model-family analysis for V1.1.

This module deliberately stops at 2022-10-29.  It may be rerun without looking
at, or changing, the fixed 2022-11-05+ final-test result.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .v1_1_advanced import COMPONENTS, TRAIN_END, choose_ensemble, load_oof
from .v2_evaluate import Experiment, decisive, feature_names, load_or_build_dataset, make_model, metrics
from .v2_features import FEATURE_GROUPS


def _unique(names: list[str]) -> list[str]:
    return list(dict.fromkeys(names))


def _validation_metrics(train: pd.DataFrame, validation: pd.DataFrame, names: list[str]) -> dict:
    experiment = Experiment("ablation", "logistic", "custom", {"C": 1.0})
    model = make_model(experiment)
    model.fit(train[names], train.fighter_a_won)
    probability = model.predict_proba(validation[names])[:, 1]
    return metrics(validation.fighter_a_won.to_numpy(), probability)


def run(root: Path) -> dict:
    frame = decisive(load_or_build_dataset(root))
    frame = frame.loc[frame.fight_date <= TRAIN_END].copy()
    train = frame.loc[frame.fight_date.dt.year <= 2020].copy()
    validation = frame.loc[frame.fight_date.dt.year >= 2021].copy()

    # A small, interpretable career-average baseline.  Each row below adds only
    # the named information group, while the final row uses the complete rich
    # 548-day feature set selected during the earlier walk-forward development.
    core = feature_names("core")
    baseline = [name for name in core if not any(token in name for token in (
        "recent", "streak", "days_since", "elo", "scheduled", "five_round",
    ))]
    rich = feature_names("rich_548")
    recent = [name for name in rich if any(token in name for token in (
        "recent", "last_1", "streak", "form_", "decay_548d", "previous_12m", "previous_24m",
    ))]
    opponent = [name for name in FEATURE_GROUPS["opponent_quality"] if "elo" not in name]
    opponent += FEATURE_GROUPS["opponent_adjusted_performance"]
    elo = FEATURE_GROUPS["elo_rating"]
    style = FEATURE_GROUPS["matchup_interactions"]
    age_decline = FEATURE_GROUPS["age"] + FEATURE_GROUPS["activity"] + [
        name for name in rich if any(token in name for token in (
            "recent_losses", "recent_ko", "finishes_suffered", "loss_streak",
        ))
    ]
    # The rich dataset represents debutant experience with the leakage-safe
    # UFC-fight-count difference. Absolute debut/low-sample flags live in the
    # separately evaluated low-data model because symmetric difference features
    # cannot represent "both debutants".
    debutant = [name for name in rich if "ufc_fights" in name]
    size = FEATURE_GROUPS["physical"] + FEATURE_GROUPS["context"]
    groups = {
        "baseline": baseline,
        "baseline_plus_recent_form": _unique(baseline + recent),
        "baseline_plus_opponent_strength": _unique(baseline + opponent),
        "baseline_plus_elo": _unique(baseline + elo),
        "baseline_plus_style_interactions": _unique(baseline + style),
        "baseline_plus_age_decline": _unique(baseline + age_decline),
        "baseline_plus_debutant_experience": _unique(baseline + debutant),
        "baseline_plus_size_weight": _unique(baseline + size),
        "full_rich_548": rich,
    }
    ablations = []
    for name, names in groups.items():
        score = _validation_metrics(train, validation, names)
        ablations.append({"feature_group": name, "features": len(names), **score})
    baseline_score = ablations[0]
    for row in ablations:
        row["accuracy_change_vs_baseline"] = row["accuracy"] - baseline_score["accuracy"]
        row["log_loss_change_vs_baseline"] = row["log_loss"] - baseline_score["log_loss"]

    oof = load_oof(root, decisive(load_or_build_dataset(root)))
    architecture, trials = choose_ensemble(oof)
    tune = oof.loc[oof.fight_date.dt.year <= 2020]
    confirm = oof.loc[oof.fight_date.dt.year >= 2021]
    families = []
    for name in [*COMPONENTS, "elo"]:
        families.append({
            "model": name,
            "tuning": metrics(tune.fighter_a_won.to_numpy(), tune[name].to_numpy()),
            "confirmation": metrics(confirm.fighter_a_won.to_numpy(), confirm[name].to_numpy()),
        })

    report = {
        "policy": {
            "development_only": True,
            "train_period": [str(train.fight_date.min().date()), str(train.fight_date.max().date())],
            "confirmation_period": [str(validation.fight_date.min().date()), str(validation.fight_date.max().date())],
            "fixed_final_test_used": False,
            "imputation": "median fitted on each training partition; missing indicators added",
        },
        "feature_group_validation": ablations,
        "model_family_walk_forward": families,
        "ensemble_trials": trials,
        "selected_before_final_test": architecture,
        "data_limitations": {
            "short_notice": "not available",
            "point_in_time_professional_record": "not available; current profile totals would leak future information",
            "promotion_history": "not available",
            "regional_elo": "not constructible reliably from the supplied UFC-only fight history",
        },
    }
    report_path = root / "reports" / "v1_1_validation_analysis.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(ablations).to_csv(root / "reports" / "v1_1_feature_ablation.csv", index=False)
    pd.DataFrame([
        {"model": row["model"], **{f"tuning_{k}": v for k, v in row["tuning"].items()},
         **{f"confirmation_{k}": v for k, v in row["confirmation"].items()}}
        for row in families
    ]).to_csv(root / "reports" / "v1_1_model_validation.csv", index=False)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run(Path(".").resolve())
