"""Validation-only ensemble selection for the accuracy-focused V2 experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .v2_evaluate import decisive, load_or_build_dataset, metrics


COMPONENTS = {
    "logistic": "logistic_rich_548_c1",
    "core_logistic": "logistic_core_c0.1",
    "xgboost": "xgboost_rich548",
    "gradient_boosting": "gradient_boosting_rich548",
    "random_forest": "random_forest_rich548",
}


def _evaluate(table: pd.DataFrame, probability: np.ndarray) -> dict[str, float]:
    return metrics(table.fighter_a_won.to_numpy(), probability)


def run(root: Path) -> dict:
    report_dir = root / "reports" / "v2_accuracy"
    merged: pd.DataFrame | None = None
    for label, experiment in COMPONENTS.items():
        path = report_dir / f"oof_{experiment}.csv"
        frame = pd.read_csv(path, parse_dates=["fight_date"])
        columns = ["fight_id", "fight_date", "fighter_a_won", "probability"]
        frame = frame[columns].rename(columns={"probability": label})
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(
                frame[["fight_id", label]], on="fight_id", validate="one_to_one"
            )
    assert merged is not None
    point_in_time = decisive(load_or_build_dataset(root))
    merged = merged.merge(
        point_in_time[["fight_id", "elo_diff"]], on="fight_id", validate="one_to_one"
    )
    merged["elo"] = 1.0 / (1.0 + 10.0 ** (-merged.elo_diff.fillna(0.0) / 400.0))
    tune = merged.loc[merged.fight_date.dt.year <= 2022].copy()
    confirmation = merged.loc[merged.fight_date.dt.year >= 2023].copy()
    base_tune = _evaluate(tune, tune.logistic.to_numpy())
    base_confirmation = _evaluate(confirmation, confirmation.logistic.to_numpy())

    specifications: list[tuple[str, tuple[str, ...], tuple[float, ...]]] = []
    for partner in ("xgboost", "gradient_boosting", "elo", "core_logistic"):
        for logistic_weight in (0.25, 0.5, 0.75):
            specifications.append(
                (
                    f"logistic+{partner}",
                    ("logistic", partner),
                    (logistic_weight, 1.0 - logistic_weight),
                )
            )
    for tree in ("xgboost", "gradient_boosting", "random_forest"):
        for first in (0.2, 0.4, 0.6):
            for second in (0.2, 0.4, 0.6):
                third = 1.0 - first - second
                if third >= 0.2 - 1e-9:
                    specifications.append(
                        (
                            f"logistic+{tree}+elo",
                            ("logistic", tree, "elo"),
                            (first, second, third),
                        )
                    )

    candidates = []
    for family, names, weights in specifications:
        tune_probability = sum(
            weight * tune[name].to_numpy() for name, weight in zip(names, weights)
        )
        tune_metrics = _evaluate(tune, tune_probability)
        candidates.append({
            "family": family,
            "components": list(names),
            "weights": list(weights),
            "tune": tune_metrics,
            "eligible_on_tuning": (
                tune_metrics["accuracy"] > base_tune["accuracy"]
                and tune_metrics["log_loss"] < base_tune["log_loss"]
                and tune_metrics["brier"] < base_tune["brier"]
            ),
        })

    chosen_by_family = []
    for family, rows in itertools.groupby(
        sorted(candidates, key=lambda row: row["family"]), key=lambda row: row["family"]
    ):
        eligible = [row for row in rows if row["eligible_on_tuning"]]
        if not eligible:
            continue
        choice = sorted(
            eligible,
            key=lambda row: (-row["tune"]["accuracy"], row["tune"]["log_loss"], row["tune"]["brier"]),
        )[0]
        probability = sum(
            weight * confirmation[name].to_numpy()
            for name, weight in zip(choice["components"], choice["weights"])
        )
        choice["confirmation"] = _evaluate(confirmation, probability)
        choice["kept_after_confirmation"] = (
            choice["confirmation"]["accuracy"] > base_confirmation["accuracy"]
            and choice["confirmation"]["log_loss"] < base_confirmation["log_loss"]
            and choice["confirmation"]["brier"] < base_confirmation["brier"]
        )
        chosen_by_family.append(choice)

    eligible_confirmation = [
        row for row in chosen_by_family if row["kept_after_confirmation"]
    ]
    chosen = (
        sorted(
            eligible_confirmation,
            key=lambda row: (-row["confirmation"]["accuracy"], row["confirmation"]["log_loss"]),
        )[0]
        if eligible_confirmation else None
    )
    standalone_confirmation = {
        name: _evaluate(confirmation, confirmation[name].to_numpy())
        for name in [*COMPONENTS, "elo"]
    }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "weight_tuning_period": "2019-2022 walk-forward out-of-fold predictions",
        "confirmation_period": "2023-2024 walk-forward out-of-fold predictions",
        "rule": "keep only if accuracy improves and both log loss and Brier decrease on tuning and confirmation",
        "base_logistic_tuning": base_tune,
        "base_logistic_confirmation": base_confirmation,
        "standalone_confirmation": standalone_confirmation,
        "chosen_by_family": chosen_by_family,
        "selected_ensemble": chosen,
    }
    (report_dir / "ensemble_analysis.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    run(Path(args.project_root).resolve())


if __name__ == "__main__":
    main()
