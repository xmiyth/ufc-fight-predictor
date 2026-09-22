"""Compare the selected V4 research candidate with V2 on allowed periods."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .v2_evaluate import metrics
from .v3_evaluate import decisive, load_or_build
from .v4_prospective import ALL_YEARS, CONFIRMATION_YEARS, DEVELOPMENT_YEARS, _baseline_predictions, _history


def _subgroups(frame: pd.DataFrame) -> dict[str, pd.Series]:
    a, b = _history(frame)
    return {
        "involves_debutant": (a == 0) | (b == 0),
        "involves_1_or_2_no_debutant": (((a.between(1, 2)) | (b.between(1, 2))) & (a > 0) & (b > 0)),
        "both_3_plus": (a >= 3) & (b >= 3),
    }


def _accuracy(y: np.ndarray, probability: np.ndarray) -> float:
    return float(((probability >= 0.5).astype(int) == y).mean())


def run(root: Path) -> dict:
    output = root / "reports/v4_prospective"
    selected = pd.read_csv(output / "selected_oof_predictions.csv", parse_dates=["fight_date"])
    baseline = _baseline_predictions(root)[["fight_id", "fight_date", "fighter_a_won", "probability"]]
    baseline = baseline.loc[baseline.fight_date.dt.year.isin(ALL_YEARS)].rename(columns={"probability": "baseline_probability"})
    selected = selected.rename(columns={"probability": "candidate_probability"})
    predictions = baseline.merge(selected, on=["fight_id", "fight_date", "fighter_a_won"], validate="one_to_one")
    frame = decisive(load_or_build(root))
    frame = frame.loc[frame.fight_date.dt.year.isin(ALL_YEARS)]
    joined = frame.merge(predictions, on=["fight_id", "fight_date", "fighter_a_won"], validate="one_to_one")
    groups = _subgroups(joined)
    # This gate is selected from development subgroup evidence only: V4 was
    # worse for debutants and better for both non-debutant experience groups.
    joined["gated_probability"] = joined.candidate_probability.where(
        ~groups["involves_debutant"], joined.baseline_probability
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gate_selected_on_development": "V2 for any debutant; V4 otherwise",
        "periods": {}, "changed_predictions": {},
    }
    for period, years in (("development", DEVELOPMENT_YEARS), ("confirmation", CONFIRMATION_YEARS)):
        period_mask = joined.fight_date.dt.year.isin(years)
        y = joined.loc[period_mask, "fighter_a_won"].to_numpy()
        base_p = joined.loc[period_mask, "baseline_probability"].to_numpy()
        cand_p = joined.loc[period_mask, "candidate_probability"].to_numpy()
        gated_p = joined.loc[period_mask, "gated_probability"].to_numpy()
        report["periods"][period] = {
            "fights": int(period_mask.sum()),
            "baseline": metrics(y, base_p),
            "candidate": metrics(y, cand_p),
            "gated_candidate": metrics(y, gated_p),
            "subgroups": {},
        }
        for name, group_mask in groups.items():
            mask = period_mask & group_mask
            group_y = joined.loc[mask, "fighter_a_won"].to_numpy()
            report["periods"][period]["subgroups"][name] = {
                "fights": int(mask.sum()),
                "baseline_accuracy": _accuracy(group_y, joined.loc[mask, "baseline_probability"].to_numpy()),
                "candidate_accuracy": _accuracy(group_y, joined.loc[mask, "candidate_probability"].to_numpy()),
            }
        base_pick = base_p >= 0.5
        cand_pick = cand_p >= 0.5
        gated_pick = gated_p >= 0.5
        changed = base_pick != cand_pick
        base_correct = base_pick == y
        cand_correct = cand_pick == y
        report["changed_predictions"][period] = {
            "changed": int(changed.sum()),
            "became_correct": int((changed & cand_correct).sum()),
            "became_incorrect": int((changed & base_correct).sum()),
            "gated_changed": int((base_pick != gated_pick).sum()),
            "gated_became_correct": int(((base_pick != gated_pick) & (gated_pick == y)).sum()),
            "gated_became_incorrect": int(((base_pick != gated_pick) & base_correct).sum()),
        }
    (output / "subgroup_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    run(Path(args.project_root).resolve())


if __name__ == "__main__":
    main()
