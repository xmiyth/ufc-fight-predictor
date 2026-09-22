"""Current-roster product filtering, isolated from predictive features."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .v2_data import stable_fighter_id


def derive_active_status(
    fights_path: str | Path,
    *,
    as_of: date,
    activity_window_days: int = 730,
    overrides_path: str | Path | None = None,
) -> pd.DataFrame:
    """Create a conservative UI filter; never use this table as a model input.

    Recent participation is only a proxy for an active UFC roster. Reviewed
    overrides can correct known releases, retirements, injuries, and newcomers.
    """
    fights = pd.read_csv(fights_path, sep=";", low_memory=False)
    fights["date"] = pd.to_datetime(fights["event_date"], format="%d/%m/%Y", errors="coerce")
    cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=activity_window_days)
    recent = fights.loc[(fights["date"] <= pd.Timestamp(as_of)) & (fights["date"] >= cutoff)]
    appearances = pd.concat(
        [
            recent[["red_fighter_name", "date"]].rename(columns={"red_fighter_name": "fighter_name"}),
            recent[["blue_fighter_name", "date"]].rename(columns={"blue_fighter_name": "fighter_name"}),
        ],
        ignore_index=True,
    )
    latest = appearances.groupby("fighter_name", as_index=False)["date"].max()
    latest["fighter_id"] = latest["fighter_name"].map(stable_fighter_id)
    latest["is_active"] = True
    latest["status_source"] = f"fight within {activity_window_days} days"

    if overrides_path is not None and Path(overrides_path).exists():
        overrides = pd.read_csv(overrides_path).fillna("")
        required = {"fighter_name", "is_active", "reason", "verified_at"}
        if missing := required - set(overrides.columns):
            raise RuntimeError(f"Active-status overrides missing: {sorted(missing)}")
        for row in overrides.itertuples(index=False):
            fighter_id = stable_fighter_id(row.fighter_name)
            value = str(row.is_active).strip().casefold() in {"1", "true", "yes", "active"}
            mask = latest["fighter_id"] == fighter_id
            if mask.any():
                latest.loc[mask, ["is_active", "status_source"]] = [
                    value,
                    f"reviewed override: {row.reason}",
                ]
            else:
                latest.loc[len(latest)] = {
                    "fighter_name": row.fighter_name,
                    "date": pd.NaT,
                    "fighter_id": fighter_id,
                    "is_active": value,
                    "status_source": f"reviewed override: {row.reason}",
                }
    return latest.sort_values("fighter_name", key=lambda values: values.str.casefold())
