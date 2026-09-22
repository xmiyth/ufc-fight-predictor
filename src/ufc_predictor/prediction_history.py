"""Immutable first-published event prediction snapshots."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Any


class PredictionHistory:
    def __init__(self, path: Path):
        self.path = path
        self._lock = Lock()

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"predictions": []}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("predictions"), list):
            raise ValueError("Prediction history must contain a predictions list.")
        return payload

    def list_predictions(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._read()["predictions"])

    def get_prediction(
        self, event_id: str, fight_id: str, model_version: str
    ) -> dict[str, Any] | None:
        key = f"{event_id}:{fight_id}:{model_version}"
        with self._lock:
            return next(
                (
                    item for item in self._read()["predictions"]
                    if item["prediction_key"] == key
                ),
                None,
            )

    def lock_prediction(
        self,
        event: dict[str, Any],
        fight: dict[str, Any],
        prediction: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        key = f"{event['event_id']}:{fight['fight_id']}:{prediction['model_version']}"
        with self._lock:
            payload = self._read()
            for existing in payload["predictions"]:
                if existing["prediction_key"] == key:
                    return existing, False
            record = {
                "prediction_key": key,
                "event_id": event["event_id"],
                "event_name": event["event_name"],
                "event_date": event["event_date"],
                "fight_id": fight["fight_id"],
                "fighter_a": prediction["fighter_a"],
                "fighter_b": prediction["fighter_b"],
                "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
                "model_version": prediction["model_version"],
                "method_model_version": prediction.get("method_model_version"),
                "fighter_a_probability": prediction["fighter_a_probability"],
                "fighter_b_probability": prediction["fighter_b_probability"],
                "predicted_winner": prediction["predicted_winner"],
                "predicted_method": prediction.get("predicted_method"),
                "actual_winner": None,
                "actual_method": None,
                "correct_winner": None,
                "correct_method": None,
                "prediction": prediction,
            }
            payload["predictions"].append(record)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temporary.replace(self.path)
            return record, True
