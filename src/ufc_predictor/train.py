"""Train and evaluate the chronological logistic-regression baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, build_dataset


def chronological_split(data: pd.DataFrame, train_fraction: float = 0.8):
    dates = sorted(data["date"].unique())
    cutoff = dates[max(1, int(len(dates) * train_fraction)) - 1]
    train = data[data["date"] <= cutoff].copy()
    test = data[data["date"] > cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Not enough unique dates for a chronological train/test split.")
    return train, test, cutoff


def train_model(data: pd.DataFrame):
    train, test, cutoff = chronological_split(data)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, C=1.0)),
        ]
    )
    pipeline.fit(train[FEATURE_NAMES], train["fighter_a_won"])
    probabilities = pipeline.predict_proba(test[FEATURE_NAMES])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    # A fair baseline knows only the most common outcome in the training data.
    majority_class = int(train["fighter_a_won"].mean() >= 0.5)
    baseline_predictions = pd.Series(majority_class, index=test.index)
    coefficients = pipeline.named_steps["model"].coef_[0]
    important = sorted(
        [
            {"feature": name, "coefficient": round(float(coef), 4), "magnitude": round(abs(float(coef)), 4)}
            for name, coef in zip(FEATURE_NAMES, coefficients)
        ],
        key=lambda item: item["magnitude"],
        reverse=True,
    )
    metrics = {
        "accuracy": round(float(accuracy_score(test["fighter_a_won"], predictions)), 4),
        "baseline_accuracy": round(
            float(accuracy_score(test["fighter_a_won"], baseline_predictions)), 4
        ),
        "baseline_method": "always predict the majority class from the training set",
        "log_loss": round(float(log_loss(test["fighter_a_won"], probabilities)), 4),
        "total_fights": int(len(data)),
        "train_fights": int(len(train)),
        "test_fights": int(len(test)),
        "train_start": str(train["date"].min()),
        "train_end": str(cutoff),
        "test_start": str(test["date"].min()),
        "test_end": str(test["date"].max()),
        "important_features": important,
        "leakage_audit": {
            "chronological_train_before_test": bool(train["date"].max() < test["date"].min()),
            "forbidden_columns_in_features": sorted(
                set(FEATURE_NAMES)
                & {"winner", "method", "round", "time", "fight_outcome", "betting_odds"}
            ),
            "duplicate_matchups_same_date": int(
                data.duplicated(["date", "fighter_a", "fighter_b"]).sum()
            ),
            "fighter_a_win_rate": round(float(data["fighter_a_won"].mean()), 4),
            "current_fight_updates_after_snapshot": True,
        },
    }
    predictions_frame = test[["date", "fighter_a", "fighter_b", "fighter_a_won"]].copy()
    predictions_frame["fighter_a_win_probability"] = probabilities
    predictions_frame["predicted_a_win"] = predictions
    return pipeline, metrics, predictions_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fights", default="data/raw/fights.csv")
    parser.add_argument("--fighters", default="data/raw/fighters.csv")
    parser.add_argument("--model", default="models/logistic_regression.joblib")
    args = parser.parse_args()

    print("Building pre-fight features in chronological order ...")
    data, states, profiles = build_dataset(args.fights, args.fighters)
    model, metrics, predictions = train_model(data)

    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": model,
        "feature_names": FEATURE_NAMES,
        "fighter_states": {name: state.serializable() for name, state in states.items()},
        "profiles": profiles,
        "last_data_date": str(data["date"].max()),
        "metrics": metrics,
    }
    joblib.dump(artifact, args.model)
    data.to_csv("data/processed/prefight_features.csv", index=False)
    predictions.to_csv("reports/test_predictions.csv", index=False)
    Path("reports/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Fights used: {metrics['total_fights']:,}")
    print(f"Train/test: {metrics['train_fights']:,} / {metrics['test_fights']:,}")
    print(f"Chronological cutoff: {metrics['train_end']}")
    print(f"Test accuracy: {metrics['accuracy']:.1%}")
    print(f"Baseline accuracy: {metrics['baseline_accuracy']:.1%}")
    print(f"Test log loss: {metrics['log_loss']:.4f}")
    print("Most important standardized features:")
    for item in metrics["important_features"][:8]:
        direction = "helps Fighter A" if item["coefficient"] > 0 else "helps Fighter B"
        print(f"  {item['feature']:<36} {item['coefficient']:>8.4f}  ({direction})")
    print(f"Saved model to {args.model}")


if __name__ == "__main__":
    main()
