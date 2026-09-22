"""Inference adapter for the frozen V2.1 ensemble (no fitting or evaluation)."""

from functools import lru_cache
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .v2_data import stable_fighter_id
from .v2_features import (
    V2FighterState, build_v2_dataset, division_slug, fighter_snapshot, matchup_features,
)


@lru_cache(maxsize=1)
def histories(root: Path, cutoff: str):
    sources = [root / 'data/raw/fights.csv', root / 'data/raw/fighters.csv',
               Path(__file__).with_name('v2_features.py'), Path(__file__).with_name('features.py'),
               Path(__file__).with_name('v2_data.py')]
    digest = hashlib.sha256(cutoff.encode())
    for source in sources:
        digest.update(source.read_bytes())
    cache = root / 'data/processed/v2_serving_histories.joblib'
    signature = digest.hexdigest()
    if cache.exists():
        saved = joblib.load(cache)
        if saved['signature'] == signature:
            return saved['states'], saved['profiles']
    _, states, profiles = build_v2_dataset(
        sources[0], sources[1], include_labels=False, stop_after=cutoff,
    )
    joblib.dump(dict(signature=signature, states=states, profiles=profiles), cache, compress=3)
    return states, profiles


def feature_frame(root, artifact, first, second, as_of, bout_type, rounds):
    states, profiles = histories(root, artifact['trained_through'])
    division = division_slug(bout_type)
    snapshots = []
    for key in (first, second):
        identity = stable_fighter_id(key)
        state = states.get(identity, V2FighterState(identity, key))
        snapshots.append(fighter_snapshot(state, profiles.get(key, {}), as_of, division))
    values = matchup_features(*snapshots, bout_type, rounds)
    return pd.DataFrame([values])[artifact['feature_names']].astype(float)


def probability(artifact, frame):
    return sum(artifact['weights'][name] * model.predict_proba(frame)[:, 1]
               for name, model in artifact['components'].items())


def analyze(artifact, frame):
    # Local sensitivity of the full ensemble to replacing each feature with its
    # training median. These effects are explanatory, not an additive decomposition.
    names = artifact['feature_names']
    batch = pd.concat([frame] * (len(names) + 1), ignore_index=True)
    medians = artifact['components']['logistic'].named_steps['imputer'].statistics_
    for index, median in enumerate(medians):
        batch.iat[index + 1, index] = median
    probabilities = probability(artifact, batch)
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    logits = np.log(clipped / (1 - clipped))
    return float(probabilities[0]), dict(zip(names, (logits[0] - logits[1:]).tolist()))
