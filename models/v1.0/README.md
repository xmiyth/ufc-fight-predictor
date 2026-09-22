# Loading UFC Predictor Model V1.0

Install the versions in `requirements.txt`, then load the frozen artifact:

```python
import joblib

artifact = joblib.load("models/v1.0/ufc_predictor_v1_0.joblib")
model = artifact["pipeline"]
feature_names = artifact["feature_names"]
```

The artifact also contains frozen fighter-history state, fighter profiles,
evaluation metadata, and the June 27, 2026 data cutoff. Use the source snapshot
in `source/` to reconstruct matchup features consistently with V1.0.

Before relying on this artifact, verify its SHA-256 against `MANIFEST.json`.

