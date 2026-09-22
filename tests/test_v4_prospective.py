import numpy as np
import pandas as pd

from ufc_predictor.v4_prospective import augment_low_history, fit_division_priors


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "weight_class": ["lightweight", "lightweight", "lightweight"],
        "ufc_fights_mean": [2.0, 2.5, 0.0],
        "ufc_fights_diff": [0.0, -5.0, 0.0],
        "sig_accuracy_mean": [0.5, 0.25, 0.0],
        "sig_accuracy_diff": [0.0, -0.5, 0.0],
    })


def test_training_priors_exclude_debutant_zero_values():
    priors = fit_division_priors(_frame())
    assert priors["global"]["sig_accuracy"] == 0.5


def test_debutant_performance_is_missing_and_shrunk_to_prior():
    frame = _frame()
    priors = fit_division_priors(frame)
    result = augment_low_history(frame, priors, prior_fights=4.0)
    assert np.isnan(result.loc[1, "sig_accuracy_diff"])
    assert result.loc[1, "v4_debutant_diff"] == 1.0
    assert result.loc[1, "v4_sig_accuracy_shrunk_diff"] == 0.0
    assert result.loc[2, "v4_both_debutants"] == 1.0


def test_orientation_swap_negates_difference_features():
    frame = _frame().iloc[[1]].copy()
    priors = fit_division_priors(_frame())
    forward = augment_low_history(frame, priors, prior_fights=4.0).iloc[0]
    swapped = frame.copy()
    swapped["ufc_fights_diff"] *= -1
    swapped["sig_accuracy_diff"] *= -1
    reverse = augment_low_history(swapped, priors, prior_fights=4.0).iloc[0]
    assert forward["v4_debutant_diff"] == -reverse["v4_debutant_diff"]
    assert np.isclose(forward["v4_sig_accuracy_shrunk_diff"], -reverse["v4_sig_accuracy_shrunk_diff"])
