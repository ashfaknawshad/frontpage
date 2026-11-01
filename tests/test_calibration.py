import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from frontpage.metrics import evaluate, expected_calibration_error
from frontpage.train import INPUT_COLUMNS, fit_calibrator, predict_prob


class FakeModel:
    """Stands in for the pipeline: the raw score is smuggled through the 'title' column."""

    def decision_function(self, frame):
        return frame["title"].to_numpy(dtype=float)

    def predict_proba(self, frame):
        p = 1 / (1 + np.exp(-frame["title"].to_numpy(dtype=float)))
        return np.column_stack([1 - p, p])


def make_frame(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1.5, n)
    y = rng.random(n) < 1 / (1 + np.exp(-(x * 0.7 - 3.2)))  # real rate far below what the raw score implies
    frame = pd.DataFrame({"title": x, "label": y.astype(int)})
    for col in INPUT_COLUMNS:
        if col not in frame:
            frame[col] = "t"
    return frame


def test_ece_zero_for_perfect_and_large_for_inflated():
    y = np.array([0, 0, 0, 1] * 500)
    assert expected_calibration_error(y, np.full(len(y), 0.25) + np.linspace(0, 1e-6, len(y))) < 0.01
    assert expected_calibration_error(y, np.full(len(y), 0.8) + np.linspace(0, 1e-6, len(y))) > 0.5


def test_platt_fixes_probabilities_without_changing_ranking():
    val, test = make_frame(seed=1), make_frame(seed=2)
    model = FakeModel()
    calibrator = fit_calibrator(model, val, "sigmoid")
    raw, cal = predict_prob(model, None, test), predict_prob(model, calibrator, test)
    assert (np.argsort(raw, kind="stable") == np.argsort(cal, kind="stable")).all()
    ev_raw = evaluate(test["label"], raw, pd.Series(pd.Timestamp("2025-10-01", tz="UTC")).repeat(len(test)), 20)
    ev_cal = evaluate(test["label"], cal, pd.Series(pd.Timestamp("2025-10-01", tz="UTC")).repeat(len(test)), 20)
    assert ev_cal["ece"] < ev_raw["ece"] / 5
    assert ev_cal["brier"] < ev_raw["brier"]
    assert abs(ev_cal["pr_auc"] - ev_raw["pr_auc"]) < 1e-3
    assert abs(ev_cal["mean_prob"] - ev_cal["base_rate"]) < 0.01


def test_none_calibration_is_passthrough():
    val = make_frame()
    assert fit_calibrator(FakeModel(), val, "none") is None
