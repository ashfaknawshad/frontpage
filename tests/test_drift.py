from datetime import datetime, timezone

import numpy as np
import pytest

from jobs import drift
from jobs.monitor import decide


def stories(n, title_len=40, text_share=0.1, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        text = rng.random() < text_share
        title = ("x" * int(rng.normal(title_len, 8))).strip() or "x"
        if rng.random() < 0.2:
            title += " 2025"
        rows.append((title, None if text else f"https://site{i % 7}.com/p", None if text else f"site{i % 7}.com",
                     "show" if rng.random() < 0.1 else "story",
                     datetime(2026, 11, 1 + int(rng.integers(0, 7)), int(rng.integers(0, 24)), tzinfo=timezone.utc)))
    return rows


def test_psi_zero_for_identical_and_large_for_shifted():
    p = np.array([0.2, 0.3, 0.5])
    assert drift.psi(p, p) == pytest.approx(0.0, abs=1e-12)
    assert drift.psi(p, np.array([0.6, 0.3, 0.1])) > 0.2


def test_roc_auc_matches_pairwise_definition_including_ties():
    rng = np.random.default_rng(3)
    y = (rng.random(300) < 0.2).astype(int)
    p = np.round(rng.random(300) * 0.7 + y * 0.3, 1)
    pos, neg = p[y == 1], p[y == 0]
    expected = (np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg))
    assert drift.roc_auc(y, p) == pytest.approx(expected)
    assert drift.roc_auc(np.zeros(5, dtype=int), np.arange(5)) is None


def test_average_precision_known_value_and_sklearn_agreement():
    y, p = [1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]
    assert drift.average_precision(y, p) == pytest.approx((1 + 2 / 3) / 2)
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(4)
    y = (rng.random(400) < 0.15).astype(int)
    p = np.round(rng.random(400) * 0.6 + y * 0.25, 2)
    assert drift.average_precision(y, p) == pytest.approx(sk.average_precision_score(y, p))
    assert drift.roc_auc(y, p) == pytest.approx(sk.roc_auc_score(y, p))


def test_extract_treats_nan_and_empty_urls_as_text_posts():
    row = datetime(2026, 11, 2, 14, tzinfo=timezone.utc)
    cols = drift.extract([("Hi", float("nan"), float("nan"), "story", row), ("Hi", "", None, "ask", row),
                          ("Yo (2025)?", "https://a.com/x", "a.com", "story", row)])
    assert cols["text_post"] == [1, 1, 0]
    assert cols["domain"] == ["(text post)", "(text post)", "a.com"]
    assert cols["has_year_tag"] == [0, 0, 1] and cols["has_question"] == [0, 0, 1]
    assert cols["dow"] == ["0"] * 3 and cols["hour_block"] == ["4"] * 3


def test_compare_is_quiet_on_same_distribution_and_flags_a_shift():
    spec = drift.profile(drift.extract(stories(6000, seed=1)))
    same = drift.compare(spec, drift.extract(stories(3000, seed=2)))
    assert all(v["psi"] < drift.PSI_ALERT for v in same.values()), {k: v["psi"] for k, v in same.items()}
    shifted = drift.compare(spec, drift.extract(stories(3000, title_len=75, text_share=0.5, seed=3)))
    flagged = {k for k, v in shifted.items() if v["psi"] >= drift.PSI_ALERT}
    assert {"title_len", "text_post"} <= flagged
    assert "->" in shifted["text_post"]["detail"]


def test_unseen_domains_fall_into_other():
    spec = drift.profile(drift.extract(stories(2000, seed=1)), top_domains=3)
    assert "other" in spec["features"]["domain"]["categories"]
    live = drift.extract([("t", "https://zzz.com", "zzz.com", "story", datetime(2026, 11, 2, tzinfo=timezone.utc))] * 50)
    assert drift.compare(spec, live)["domain"]["psi"] > drift.PSI_ALERT


@pytest.mark.parametrize("args,expected", [
    (dict(n_drifted=1, n_live=900, champion_roc=0.7, reference_roc=0.72, n_graded=500, previous_drift=False,
          days_since_trigger=None), (None, "none")),
    (dict(n_drifted=5, n_live=900, champion_roc=0.7, reference_roc=0.72, n_graded=500, previous_drift=False,
          days_since_trigger=None), (None, "none")),
    (dict(n_drifted=5, n_live=900, champion_roc=0.7, reference_roc=0.72, n_graded=500, previous_drift=True,
          days_since_trigger=None), ("drift", "would_trigger")),
    (dict(n_drifted=5, n_live=900, champion_roc=0.7, reference_roc=0.72, n_graded=500, previous_drift=True,
          days_since_trigger=2), ("drift", "cooldown")),
    (dict(n_drifted=0, n_live=900, champion_roc=0.60, reference_roc=0.72, n_graded=500, previous_drift=False,
          days_since_trigger=None), ("performance", "would_trigger")),
    (dict(n_drifted=0, n_live=900, champion_roc=0.60, reference_roc=0.72, n_graded=100, previous_drift=False,
          days_since_trigger=None), (None, "none")),
    (dict(n_drifted=9, n_live=100, champion_roc=None, reference_roc=None, n_graded=0, previous_drift=True,
          days_since_trigger=None), (None, "none")),
])
def test_decision_rules(args, expected):
    assert decide(auto_retrain=False, **args) == expected


def test_auto_retrain_flag_turns_would_trigger_into_triggered():
    args = dict(n_drifted=5, n_live=900, champion_roc=0.7, reference_roc=0.72, n_graded=500, previous_drift=True,
                days_since_trigger=None)
    assert decide(auto_retrain=True, **args) == ("drift", "triggered")
