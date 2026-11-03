import os
from datetime import date, datetime, timedelta, timezone

import numpy as np
import psycopg
import pytest

import jobs.model_compare as mc
from jobs import drift
from jobs.common import SCHEMA as BASE_SCHEMA

DSN = os.environ.get("TEST_DATABASE_URL")
PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY, story_id BIGINT, model_name TEXT NOT NULL, model_version TEXT NOT NULL,
    role TEXT NOT NULL, prob DOUBLE PRECISION NOT NULL, latency_ms REAL, source TEXT NOT NULL DEFAULT 'api',
    request_id TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
"""


def synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    rows, items = [], {}
    for i in range(n):
        hit = rng.random() < 0.05
        signal = rng.random() * 0.4 + (0.3 if hit else 0)
        champion = min(signal + 0.4, 0.99)
        challenger = signal * 0.12
        late = (not hit) and rng.random() < 0.03
        rows.append((i, date(2026, 11, 1 + i % 3), champion, challenger, int(hit)))
        items[i] = {"id": i, "score": 150 if (hit or late) else 4}
    return rows, items


def test_calibrated_model_wins_on_brier_and_ece_while_ranking_matches():
    rows, items = synthetic()
    res = {(r["label_def"], r["role"]): r for r in mc.compare(rows, items)}
    assert set(res) == {("24h", "champion"), ("24h", "challenger"), ("now", "champion"), ("now", "challenger")}
    for label in ("24h", "now"):
        champ, chall = res[(label, "champion")], res[(label, "challenger")]
        assert chall["brier"] < champ["brier"] / 2 and chall["ece"] < champ["ece"] / 2
        assert chall["roc_auc"] == pytest.approx(champ["roc_auc"], abs=1e-9)
        assert chall["p_at_20_day"] == pytest.approx(champ["p_at_20_day"])
    assert res[("now", "champion")]["hits"] > res[("24h", "champion")]["hits"]


def test_missing_items_and_single_class_do_not_crash():
    rows, _ = synthetic(50)
    out = mc.compare(rows, {})
    assert {r["label_def"] for r in out} <= {"24h"}
    assert mc.compare([(1, date(2026, 11, 1), 0.5, 0.1, 0)] * 5, {}) == []


def test_ece_and_precision_at_k_helpers():
    assert drift.expected_calibration_error([0, 1] * 50, [0.5] * 100) == pytest.approx(0.0, abs=1e-9)
    assert drift.expected_calibration_error([0] * 100, [0.9] * 100) == pytest.approx(0.9)
    assert drift.precision_at_k_per_day([1, 1, 1, 2, 2], [1, 0, 0, 1, 1], [0.9, 0.8, 0.1, 0.5, 0.4], k=2) == pytest.approx(0.75)


@pytest.mark.skipif(not DSN, reason="set TEST_DATABASE_URL")
def test_main_stores_rows_for_both_models(monkeypatch):
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS model_compare, label_audit, monitor_model, monitor_psi, monitor_runs, predictions, labels, snapshots, stories CASCADE")
        conn.execute(BASE_SCHEMA)
        conn.execute(PREDICTIONS)
        rows, items = synthetic(300, seed=2)
        now = datetime.now(timezone.utc)
        for sid, _, champ, chall, label in rows:
            sid += 1
            conn.execute("INSERT INTO stories (id, title, kind, posted_at) VALUES (%s, 't', 'story', %s)", (sid, now - timedelta(hours=40 + sid % 30)))
            conn.execute("INSERT INTO labels (story_id, score_24h, label, status, age_hours) VALUES (%s, 5, %s, 'ok', 24.5)", (sid, bool(label)))
            conn.execute("INSERT INTO predictions (story_id, model_name, model_version, role, prob) VALUES (%s, 'frontpage', '1', 'champion', %s)", (sid, champ))
            conn.execute("INSERT INTO predictions (story_id, model_name, model_version, role, prob) VALUES (%s, 'frontpage', '3', 'challenger', %s)", (sid, chall))
    monkeypatch.setattr(mc, "connect", lambda: psycopg.connect(DSN))
    monkeypatch.setattr(mc, "fetch_items", lambda ids: [{"id": i, "score": 150 if i % 20 == 0 else 3} for i in ids])
    mc.main()
    with psycopg.connect(DSN) as conn:
        got = conn.execute("SELECT label_def, role, model_version, n FROM model_compare ORDER BY 1, 2").fetchall()
    assert got == [("24h", "challenger", "3", 300), ("24h", "champion", "1", 300),
                   ("now", "challenger", "3", 300), ("now", "champion", "1", 300)]
