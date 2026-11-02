import os
from datetime import datetime, timedelta, timezone

import numpy as np
import psycopg
import pytest

import jobs.monitor as monitor
from jobs import drift
from jobs.common import SCHEMA as BASE_SCHEMA

DSN = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="set TEST_DATABASE_URL to run monitor tests against Postgres")

PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY, story_id BIGINT, model_name TEXT NOT NULL, model_version TEXT NOT NULL,
    role TEXT NOT NULL, prob DOUBLE PRECISION NOT NULL, latency_ms REAL, source TEXT NOT NULL DEFAULT 'api',
    request_id TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
"""


def seed(conn, n=700, anti=False, seed_=0):
    rng = np.random.default_rng(seed_)
    now = datetime.now(timezone.utc)
    for i in range(1, n + 1):
        posted = now - timedelta(hours=float(rng.uniform(26, 120)))
        hit = bool(rng.random() < 0.12)
        text = rng.random() < 0.5
        conn.execute("INSERT INTO stories (id, title, url, domain, kind, posted_at) VALUES (%s, %s, %s, %s, 'story', %s)",
                     (i, "w" * int(rng.integers(60, 90)), None if text else "https://x.com/a", None if text else "x.com", posted))
        conn.execute("INSERT INTO labels (story_id, score_24h, label, status, age_hours) VALUES (%s, %s, %s, 'ok', 24.5)",
                     (i, 150 if hit else 3, hit))
        signal = (0.3 if hit else 0.0) * (-1 if anti else 1)
        conn.execute("INSERT INTO predictions (story_id, model_name, model_version, role, prob) VALUES (%s, 'frontpage', '1', 'champion', %s)",
                     (i, float(rng.random() * 0.6 + 0.3 + signal)))


@pytest.fixture
def db(monkeypatch):
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS monitor_model, monitor_psi, monitor_runs, predictions, labels, snapshots, stories CASCADE")
        conn.execute(BASE_SCHEMA)
        conn.execute(PREDICTIONS)
    monkeypatch.setattr(monitor, "connect", lambda: psycopg.connect(DSN))
    monkeypatch.setattr(monitor, "MIN_LIVE", 100)
    monkeypatch.setattr(monitor, "MIN_GRADED", 100)
    monkeypatch.setattr(monitor, "champion_reference", lambda: ("1", 0.72))
    reference = drift.profile(drift.extract(
        [("t" * 30, "https://a.com", "a.com", "story", datetime(2025, 5, 5, 12, tzinfo=timezone.utc))] * 300))
    reference["base_rate"] = 0.047
    monkeypatch.setattr(monitor, "load_reference", lambda: reference)
    return DSN


def runs():
    with psycopg.connect(DSN) as conn:
        return conn.execute("SELECT drifted_features, drift, action, reason, champion_roc_auc, n_graded, base_rate_live "
                            "FROM monitor_runs ORDER BY id").fetchall()


def test_drift_needs_two_runs_then_triggers_then_respects_cooldown(db, monkeypatch):
    with psycopg.connect(DSN, autocommit=True) as conn:
        seed(conn)
    sent = []
    monkeypatch.setattr(monitor, "dispatch_retrain", lambda reason: sent.append(reason) or 204)
    monitor.main()
    monitor.main()
    monkeypatch.setattr(monitor, "AUTO_RETRAIN", True)
    monitor.main()
    monitor.main()
    rows = runs()
    assert [r[2] for r in rows] == ["none", "would_trigger", "triggered", "cooldown"]
    assert rows[1][0] >= 3 and rows[1][1] is True and rows[1][3] == "drift"
    assert sent == ["drift"]
    with psycopg.connect(DSN) as conn:
        assert conn.execute("SELECT count(*) FROM monitor_psi").fetchone()[0] == 4 * len(drift.FEATURES)
        role, version, n, roc = conn.execute("SELECT role, model_version, n, roc_auc FROM monitor_model LIMIT 1").fetchone()
    assert (role, version, n) == ("champion", "1", 700) and roc is not None


def test_performance_rule_fires_when_champion_ranks_backwards(db):
    with psycopg.connect(DSN, autocommit=True) as conn:
        seed(conn, anti=True)
    monitor.main()
    drifted, drift_flag, action, reason, roc, n_graded, base = runs()[0]
    assert (reason, action) == ("performance", "would_trigger")
    assert roc < 0.5 < 0.72 and n_graded == 700 and 0.05 < base < 0.2


def test_dispatch_failure_is_recorded_not_raised(db, monkeypatch):
    with psycopg.connect(DSN, autocommit=True) as conn:
        seed(conn, anti=True)
    monkeypatch.setattr(monitor, "AUTO_RETRAIN", True)

    def boom(reason):
        raise OSError("network down")

    monkeypatch.setattr(monitor, "dispatch_retrain", boom)
    monitor.main()
    assert runs()[0][2] == "dispatch_failed"
