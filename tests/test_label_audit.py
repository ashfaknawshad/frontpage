import os
from datetime import datetime, timedelta, timezone

import numpy as np
import psycopg
import pytest

import jobs.label_audit as audit_job
from jobs.common import SCHEMA as BASE_SCHEMA

DSN = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="set TEST_DATABASE_URL to run audit tests against Postgres")

PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY, story_id BIGINT, model_name TEXT NOT NULL, model_version TEXT NOT NULL,
    role TEXT NOT NULL, prob DOUBLE PRECISION NOT NULL, latency_ms REAL, source TEXT NOT NULL DEFAULT 'api',
    request_id TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
"""


def test_audit_counts_late_risers_and_compares_both_label_definitions():
    rows = [(i, 0.9 if i < 20 else 0.1, 1 if i < 10 else 0) for i in range(200)]
    items = {i: {"id": i, "score": 150 if i < 20 else 5} for i in range(200)}
    items[3] = {"id": 3, "score": 7}
    items[500] = {"id": 500, "score": 999}
    out = audit_job.audit(rows, items)
    assert out["n"] == 200
    assert out["hits_24h"] == 10 and out["hits_final"] == 19 and out["late_risers"] == 10
    assert out["auc_final"] > out["auc_24h"] > 0.5
    assert out["auc_24h_se"] > 0


def test_audit_skips_dead_and_missing_items():
    rows = [(1, 0.5, 0), (2, 0.6, 1), (3, 0.7, 0)]
    items = {1: {"id": 1, "score": 3, "dead": True}, 2: {"id": 2, "score": None}, 3: {"id": 3, "score": 120}}
    assert audit_job.audit(rows, items)["n"] == 1
    assert audit_job.audit(rows, {}) is None


def test_main_writes_one_row(monkeypatch):
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS label_audit, monitor_model, monitor_psi, monitor_runs, predictions, labels, snapshots, stories CASCADE")
        conn.execute(BASE_SCHEMA)
        conn.execute(PREDICTIONS)
        rng = np.random.default_rng(0)
        now = datetime.now(timezone.utc)
        for i in range(1, 301):
            posted = now - timedelta(days=float(rng.uniform(4, 10)))
            hit = bool(rng.random() < 0.1)
            conn.execute("INSERT INTO stories (id, title, kind, posted_at) VALUES (%s, 't', 'story', %s)", (i, posted))
            conn.execute("INSERT INTO labels (story_id, score_24h, label, status, age_hours) VALUES (%s, %s, %s, 'ok', 24.5)", (i, 5, hit))
            conn.execute("INSERT INTO predictions (story_id, model_name, model_version, role, prob) VALUES (%s, 'frontpage', '1', 'champion', %s)",
                         (i, float(rng.random() * 0.5 + (0.3 if hit else 0))))
    monkeypatch.setattr(audit_job, "connect", lambda: psycopg.connect(DSN))
    monkeypatch.setattr(audit_job, "fetch_items", lambda ids: [{"id": i, "score": 120 if i % 7 == 0 else 4} for i in ids])
    audit_job.main()
    with psycopg.connect(DSN) as conn:
        n, version, late = conn.execute("SELECT n, champion_version, late_risers FROM label_audit").fetchone()
    assert n == 300 and version == "1" and late > 0
