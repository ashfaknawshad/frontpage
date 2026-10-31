import importlib
import os
import sys
import time
from pathlib import Path

import httpx
import psycopg
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
DSN = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="set TEST_DATABASE_URL to run gateway tests against Postgres")


def fake_predictor(version, fail=False):
    def handler(request):
        if fail:
            return httpx.Response(503, json={"detail": "down"})
        n = len(httpx.Response(200, content=request.content).json()["instances"])
        return httpx.Response(200, json={"predictions": [0.1 * (i + 1) for i in range(n)],
                                         "model_name": "frontpage", "model_version": version})
    return handler


def load_app(monkeypatch, mode="shadow", challenger=True, champion_fail=False, canary=0):
    monkeypatch.setenv("DATABASE_URL", DSN)
    monkeypatch.setenv("API_KEYS", "k-ingest,k-public")
    monkeypatch.setenv("ROUTING_MODE", mode)
    monkeypatch.setenv("CANARY_PERCENT", str(canary))
    monkeypatch.setenv("CHAMPION_URL", "http://champion/v1/models/frontpage:predict")
    monkeypatch.setenv("CHALLENGER_URL", "http://challenger/v1/models/frontpage:predict" if challenger else "")
    from prometheus_client import REGISTRY

    for collector in list(REGISTRY._collector_to_names):
        if any(name.startswith("gateway_") for name in REGISTRY._collector_to_names[collector]):
            REGISTRY.unregister(collector)
    gateway_dir = str(ROOT / "services" / "gateway")
    if gateway_dir not in sys.path:
        sys.path.insert(0, gateway_dir)
    sys.modules.pop("app", None)
    gw = importlib.import_module("app")
    routes = {"champion": fake_predictor("1", fail=champion_fail), "challenger": fake_predictor("2")}
    transport = httpx.MockTransport(lambda req: routes[req.url.host](req))
    gw.state["transport_client"] = httpx.AsyncClient(transport=transport)
    return gw


@pytest.fixture(autouse=True)
def clean_db():
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS predictions, labels, snapshots, stories CASCADE")
        conn.execute("""
            CREATE TABLE stories (id BIGINT PRIMARY KEY, title TEXT NOT NULL, url TEXT, domain TEXT, by TEXT,
                kind TEXT NOT NULL, posted_at TIMESTAMPTZ NOT NULL, ingested_at TIMESTAMPTZ NOT NULL DEFAULT now());
            CREATE TABLE labels (story_id BIGINT PRIMARY KEY, score_24h INTEGER, label BOOLEAN, status TEXT NOT NULL,
                age_hours REAL NOT NULL, labeled_at TIMESTAMPTZ NOT NULL DEFAULT now());
        """)
    yield


def rows(sql):
    with psycopg.connect(DSN) as conn:
        return conn.execute(sql).fetchall()


def wait_for(sql, n, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if rows(sql)[0][0] >= n:
            return True
        time.sleep(0.05)
    return False


def test_auth_required(monkeypatch):
    gw = load_app(monkeypatch)
    with TestClient(gw.app) as c:
        assert c.post("/predict", json={"title": "x"}).status_code == 401
        assert c.post("/predict", json={"title": "x"}, headers={"X-API-Key": "nope"}).status_code == 401


def test_batch_logs_champion_and_shadow(monkeypatch):
    gw = load_app(monkeypatch, mode="shadow")
    stories = [{"title": f"t{i}", "story_id": 100 + i, "source": "ingest",
                "posted_at": "2026-10-31T10:00:00Z"} for i in range(3)]
    with TestClient(gw.app) as c:
        r = c.post("/predict/batch", json={"stories": stories}, headers={"X-API-Key": "k-ingest"})
        assert r.status_code == 200
        body = r.json()
        assert body["role"] == "champion" and body["model_version"] == "1"
        assert [p["prob"] for p in body["predictions"]] == pytest.approx([0.1, 0.2, 0.3])
        assert wait_for("SELECT count(*) FROM predictions WHERE role='challenger'", 3)
        again = c.post("/predict/batch", json={"stories": stories}, headers={"X-API-Key": "k-ingest"})
        assert again.status_code == 200
        assert "gateway_predictions_total" in c.get("/metrics").text
    got = rows("SELECT role, model_version, count(*), min(source) FROM predictions GROUP BY 1, 2 ORDER BY 1")
    assert got == [("challenger", "2", 3, "ingest"), ("champion", "1", 3, "ingest")]


def test_canary_routes_to_challenger(monkeypatch):
    gw = load_app(monkeypatch, mode="canary", canary=100)
    with TestClient(gw.app) as c:
        r = c.post("/predict", json={"title": "x"}, headers={"X-API-Key": "k-public"})
        assert r.json()["role"] == "challenger"
    assert rows("SELECT role, story_id FROM predictions") == [("challenger", None)]


def test_champion_down_returns_502(monkeypatch):
    gw = load_app(monkeypatch, challenger=False, champion_fail=True)
    with TestClient(gw.app) as c:
        r = c.post("/predict", json={"title": "x"}, headers={"X-API-Key": "k-public"})
        assert r.status_code == 502
        assert 'gateway_upstream_errors_total{role="champion"} 1.0' in c.get("/metrics").text


def test_demo_page_renders_picks_and_report_card(monkeypatch):
    gw = load_app(monkeypatch, challenger=False)
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("INSERT INTO stories (id, title, url, domain, kind, posted_at) VALUES "
                     "(1, 'Fresh <story>', 'https://a.com/x', 'a.com', 'story', now() - interval '2 hours'),"
                     "(2, 'Old hit', NULL, NULL, 'ask', now() - interval '30 hours'),"
                     "(3, 'Old miss', 'https://b.com', 'b.com', 'story', now() - interval '30 hours')")
        conn.execute("INSERT INTO labels (story_id, score_24h, label, status, age_hours) VALUES "
                     "(2, 250, true, 'ok', 24.5), (3, 4, false, 'ok', 24.5)")
    with TestClient(gw.app) as c:
        for sid in (1, 2, 3):
            c.post("/predict", json={"title": "t", "story_id": sid}, headers={"X-API-Key": "k-ingest"})
        page = c.get("/").text
    assert "Fresh &lt;story&gt;" in page
    assert "1 of 2 picks reached 100 points" in page
    assert "model v1" in page
