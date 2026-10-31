import asyncio
import hmac
import html
import logging
import os
import random
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("frontpage.gateway")

CHAMPION_URL = os.environ.get("CHAMPION_URL", "http://frontpage-champion-predictor.frontpage.svc.cluster.local/v1/models/frontpage:predict")
CHALLENGER_URL = os.environ.get("CHALLENGER_URL", "")
ROUTING_MODE = os.environ.get("ROUTING_MODE", "shadow")
CANARY_PERCENT = float(os.environ.get("CANARY_PERCENT", "0"))
API_KEYS = [k.strip() for k in os.environ.get("API_KEYS", "").split(",") if k.strip()]
MAX_BATCH = int(os.environ.get("MAX_BATCH", "500"))
UPSTREAM_TIMEOUT = float(os.environ.get("UPSTREAM_TIMEOUT_SECONDS", "20"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            BIGSERIAL PRIMARY KEY,
    story_id      BIGINT,
    model_name    TEXT NOT NULL,
    model_version TEXT NOT NULL,
    role          TEXT NOT NULL,
    prob          DOUBLE PRECISION NOT NULL,
    latency_ms    REAL,
    source        TEXT NOT NULL DEFAULT 'api',
    request_id    TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS predictions_story_role_version
    ON predictions (story_id, role, model_version) WHERE story_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS predictions_created_at_idx ON predictions (created_at);
"""

REQUESTS = Counter("gateway_requests_total", "Requests", ["route", "status"])
LATENCY = Histogram("gateway_request_latency_seconds", "End-to-end latency", ["route"],
                    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5))
UPSTREAM_ERRORS = Counter("gateway_upstream_errors_total", "Predictor call failures", ["role"])
SCORED = Counter("gateway_predictions_total", "Rows scored", ["role", "model_version"])
SCORE = Histogram("gateway_prediction_score", "Predicted probabilities", ["role", "model_version"],
                  buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0))

state = {"pool": None, "http": None, "tasks": set()}


class Story(BaseModel):
    title: str = Field(min_length=1, max_length=400)
    url: str | None = Field(default=None, max_length=2000)
    posted_at: str | None = None
    story_id: int | None = None
    source: str = Field(default="api", max_length=20)


class Batch(BaseModel):
    stories: list[Story] = Field(min_length=1)


@asynccontextmanager
async def lifespan(app):
    pool = AsyncConnectionPool(conninfo=os.environ.get("DATABASE_URL", ""), min_size=1, max_size=5, open=False)
    await pool.open(wait=True, timeout=30)
    async with pool.connection() as conn:
        await conn.execute(SCHEMA)
    state["pool"] = pool
    state["http"] = state.get("transport_client") or httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT)
    log.info("gateway ready: mode=%s challenger=%s keys=%d", ROUTING_MODE, bool(CHALLENGER_URL), len(API_KEYS))
    yield
    if state["tasks"]:
        await asyncio.gather(*state["tasks"], return_exceptions=True)
    await state["http"].aclose()
    await pool.close()


app = FastAPI(title="frontpage-gateway", lifespan=lifespan)


def check_key(key):
    if not API_KEYS:
        raise HTTPException(503, "no API keys configured")
    if not key or not any(hmac.compare_digest(key, k) for k in API_KEYS):
        raise HTTPException(401, "missing or invalid X-API-Key")


async def call_predictor(url, role, stories):
    payload = {"instances": [{"title": s.title, "url": s.url, "posted_at": s.posted_at} for s in stories]}
    t0 = time.perf_counter()
    try:
        resp = await state["http"].post(url, json=payload)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        UPSTREAM_ERRORS.labels(role).inc()
        raise HTTPException(502, f"{role} predictor failed: {exc}") from exc
    body = resp.json()
    latency_ms = (time.perf_counter() - t0) * 1000
    version = str(body.get("model_version", "unknown"))
    probs = body["predictions"]
    if len(probs) != len(stories):
        UPSTREAM_ERRORS.labels(role).inc()
        raise HTTPException(502, f"{role} predictor returned {len(probs)} predictions for {len(stories)} stories")
    for p in probs:
        SCORED.labels(role, version).inc()
        SCORE.labels(role, version).observe(p)
    return {"role": role, "model_version": version, "model_name": body.get("model_name", "frontpage"),
            "predictions": probs, "latency_ms": latency_ms}


async def log_predictions(result, stories, request_id):
    rows = [(s.story_id, result["model_name"], result["model_version"], result["role"], p,
             result["latency_ms"] / len(stories), s.source, request_id)
            for s, p in zip(stories, result["predictions"])]
    async with state["pool"].connection() as conn:
        async with conn.cursor() as cur:
            await cur.executemany(
                "INSERT INTO predictions (story_id, model_name, model_version, role, prob, latency_ms, source, request_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (story_id, role, model_version) WHERE story_id IS NOT NULL DO NOTHING",
                rows,
            )


async def shadow(stories, request_id):
    try:
        result = await call_predictor(CHALLENGER_URL, "challenger", stories)
        await log_predictions(result, stories, request_id)
    except Exception as exc:
        log.warning("shadow call failed: %s", exc)


def pick_primary():
    if CHALLENGER_URL and ROUTING_MODE == "canary" and random.random() * 100 < CANARY_PERCENT:
        return CHALLENGER_URL, "challenger"
    return CHAMPION_URL, "champion"


async def score(stories, route):
    t0 = time.perf_counter()
    request_id = uuid.uuid4().hex[:16]
    status = "500"
    try:
        url, role = pick_primary()
        result = await call_predictor(url, role, stories)
        await log_predictions(result, stories, request_id)
        if CHALLENGER_URL and ROUTING_MODE == "shadow":
            task = asyncio.create_task(shadow(stories, request_id))
            state["tasks"].add(task)
            task.add_done_callback(state["tasks"].discard)
        status = "200"
        return {
            "request_id": request_id,
            "model_version": result["model_version"],
            "role": role,
            "predictions": [{"story_id": s.story_id, "prob": p} for s, p in zip(stories, result["predictions"])],
        }
    except HTTPException as exc:
        status = str(exc.status_code)
        raise
    finally:
        REQUESTS.labels(route, status).inc()
        LATENCY.labels(route).observe(time.perf_counter() - t0)


@app.post("/predict")
async def predict(story: Story, x_api_key: str | None = Header(default=None)):
    check_key(x_api_key)
    out = await score([story], "predict")
    return {**out, "prob": out["predictions"][0]["prob"]}


@app.post("/predict/batch")
async def predict_batch(batch: Batch, x_api_key: str | None = Header(default=None)):
    check_key(x_api_key)
    if len(batch.stories) > MAX_BATCH:
        raise HTTPException(413, f"at most {MAX_BATCH} stories per request")
    return await score(batch.stories, "batch")


@app.get("/healthz")
async def healthz():
    async with state["pool"].connection() as conn:
        await conn.execute("SELECT 1")
    return {"status": "ok", "mode": ROUTING_MODE, "challenger": bool(CHALLENGER_URL)}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


TOP_SQL = """
SELECT DISTINCT ON (s.id) s.id, s.title, s.url, s.domain, s.posted_at, p.prob
FROM predictions p JOIN stories s ON s.id = p.story_id
WHERE p.role = 'champion' AND s.posted_at > now() - interval '24 hours'
ORDER BY s.id, p.created_at DESC
"""

CARD_SQL = """
SELECT DISTINCT ON (s.id) s.id, s.title, s.url, p.prob, l.score_24h, l.label
FROM predictions p JOIN stories s ON s.id = p.story_id JOIN labels l ON l.story_id = s.id
WHERE p.role = 'champion' AND l.status = 'ok'
  AND s.posted_at > now() - interval '48 hours' AND s.posted_at <= now() - interval '24 hours'
ORDER BY s.id, p.created_at DESC
"""


def _link(title, url, story_id):
    href = url or f"https://news.ycombinator.com/item?id={story_id}"
    return f'<a href="{html.escape(href)}">{html.escape(title)}</a>'


@app.get("/", response_class=HTMLResponse)
async def demo(request: Request):
    async with state["pool"].connection() as conn:
        top = sorted(await (await conn.execute(TOP_SQL)).fetchall(), key=lambda r: -r[5])[:20]
        card_rows = await (await conn.execute(CARD_SQL)).fetchall()
        row = await (await conn.execute(
            "SELECT model_version FROM predictions WHERE role = 'champion' ORDER BY created_at DESC LIMIT 1")).fetchone()
    version = row[0] if row else "?"
    card = sorted(card_rows, key=lambda r: -r[3])
    picks = card[:20]
    hits = sum(1 for r in picks if r[5])
    base = (sum(1 for r in card if r[5]) / len(card)) if card else 0.0
    top_html = "".join(
        f"<tr><td>{p:.0%}</td><td>{_link(t, u, i)}<div class='m'>{html.escape(d or 'text post')}"
        f" · {posted:%H:%M} UTC</div></td></tr>"
        for i, t, u, d, posted, p in top) or "<tr><td colspan=2>No scored stories yet.</td></tr>"
    card_html = "".join(
        f"<tr class='{'hit' if lab else ''}'><td>{p:.0%}</td><td>{_link(t, u, i)}</td><td>{s}</td></tr>"
        for i, t, u, p, s, lab in picks) or "<tr><td colspan=3>Labels arrive 24 hours after a story is posted.</td></tr>"
    summary = (f"{hits} of {len(picks)} picks reached 100 points; a random pick would hit {base:.1%} of the time."
               if picks else "No graded picks yet.")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Frontpage</title>
<style>
body{{font:15px/1.45 system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 16px;color:#1d1d1f;background:#fff}}
h1{{font-size:1.5rem;margin:0}} h2{{font-size:1.1rem;margin-top:2rem}} .sub,.m{{color:#6e6e73;font-size:.85rem}}
table{{width:100%;border-collapse:collapse}} td{{padding:.4rem .3rem;border-bottom:1px solid #e5e5ea;vertical-align:top}}
td:first-child{{width:3.5rem;font-variant-numeric:tabular-nums;font-weight:600}} a{{color:inherit}}
tr.hit td:first-child{{color:#1a7f37}} .card{{background:#f5f5f7;padding:.6rem .8rem;border-radius:8px}}
@media (prefers-color-scheme:dark){{body{{background:#111;color:#eee}} td{{border-color:#333}} .card{{background:#1c1c1e}}}}
</style></head><body>
<h1>Frontpage</h1><div class="sub">Which new Hacker News stories will reach 100 points? Scored live by model v{html.escape(version)}, graded 24 hours later.</div>
<h2>Today's top picks</h2><table>{top_html}</table>
<h2>Yesterday's report card</h2><p class="card">{html.escape(summary)}</p><table>{card_html}</table>
<p class="sub">API: POST /predict with an X-API-Key. Source: github.com/ashfaknawshad/frontpage</p>
</body></html>"""
    return HTMLResponse(page)
