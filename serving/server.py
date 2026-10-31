import logging
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("frontpage.serving")

MODEL_NAME = os.environ.get("MODEL_NAME", "frontpage")
MODEL_URI = os.environ.get("MODEL_URI", f"models:/{MODEL_NAME}/1")
MODEL_ROLE = os.environ.get("MODEL_ROLE", "champion")
MAX_BATCH = int(os.environ.get("MAX_BATCH", "1000"))


def version_from_uri(uri):
    match = re.match(r"^models:/[^/@]+/(\d+)$", uri)
    return match.group(1) if match else os.environ.get("MODEL_VERSION", "unknown")


MODEL_VERSION = version_from_uri(MODEL_URI)
LABELS = {"model_version": MODEL_VERSION, "role": MODEL_ROLE}

LATENCY = Histogram("serving_predict_latency_seconds", "Model predict latency per request",
                    list(LABELS), buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
ROWS = Counter("serving_predictions_total", "Rows scored", list(LABELS))
SCORES = Histogram("serving_prediction_score", "Predicted probabilities", list(LABELS),
                   buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0))
LOADED = Gauge("serving_model_loaded", "1 when the model is loaded", list(LABELS))

state = {"model": None}


class Instance(BaseModel):
    title: str = Field(min_length=1, max_length=400)
    url: str | None = Field(default=None, max_length=2000)
    posted_at: str | None = None


class PredictRequest(BaseModel):
    instances: list[Instance] = Field(min_length=1)


def to_frame(instances):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    posted = pd.to_datetime(pd.Series([i.posted_at or now for i in instances]), utc=True, format="ISO8601",
                            errors="coerce")
    bad = [n for n, ok in enumerate(posted.notna()) if not ok]
    if bad:
        raise HTTPException(422, f"posted_at is not an ISO 8601 timestamp for instance(s) {bad[:10]}")
    return pd.DataFrame({
        "title": [i.title for i in instances],
        "url": [i.url or None for i in instances],
        "posted_at": posted.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
    })


def load_model():
    import mlflow.pyfunc

    t0 = time.time()
    state["model"] = mlflow.pyfunc.load_model(MODEL_URI)
    LOADED.labels(**LABELS).set(1)
    log.info("loaded %s (version %s, role %s) in %.1fs", MODEL_URI, MODEL_VERSION, MODEL_ROLE, time.time() - t0)


@asynccontextmanager
async def lifespan(app):
    load_model()
    yield


app = FastAPI(title="frontpage-predictor", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    if state["model"] is None:
        raise HTTPException(503, "model not loaded")
    return {"status": "ok", "model_version": MODEL_VERSION, "role": MODEL_ROLE}


@app.get(f"/v1/models/{MODEL_NAME}")
def model_status():
    return {"name": MODEL_NAME, "ready": state["model"] is not None, "version": MODEL_VERSION, "role": MODEL_ROLE}


@app.post(f"/v1/models/{MODEL_NAME}:predict")
def predict(req: PredictRequest):
    if state["model"] is None:
        raise HTTPException(503, "model not loaded")
    if len(req.instances) > MAX_BATCH:
        raise HTTPException(413, f"at most {MAX_BATCH} instances per request")
    frame = to_frame(req.instances)
    t0 = time.perf_counter()
    probs = state["model"].predict(frame)["prob"].astype(float).round(6).tolist()
    LATENCY.labels(**LABELS).observe(time.perf_counter() - t0)
    ROWS.labels(**LABELS).inc(len(probs))
    for p in probs:
        SCORES.labels(**LABELS).observe(p)
    return {"predictions": probs, "model_name": MODEL_NAME, "model_version": MODEL_VERSION, "role": MODEL_ROLE}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
