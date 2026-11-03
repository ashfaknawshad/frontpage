import json
import os
import urllib.request
from collections import defaultdict
from pathlib import Path

from jobs import drift
from jobs.common import connect, ensure_schema, log

REFERENCE = Path(__file__).with_name("reference.json")
WINDOW_DAYS = int(os.environ.get("MONITOR_WINDOW_DAYS", "7"))
PERF_DAYS = int(os.environ.get("MONITOR_PERF_DAYS", "14"))
MIN_LIVE = int(os.environ.get("MONITOR_MIN_LIVE", "500"))
MIN_GRADED = int(os.environ.get("MONITOR_MIN_GRADED", "300"))
MIN_ROWS_PER_MODEL = int(os.environ.get("MONITOR_MIN_ROWS_PER_MODEL", "30"))
DRIFT_MIN_FEATURES = int(os.environ.get("MONITOR_DRIFT_MIN_FEATURES", "3"))
PERF_RATIO = float(os.environ.get("MONITOR_PERF_RATIO", "0.9"))
PERF_Z = float(os.environ.get("MONITOR_PERF_Z", "1.64"))
COOLDOWN_DAYS = int(os.environ.get("MONITOR_COOLDOWN_DAYS", "7"))
AUTO_RETRAIN = os.environ.get("AUTO_RETRAIN", "false").lower() == "true"
MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://mlflow.mlflow.svc.cluster.local").rstrip("/")
MODEL_NAME = os.environ.get("MODEL_NAME", "frontpage")
REPO = os.environ.get("GITHUB_REPOSITORY", "ashfaknawshad/frontpage")
TOKEN = os.environ.get("GITHUB_DISPATCH_TOKEN", "")

SCHEMA = """
CREATE TABLE IF NOT EXISTS monitor_runs (
    id                BIGSERIAL PRIMARY KEY,
    ts                TIMESTAMPTZ NOT NULL DEFAULT now(),
    window_days       INTEGER NOT NULL,
    n_live            INTEGER NOT NULL,
    drifted_features  INTEGER,
    drift             BOOLEAN,
    champion_version  TEXT,
    champion_roc_auc  DOUBLE PRECISION,
    reference_roc_auc DOUBLE PRECISION,
    n_graded          INTEGER,
    base_rate_live    DOUBLE PRECISION,
    base_rate_ref     DOUBLE PRECISION,
    reason            TEXT,
    action            TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS monitor_psi (
    run_id  BIGINT NOT NULL REFERENCES monitor_runs(id) ON DELETE CASCADE,
    feature TEXT NOT NULL,
    psi     DOUBLE PRECISION NOT NULL,
    detail  TEXT,
    PRIMARY KEY (run_id, feature)
);
CREATE TABLE IF NOT EXISTS monitor_model (
    run_id        BIGINT NOT NULL REFERENCES monitor_runs(id) ON DELETE CASCADE,
    role          TEXT NOT NULL,
    model_version TEXT NOT NULL,
    n             INTEGER NOT NULL,
    roc_auc       DOUBLE PRECISION,
    pr_auc        DOUBLE PRECISION,
    brier         DOUBLE PRECISION,
    mean_prob     DOUBLE PRECISION,
    base_rate     DOUBLE PRECISION,
    PRIMARY KEY (run_id, role, model_version)
);
ALTER TABLE monitor_runs ADD COLUMN IF NOT EXISTS champion_roc_auc_se DOUBLE PRECISION;
ALTER TABLE monitor_model ADD COLUMN IF NOT EXISTS roc_auc_se DOUBLE PRECISION;
"""


def load_reference():
    if not REFERENCE.exists():
        log.warning("monitor: %s missing; skipping drift (run the make-reference workflow)", REFERENCE.name)
        return None
    return json.loads(REFERENCE.read_text())


def live_drift(conn, spec):
    rows = conn.execute(
        "SELECT title, url, domain, kind, posted_at FROM stories "
        "WHERE posted_at > now() - make_interval(days => %s)", (WINDOW_DAYS,)).fetchall()
    if len(rows) < MIN_LIVE or spec is None:
        return len(rows), None
    return len(rows), drift.compare(spec, drift.extract(rows))


def model_performance(conn):
    rows = conn.execute(
        "SELECT p.role, p.model_version, p.prob, l.label::int FROM predictions p "
        "JOIN labels l ON l.story_id = p.story_id JOIN stories s ON s.id = p.story_id "
        "WHERE l.status = 'ok' AND s.posted_at > now() - make_interval(days => %s)", (PERF_DAYS,)).fetchall()
    groups = defaultdict(lambda: ([], []))
    for role, version, prob, label in rows:
        groups[(role, version)][0].append(label)
        groups[(role, version)][1].append(prob)
    out = []
    for (role, version), (y, p) in sorted(groups.items()):
        full = len(y) >= MIN_ROWS_PER_MODEL
        auc = drift.roc_auc(y, p) if full else None
        out.append({"role": role, "model_version": version, "n": len(y),
                    "roc_auc": auc, "roc_auc_se": drift.roc_auc_se(auc, sum(y), len(y) - sum(y)),
                    "pr_auc": drift.average_precision(y, p) if full else None,
                    "brier": drift.brier(y, p) if full else None,
                    "mean_prob": sum(p) / len(p), "base_rate": sum(y) / len(y)})
    base = conn.execute(
        "SELECT avg(l.label::int) FROM labels l JOIN stories s ON s.id = l.story_id "
        "WHERE l.status = 'ok' AND s.posted_at > now() - make_interval(days => %s)", (PERF_DAYS,)).fetchone()[0]
    return out, (float(base) if base is not None else None)


def _get(path):
    with urllib.request.urlopen(f"{MLFLOW_URL}{path}", timeout=15) as resp:
        return json.load(resp)


def champion_reference():
    """Test ROC-AUC the champion was registered with, read from the MLflow REST API."""
    try:
        version = _get(f"/api/2.0/mlflow/registered-models/alias?name={MODEL_NAME}&alias=champion")["model_version"]
        run = _get(f"/api/2.0/mlflow/runs/get?run_id={version['run_id']}")["run"]
        metrics = {m["key"]: m["value"] for m in run["data"]["metrics"]}
        return str(version["version"]), metrics.get("test_roc_auc")
    except Exception as exc:
        log.warning("monitor: could not read the champion from MLflow: %s", exc)
        return None, None


def decide(n_drifted, n_live, champion_roc, reference_roc, n_graded, previous_drift, days_since_trigger,
           auto_retrain, champion_se=0.0):
    """Returns (reason, action). Two rules: live ranking quality is significantly lower, or the inputs drifted twice.

    Quality counts as lower only when even the optimistic end of the estimate (AUC + Z standard errors) is
    below 90% of the reference, so a handful of hits cannot trigger a retrain by chance."""
    perf_bad = (champion_roc is not None and reference_roc is not None and n_graded >= MIN_GRADED
                and champion_roc + PERF_Z * (champion_se or 0.0) < PERF_RATIO * reference_roc)
    drift_bad = n_drifted is not None and n_live >= MIN_LIVE and n_drifted >= DRIFT_MIN_FEATURES
    reason = "performance" if perf_bad else ("drift" if drift_bad and previous_drift else None)
    if reason is None:
        return None, "none"
    if days_since_trigger is not None and days_since_trigger < COOLDOWN_DAYS:
        return reason, "cooldown"
    return reason, "triggered" if auto_retrain else "would_trigger"


def dispatch_retrain(reason):
    body = json.dumps({"event_type": "retrain", "client_payload": {"reason": reason}}).encode()
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/dispatches", data=body, method="POST",
        headers={"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "frontpage-monitor"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.status


def main():
    with connect() as conn:
        ensure_schema(conn)
        conn.execute(SCHEMA)
        conn.commit()
        spec = load_reference()
        n_live, drift_result = live_drift(conn, spec)
        models, base_live = model_performance(conn)
        champ_version, reference_roc = champion_reference()
        # Grade the champion version under whichever role logged it most: right after a promotion its
        # graded history sits under "challenger" from its time in shadow.
        champ = max((m for m in models if m["model_version"] == champ_version), key=lambda m: m["n"], default=None)
        champ_roc, n_graded = (champ["roc_auc"], champ["n"]) if champ else (None, 0)
        champ_se = champ["roc_auc_se"] if champ else None

        n_drifted = None if drift_result is None else sum(1 for v in drift_result.values() if v["psi"] >= drift.PSI_ALERT)
        previous = conn.execute("SELECT drift FROM monitor_runs WHERE ts > now() - interval '3 days' "
                                "ORDER BY ts DESC LIMIT 1").fetchone()
        last_trigger = conn.execute("SELECT extract(epoch FROM now() - max(ts)) / 86400 FROM monitor_runs "
                                    "WHERE action = 'triggered'").fetchone()[0]
        reason, action = decide(n_drifted, n_live, champ_roc, reference_roc, n_graded,
                                bool(previous and previous[0]), last_trigger, AUTO_RETRAIN, champ_se)
        if action == "triggered":
            try:
                status = dispatch_retrain(reason)
                log.info("monitor: retrain dispatched (%s), GitHub answered %s", reason, status)
            except Exception as exc:
                log.error("monitor: dispatch failed: %s", exc)
                action = "dispatch_failed"

        run_id = conn.execute(
            "INSERT INTO monitor_runs (window_days, n_live, drifted_features, drift, champion_version, "
            "champion_roc_auc, champion_roc_auc_se, reference_roc_auc, n_graded, base_rate_live, base_rate_ref, "
            "reason, action) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (WINDOW_DAYS, n_live, n_drifted, None if n_drifted is None else n_drifted >= DRIFT_MIN_FEATURES,
             champ_version, champ_roc, champ_se, reference_roc, n_graded, base_live,
             spec.get("base_rate") if spec else None, reason, action)).fetchone()[0]
        for feature, result in (drift_result or {}).items():
            conn.execute("INSERT INTO monitor_psi (run_id, feature, psi, detail) VALUES (%s, %s, %s, %s)",
                         (run_id, feature, result["psi"], result["detail"]))
        for m in models:
            conn.execute(
                "INSERT INTO monitor_model (run_id, role, model_version, n, roc_auc, roc_auc_se, pr_auc, brier, "
                "mean_prob, base_rate) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (run_id, m["role"], m["model_version"], m["n"], m["roc_auc"], m["roc_auc_se"], m["pr_auc"], m["brier"],
                 m["mean_prob"], m["base_rate"]))
        conn.commit()
    log.info("monitor: live=%d drifted=%s champion v%s roc_auc=%s +/- %s (reference %s, n=%d) -> %s%s",
             n_live, n_drifted, champ_version, champ_roc, champ_se, reference_roc, n_graded, action,
             f" ({reason})" if reason else "")


if __name__ == "__main__":
    main()
