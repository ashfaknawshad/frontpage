import os

from jobs import drift
from jobs.common import connect, ensure_schema, fetch_items, log

MIN_AGE_HOURS = float(os.environ.get("COMPARE_MIN_AGE_HOURS", "25"))
MAX_AGE_DAYS = int(os.environ.get("COMPARE_MAX_AGE_DAYS", "14"))
THRESHOLD = 100

SCHEMA = """
CREATE TABLE IF NOT EXISTS model_compare (
    id            BIGSERIAL PRIMARY KEY,
    ts            TIMESTAMPTZ NOT NULL DEFAULT now(),
    label_def     TEXT NOT NULL,
    role          TEXT NOT NULL,
    model_version TEXT NOT NULL,
    n             INTEGER NOT NULL,
    hits          INTEGER NOT NULL,
    base_rate     DOUBLE PRECISION,
    mean_prob     DOUBLE PRECISION,
    brier         DOUBLE PRECISION,
    ece           DOUBLE PRECISION,
    roc_auc       DOUBLE PRECISION,
    roc_auc_se    DOUBLE PRECISION,
    pr_auc        DOUBLE PRECISION,
    p_at_20_day   DOUBLE PRECISION
);
"""

QUERY = """
WITH c AS (SELECT model_version FROM predictions WHERE role = 'champion' ORDER BY created_at DESC LIMIT 1),
     h AS (SELECT model_version FROM predictions WHERE role = 'challenger' ORDER BY created_at DESC LIMIT 1)
SELECT s.id, s.posted_at::date, pc.prob, ph.prob, l.label::int, c.model_version, h.model_version
FROM stories s
JOIN c ON TRUE JOIN h ON TRUE
JOIN predictions pc ON pc.story_id = s.id AND pc.role = 'champion' AND pc.model_version = c.model_version
JOIN predictions ph ON ph.story_id = s.id AND ph.role = 'challenger' AND ph.model_version = h.model_version
JOIN labels l ON l.story_id = s.id AND l.status = 'ok'
WHERE s.posted_at BETWEEN now() - make_interval(days => %s) AND now() - %s * interval '1 hour'
"""


def score_model(days, y, p):
    auc = drift.roc_auc(y, p)
    return {"n": len(y), "hits": int(sum(y)), "base_rate": sum(y) / len(y), "mean_prob": float(sum(p) / len(p)),
            "brier": drift.brier(y, p), "ece": drift.expected_calibration_error(y, p), "roc_auc": auc,
            "roc_auc_se": drift.roc_auc_se(auc, sum(y), len(y) - sum(y)), "pr_auc": drift.average_precision(y, p),
            "p_at_20_day": drift.precision_at_k_per_day(days, y, p)}


def compare(rows, items):
    """rows: (story_id, day, champion_prob, challenger_prob, label_24h). Both models scored the same stories."""
    days = [r[1] for r in rows]
    out = []
    defs = {"24h": [r[4] for r in rows]}
    keep = [i for i, r in enumerate(rows) if r[0] in items and items[r[0]].get("score") is not None
            and not items[r[0]].get("dead") and not items[r[0]].get("deleted")]
    if keep:
        defs["now"] = None
    for name in defs:
        if name == "24h":
            idx = list(range(len(rows)))
            y = defs["24h"]
        else:
            idx = keep
            y = [int(items[rows[i][0]]["score"] >= THRESHOLD) for i in idx]
        for role, col in (("champion", 2), ("challenger", 3)):
            p = [rows[i][col] for i in idx]
            if len(y) >= 2 and 0 < sum(y) < len(y):
                out.append({"label_def": name, "role": role, **score_model([days[i] for i in idx], y, p)})
    return out


def main():
    with connect() as conn:
        ensure_schema(conn)
        conn.execute(SCHEMA)
        conn.commit()
        found = conn.execute(QUERY, (MAX_AGE_DAYS, MIN_AGE_HOURS)).fetchall()
        if not found:
            log.warning("model_compare: no stories scored by both champion and challenger yet")
            return
        champ_v, chall_v = found[0][5], found[0][6]
        items = {item["id"]: item for item in fetch_items([r[0] for r in found])}
        results = compare([r[:5] for r in found], items)
        versions = {"champion": champ_v, "challenger": chall_v}
        for r in results:
            conn.execute(
                "INSERT INTO model_compare (label_def, role, model_version, n, hits, base_rate, mean_prob, brier, ece, "
                "roc_auc, roc_auc_se, pr_auc, p_at_20_day) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (r["label_def"], r["role"], versions[r["role"]], r["n"], r["hits"], r["base_rate"], r["mean_prob"],
                 r["brier"], r["ece"], r["roc_auc"], r["roc_auc_se"], r["pr_auc"], r["p_at_20_day"]))
        conn.commit()
    for r in results:
        log.info("model_compare [%s labels] %s v%s: n=%d hits=%d base=%.3f mean_prob=%.3f brier=%.4f ece=%.4f "
                 "roc_auc=%s pr_auc=%s p@20/day=%s", r["label_def"], r["role"], versions[r["role"]], r["n"],
                 r["hits"], r["base_rate"], r["mean_prob"], r["brier"], r["ece"],
                 None if r["roc_auc"] is None else round(r["roc_auc"], 3),
                 None if r["pr_auc"] is None else round(r["pr_auc"], 3),
                 None if r["p_at_20_day"] is None else round(r["p_at_20_day"], 3))


if __name__ == "__main__":
    main()
