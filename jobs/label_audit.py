import os

from jobs import drift
from jobs.common import connect, ensure_schema, fetch_items, log

MIN_AGE_DAYS = int(os.environ.get("AUDIT_MIN_AGE_DAYS", "3"))
MAX_AGE_DAYS = int(os.environ.get("AUDIT_MAX_AGE_DAYS", "14"))
LIMIT = int(os.environ.get("AUDIT_LIMIT", "3000"))
THRESHOLD = 100

SCHEMA = """
CREATE TABLE IF NOT EXISTS label_audit (
    id              BIGSERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
    champion_version TEXT,
    min_age_days    INTEGER NOT NULL,
    n               INTEGER NOT NULL,
    hits_24h        INTEGER,
    hits_final      INTEGER,
    late_risers     INTEGER,
    base_24h        DOUBLE PRECISION,
    base_final      DOUBLE PRECISION,
    auc_24h         DOUBLE PRECISION,
    auc_24h_se      DOUBLE PRECISION,
    auc_final       DOUBLE PRECISION,
    auc_final_se    DOUBLE PRECISION
);
"""

QUERY = """
WITH v AS (SELECT model_version FROM predictions WHERE role = 'champion' ORDER BY created_at DESC LIMIT 1)
SELECT s.id, p.prob, l.label::int, v.model_version
FROM predictions p
JOIN v ON v.model_version = p.model_version
JOIN labels l ON l.story_id = p.story_id AND l.status = 'ok'
JOIN stories s ON s.id = p.story_id
WHERE p.role = 'champion'
  AND s.posted_at BETWEEN now() - make_interval(days => %s) AND now() - make_interval(days => %s)
ORDER BY random() LIMIT %s
"""


def auc_with_se(y, p):
    auc = drift.roc_auc(y, p)
    return auc, drift.roc_auc_se(auc, sum(y), len(y) - sum(y))


def audit(rows, items):
    """rows: (story_id, prob, label_24h). items: story_id -> HN item with today's score."""
    kept = [(prob, label, int(items[sid]["score"] >= THRESHOLD)) for sid, prob, label in rows
            if sid in items and items[sid].get("score") is not None
            and not items[sid].get("dead") and not items[sid].get("deleted")]
    if not kept:
        return None
    p = [k[0] for k in kept]
    y24 = [k[1] for k in kept]
    yfinal = [k[2] for k in kept]
    a24, s24 = auc_with_se(y24, p)
    afin, sfin = auc_with_se(yfinal, p)
    return {"n": len(kept), "hits_24h": sum(y24), "hits_final": sum(yfinal),
            "late_risers": sum(1 for a, b in zip(y24, yfinal) if b and not a),
            "base_24h": sum(y24) / len(kept), "base_final": sum(yfinal) / len(kept),
            "auc_24h": a24, "auc_24h_se": s24, "auc_final": afin, "auc_final_se": sfin}


def main():
    with connect() as conn:
        ensure_schema(conn)
        conn.execute(SCHEMA)
        conn.commit()
        found = conn.execute(QUERY, (MAX_AGE_DAYS, MIN_AGE_DAYS, LIMIT)).fetchall()
        if not found:
            log.warning("label_audit: no graded stories older than %d days with champion predictions", MIN_AGE_DAYS)
            return
        version = found[0][3]
        items = {item["id"]: item for item in fetch_items([r[0] for r in found])}
        result = audit([r[:3] for r in found], items)
        if result is None:
            log.warning("label_audit: HN returned nothing usable")
            return
        conn.execute(
            "INSERT INTO label_audit (champion_version, min_age_days, n, hits_24h, hits_final, late_risers, "
            "base_24h, base_final, auc_24h, auc_24h_se, auc_final, auc_final_se) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (version, MIN_AGE_DAYS, result["n"], result["hits_24h"], result["hits_final"], result["late_risers"],
             result["base_24h"], result["base_final"], result["auc_24h"], result["auc_24h_se"],
             result["auc_final"], result["auc_final_se"]))
        conn.commit()
    log.info("label_audit: champion v%s on %d stories older than %dd. Hits at 24h %d (%.2f%%), final %d (%.2f%%), "
             "late risers %d. ROC-AUC with 24h labels %.3f +/- %.3f, with final labels %.3f +/- %.3f",
             version, result["n"], MIN_AGE_DAYS, result["hits_24h"], 100 * result["base_24h"], result["hits_final"],
             100 * result["base_final"], result["late_risers"], result["auc_24h"] or float("nan"),
             result["auc_24h_se"] or float("nan"), result["auc_final"] or float("nan"),
             result["auc_final_se"] or float("nan"))


if __name__ == "__main__":
    main()
