import json
import os
import urllib.request
from datetime import datetime, timezone

from jobs.common import (connect, domain_of, ensure_schema, fetch_items, get_json,
                         is_live_story, kind_of, log, to_utc)

SNAPSHOT_WINDOW_MIN = 90
GATEWAY_URL = os.environ.get("GATEWAY_URL", "")
GATEWAY_API_KEY = os.environ.get("GATEWAY_API_KEY", "")
SCORE_BATCH = 200
SCORE_LIMIT = 1000
SCORE_LOOKBACK_DAYS = 7


def ingest_new(conn):
    ids = get_json("newstories") or []
    known = {row[0] for row in conn.execute("SELECT id FROM stories WHERE id = ANY(%s)", (ids,))}
    fresh = [i for i in ids if i not in known]
    rows = []
    for item in fetch_items(fresh):
        if not is_live_story(item):
            continue
        title = item["title"].strip()
        url = item.get("url")
        rows.append((item["id"], title, url, domain_of(url), item.get("by"), kind_of(title), to_utc(item["time"])))
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO stories (id, title, url, domain, by, kind, posted_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
            rows,
        )
    conn.commit()
    return len(ids), len(fresh), len(rows)


def snapshot_recent(conn):
    recent = conn.execute(
        "SELECT id, posted_at FROM stories WHERE posted_at > now() - make_interval(mins => %s)",
        (SNAPSHOT_WINDOW_MIN,),
    ).fetchall()
    posted = dict(recent)
    now = datetime.now(timezone.utc)
    rows = []
    for item in fetch_items(list(posted)):
        age_min = int((now - posted[item["id"]]).total_seconds() // 60)
        rows.append((item["id"], age_min, item.get("score"), item.get("descendants")))
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO snapshots (story_id, age_min, score, descendants) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
            rows,
        )
    conn.commit()
    return len(rows)


def post_json(url, payload, key, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "X-API-Key": key})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def score_pending(conn):
    if not GATEWAY_URL or not GATEWAY_API_KEY:
        return 0, "scoring disabled (GATEWAY_URL or GATEWAY_API_KEY not set)"
    if conn.execute("SELECT to_regclass('predictions')").fetchone()[0] is None:
        return 0, "predictions table not created yet (gateway not started)"
    pending = conn.execute(
        "SELECT s.id, s.title, s.url, s.posted_at, s.ingested_at > now() - interval '15 minutes' "
        "FROM stories s WHERE s.posted_at > now() - make_interval(days => %s) "
        "AND NOT EXISTS (SELECT 1 FROM predictions p WHERE p.story_id = s.id AND p.role = 'champion') "
        "ORDER BY s.posted_at DESC LIMIT %s",
        (SCORE_LOOKBACK_DAYS, SCORE_LIMIT),
    ).fetchall()
    scored = 0
    for start in range(0, len(pending), SCORE_BATCH):
        chunk = pending[start:start + SCORE_BATCH]
        stories = [{"story_id": sid, "title": title, "url": url, "posted_at": posted.isoformat(),
                    "source": "ingest" if fresh else "backfill"} for sid, title, url, posted, fresh in chunk]
        out = post_json(f"{GATEWAY_URL}/predict/batch", {"stories": stories}, GATEWAY_API_KEY)
        scored += len(out["predictions"])
    return scored, f"scored {scored} of {len(pending)} pending"


def main():
    with connect() as conn:
        ensure_schema(conn)
        seen, fresh, inserted = ingest_new(conn)
        snaps = snapshot_recent(conn)
        try:
            scored, note = score_pending(conn)
        except Exception as exc:
            scored, note = 0, f"scoring failed: {exc}"
        total = conn.execute("SELECT count(*) FROM stories").fetchone()[0]
    log.info("ingest: listed=%d new_ids=%d inserted=%d snapshots=%d total_stories=%d scored=%d (%s)",
             seen, fresh, inserted, snaps, total, scored, note)


if __name__ == "__main__":
    main()
