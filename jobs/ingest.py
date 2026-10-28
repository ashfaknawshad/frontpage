from datetime import datetime, timezone

from jobs.common import (connect, domain_of, ensure_schema, fetch_items, get_json,
                         is_live_story, kind_of, log, to_utc)

SNAPSHOT_WINDOW_MIN = 90


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


def main():
    with connect() as conn:
        ensure_schema(conn)
        seen, fresh, inserted = ingest_new(conn)
        snaps = snapshot_recent(conn)
        total = conn.execute("SELECT count(*) FROM stories").fetchone()[0]
    log.info("ingest: listed=%d new_ids=%d inserted=%d snapshots=%d total_stories=%d",
             seen, fresh, inserted, snaps, total)


if __name__ == "__main__":
    main()
