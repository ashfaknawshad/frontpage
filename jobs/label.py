from datetime import datetime, timezone

from jobs.common import connect, ensure_schema, fetch_items, log

THRESHOLD = 100
LOOKBACK_DAYS = 7


def main():
    with connect() as conn:
        ensure_schema(conn)
        due = conn.execute(
            "SELECT s.id, s.posted_at FROM stories s LEFT JOIN labels l ON l.story_id = s.id "
            "WHERE l.story_id IS NULL AND s.posted_at <= now() - interval '24 hours' "
            "AND s.posted_at > now() - make_interval(days => %s)",
            (LOOKBACK_DAYS,),
        ).fetchall()
        posted = dict(due)
        items = {item["id"]: item for item in fetch_items(list(posted))}
        now = datetime.now(timezone.utc)
        rows = []
        for story_id, posted_at in posted.items():
            item = items.get(story_id)
            if item is None:
                continue
            age_hours = (now - posted_at).total_seconds() / 3600
            if item.get("dead") or item.get("deleted"):
                rows.append((story_id, None, None, "removed", age_hours))
                continue
            score = item.get("score")
            rows.append((story_id, score, score is not None and score >= THRESHOLD, "ok", age_hours))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO labels (story_id, score_24h, label, status, age_hours) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (story_id) DO NOTHING",
                rows,
            )
        conn.commit()
        positives = sum(1 for r in rows if r[2])
    log.info("label: due=%d labeled=%d positives=%d", len(posted), len(rows), positives)


if __name__ == "__main__":
    main()
