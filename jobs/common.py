import json
import logging
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlparse

import psycopg

HN_API = os.environ.get("HN_API", "https://hacker-news.firebaseio.com/v0")
USER_AGENT = "frontpage-mlops/0.1 (+https://github.com/ashfaknawshad/frontpage)"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("frontpage")

SCHEMA = """
CREATE TABLE IF NOT EXISTS stories (
    id          BIGINT PRIMARY KEY,
    title       TEXT NOT NULL,
    url         TEXT,
    domain      TEXT,
    by          TEXT,
    kind        TEXT NOT NULL,
    posted_at   TIMESTAMPTZ NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS stories_posted_at_idx ON stories (posted_at);

CREATE TABLE IF NOT EXISTS snapshots (
    story_id    BIGINT NOT NULL REFERENCES stories(id),
    age_min     INTEGER NOT NULL,
    score       INTEGER,
    descendants INTEGER,
    observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (story_id, age_min)
);

CREATE TABLE IF NOT EXISTS labels (
    story_id   BIGINT PRIMARY KEY REFERENCES stories(id),
    score_24h  INTEGER,
    label      BOOLEAN,
    status     TEXT NOT NULL,
    age_hours  REAL NOT NULL,
    labeled_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def get_json(path, retries=3):
    url = f"{HN_API}/{path}.json"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.load(resp)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def _safe_item(item_id):
    try:
        return get_json(f"item/{item_id}")
    except Exception as exc:
        log.warning("item %s failed: %s", item_id, exc)
        return None


def fetch_items(ids, workers=8):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [item for item in pool.map(_safe_item, ids) if item]


def kind_of(title):
    for prefix, kind in (("Show HN", "show"), ("Ask HN", "ask"), ("Launch HN", "launch")):
        if title.startswith(prefix):
            return kind
    return "story"


def domain_of(url):
    if not url:
        return None
    host = urlparse(url).netloc.lower().split("@")[-1].split(":")[0]
    return host[4:] if host.startswith("www.") else host or None


def is_live_story(item):
    return (
        item.get("type") == "story"
        and not item.get("dead")
        and not item.get("deleted")
        and bool(item.get("title"))
        and item.get("time") is not None
    )


def to_utc(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def connect():
    return psycopg.connect(connect_timeout=10)


def ensure_schema(conn):
    conn.execute(SCHEMA)
    conn.commit()
