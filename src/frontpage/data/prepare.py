import argparse
import json
import logging
import os
from pathlib import Path

import duckdb
import pandas as pd
import pandera.pandas as pa

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prepare")

HF_REPO = "open-index/hacker-news"
LABEL_THRESHOLD = 100

STORY_SQL = """
SELECT
    CAST(id AS BIGINT)            AS id,
    trim(title)                   AS title,
    NULLIF(url, '')               AS url,
    "by"                          AS "by",
    CAST(time AS TIMESTAMPTZ)     AS posted_at,
    CAST(score AS INTEGER)        AS score,
    CAST(coalesce(descendants, 0) AS INTEGER) AS descendants
FROM read_parquet(?)
WHERE type = 1
  AND coalesce(deleted, 0) = 0
  AND coalesce(dead, 0) = 0
  AND title IS NOT NULL AND trim(title) <> ''
  AND score IS NOT NULL
"""

SCHEMA = pa.DataFrameSchema(
    {
        "id": pa.Column("int64", unique=True),
        "title": pa.Column(str, pa.Check.str_length(min_value=1)),
        "url": pa.Column(str, nullable=True),
        "domain": pa.Column(str, nullable=True),
        "by": pa.Column(str, nullable=True),
        "kind": pa.Column(str, pa.Check.isin(["story", "show", "ask", "launch"])),
        "posted_at": pa.Column("datetime64[ns, UTC]"),
        "month": pa.Column(str, pa.Check.str_matches(r"^\d{4}-\d{2}$")),
        "score": pa.Column("int32", pa.Check.ge(0)),
        "descendants": pa.Column("int32", pa.Check.ge(0)),
        "label": pa.Column(bool),
    },
    checks=[
        pa.Check(lambda df: 0.01 <= df["label"].mean() <= 0.30, error="positive rate outside 1%-30%"),
    ],
    strict=True,
    coerce=False,
)


def months_between(start, end):
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def kind_of(title):
    for prefix, kind in (("Show HN", "show"), ("Ask HN", "ask"), ("Launch HN", "launch")):
        if title.startswith(prefix):
            return kind
    return "story"


def add_derived(df, month):
    host = df["url"].str.extract(r"^[a-zA-Z][a-zA-Z0-9+.-]*://(?:[^@/]*@)?([^/:?#]+)", expand=False).str.lower()
    df["domain"] = host.str.replace(r"^www\.", "", regex=True)
    df["kind"] = df["title"].map(kind_of)
    df["month"] = month
    df["posted_at"] = pd.to_datetime(df["posted_at"], utc=True).astype("datetime64[ns, UTC]")
    df["score"] = df["score"].clip(lower=0).astype("int32")
    df["descendants"] = df["descendants"].clip(lower=0).astype("int32")
    df["id"] = df["id"].astype("int64")
    df["label"] = df["score"] >= LABEL_THRESHOLD
    return df


def stories_from_file(path, month):
    df = duckdb.execute(STORY_SQL, [str(path)]).df()
    df = df[df["posted_at"].dt.strftime("%Y-%m") == month]
    return add_derived(df, month)


def download_month(month, cache_dir):
    from huggingface_hub import hf_hub_download

    year = month[:4]
    return hf_hub_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        filename=f"data/{year}/{month}.parquet",
        local_dir=cache_dir,
    )


def build(months, fetch, cache_dir):
    frames = []
    for month in months:
        try:
            path = fetch(month, cache_dir)
        except Exception as exc:
            if "404" in str(exc) or "EntryNotFound" in type(exc).__name__:
                log.warning("%s: not in the dataset yet, skipping", month)
                continue
            raise
        df = stories_from_file(path, month)
        log.info("%s: %d stories, %.2f%% >= %d points", month, len(df), 100 * df["label"].mean(), LABEL_THRESHOLD)
        frames.append(df)
        if os.environ.get("PREPARE_DELETE_RAW", "1") == "1":
            Path(path).unlink(missing_ok=True)
    if not frames:
        raise SystemExit("no months downloaded")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("posted_at").drop_duplicates("id", keep="last").reset_index(drop=True)
    cols = ["id", "title", "url", "domain", "by", "kind", "posted_at", "month", "score", "descendants", "label"]
    return SCHEMA.validate(out[cols])


def summarize(df):
    per_month = df.groupby("month").agg(stories=("id", "size"), positive_rate=("label", "mean")).reset_index()
    return {
        "rows": int(len(df)),
        "months": len(per_month),
        "first": str(df["posted_at"].min()),
        "last": str(df["posted_at"].max()),
        "positive_rate": round(float(df["label"].mean()), 4),
        "kinds": {k: int(v) for k, v in df["kind"].value_counts().items()},
        "per_month": [
            {"month": r.month, "stories": int(r.stories), "positive_rate": round(float(r.positive_rate), 4)}
            for r in per_month.itertuples()
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-10")
    parser.add_argument("--end", default="2026-09")
    parser.add_argument("--out", default="data/hn_stories.parquet")
    parser.add_argument("--stats", default="data/hn_stories.stats.json")
    parser.add_argument("--cache-dir", default=".hf-cache")
    args = parser.parse_args()

    df = build(months_between(args.start, args.end), download_month, args.cache_dir)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    stats = summarize(df)
    Path(args.stats).write_text(json.dumps(stats, indent=2))
    log.info("wrote %s: %d rows, positive rate %.2f%%", args.out, stats["rows"], 100 * stats["positive_rate"])


if __name__ == "__main__":
    main()
