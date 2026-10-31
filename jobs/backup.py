import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
from botocore.config import Config

from jobs.common import log

BACKUP_DIR = Path(os.environ.get("BACKUP_DIR", "/backup"))
BUCKET = os.environ.get("BACKUP_BUCKET", "frontpage")
PREFIX = os.environ.get("BACKUP_PREFIX", "backups/postgres")
ENDPOINT = os.environ.get("BACKUP_S3_ENDPOINT", "https://dagshub.com/api/v1/repo-buckets/s3/ashfaknawshad")
RETENTION_DAYS = int(os.environ.get("BACKUP_RETENTION_DAYS", "14"))
MIN_BYTES = int(os.environ.get("BACKUP_MIN_BYTES", "1024"))


def client():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 5, "mode": "standard"}),
    )


def upload_and_verify(s3, path, key):
    size = path.stat().st_size
    if size < MIN_BYTES:
        raise SystemExit(f"{path.name} is only {size} bytes; refusing to upload a suspiciously small dump")
    s3.upload_file(str(path), BUCKET, key)
    remote = s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
    if remote != size:
        raise SystemExit(f"verification failed for {key}: local {size} bytes, remote {remote} bytes")
    log.info("uploaded and verified %s (%.1f MB)", key, size / 1e6)


def prune(s3, now):
    cutoff = (now - timedelta(days=RETENTION_DAYS)).strftime("%Y-%m-%d")
    paginator = s3.get_paginator("list_objects_v2")
    old = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{PREFIX}/"):
        for obj in page.get("Contents", []):
            day = obj["Key"][len(PREFIX) + 1:].split("/", 1)[0]
            if len(day) == 10 and day < cutoff:
                old.append(obj["Key"])
    for key in old:
        s3.delete_object(Bucket=BUCKET, Key=key)
    log.info("pruned %d object(s) older than %s", len(old), cutoff)
    return old


def main(now=None):
    now = now or datetime.now(timezone.utc)
    dumps = sorted(BACKUP_DIR.glob("*.dump"))
    if not dumps:
        raise SystemExit(f"no dumps found in {BACKUP_DIR}")
    s3 = client()
    day, stamp = now.strftime("%Y-%m-%d"), now.strftime("%H%M%S")
    for path in dumps:
        upload_and_verify(s3, path, f"{PREFIX}/{day}/{stamp}-{path.name}")
    prune(s3, now)


if __name__ == "__main__":
    sys.exit(main())
