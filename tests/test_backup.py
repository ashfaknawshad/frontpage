from datetime import datetime, timezone

import boto3
import pytest
from moto import mock_aws

import jobs.backup as backup


@pytest.fixture
def s3(monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setattr(backup, "ENDPOINT", None)
    monkeypatch.setattr(backup, "BACKUP_DIR", tmp_path)
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=backup.BUCKET)
        yield client


def test_uploads_verifies_and_prunes(s3, tmp_path):
    (tmp_path / "frontpage.dump").write_bytes(b"x" * 5000)
    (tmp_path / "mlflow.dump").write_bytes(b"y" * 3000)
    s3.put_object(Bucket=backup.BUCKET, Key=f"{backup.PREFIX}/2026-10-01/000000-frontpage.dump", Body=b"old")
    s3.put_object(Bucket=backup.BUCKET, Key=f"{backup.PREFIX}/2026-10-30/000000-frontpage.dump", Body=b"recent")
    backup.main(now=datetime(2026, 11, 1, 20, 30, tzinfo=timezone.utc))
    keys = sorted(o["Key"] for o in s3.list_objects_v2(Bucket=backup.BUCKET)["Contents"])
    assert keys == [
        f"{backup.PREFIX}/2026-10-30/000000-frontpage.dump",
        f"{backup.PREFIX}/2026-11-01/203000-frontpage.dump",
        f"{backup.PREFIX}/2026-11-01/203000-mlflow.dump",
    ]


def test_refuses_tiny_dump(s3, tmp_path):
    (tmp_path / "frontpage.dump").write_bytes(b"x" * 10)
    with pytest.raises(SystemExit, match="suspiciously small"):
        backup.main()


def test_fails_without_dumps(s3):
    with pytest.raises(SystemExit, match="no dumps"):
        backup.main()
