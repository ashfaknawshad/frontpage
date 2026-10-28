from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

from frontpage.data.prepare import build, months_between


def _write_month(path, rows):
    schema = pa.schema([
        ("id", pa.uint32()), ("deleted", pa.uint8()), ("type", pa.int8()), ("by", pa.string()),
        ("time", pa.timestamp("ms", tz="UTC")), ("text", pa.string()), ("dead", pa.uint8()),
        ("url", pa.string()), ("score", pa.int32()), ("title", pa.string()), ("descendants", pa.int32()),
    ])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _row(i, title="A title", type_=1, score=10, dead=0, deleted=0, url="https://www.Example.com/p", day=5):
    return {"id": i, "deleted": deleted, "type": type_, "by": "u", "time": datetime(2025, 1, day, tzinfo=timezone.utc),
            "text": "", "dead": dead, "url": url, "score": score, "title": title, "descendants": None}


def test_months_between():
    assert months_between("2024-11", "2025-02") == ["2024-11", "2024-12", "2025-01", "2025-02"]


def test_build_filters_and_derives(tmp_path):
    rows = [_row(1, "Show HN: x", score=150), _row(2, score=3, url=""), _row(3, type_=2),
            _row(4, dead=1), _row(5, deleted=1), _row(6, title="  "), _row(7, "Ask HN: y", score=99)]
    rows += [_row(100 + i, score=1) for i in range(20)]
    f = tmp_path / "2025-01.parquet"
    _write_month(f, rows)
    df = build(["2025-01"], lambda month, cache: str(f), str(tmp_path))
    assert set(df["id"]) == {1, 2, 7} | {100 + i for i in range(20)}
    one = df.set_index("id").loc[1]
    assert one["kind"] == "show" and one["domain"] == "example.com" and bool(one["label"])
    assert df.set_index("id").loc[2, "url"] is None or str(df.set_index("id").loc[2, "url"]) in ("None", "nan", "<NA>")
    assert not bool(df.set_index("id").loc[7, "label"])
    assert (df["descendants"] == 0).all()


def test_stale_month_flagged(tmp_path):
    good = [_row(i, score=(150 if i % 10 == 0 else 5)) for i in range(1, 101)]
    stale = [dict(_row(1000 + i, score=1), time=datetime(2025, 2, 3, tzinfo=timezone.utc)) for i in range(60)]
    f1, f2 = tmp_path / "2025-01.parquet", tmp_path / "2025-02.parquet"
    _write_month(f1, good)
    _write_month(f2, stale)
    files = {"2025-01": str(f1), "2025-02": str(f2)}
    df = build(["2025-01", "2025-02", "2025-03"], lambda m, c: files[m] if m in files else (_ for _ in ()).throw(FileNotFoundError("404 Client Error")), str(tmp_path))
    by_month = df.groupby("month")["score_reliable"].all().to_dict()
    assert by_month == {"2025-01": True, "2025-02": False}
