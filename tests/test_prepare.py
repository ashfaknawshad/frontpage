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
