"""Builds jobs/reference.json: the distribution of the training data, which the daily monitor compares live data to."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jobs import drift  # noqa: E402


def data_md5(dvc_file="data/hn_stories.parquet.dvc"):
    try:
        return yaml.safe_load(open(dvc_file))["outs"][0]["md5"]
    except Exception:
        return "unknown"


def build(params):
    df = pd.read_parquet(params["data"]["path"])
    df = df[df["score_reliable"]]
    lo, hi = params["data"]["train"]
    df = df[(df["month"] >= lo) & (df["month"] <= hi)]
    def values(column):
        return [None if pd.isna(v) else v for v in df[column]]

    rows = zip(values("title"), values("url"), values("domain"), values("kind"), df["posted_at"])
    spec = drift.profile(drift.extract(rows))
    spec.update({
        "base_rate": float(df["label"].mean()),
        "months": [lo, hi],
        "data_md5": data_md5(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    return spec


def main():
    params = yaml.safe_load(open("params.yaml"))
    spec = build(params)
    out = ROOT / "jobs" / "reference.json"
    out.write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out} from {spec['n']:,} training stories ({spec['months'][0]}..{spec['months'][1]}), "
          f"base rate {100 * spec['base_rate']:.2f}%")


if __name__ == "__main__":
    main()
