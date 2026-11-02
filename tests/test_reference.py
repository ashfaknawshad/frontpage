from datetime import datetime, timezone

import pandas as pd

from frontpage.features import add_features
from jobs import drift

TITLES = ["Show HN: A tiny database", "Ask HN: Is this normal?", "Launch HN: Acme (YC S25)", "Why 2026 will be odd (2026)",
          "plain title", "  padded title  ", "Numbers 42 and a question?"]


def test_live_features_match_training_features():
    posted = [datetime(2026, 11, 1 + i, 3 * i, 30, tzinfo=timezone.utc) for i in range(len(TITLES))]
    urls = [None if i % 2 else f"https://www.site{i}.com/path?x=1" for i in range(len(TITLES))]
    frame = pd.DataFrame({"title": TITLES, "url": urls, "posted_at": [p.isoformat() for p in posted]})
    trained = add_features(frame)
    kinds = [t.split()[0].lower() if t.split()[0] in ("Show", "Ask", "Launch") else "story" for t in TITLES]
    domains = [None if u is None else u.split("/")[2].removeprefix("www.") for u in urls]
    live = drift.extract(zip(TITLES, urls, domains, kinds, posted))
    assert live["title_len"] == trained["title_len"].tolist()
    for name in ("has_digit", "has_question", "has_year_tag", "text_post"):
        assert live[name] == trained[name].astype(int).tolist(), name
    assert live["kind"] == trained["kind"].tolist()
    assert live["dow"] == trained["dow"].tolist()
    assert live["domain"] == trained["domain"].tolist()
