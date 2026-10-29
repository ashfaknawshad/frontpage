import numpy as np
import pandas as pd
import yaml

from frontpage.features import INPUT_COLUMNS, add_features
from frontpage.metrics import precision_at_k_per_day
from frontpage.train import make_model


def test_add_features_handles_text_posts_and_tags():
    df = pd.DataFrame({
        "title": ["Show HN: A thing (2019)", "Ask HN: Why?", "Plain title 42"],
        "url": ["https://www.Example.com/x", None, ""],
        "posted_at": ["2025-01-05T23:30:00Z", "2025-01-06T00:00:00+00:00", "2025-01-06T12:00:00Z"],
    })
    out = add_features(df)
    assert out["domain"].tolist() == ["example.com", "(text post)", "(text post)"]
    assert out["kind"].tolist() == ["show", "ask", "story"]
    assert out["has_year_tag"].tolist() == [1.0, 0.0, 0.0]
    assert out["has_question"].tolist() == [0.0, 1.0, 0.0]
    assert out["has_digit"].tolist() == [1.0, 0.0, 1.0]
    assert out["text_post"].tolist() == [0.0, 1.0, 1.0]
    assert out["dow"].tolist() == ["6", "0", "0"]


def test_pipeline_fits_and_scores_without_url_column():
    params = yaml.safe_load(open("params.yaml"))
    params["features"].update(word_hash_bits=12, char_hash_bits=12, domain_min_count=1)
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "title": [f"rust lisp {i}" if i % 5 == 0 else f"earnings report {i}" for i in range(n)],
        "url": [f"https://site{i % 7}.com/p" for i in range(n)],
        "posted_at": pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC").astype(str),
    })
    y = np.array([i % 5 == 0 for i in range(n)]) ^ (rng.random(n) < 0.05)
    model = make_model(params, include=("word", "char", "domain", "categorical", "numeric")).fit(df[INPUT_COLUMNS], y)
    prob = model.predict_proba(pd.DataFrame({"title": ["rust lisp new"], "url": [None],
                                             "posted_at": ["2025-02-01T00:00:00Z"]}))[:, 1]
    assert 0.5 < prob[0] <= 1


def test_precision_at_k_per_day():
    posted = ["2025-01-01T01:00Z", "2025-01-01T02:00Z", "2025-01-02T01:00Z", "2025-01-02T02:00Z"]
    assert precision_at_k_per_day(posted, [1, 0, 0, 1], [0.9, 0.1, 0.8, 0.2], k=1) == 0.5
