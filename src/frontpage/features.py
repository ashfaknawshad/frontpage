import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import FunctionTransformer, Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

INPUT_COLUMNS = ["title", "url", "posted_at"]
NUMERIC = ["title_len", "has_digit", "has_question", "has_year_tag", "hour_sin", "hour_cos", "text_post"]
KINDS = [("Show HN", "show"), ("Ask HN", "ask"), ("Launch HN", "launch")]
DOMAIN_RE = r"^[a-zA-Z][a-zA-Z0-9+.-]*://(?:[^@/]*@)?([^/:?#]+)"


def kind_of(title):
    for prefix, kind in KINDS:
        if title.startswith(prefix):
            return kind
    return "story"


def add_features(df):
    out = pd.DataFrame(index=df.index)
    title = df["title"].fillna("").astype(str).str.strip()
    url = df["url"].astype("string") if "url" in df else pd.Series(pd.NA, index=df.index, dtype="string")
    url = url.where(url.str.len() > 0)
    posted = pd.to_datetime(df["posted_at"], utc=True)
    host = url.str.extract(DOMAIN_RE, expand=False).str.lower().str.replace(r"^www\.", "", regex=True)
    hour = posted.dt.hour + posted.dt.minute / 60.0
    out["title"] = title
    out["domain"] = host.fillna("(text post)").astype(object)
    out["kind"] = title.map(kind_of).astype(object)
    out["dow"] = posted.dt.dayofweek.astype(str).astype(object)
    out["title_len"] = title.str.len().astype(float)
    out["has_digit"] = title.str.contains(r"\d", regex=True).astype(float)
    out["has_question"] = title.str.contains("?", regex=False).astype(float)
    out["has_year_tag"] = title.str.contains(r"\((?:19|20)\d{2}\)", regex=True).astype(float)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["text_post"] = url.isna().astype(float)
    return out


def _text_branch(analyzer, ngram, bits):
    return make_pipeline(
        HashingVectorizer(analyzer=analyzer, ngram_range=tuple(ngram), n_features=2 ** bits,
                          alternate_sign=False, norm=None, lowercase=True, dtype=np.float32),
        TfidfTransformer(sublinear_tf=True),
    )


def build_features(params, include=("word", "char", "domain", "categorical", "numeric")):
    p = params["features"]
    branches = {
        "word": (_text_branch("word", (1, p["word_ngram_max"]), p["word_hash_bits"]), "title"),
        "char": (_text_branch("char_wb", p["char_ngram"], p["char_hash_bits"]), "title"),
        "domain": (OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=p["domain_min_count"],
                                 dtype=np.float32), ["domain"]),
        "categorical": (OneHotEncoder(handle_unknown="ignore", dtype=np.float32), ["kind", "dow"]),
        "numeric": (StandardScaler(), NUMERIC),
    }
    transformers = [(name, *branches[name]) for name in include]
    return Pipeline([
        ("derive", FunctionTransformer(add_features)),
        ("columns", ColumnTransformer(transformers, sparse_threshold=1.0)),
    ])
