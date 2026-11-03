import re

import numpy as np

DIGIT = re.compile(r"\d")
YEAR_TAG = re.compile(r"\((?:19|20)\d{2}\)")
NUMERIC = ["title_len"]
BINARY = ["has_digit", "has_question", "has_year_tag", "text_post"]
CATEGORICAL = ["kind", "dow", "hour_block", "domain"]
FEATURES = NUMERIC + BINARY + CATEGORICAL
PSI_ALERT = 0.2
EPS = 1e-4
OTHER = "other"


def _clean(value):
    """None for missing values, including float NaN coming out of pandas."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    return value


def extract(rows):
    """rows: iterable of (title, url, domain, kind, posted_at). Returns one list per feature."""
    cols = {name: [] for name in FEATURES}
    for title, url, domain, kind, posted_at in rows:
        title, url, domain, kind = _clean(title), _clean(url), _clean(domain), _clean(kind)
        title = (title or "").strip()
        has_url = bool(url) and bool(str(url).strip())
        cols["title_len"].append(len(title))
        cols["has_digit"].append(1 if DIGIT.search(title) else 0)
        cols["has_question"].append(1 if "?" in title else 0)
        cols["has_year_tag"].append(1 if YEAR_TAG.search(title) else 0)
        cols["text_post"].append(0 if has_url else 1)
        cols["kind"].append(kind or "story")
        cols["dow"].append(str(posted_at.weekday()))
        cols["hour_block"].append(str(posted_at.hour // 3))
        cols["domain"].append(domain if has_url and domain else "(text post)")
    return cols


def _shares(counts):
    counts = np.asarray(counts, dtype=float)
    return counts / max(counts.sum(), 1.0)


def profile(cols, bins=10, top_domains=30):
    """Reference distribution for every feature, stored as plain JSON-able lists."""
    spec = {"n": len(cols[FEATURES[0]]), "features": {}}
    for name in NUMERIC:
        x = np.asarray(cols[name], dtype=float)
        edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1])).tolist()
        idx = np.searchsorted(edges, x, side="right")
        spec["features"][name] = {"type": "numeric", "edges": edges, "mean": float(x.mean()),
                                  "p": _shares(np.bincount(idx, minlength=len(edges) + 1)).tolist()}
    for name in BINARY:
        spec["features"][name] = {"type": "binary", "p1": float(np.mean(cols[name]))}
    for name in CATEGORICAL:
        values, counts = np.unique(np.asarray(cols[name], dtype=object).astype(str), return_counts=True)
        order = np.argsort(-counts, kind="mergesort")
        limit = top_domains if name == "domain" else len(values)
        keep = [str(values[i]) for i in order[:limit]]
        total = counts.sum()
        p = [float(counts[list(values).index(k)] / total) for k in keep]
        rest = 1.0 - sum(p)
        if rest > 1e-9:
            keep.append(OTHER)
            p.append(rest)
        spec["features"][name] = {"type": "categorical", "categories": keep, "p": p}
    return spec


def psi(expected, actual):
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _live_shares(feature, values):
    if feature["type"] == "numeric":
        x = np.asarray(values, dtype=float)
        idx = np.searchsorted(feature["edges"], x, side="right")
        return _shares(np.bincount(idx, minlength=len(feature["edges"]) + 1))
    if feature["type"] == "binary":
        p1 = float(np.mean(values))
        return np.array([1 - p1, p1])
    cats = feature["categories"]
    pos = {c: i for i, c in enumerate(cats)}
    unseen = len(cats)
    counts = np.zeros(len(cats) + (0 if OTHER in pos else 1))
    for v in values:
        key = str(v) if str(v) in pos else OTHER
        counts[pos.get(key, unseen)] += 1
    return _shares(counts)


def compare(spec, cols):
    """PSI per feature plus a short human-readable 'reference -> live' note."""
    out = {}
    for name, feature in spec["features"].items():
        live = _live_shares(feature, cols[name])
        if feature["type"] == "binary":
            ref = np.array([1 - feature["p1"], feature["p1"]])
            detail = f"{100 * feature['p1']:.1f}% -> {100 * live[1]:.1f}%"
        elif feature["type"] == "numeric":
            ref = np.asarray(feature["p"])
            detail = f"mean {feature['mean']:.1f} -> {float(np.mean(cols[name])):.1f}"
        else:
            names = list(feature["categories"])
            ref = np.asarray(feature["p"])
            if len(live) > len(ref):
                names.append("(unseen)")
                ref = np.append(ref, 0.0)
            k = int(np.argmax(np.abs(live - ref)))
            detail = f"{names[k]}: {100 * ref[k]:.1f}% -> {100 * live[k]:.1f}%"
        out[name] = {"psi": psi(ref, live), "detail": detail}
    return out


def _average_ranks(x):
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    _, start, counts = np.unique(xs, return_index=True, return_counts=True)
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.repeat(start + (counts + 1) / 2.0, counts)
    return ranks


def roc_auc(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(p)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_auc_se(auc, n_pos, n_neg):
    """Hanley-McNeil standard error of an ROC-AUC estimate."""
    if auc is None or n_pos < 2 or n_neg < 2:
        return None
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    return float(var ** 0.5)


def average_precision(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if y.sum() == 0:
        return None
    order = np.argsort(-p, kind="mergesort")
    y, p = y[order], p[order]
    tps, fps = np.cumsum(y), np.cumsum(1 - y)
    last = np.r_[np.where(np.diff(p))[0], len(p) - 1]
    tps, fps = tps[last], fps[last]
    precision, recall = tps / (tps + fps), tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def brier(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))
