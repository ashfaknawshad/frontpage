import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def precision_at_k_per_day(posted_at, y, prob, k):
    day = pd.Series(pd.to_datetime(pd.Series(posted_at).to_numpy(), utc=True)).dt.date.to_numpy()
    frame = pd.DataFrame({"day": day, "y": np.asarray(y), "p": np.asarray(prob)})
    top = frame.sort_values("p", ascending=False).groupby("day", sort=False).head(k)
    return float(top.groupby("day")["y"].mean().mean())


def evaluate(y, prob, posted_at, k):
    y = np.asarray(y).astype(int)
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "pr_auc": float(average_precision_score(y, prob)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
        "log_loss": float(log_loss(y, prob)),
        f"precision_at_{k}_per_day": precision_at_k_per_day(posted_at, y, prob, k),
        "base_rate": float(y.mean()),
    }
