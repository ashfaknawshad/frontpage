import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline

from frontpage.features import INPUT_COLUMNS, build_features
from frontpage.metrics import evaluate

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")
ROOT = Path(__file__).resolve().parents[2]


def load_params(path="params.yaml"):
    return yaml.safe_load(open(path))


def data_md5(dvc_file="data/hn_stories.parquet.dvc"):
    try:
        return yaml.safe_load(open(dvc_file))["outs"][0]["md5"]
    except Exception:
        return "unknown"


def git_sha():
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def split(df, params):
    df = df[df["score_reliable"]].copy()
    d = params["data"]
    pick = lambda r: df[(df["month"] >= r[0]) & (df["month"] <= r[1])].reset_index(drop=True)
    return pick(d["train"]), pick(d["val"]), pick(d["test"])


def make_model(params, include):
    m = params["model"]
    return Pipeline([
        ("features", build_features(params, include)),
        ("clf", LogisticRegression(C=m["C"], class_weight=m["class_weight"], max_iter=m["max_iter"],
                                   solver="liblinear")),
    ])


def fit_and_score(name, model, train, val, test, k):
    t0 = time.time()
    model.fit(train[INPUT_COLUMNS], train["label"])
    fit_s = time.time() - t0
    scores = {}
    for split_name, frame in (("val", val), ("test", test)):
        prob = model.predict_proba(frame[INPUT_COLUMNS])[:, 1]
        scores[split_name] = evaluate(frame["label"], prob, frame["posted_at"], k)
        scores[split_name + "_prob"] = prob
    t0 = time.time()
    model.predict_proba(test[INPUT_COLUMNS].head(1))
    scores["single_row_ms"] = (time.time() - t0) * 1000
    scores["fit_seconds"] = fit_s
    log.info("%s: val PR-AUC %.4f, test PR-AUC %.4f, test P@%d/day %.3f, fit %.0fs", name,
             scores["val"]["pr_auc"], scores["test"]["pr_auc"], k,
             scores["test"][f"precision_at_{k}_per_day"], fit_s)
    return scores


def plots(y, prob, out_dir, label):
    precision, recall, _ = precision_recall_curve(y, prob)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(recall, precision)
    ax[0].axhline(np.mean(y), ls="--", c="grey", label=f"base rate {np.mean(y):.3f}")
    ax[0].set(xlabel="recall", ylabel="precision", title=f"PR curve ({label})")
    ax[0].legend()
    frac_pos, mean_pred = calibration_curve(y, prob, n_bins=15, strategy="quantile")
    ax[1].plot(mean_pred, frac_pos, marker="o")
    ax[1].plot([0, 1], [0, 1], ls="--", c="grey")
    ax[1].set(xlabel="predicted probability", ylabel="observed rate", title="Calibration", xscale="log", yscale="log")
    fig.tight_layout()
    path = Path(out_dir) / "pr_and_calibration.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def top_terms(model, train, n=25):
    from sklearn.feature_extraction.text import CountVectorizer

    words = CountVectorizer(min_df=50).fit(train["title"]).get_feature_names_out()
    cols = model.named_steps["features"].named_steps["columns"]
    hasher = cols.named_transformers_["word"].steps[0][1]
    idx = hasher.transform(list(words)).tocsr().indices
    offset = cols.output_indices_["word"].start
    coef = model.named_steps["clf"].coef_.ravel()
    word_coef = pd.Series(coef[offset + idx], index=words)
    return pd.DataFrame({"hit_words": word_coef.nlargest(n).index, "miss_words": word_coef.nsmallest(n).index})


def baseline_rate_scores(train, val, test, k):
    rate = train["label"].mean()
    out = {}
    for split_name, frame in (("val", val), ("test", test)):
        out[split_name] = evaluate(frame["label"], np.full(len(frame), rate), frame["posted_at"], k)
    return out


def champion_test_pr_auc(client, model_name):
    try:
        mv = client.get_model_version_by_alias(model_name, "champion")
        return client.get_run(mv.run_id).data.metrics.get("test_pr_auc"), mv.version
    except Exception:
        return None, None


def main():
    params = load_params()
    k = params["eval"]["top_k_per_day"]
    df = pd.read_parquet(params["data"]["path"])
    train, val, test = split(df, params)
    log.info("rows: train %d, val %d, test %d (positive rate %.3f / %.3f / %.3f)", len(train), len(val), len(test),
             train["label"].mean(), val["label"].mean(), test["label"].mean())
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    tracking = bool(os.environ.get("MLFLOW_TRACKING_URI"))
    tags = {"git_sha": git_sha(), "data_md5": data_md5(), "trigger": os.environ.get("TRAIN_TRIGGER", "manual")}
    if tracking:
        import mlflow

        mlflow.set_experiment(params["mlflow"]["experiment"])
        mlflow.enable_system_metrics_logging()
        mlflow.start_run(run_name="logreg-full", tags={**tags, "kind": "candidate"})
    try:
        results = {"base_rate": baseline_rate_scores(train, val, test, k)}
        domain_model = make_model(params, include=("domain", "numeric"))
        results["domain_only"] = fit_and_score("domain_only", domain_model, train, val, test, k)
        model = make_model(params, include=("word", "char", "domain", "categorical", "numeric"))
        results["logreg_full"] = fit_and_score("logreg_full", model, train, val, test, k)
        joblib.dump(model, "models/pipeline.joblib", compress=3)

        lift = results["logreg_full"]["val"]["pr_auc"] - results["domain_only"]["val"]["pr_auc"]
        best = results["logreg_full"]
        metrics = {
            "rows_train": len(train), "rows_val": len(val), "rows_test": len(test),
            "model_size_mb": round(os.path.getsize("models/pipeline.joblib") / 1e6, 2),
            "single_row_ms": round(best["single_row_ms"], 2), "fit_seconds": round(best["fit_seconds"], 1),
            "lift_over_domain_val_pr_auc": round(lift, 4),
        }
        for name in ("base_rate", "domain_only", "logreg_full"):
            for split_name in ("val", "test"):
                for metric, value in results[name][split_name].items():
                    metrics[f"{name}.{split_name}_{metric}"] = round(value, 5)
        Path("reports/metrics.json").write_text(json.dumps(metrics, indent=2))

        if not tracking:
            log.info("MLFLOW_TRACKING_URI not set; skipping MLflow. Metrics in reports/metrics.json")
            return
        log_candidate(params, results, model, train, test, metrics, lift, tags)
        mlflow.end_run()
        log_baselines(params, results, tags)
    except Exception:
        if tracking:
            mlflow.end_run(status="FAILED")
        raise


def log_baselines(params, results, tags):
    import mlflow

    for name in ("base_rate", "domain_only"):
        with mlflow.start_run(run_name=f"baseline-{name}", tags={**tags, "kind": "baseline"}):
            mlflow.log_params({"model": name, **{f"data.{s}": "..".join(params["data"][s]) for s in ("train", "val", "test")}})
            for split_name in ("val", "test"):
                mlflow.log_metrics({f"{split_name}_{m}": v for m, v in results[name][split_name].items()})


def log_candidate(params, results, model, train, test, metrics, lift, tags):
    import mlflow
    from mlflow.models import ModelSignature
    from mlflow.tracking import MlflowClient
    from mlflow.types import ColSpec, Schema

    from frontpage.serving import FrontpageModel

    model_name = params["mlflow"]["registered_model"]
    client = MlflowClient()
    run = mlflow.active_run()
    md5, sha = tags["data_md5"], tags["git_sha"]
    dataset = mlflow.data.from_pandas(train.head(1000), name="hn_stories", digest=md5[:12],
                                      source="https://dagshub.com/ashfaknawshad/frontpage/src/main/data/hn_stories.parquet")
    mlflow.log_input(dataset, context="training")
    flat = {f"{section}.{key}": value for section in ("data", "features", "model", "eval", "gate")
            for key, value in params[section].items()}
    mlflow.log_params({k_: (str(v) if isinstance(v, list) else v) for k_, v in flat.items()})
    for split_name in ("val", "test"):
        mlflow.log_metrics({f"{split_name}_{m}": v for m, v in results["logreg_full"][split_name].items()})
    mlflow.log_metrics({"model_size_mb": metrics["model_size_mb"], "single_row_ms": metrics["single_row_ms"],
                        "fit_seconds": metrics["fit_seconds"], "lift_over_domain_val_pr_auc": lift,
                        "domain_only_val_pr_auc": results["domain_only"]["val"]["pr_auc"],
                        "base_rate_val_pr_auc": results["base_rate"]["val"]["pr_auc"]})
    with tempfile.TemporaryDirectory() as tmp:
        mlflow.log_artifact(str(plots(test["label"].to_numpy(), results["logreg_full"]["test_prob"], tmp, "test")))
        try:
            p = Path(tmp) / "top_terms.csv"
            top_terms(model, train).to_csv(p, index=False)
            mlflow.log_artifact(str(p))
        except Exception as exc:
            log.warning("top terms skipped: %s", exc)
    mlflow.log_artifact("reports/metrics.json")

    example = test[INPUT_COLUMNS].head(3).assign(posted_at=lambda d: d["posted_at"].astype(str))
    signature = ModelSignature(
        inputs=Schema([ColSpec("string", "title"), ColSpec("string", "url", required=False),
                       ColSpec("string", "posted_at")]),
        outputs=Schema([ColSpec("double", "prob")]),
    )
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=FrontpageModel(),
        artifacts={"pipeline": "models/pipeline.joblib"},
        code_paths=[str(ROOT / "src" / "frontpage")],
        signature=signature,
        input_example=example,
        pip_requirements=["mlflow==3.16.0", "scikit-learn==1.9.1", "pandas==3.0.5", "numpy==2.5.3",
                          "scipy==1.18.1", "joblib"],
    )

    val_pr = results["logreg_full"]["val"]["pr_auc"]
    gate_ok = lift >= params["gate"]["min_lift_over_domain_baseline"] and val_pr > results["base_rate"]["val"]["pr_auc"]
    champ_pr, champ_version = champion_test_pr_auc(client, model_name)
    test_pr = results["logreg_full"]["test"]["pr_auc"]
    beats_champion = champ_pr is None or test_pr >= champ_pr
    passed = gate_ok and beats_champion
    mlflow.set_tags({"gate": "passed" if passed else "failed", "champion_test_pr_auc": str(champ_pr)})
    gate = {"passed": passed, "run_id": run.info.run_id, "lift": lift, "champion_version": champ_version,
            "champion_test_pr_auc": champ_pr, "test_pr_auc": test_pr}
    if not passed:
        log.warning("gate failed (lift %.4f, champion %s vs %.4f); not registering", lift, champ_pr, test_pr)
        Path("reports/gate.json").write_text(json.dumps(gate, indent=2))
        return
    mv = mlflow.register_model(info.model_uri, model_name, tags={"git_sha": sha, "data_md5": md5})
    client.set_registered_model_alias(model_name, "challenger", mv.version)
    if champ_version is None:
        client.set_registered_model_alias(model_name, "champion", mv.version)
    gate.update(version=mv.version, artifact_uri=client.get_model_version_download_uri(model_name, mv.version))
    Path("reports/gate.json").write_text(json.dumps(gate, indent=2))
    log.info("registered %s v%s as challenger%s", model_name, mv.version,
             " and champion (first model)" if champ_version is None else "")


if __name__ == "__main__":
    main()
