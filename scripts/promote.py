#!/usr/bin/env python3
"""Model promotion helpers.

plan          markdown comparison of a candidate version and the current champion (for the pull request body)
apply         edit the serving manifests so the candidate becomes champion and the old champion becomes challenger
sync-aliases  make the MLflow champion/challenger aliases match what the manifests say (Git is the source of truth)
"""
import argparse
import base64
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

SERVING = Path("deploy/frontpage/serving")
MODEL = "frontpage"
URI = re.compile(r'(\{ name: MODEL_URI, value: "models:/frontpage/)(\d+)(" \})')
SHOW_PROB = re.compile(r'^(\s*- \{ name: SHOW_PROBABILITY, value: ")(true|false)(" \})$', re.M)
CANARY = re.compile(r'^(\s*)- \{ name: CANARY_PERCENT, value: "[^"]*" \}$', re.M)
METRICS = [("test_pr_auc", "PR-AUC"), ("test_roc_auc", "ROC-AUC"), ("test_precision_at_20_per_day", "Precision@20 per day"),
           ("test_brier", "Brier (lower is better)"), ("test_ece", "Calibration error (lower is better)"),
           ("test_mean_prob", "Mean predicted probability"), ("test_base_rate", "Base rate of the test months")]


def read_version(text):
    found = URI.search(text)
    if not found:
        raise ValueError("no MODEL_URI line found")
    return int(found.group(2))


def set_version(text, version):
    new, count = URI.subn(lambda m: f"{m.group(1)}{version}{m.group(3)}", text)
    if count != 1:
        raise ValueError(f"expected exactly one MODEL_URI line, found {count}")
    return new


def set_show_probability(text, on):
    value = "true" if on else "false"
    if SHOW_PROB.search(text):
        return SHOW_PROB.sub(lambda m: f"{m.group(1)}{value}{m.group(3)}", text)
    found = CANARY.search(text)
    if not found:
        raise ValueError("no CANARY_PERCENT line to anchor SHOW_PROBABILITY on")
    line = f'{found.group(1)}- {{ name: SHOW_PROBABILITY, value: "{value}" }}'
    return text[:found.end()] + "\n" + line + text[found.end():]


def apply(version, show_probability, root=Path(".")):
    champion, challenger, gateway = (root / SERVING / n for n in ("champion.yaml", "challenger.yaml", "gateway.yaml"))
    current = read_version(champion.read_text())
    if version == current:
        raise SystemExit(f"v{version} is already the champion")
    champion.write_text(set_version(champion.read_text(), version))
    challenger.write_text(set_version(challenger.read_text(), current))
    if show_probability:
        gateway.write_text(set_show_probability(gateway.read_text(), True))
    return current


class MLflow:
    def __init__(self):
        self.base = os.environ["MLFLOW_TRACKING_URI"].rstrip("/")
        user, password = os.environ.get("MLFLOW_TRACKING_USERNAME", ""), os.environ.get("MLFLOW_TRACKING_PASSWORD", "")
        self.auth = "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()

    def call(self, path, body=None):
        req = urllib.request.Request(self.base + path, headers={"Authorization": self.auth, "Content-Type": "application/json"},
                                     data=None if body is None else json.dumps(body).encode(),
                                     method="GET" if body is None else "POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)

    def metrics(self, version):
        mv = self.call(f"/api/2.0/mlflow/model-versions/get?name={MODEL}&version={version}")["model_version"]
        run = self.call("/api/2.0/mlflow/runs/get?run_id=" + urllib.parse.quote(mv["run_id"]))["run"]
        return {m["key"]: m["value"] for m in run["data"]["metrics"]}, mv.get("tags", [])

    def set_alias(self, alias, version):
        self.call("/api/2.0/mlflow/registered-models/alias", {"name": MODEL, "alias": alias, "version": str(version)})


def plan(version, mlflow=None, root=Path(".")):
    mlflow = mlflow or MLflow()
    current = read_version((root / SERVING / "champion.yaml").read_text())
    new_metrics, _ = mlflow.metrics(version)
    old_metrics, _ = mlflow.metrics(current)
    fmt = lambda m, key: "n/a" if key not in m else f"{m[key]:.4f}"
    lines = [f"## Promote v{version} to champion", "",
             f"v{current} (the current champion) becomes the challenger and keeps scoring live stories in shadow, "
             f"so a rollback is one `git revert` away.", "",
             f"| Test-set metric | champion v{current} | candidate v{version} |", "|---|---|---|"]
    lines += [f"| {label} | {fmt(old_metrics, key)} | {fmt(new_metrics, key)} |" for key, label in METRICS]
    lines += ["", "Decision rule: promote when ranking quality (PR-AUC, ROC-AUC) is not lower beyond noise and "
              "probability quality (Brier, calibration error) is better. Check the head-to-head table on the model "
              "dashboard, which scores both models on the same live stories, before merging."]
    return "\n".join(lines) + "\n"


def sync_aliases(mlflow=None, root=Path(".")):
    mlflow = mlflow or MLflow()
    wanted = {"champion": read_version((root / SERVING / "champion.yaml").read_text()),
              "challenger": read_version((root / SERVING / "challenger.yaml").read_text())}
    for alias, version in wanted.items():
        mlflow.set_alias(alias, version)
    return wanted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "apply"):
        p = sub.add_parser(name)
        p.add_argument("--version", type=int, required=True)
        if name == "apply":
            p.add_argument("--show-probability", action="store_true")
    sub.add_parser("sync-aliases")
    args = parser.parse_args(argv)
    if args.cmd == "plan":
        sys.stdout.write(plan(args.version))
    elif args.cmd == "apply":
        old = apply(args.version, args.show_probability)
        print(f"champion v{old} -> v{args.version}; challenger is now v{old}")
    else:
        print("aliases now:", sync_aliases())


if __name__ == "__main__":
    main()
