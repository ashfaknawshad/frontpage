import re
import sys
from pathlib import Path

DOMAIN = "frontpage-ashfak.duckdns.org"

argo = Path("infra/bootstrap/argocd-values.yaml")
t = argo.read_text()
t = re.sub(r"\n    annotations:\n      cert-manager\.io/cluster-issuer: letsencrypt-prod\n", "\n", t)
t = t.replace("    tls: true\n", "    tls: false\n")
argo.write_text(t)

ml = Path("deploy/argocd/apps/mlflow.yaml")
t = ml.read_text()
t = t.replace("              cert-manager.io/cluster-issuer: letsencrypt-prod\n", "")
t = re.sub(r"            tls:\n              - secretName: mlflow-tls\n                hosts:\n                  - mlflow\.[^\n]+\n",
           "            tls: []\n", t)
if "corsAllowedOrigins" not in t:
    t = t.replace("          serverAllowedHosts:\n",
                  f"          corsAllowedOrigins:\n            - https://mlflow.{DOMAIN}\n          serverAllowedHosts:\n", 1)
ml.write_text(t)

problems = []
if "cert-manager.io/cluster-issuer" in argo.read_text() + ml.read_text():
    problems.append("a cert-manager annotation is still present")
if "tls: true" in argo.read_text():
    problems.append("argocd ingress still has tls: true")
if "mlflow-tls" in ml.read_text():
    problems.append("mlflow ingress still has a TLS block")
if problems:
    sys.exit("NOT patched cleanly: " + "; ".join(problems))
print("ok: Argo CD and MLflow ingresses now plain HTTP behind Caddy")
