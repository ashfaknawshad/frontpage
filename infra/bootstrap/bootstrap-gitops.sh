#!/usr/bin/env bash
# Installs Argo CD, restores the Sealed Secrets key if a backup exists, then hands the cluster to Git.
# Run as the normal user from the repo root:  bash infra/bootstrap/bootstrap-gitops.sh [path/to/sealed-secrets-key.yaml]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
KEY_BACKUP="${1:-$HOME/sealed-secrets-key.yaml}"
ARGOCD_CHART_VERSION="10.11.6"

if [[ -f "$KEY_BACKUP" ]]; then
  echo "==> Restoring Sealed Secrets key from $KEY_BACKUP"
  sed -E '/^\s+(resourceVersion|uid|creationTimestamp|selfLink):/d' "$KEY_BACKUP" | kubectl apply -f -
  kubectl -n kube-system get secret -l sealedsecrets.bitnami.com/sealed-secrets-key
else
  echo "==> No key backup at $KEY_BACKUP: the controller will make a new key; re-run make-secrets.sh --force afterwards."
fi

echo "==> Argo CD ${ARGOCD_CHART_VERSION}"
helm repo add argo https://argoproj.github.io/argo-helm >/dev/null 2>&1 || true
helm repo update argo >/dev/null
helm upgrade --install argocd argo/argo-cd -n argocd --create-namespace \
  --version "$ARGOCD_CHART_VERSION" -f infra/bootstrap/argocd-values.yaml --wait --timeout 10m

echo "==> Handing over to Git"
kubectl apply -f deploy/argocd/root.yaml
echo "Watch it rebuild:  kubectl get applications -n argocd -w"
echo "Argo CD admin password:"
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath='{.data.password}' | base64 -d; echo
