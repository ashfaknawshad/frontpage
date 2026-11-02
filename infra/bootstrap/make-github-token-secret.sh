#!/usr/bin/env bash
# Seals the GitHub token the monitor job uses to start the train workflow. Typed hidden, never printed.
# Run from the repo on the server:  bash infra/bootstrap/make-github-token-secret.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
for bin in kubeseal kubectl; do
  command -v "$bin" >/dev/null || { echo "Missing $bin"; exit 1; }
done

read -rsp "Paste the GitHub fine-grained token (hidden), then press Enter: " TOKEN
echo
case "$TOKEN" in
  github_pat_*|ghp_*) ;;
  *) echo "That does not look like a GitHub token (it should start with github_pat_)."; exit 1 ;;
esac

mkdir -p deploy/frontpage/secrets
kubectl create secret generic github-dispatch -n frontpage \
  --from-literal=GITHUB_DISPATCH_TOKEN="$TOKEN" --dry-run=client -o yaml \
  | kubeseal --format yaml --controller-namespace kube-system --controller-name sealed-secrets-controller \
  > deploy/frontpage/secrets/github-dispatch.yaml
echo "Wrote deploy/frontpage/secrets/github-dispatch.yaml"
