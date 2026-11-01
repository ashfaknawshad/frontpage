#!/usr/bin/env bash
# Seals the Grafana admin login, the Grafana environment (read-only Postgres password) and the Postgres role password. Plaintext copies are appended to ~/frontpage-credentials.txt only.
# Run from the repo on the server:  bash infra/bootstrap/make-monitoring-secrets.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [[ -e deploy/platform/monitoring/secrets/grafana-admin.yaml && "${1:-}" != "--force" ]]; then
  echo "Monitoring secrets already exist. Re-running would change live passwords. Use --force only if you mean it."
  exit 1
fi
for bin in kubeseal openssl kubectl; do
  command -v "$bin" >/dev/null || { echo "Missing $bin"; exit 1; }
done

gen() { openssl rand -hex 16; }
seal() { kubeseal --format yaml --controller-namespace kube-system --controller-name sealed-secrets-controller; }

GRAFANA_ADMIN="$(gen)"
PG_GRAFANA="$(gen)"

mkdir -p deploy/platform/monitoring/secrets deploy/platform/data/secrets

kubectl create secret generic grafana-admin -n monitoring \
  --from-literal=admin-user=admin --from-literal=admin-password="$GRAFANA_ADMIN" \
  --dry-run=client -o yaml | seal > deploy/platform/monitoring/secrets/grafana-admin.yaml

kubectl create secret generic grafana-env -n monitoring \
  --from-literal=GF_PG_PASSWORD="$PG_GRAFANA" \
  --dry-run=client -o yaml | seal > deploy/platform/monitoring/secrets/grafana-env.yaml

kubectl create secret generic pg-grafana -n data --type=kubernetes.io/basic-auth \
  --from-literal=username=grafana_ro --from-literal=password="$PG_GRAFANA" \
  --dry-run=client -o yaml | seal > deploy/platform/data/secrets/pg-grafana.yaml

umask 077
cat >> ~/frontpage-credentials.txt <<CREDS

# Monitoring, generated $(date -u +%Y-%m-%dT%H:%MZ)
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=$GRAFANA_ADMIN
PG_GRAFANA_RO_PASSWORD=$PG_GRAFANA
CREDS

echo "Sealed secrets written. Grafana login and password appended to ~/frontpage-credentials.txt."
git status --short deploy/
