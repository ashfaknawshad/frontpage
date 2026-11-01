#!/usr/bin/env bash
# Seals the Discord webhook URL for Grafana alerts. The URL is typed hidden and never printed or stored in plaintext.
# Run from the repo on the server:  bash infra/bootstrap/make-discord-secret.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
for bin in kubeseal kubectl; do
  command -v "$bin" >/dev/null || { echo "Missing $bin"; exit 1; }
done

read -rsp "Paste the Discord webhook URL (hidden), then press Enter: " URL
echo
case "$URL" in
  https://discord.com/api/webhooks/*|https://discordapp.com/api/webhooks/*) ;;
  *) echo "That does not look like a Discord webhook URL (it should start with https://discord.com/api/webhooks/)."; exit 1 ;;
esac

mkdir -p deploy/platform/monitoring/secrets
kubectl create secret generic grafana-discord -n monitoring \
  --from-literal=DISCORD_WEBHOOK_URL="$URL" --dry-run=client -o yaml \
  | kubeseal --format yaml --controller-namespace kube-system --controller-name sealed-secrets-controller \
  > deploy/platform/monitoring/secrets/grafana-discord.yaml
echo "Wrote deploy/platform/monitoring/secrets/grafana-discord.yaml"
