#!/usr/bin/env bash
# Creates the gateway API keys (one for the in-cluster ingestor, one public), seals them for the frontpage namespace.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT=deploy/frontpage/secrets/gateway-api-keys.yaml
if [[ -f "$OUT" && "${1:-}" != "--force" ]]; then
  echo "$OUT exists; use --force to rotate the keys."; exit 1
fi
INGEST="ing_$(openssl rand -hex 16)"
PUBLIC="pub_$(openssl rand -hex 16)"
kubectl create secret generic gateway-api-keys -n frontpage \
  --from-literal=API_KEYS="${INGEST},${PUBLIC}" --from-literal=INGEST_KEY="${INGEST}" --dry-run=client -o yaml \
  | kubeseal --format yaml --controller-namespace kube-system --controller-name sealed-secrets-controller > "$OUT"
umask 077
touch ~/frontpage-credentials.txt
sed -i '/^GATEWAY_PUBLIC_API_KEY=/d' ~/frontpage-credentials.txt
echo "GATEWAY_PUBLIC_API_KEY=${PUBLIC}" >> ~/frontpage-credentials.txt
grep -c "kind: SealedSecret" "$OUT" >/dev/null && echo "Sealed $OUT. Public key saved to ~/frontpage-credentials.txt"
