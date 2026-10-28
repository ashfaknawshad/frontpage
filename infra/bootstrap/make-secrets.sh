#!/usr/bin/env bash
# Generates every Day-2 password, seals each one for its namespace, and writes
# the SealedSecrets into the repo. Plaintext copies go to ~/frontpage-credentials.txt only.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if ls deploy/platform/data/secrets/*.yaml >/dev/null 2>&1 && [[ "${1:-}" != "--force" ]]; then
  echo "Sealed secrets already exist. Re-running would change live passwords. Use --force only if you mean it."
  exit 1
fi

for bin in kubeseal htpasswd openssl kubectl; do
  command -v "$bin" >/dev/null || { echo "Missing $bin"; exit 1; }
done

gen() { openssl rand -hex 16; }
seal() { kubeseal --format yaml --controller-namespace kube-system --controller-name sealed-secrets-controller; }
basic_auth() { kubectl create secret generic "$1" -n "$2" --type=kubernetes.io/basic-auth \
  --from-literal=username="$3" --from-literal=password="$4" --dry-run=client -o yaml | seal; }
s3_creds() { kubectl create secret generic "$1" -n "$2" \
  --from-literal=AWS_ACCESS_KEY_ID="$3" --from-literal=AWS_SECRET_ACCESS_KEY="$4" --dry-run=client -o yaml | seal; }

PG_MLFLOW="$(gen)"
PG_FRONTPAGE="$(gen)"
S3_KEY="fp$(openssl rand -hex 8)"
S3_SECRET="$(gen)"
MLFLOW_UI="$(gen)"

basic_auth pg-mlflow     data      mlflow    "$PG_MLFLOW"    > deploy/platform/data/secrets/pg-mlflow.yaml
basic_auth pg-frontpage  data      frontpage "$PG_FRONTPAGE" > deploy/platform/data/secrets/pg-frontpage.yaml
s3_creds   seaweedfs-s3  data      "$S3_KEY" "$S3_SECRET"    > deploy/platform/data/secrets/seaweedfs-s3.yaml

basic_auth pg-mlflow     mlflow    mlflow    "$PG_MLFLOW"    > deploy/platform/mlflow/secrets/pg-mlflow.yaml
s3_creds   mlflow-s3     mlflow    "$S3_KEY" "$S3_SECRET"    > deploy/platform/mlflow/secrets/mlflow-s3.yaml
kubectl create secret generic mlflow-basic-auth -n mlflow \
  --from-literal=users="$(htpasswd -nbB mlflow "$MLFLOW_UI")" --dry-run=client -o yaml | seal \
  > deploy/platform/mlflow/secrets/mlflow-basic-auth.yaml

basic_auth pg-frontpage  frontpage frontpage "$PG_FRONTPAGE" > deploy/frontpage/secrets/pg-frontpage.yaml
s3_creds   frontpage-s3  frontpage "$S3_KEY" "$S3_SECRET"    > deploy/frontpage/secrets/frontpage-s3.yaml

umask 077
cat > ~/frontpage-credentials.txt <<CREDS
# Frontpage credentials, generated $(date -u +%Y-%m-%dT%H:%MZ). Never commit this file.
MLFLOW_TRACKING_USERNAME=mlflow
MLFLOW_TRACKING_PASSWORD=$MLFLOW_UI
PG_MLFLOW_PASSWORD=$PG_MLFLOW
PG_FRONTPAGE_PASSWORD=$PG_FRONTPAGE
S3_ACCESS_KEY_ID=$S3_KEY
S3_SECRET_ACCESS_KEY=$S3_SECRET
CREDS

echo "Sealed secrets written. Plaintext copy: ~/frontpage-credentials.txt (chmod 600)."
git status --short deploy/
