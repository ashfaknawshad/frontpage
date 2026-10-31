#!/usr/bin/env bash
# Adds a single-node k3s for Frontpage to a server that already runs other services.
# Leaves ports 80/443, the firewall policy, SSH settings, hostname and timezone alone.
# k3s Traefik listens on NodePort 30080 (HTTP) bound only to Docker bridge addresses,
# so the existing reverse proxy (Caddy) can forward to it and the internet cannot.
# Run as root:  bash bootstrap-node-shared.sh <new-username> <public-ip>
set -euo pipefail

NEW_USER="${1:?usage: bash bootstrap-node-shared.sh <new-username> <public-ip>}"
PUBLIC_IP="${2:?usage: bash bootstrap-node-shared.sh <new-username> <public-ip>}"
NODE_NAME="frontpage"
SWAP_GB=2
K3S_GOMEMLIMIT="1GiB"
TRAEFIK_HTTP_NODEPORT=30080
DOCKER_NETS="172.16.0.0/12"

[[ $EUID -eq 0 ]] || { echo "Run as root"; exit 1; }
. /etc/os-release
[[ "${ID}" == "ubuntu" || "${ID}" == "debian" ]] || { echo "Expected Ubuntu/Debian, found ${ID}"; exit 1; }

echo "==> Packages (no upgrades of existing software)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y curl git jq unzip apache2-utils

echo "==> Swap (${SWAP_GB} GB file, swappiness 10)"
if ! swapon --show=NAME --noheadings | grep -qx /swapfile; then
  [[ -f /swapfile ]] || fallocate -l "${SWAP_GB}G" /swapfile
  chmod 600 /swapfile
  mkswap /swapfile >/dev/null
  swapon /swapfile
fi
grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
echo 'vm.swappiness=10' > /etc/sysctl.d/99-swap.conf
sysctl --system >/dev/null

echo "==> User ${NEW_USER} (sudo, same SSH keys as root). Root login is NOT changed."
id "${NEW_USER}" &>/dev/null || adduser --disabled-password --gecos "" "${NEW_USER}"
usermod -aG sudo "${NEW_USER}"
echo "${NEW_USER} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/90-${NEW_USER}"
chmod 440 "/etc/sudoers.d/90-${NEW_USER}"
visudo -cf "/etc/sudoers.d/90-${NEW_USER}" >/dev/null
USER_HOME="$(getent passwd "${NEW_USER}" | cut -d: -f6)"
install -d -m 700 -o "${NEW_USER}" -g "${NEW_USER}" "${USER_HOME}/.ssh"
touch "${USER_HOME}/.ssh/authorized_keys"
if [[ -s /root/.ssh/authorized_keys ]]; then cat /root/.ssh/authorized_keys >> "${USER_HOME}/.ssh/authorized_keys"; fi
sort -u -o "${USER_HOME}/.ssh/authorized_keys" "${USER_HOME}/.ssh/authorized_keys"
chown "${NEW_USER}:${NEW_USER}" "${USER_HOME}/.ssh/authorized_keys"
chmod 600 "${USER_HOME}/.ssh/authorized_keys"
echo "${NEW_USER} has $(grep -c . "${USER_HOME}/.ssh/authorized_keys" || true) SSH key(s)."

if command -v ufw >/dev/null && ufw status | grep -q "Status: active"; then
  echo "==> ufw: allow k3s pod/service ranges (existing rules untouched)"
  ufw allow from 10.42.0.0/16 to any comment 'k3s pods' >/dev/null
  ufw allow from 10.43.0.0/16 to any comment 'k3s services' >/dev/null
fi

echo "==> k3s config: no ServiceLB, Traefik on NodePort ${TRAEFIK_HTTP_NODEPORT}, NodePorts only on ${DOCKER_NETS}"
mkdir -p /etc/rancher/k3s /var/lib/rancher/k3s/server/manifests
cat > /etc/rancher/k3s/config.yaml <<K3S
write-kubeconfig-mode: "0644"
node-name: ${NODE_NAME}
tls-san:
  - ${PUBLIC_IP}
disable:
  - servicelb
kubelet-arg:
  - "fail-swap-on=false"
kube-proxy-arg:
  - "nodeport-addresses=${DOCKER_NETS}"
K3S
cat > /var/lib/rancher/k3s/server/manifests/traefik-config.yaml <<TRAEFIK
apiVersion: helm.cattle.io/v1
kind: HelmChartConfig
metadata:
  name: traefik
  namespace: kube-system
spec:
  valuesContent: |-
    service:
      type: NodePort
    ports:
      web:
        nodePort: ${TRAEFIK_HTTP_NODEPORT}
      websecure:
        nodePort: 30443
TRAEFIK

echo "==> Installing k3s"
curl -sfL https://get.k3s.io | sh -
grep -q '^GOMEMLIMIT=' /etc/systemd/system/k3s.service.env 2>/dev/null || \
  echo "GOMEMLIMIT=${K3S_GOMEMLIMIT}" >> /etc/systemd/system/k3s.service.env
systemctl restart k3s
for _ in $(seq 1 60); do
  k3s kubectl get node "${NODE_NAME}" 2>/dev/null | grep -q ' Ready' && break
  sleep 5
done
k3s kubectl get nodes

echo "==> kubectl, Helm, kubeseal for ${NEW_USER}"
install -d -m 700 -o "${NEW_USER}" -g "${NEW_USER}" "${USER_HOME}/.kube"
install -m 600 -o "${NEW_USER}" -g "${NEW_USER}" /etc/rancher/k3s/k3s.yaml "${USER_HOME}/.kube/config"
grep -q 'KUBECONFIG=' "${USER_HOME}/.bashrc" || echo 'export KUBECONFIG=~/.kube/config' >> "${USER_HOME}/.bashrc"
command -v helm >/dev/null || curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
KUBESEAL_VERSION=0.41.2
if ! command -v kubeseal >/dev/null; then
  curl -sSL "https://github.com/bitnami-labs/sealed-secrets/releases/download/v${KUBESEAL_VERSION}/kubeseal-${KUBESEAL_VERSION}-linux-amd64.tar.gz" \
    | tar xz -C /tmp kubeseal
  install /tmp/kubeseal /usr/local/bin/kubeseal && rm -f /tmp/kubeseal
fi

echo "==> Waiting for Traefik"
for _ in $(seq 1 60); do
  k3s kubectl -n kube-system get pods -l app.kubernetes.io/name=traefik 2>/dev/null | grep -q '1/1 *Running' && break
  sleep 5
done
k3s kubectl -n kube-system get svc traefik

echo
echo "==> Checks"
echo "Ports 80/443 should still belong to docker-proxy (sme-gpt Caddy):"
ss -tlnp | grep -E ':(80|443) ' || true
echo "Traefik NodePort from a Docker bridge address (expect 404 = Traefik answering):"
for ip in $(ip -4 -o addr show | awk '{print $4}' | cut -d/ -f1 | grep -E '^172\.(1[6-9]|2[0-9]|3[01])\.'); do
  printf '  %s:%s -> ' "$ip" "$TRAEFIK_HTTP_NODEPORT"
  curl -s -o /dev/null -m 5 -w '%{http_code}\n' "http://${ip}:${TRAEFIK_HTTP_NODEPORT}/" || echo "no answer"
done
printf 'Traefik NodePort on the public IP (expect no answer): '
curl -s -o /dev/null -m 5 -w '%{http_code}\n' "http://${PUBLIC_IP}:${TRAEFIK_HTTP_NODEPORT}/" || echo "no answer (good)"
echo
echo "Done. Test in a NEW terminal:  ssh ${NEW_USER}@${PUBLIC_IP}   then   kubectl get nodes"
