#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/deploy/observa-infancia.env"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ENV_FILE}"
  set +a
fi

OBSERVA_FRONTEND_DIST="${OBSERVA_FRONTEND_DIST:-${REPO_ROOT}/apps/observa-infancia-web/dist}"
OBSERVA_API_HOST="${OBSERVA_API_HOST:-127.0.0.1}"
OBSERVA_API_PORT="${OBSERVA_API_PORT:-8000}"
OBSERVA_CADDY_HOST="${OBSERVA_CADDY_HOST:-127.0.0.1}"
OBSERVA_CADDY_PORT="${OBSERVA_CADDY_PORT:-8080}"
OBSERVA_PUBLIC_HEALTHCHECK_URL="${OBSERVA_PUBLIC_HEALTHCHECK_URL:-}"

echo "Checking frontend build..."
test -f "${OBSERVA_FRONTEND_DIST}/index.html"

echo "Checking API health..."
curl -sf "http://${OBSERVA_API_HOST}:${OBSERVA_API_PORT}/healthz" >/dev/null

echo "Checking proxied manifest through Caddy..."
curl -sf "http://${OBSERVA_CADDY_HOST}:${OBSERVA_CADDY_PORT}/api/v1/meta/manifest" >/dev/null

echo "Checking proxied frontend through Caddy..."
curl -sf "http://${OBSERVA_CADDY_HOST}:${OBSERVA_CADDY_PORT}/overview" >/dev/null

if [[ -n "${OBSERVA_PUBLIC_HEALTHCHECK_URL}" ]]; then
  echo "Checking public hostname..."
  curl -I -sS "${OBSERVA_PUBLIC_HEALTHCHECK_URL}" | head -n 1
fi

echo "Dashboard stack checks passed."
