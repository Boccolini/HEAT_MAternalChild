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

mkdir -p "${REPO_ROOT}/tmp/logs"

OBSERVA_API_HOST="${OBSERVA_API_HOST:-127.0.0.1}"
OBSERVA_API_PORT="${OBSERVA_API_PORT:-8000}"
OBSERVA_DASHBOARD_DIR="${OBSERVA_DASHBOARD_DIR:-${REPO_ROOT}/artifacts/dashboard/v1}"

exec "${REPO_ROOT}/.venv/bin/heat-mnch" \
  serve-dashboard-api \
  --dashboard-dir "${OBSERVA_DASHBOARD_DIR}" \
  --host "${OBSERVA_API_HOST}" \
  --port "${OBSERVA_API_PORT}"
