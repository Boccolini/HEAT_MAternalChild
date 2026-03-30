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

FRONTEND_DIR="${OBSERVA_FRONTEND_DIR:-${REPO_ROOT}/apps/observa-infancia-web}"

if [[ ! -d "${FRONTEND_DIR}" ]]; then
  echo "Frontend directory not found: ${FRONTEND_DIR}" >&2
  exit 1
fi

cd "${FRONTEND_DIR}"

if [[ ! -d node_modules ]]; then
  npm ci
fi

npm run build

echo "Frontend build completed at ${FRONTEND_DIR}/dist"
