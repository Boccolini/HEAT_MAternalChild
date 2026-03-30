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

CADDY_BIN="${OBSERVA_CADDY_BIN:-/opt/homebrew/opt/caddy/bin/caddy}"
CADDY_CONFIG="${OBSERVA_CADDY_CONFIG:-${REPO_ROOT}/deploy/Caddyfile}"

if [[ ! -x "${CADDY_BIN}" ]]; then
  echo "Caddy binary not found at ${CADDY_BIN}" >&2
  exit 1
fi

export XDG_DATA_HOME="${OBSERVA_CADDY_DATA_HOME:-/opt/homebrew/var/lib}"
export HOME="${OBSERVA_CADDY_HOME:-/opt/homebrew/var/lib}"

exec "${CADDY_BIN}" run --config "${CADDY_CONFIG}" --adapter caddyfile
