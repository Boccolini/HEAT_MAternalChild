#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/heat_mnch_remote"
OUTDIR="$ROOT/artifacts/remote_benchmarks/e002/output"
LOGDIR="$ROOT/artifacts/remote_benchmarks/e002/logs"
STATUS="$OUTDIR/gcp_year2024_diagnostics_status.txt"
FIT_LOG="$LOGDIR/year2024_all_ufs_diagnostics_rung.log"
TIME_BIN="/usr/bin/time"
SHUTDOWN_DELAY_MINUTES="${SHUTDOWN_DELAY_MINUTES:-5}"
CURRENT_STEP="init"

mark_step() {
  CURRENT_STEP="$1"
  printf "current_step=%s\n" "$CURRENT_STEP" >> "$STATUS"
}

fail() {
  local rc="$1"
  local line="$2"
  printf "failure_step=%s\n" "$CURRENT_STEP" >> "$STATUS"
  printf "failure_line=%s\n" "$line" >> "$STATUS"
  printf "finished_at=%s\n" "$(date -Iseconds)" >> "$STATUS"
  printf "workflow_exit_code=%s\n" "$rc" >> "$STATUS"
  sudo -n shutdown -h +"${SHUTDOWN_DELAY_MINUTES}" >/dev/null 2>&1 || true
  exit "$rc"
}

mkdir -p "$OUTDIR" "$LOGDIR"

: > "$STATUS"
printf "started_at=%s\n" "$(date -Iseconds)" >> "$STATUS"
printf "policy=gcp_autoshutdown_authorized_by_user_for_this_host_only\n" >> "$STATUS"
printf "target=year2024_all_ufs_diagnostics_models_2_and_3\n" >> "$STATUS"

mark_step "verify_existing_input"
if [ ! -f "$OUTDIR/year2024_all_ufs_base_input.csv" ]; then
  printf "missing_input=%s\n" "$OUTDIR/year2024_all_ufs_base_input.csv" >> "$STATUS"
  fail 2 "$LINENO"
fi

mark_step "fit_models_2_and_3"
"$TIME_BIN" -v Rscript "$ROOT/scripts/e002_fit_2024_diagnostics_remote.R" \
  --lib-dir="$HOME/R/x86_64-pc-linux-gnu-library/4.2" \
  --input="$OUTDIR/year2024_all_ufs_base_input.csv" \
  --slice-name="year2024_all_ufs_base" \
  --output-dir="$OUTDIR" > "$FIT_LOG" 2>&1 || fail "$?" "$LINENO"

printf "result_path=%s\n" "$OUTDIR/year2024_all_ufs_base_diagnostics_result.csv" >> "$STATUS"
printf "finished_at=%s\n" "$(date -Iseconds)" >> "$STATUS"
printf "workflow_exit_code=0\n" >> "$STATUS"
printf "shutdown_delay_minutes=%s\n" "$SHUTDOWN_DELAY_MINUTES" >> "$STATUS"

sudo -n shutdown -h +"${SHUTDOWN_DELAY_MINUTES}" >/dev/null 2>&1 || true
