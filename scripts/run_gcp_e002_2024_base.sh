#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/heat_mnch_remote"
OUTDIR="$ROOT/artifacts/remote_benchmarks/e002/output"
LOGDIR="$ROOT/artifacts/remote_benchmarks/e002/logs"
TMPDIR="$ROOT/artifacts/remote_benchmarks/e002/tmp/year2024_all_ufs_base"
STATUS="$OUTDIR/gcp_year2024_base_status.txt"
ASSEMBLE_LOG="$LOGDIR/year2024_all_ufs_base_assemble.log"
FIT_LOG="$LOGDIR/year2024_all_ufs_base_rung.log"
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

mkdir -p "$OUTDIR" "$LOGDIR" "$TMPDIR"

: > "$STATUS"
printf "started_at=%s\n" "$(date -Iseconds)" >> "$STATUS"
printf "policy=gcp_autoshutdown_authorized_by_user_for_this_host_only\n" >> "$STATUS"
printf "target=year2024_all_ufs_base_simple\n" >> "$STATUS"

mark_step "sync_check"
find "$ROOT/artifacts/remote_benchmarks/e002/block_stage/YEAR=2024" -type f -name '*.parquet' | wc -l > "$LOGDIR/year2024_all_ufs_base_block_count.txt"

mark_step "assemble_year2024_input"
"$ROOT/.venv/bin/python" "$ROOT/scripts/e002_assemble_year_remote.py" \
  --input-glob "$ROOT/artifacts/remote_benchmarks/e002/block_stage/YEAR=2024/*.parquet" \
  --output-csv "$OUTDIR/year2024_all_ufs_base_input.csv" \
  --summary-csv "$OUTDIR/year2024_all_ufs_base_input_summary.csv" \
  --label "year2024_all_ufs_base" \
  --threads 16 \
  --memory-limit 32GB \
  --temp-dir "$TMPDIR" \
  --column-set minfit > "$ASSEMBLE_LOG" 2>&1 || fail "$?" "$LINENO"

printf "year2024_input_summary=%s\n" "$OUTDIR/year2024_all_ufs_base_input_summary.csv" >> "$STATUS"

mark_step "fit_year2024_base_simple"
"$TIME_BIN" -v Rscript "$ROOT/scripts/e002_fit_simple_remote.R" \
  --lib-dir="$HOME/R/x86_64-pc-linux-gnu-library/4.2" \
  --input="$OUTDIR/year2024_all_ufs_base_input.csv" \
  --slice-name="year2024_all_ufs_base" \
  --output-dir="$OUTDIR" > "$FIT_LOG" 2>&1 || fail "$?" "$LINENO"

printf "result_path=%s\n" "$OUTDIR/year2024_all_ufs_base_simple_result.csv" >> "$STATUS"
printf "finished_at=%s\n" "$(date -Iseconds)" >> "$STATUS"
printf "workflow_exit_code=0\n" >> "$STATUS"
printf "shutdown_delay_minutes=%s\n" "$SHUTDOWN_DELAY_MINUTES" >> "$STATUS"

sudo -n shutdown -h +"${SHUTDOWN_DELAY_MINUTES}" >/dev/null 2>&1 || true
