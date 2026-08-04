#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 LOG_DIR COMMAND [ARGS...]" >&2
  exit 2
fi

log_dir="$1"
shift
mkdir -p "$log_dir"

sample_interval="${MONITOR_INTERVAL_SECONDS:-15}"

monitor_loop() {
  while kill -0 "$1" 2>/dev/null; do
    ts="$(date -Is)"
    {
      echo "===== $ts ps ====="
      ps -eo pid,ppid,stat,pcpu,pmem,wchan:24,comm,args --sort=-pcpu | head -80
    } >> "$log_dir/ps.log" 2>&1 || true
    {
      echo "===== $ts vmstat ====="
      vmstat 1 2
    } >> "$log_dir/vmstat.log" 2>&1 || true
    {
      echo "===== $ts iostat ====="
      iostat -xz 1 2
    } >> "$log_dir/iostat.log" 2>&1 || true
    {
      echo "===== $ts sar-dev ====="
      sar -n DEV 1 2
    } >> "$log_dir/sar-dev.log" 2>&1 || true
    {
      echo "===== $ts nfsiostat ====="
      nfsiostat 1 2
    } >> "$log_dir/nfsiostat.log" 2>&1 || true
    {
      echo "===== $ts ss ====="
      ss -ntp
    } >> "$log_dir/ss.log" 2>&1 || true
    sleep "$sample_interval"
  done
}

"$@" &
cmd_pid="$!"
monitor_loop "$cmd_pid" &
mon_pid="$!"

set +e
wait "$cmd_pid"
status="$?"
kill "$mon_pid" 2>/dev/null
wait "$mon_pid" 2>/dev/null
set -e

exit "$status"
