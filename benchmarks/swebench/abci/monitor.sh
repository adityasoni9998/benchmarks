#!/usr/bin/env bash
# Read-only monitor for a launch_eval.sh run: every 5 minutes append a snapshot (records, running
# run_infer processes, node load, failed attempts by reason) to <run dir>/monitor.log; exit after
# LIMIT seconds or when every instance has a record, printing the last snapshots.
#
# usage: monitor.sh <run dir> <eval node> [limit seconds, default 900] [instances=500]
RUN="$1"; NODE="$2"; LIMIT="${3:-900}"; TOTAL="${4:-500}"
LOG=$RUN/monitor.log
records() { cat "$RUN"/shard_*/*/*/*/output.jsonl "$RUN"/shard_*/*/*/*/output_errors.jsonl 2>/dev/null | wc -l; }
reasons() {
  cat "$RUN"/shard_*.log 2>/dev/null | grep -E "failed \(attempt [0-9]/[0-9]\)|episode ended by the agent" | awk '
    {r="other"} /episode ended by the agent \(/{match($0,/agent \([a-z_]+\)/); r="ended:" substr($0,RSTART+7,RLENGTH-8)}
    /failed \(attempt/{r="retried:other"} /failed \(attempt/&&/got stuck/{r="retried:stuck"}
    /failed \(attempt/&&/context length/{r="retried:context_window"} /failed \(attempt/&&/MaxIterationsReached/{r="retried:turn_limit"}
    /failed \(attempt/&&/Run timed out after/{r="retried:timeout"} /failed \(attempt/&&/ended with error/{r="retried:ended_with_error"}
    /failed \(attempt/&&/function\.arguments must be a JSON object/{r="retried:malformed_tool_call"}
    {c[r]++} END{for(k in c) printf "%s=%d ", k, c[k]}'
}
snapshot() {
  local node
  node=$(timeout 20 ssh -n "$NODE" 'pgrep -fc "python -m benchmarks[.]swebench[.]run_infer"; cut -d" " -f1 /proc/loadavg' 2>/dev/null | tr '\n' ' ')
  echo "$(date +%T) records=$(records)/$TOTAL started=$(cat "$RUN"/shard_*.log 2>/dev/null | grep -c 'Evaluation Started') procs+load=[$node] $(reasons)" >> "$LOG"
}
while [ $SECONDS -lt "$LIMIT" ]; do
  snapshot
  [ "$(records)" -ge "$TOTAL" ] && break
  sleep 300
done
snapshot
tail -n 4 "$LOG"
