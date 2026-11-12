#!/usr/bin/env bash
# Launch a SWE-bench Verified inference run on this node: P run_infer processes x W workers
# (each run_infer process runs at most 20 instances at once), over P balanced shards of the
# instance list, against an OpenAI-compatible LLM server (e.g. serve_sglang.sh).
#
# Modes:
#   harness  the settings of the first evaluations: up to 500 iterations per run() call, the
#            fake user ("please continue", up to 10), 1-hour conversation timeout, and up to 4
#            attempts per instance for any error.
#   pass1    genuine pass@1, matching hyperion training: 75 turns, no fake user, 1-hour
#            conversation timeout; an agent-caused conversation error (turn limit, stuck,
#            context window, timeout, malformed tool call) ends the episode and its patch is
#            graded; only other failures (sandbox or LLM server) are retried, up to 3 attempts.
# Both: default2.j2 prompt, condenser off, prebuilt agent SIFs, sandboxes on node-local scratch.
#
# usage: launch_eval.sh <harness|pass1> <llm config json> <run dir> <node-local dir> [P=16] [W=16]
# env:   OPENHANDS_APPTAINER_SIF_DIR (agent SIFs, <instance_id>.sif; must contain all_ids.txt
#        or set IDS_FILE)
set -euo pipefail
MODE="$1"; LLM_CONFIG="$2"; RUN="$3"; LOCAL="$4"; P="${5:-16}"; W="${6:-16}"
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
: "${OPENHANDS_APPTAINER_SIF_DIR:?set OPENHANDS_APPTAINER_SIF_DIR}"
IDS="${IDS_FILE:-$OPENHANDS_APPTAINER_SIF_DIR/all_ids.txt}"
[[ -f "$IDS" ]] || { echo "missing instance list $IDS" >&2; exit 1; }
[[ -e "$RUN" ]] && { echo "run dir exists, refusing to reuse it: $RUN" >&2; exit 1; }
case "$MODE" in
  harness) ITER=500; RETRIES=3; FAKE=10; END_ON_AGENT_ERROR=0 ;;
  pass1)   ITER=75;  RETRIES=2; FAKE=0;  END_ON_AGENT_ERROR=1 ;;
  *) echo "mode must be harness or pass1" >&2; exit 1 ;;
esac
mkdir -p "$RUN" "$LOCAL/oh-scratch" "$LOCAL/apptainer-cache-run"
awk -v D="$RUN" -v P="$P" '{print > (D "/shard_" (NR-1)%P ".txt")}' "$IDS"
cat > "$RUN/launch.json" <<EOF
{"mode": "$MODE", "llm_config": "$LLM_CONFIG", "processes": $P, "workers_per_process": $W,
 "max_iterations": $ITER, "max_retries": $RETRIES, "max_fake_responses": $FAKE,
 "conversation_timeout": 3600, "end_episode_on_agent_error": $END_ON_AGENT_ERROR,
 "prompt": "default2.j2", "host": "$(hostname -s)", "started": "$(date -Is)",
 "benchmarks_commit": "$(git -C "$REPO" rev-parse HEAD)", "benchmarks_dirty": $(git -C "$REPO" diff --quiet && echo false || echo true)}
EOF
cd "$REPO"
export PATH=$HOME/.local/bin:$PATH OPENHANDS_SUPPRESS_BANNER=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OPENHANDS_APPTAINER_SCRATCH_DIR=$LOCAL/oh-scratch APPTAINER_CACHEDIR=$LOCAL/apptainer-cache-run
export CONVERSATION_TIMEOUT=3600 MAX_FAKE_RESPONSES=$FAKE
if [[ $END_ON_AGENT_ERROR == 1 ]]; then export SWEBENCH_END_EPISODE_ON_AGENT_ERROR=1
else unset SWEBENCH_END_EPISODE_ON_AGENT_ERROR; fi
for i in $(seq 0 $((P - 1))); do
  setsid nohup uv run --frozen python -m benchmarks.swebench.run_infer "$LLM_CONFIG" \
    --workspace apptainer --prompt-path benchmarks/swebench/prompts/default2.j2 \
    --select "$RUN/shard_$i.txt" --num-workers "$W" --n-critic-runs 1 \
    --max-iterations "$ITER" --max-retries "$RETRIES" --output-dir "$RUN/shard_$i" \
    > "$RUN/shard_$i.log" 2>&1 < /dev/null &
done
echo "launched $P run_infer processes ($MODE) into $RUN"
