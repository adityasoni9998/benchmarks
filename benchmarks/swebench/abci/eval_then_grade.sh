#!/usr/bin/env bash
# Grade an inference run once it has completely finished: wait until every run_infer process
# writing to <run dir> on this node has exited, then grade all patches once with grade_run
# into <run dir>_grading (which must not exist: no resumed grading, so no report can be reused
# for a patch that changed). Run it detached on the inference node.
#
# usage: eval_then_grade.sh <run dir> <node-local dir> [apptainer|modal, default apptainer]
# env:   GRADING_SIF_DIR (apptainer: plain grading SIFs, <instance_id>.sif)
set -uo pipefail
RUN="$(cd "$1" && pwd)"; LOCAL="$2"; BACKEND="${3:-apptainer}"
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
[[ "$BACKEND" == modal ]] || : "${GRADING_SIF_DIR:?set GRADING_SIF_DIR}"
OUT=${RUN}_grading
[[ -e "$OUT" ]] && { echo "grading dir exists, refusing to reuse it: $OUT" >&2; exit 1; }
running() { pgrep -fc "python -m benchmarks[.]swebench[.]run_infer .* --output-dir $RUN/shard_" || true; }
seen=0
while true; do
  n=$(running)
  [[ "$n" -gt 0 ]] && seen=1
  [[ "$seen" -eq 1 && "$n" -eq 0 ]] && break
  sleep 60
done
records=$(cat "$RUN"/shard_*/*/*/*/output.jsonl "$RUN"/shard_*/*/*/*/output_errors.jsonl 2>/dev/null | wc -l)
echo "$(date +%T) all run_infer processes exited; $records records -> grading"
cd "$REPO"
export PATH=$HOME/.local/bin:$PATH OPENHANDS_SUPPRESS_BANNER=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
if [[ "$BACKEND" == modal ]]; then
  # Official images, hyperion's sandbox requests, all instances in parallel.
  uv run --frozen python -m benchmarks.swebench.grade_run --run-dir "$RUN" --out "$OUT" \
    --backend modal --workers 500 > "$OUT.log" 2>&1
else
  uv run --frozen python -m benchmarks.swebench.grade_run --run-dir "$RUN" --out "$OUT" \
    --backend apptainer --sif-dir "$GRADING_SIF_DIR" --scratch "$LOCAL/grade-scratch-$(basename "$RUN")" \
    --workers 48 > "$OUT.log" 2>&1
fi
echo "$(date +%T) grading finished"
cat "$OUT/summary.json" 2>/dev/null || tail -5 "$OUT.log"
