#!/usr/bin/env bash
# One full pass@1 evaluation of a model on this node: serve it with serve_sglang.sh (patched,
# training-matched tool-call parsing), wait until it answers, launch launch_eval.sh pass1, then
# grade on Modal once every run_infer process has exited (eval_then_grade.sh). Run on the node
# (inside its PBS allocation); everything is started detached, logs go next to the run dir.
#
# usage: run_model_eval.sh <hf model dir> <served name> <llm config> <run dir> <node-local dir>
# env:   OPENHANDS_APPTAINER_SIF_DIR (agent SIFs, with all_ids.txt)
set -euo pipefail
MODEL="$1"; SERVED="$2"; LLM_CONFIG="$3"; RUN="$4"; LOCAL="$5"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
: "${OPENHANDS_APPTAINER_SIF_DIR:?set OPENHANDS_APPTAINER_SIF_DIR}"
[[ -e "$RUN" ]] && { echo "run dir exists, refusing to reuse it: $RUN" >&2; exit 1; }
LOGS="$(dirname "$RUN")/logs"
mkdir -p "$LOGS"
NAME="$(basename "$RUN")"

if curl -s -m 3 localhost:30000/v1/models | grep -q .; then
  echo "an LLM server already answers on port 30000; stop it first" >&2; exit 1
fi
PBS_LOCALDIR="$LOCAL" setsid nohup bash "$HERE/serve_sglang.sh" "$MODEL" "$SERVED" 30000 \
  > "$LOGS/sglang_$NAME.log" 2>&1 < /dev/null &
for _ in $(seq 1 120); do
  curl -s -m 3 localhost:30000/v1/models | grep -q "\"$SERVED\"" && break
  sleep 5
done
curl -s -m 3 localhost:30000/v1/models | grep -q "\"$SERVED\"" || { echo "SGLang did not come up; see $LOGS/sglang_$NAME.log" >&2; exit 1; }
echo "$(date +%T) SGLang serving $SERVED"

cd "$REPO"
bash "$HERE/launch_eval.sh" pass1 "$LLM_CONFIG" "$RUN" "$LOCAL"
setsid nohup bash "$HERE/eval_then_grade.sh" "$RUN" "$LOCAL" modal \
  > "$LOGS/eval_then_grade_$NAME.log" 2>&1 < /dev/null &
echo "$(date +%T) inference running; Modal grading starts when it finishes ($LOGS/eval_then_grade_$NAME.log)"
