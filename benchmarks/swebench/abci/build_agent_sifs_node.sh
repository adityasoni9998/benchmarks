#!/usr/bin/env bash
# Build SWE-bench agent-server SIFs on this node with N parallel workers, sharing the work with
# other nodes running this script on the same instance list.
#
# Each build runs build_apptainer_sifs (official image + agent server from the vendored SDK) for
# one instance, linking <SIF_DIR>/<instance_id>.sif, the layout run_infer reads via
# OPENHANDS_APPTAINER_SIF_DIR.
# Cache: Apptainer 1.5.5 rewrites $APPTAINER_CACHEDIR/cache/blob/index.json non-atomically, so
# concurrent builds must not share it. As in scripts/pull_apptainer_images.py, each build gets a
# private cache dir whose cache/blob/blobs links to this node's shared, digest-addressed blobs.
# Work sharing: workers walk $SIF_DIR/all_ids.txt (sorted, so one repo's images are adjacent)
# and claim an instance with an atomic mkdir in $SIF_DIR/claims; linked instances are skipped.
#
# usage: build_agent_sifs_node.sh <node-local dir, e.g. /local/$PBS_JOBID> <workers>
# env:   SIF_DIR (links + all_ids.txt + claims), OPENHANDS_APPTAINER_BUILD_ROOT (SIF files)
set -euo pipefail
LOCAL="$1"
WORKERS="$2"
SIF_DIR="${SIF_DIR:?set SIF_DIR}"
: "${OPENHANDS_APPTAINER_BUILD_ROOT:?set OPENHANDS_APPTAINER_BUILD_ROOT}"
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
IDS=$SIF_DIR/all_ids.txt
CLAIMS=$SIF_DIR/claims
CACHE=$LOCAL/apptainer-cache
LOGS=$SIF_DIR/node_logs/$(hostname -s)
[[ -f "$IDS" ]] || { echo "missing $IDS (one instance id per line)" >&2; exit 1; }
mkdir -p "$CLAIMS" "$CACHE/cache/blob/blobs" "$CACHE/build-metadata" "$LOCAL/apptainer-tmp" "$LOGS"
export PATH=$HOME/.local/bin:$PATH
export OPENHANDS_APPTAINER_BUILD_ROOT
export APPTAINER_TMPDIR=$LOCAL/apptainer-tmp
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset APPTAINER_DISABLE_CACHE

build_one() {
  local id="$1" private
  private=$(mktemp -d "$CACHE/build-metadata/build-XXXXXX")
  mkdir -p "$private/cache/blob"
  ln -s "$CACHE/cache/blob/blobs" "$private/cache/blob/blobs"
  printf '%s\n' "$id" > "$private/select.txt"
  APPTAINER_CACHEDIR="$private" uv run --frozen --project "$REPO" \
    python -m benchmarks.swebench.build_apptainer_sifs \
    --sif-dir "$SIF_DIR" --select "$private/select.txt" --workers 1 \
    >> "$LOGS/$id.log" 2>&1 || true
  rm -rf "$private"
}

worker() {
  local id
  while read -r id; do
    [[ -e "$SIF_DIR/$id.sif" ]] && continue
    mkdir "$CLAIMS/$id" 2>/dev/null || continue
    echo "$(date +%T) $(hostname -s) start $id"
    build_one "$id"
    if [[ -e "$SIF_DIR/$id.sif" ]]; then echo "$(date +%T) $(hostname -s) built $id"
    else echo "$(date +%T) $(hostname -s) FAILED $id"; fi
  done < "$IDS"
}

cd "$REPO"
for _ in $(seq "$WORKERS"); do worker & done
wait
echo "$(date +%T) $(hostname -s) done"
