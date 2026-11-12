#!/usr/bin/env bash
# Build plain SWE-bench grading SIFs: the official instance images converted as they are (no
# definition file, no %post, no agent server), which is what the SWE-bench harness grades on.
# Output: $GRADING_SIF_DIR/<instance_id>.sif, written under a temporary name and renamed only
# on success. Same cache scheme and work sharing as build_agent_sifs_node.sh.
#
# usage: build_grading_sifs_node.sh <node-local dir, e.g. /local/$PBS_JOBID> <workers>
# env:   GRADING_SIF_DIR, containing images.txt ("<instance_id> docker://<image>" per line;
#        write it with: python -m benchmarks.swebench.abci.instance_lists grading-images FILE)
#        APPTAINER (default: apptainer on PATH)
set -euo pipefail
LOCAL="$1"
WORKERS="$2"
OUT="${GRADING_SIF_DIR:?set GRADING_SIF_DIR}"
IMAGES=$OUT/images.txt
CLAIMS=$OUT/claims
LOGS=$OUT/logs
CACHE=$LOCAL/apptainer-cache
APPTAINER="${APPTAINER:-$(command -v apptainer || echo "$HOME/.local/apptainer/x86_64/bin/apptainer")}"
[[ -f "$IMAGES" ]] || { echo "missing $IMAGES" >&2; exit 1; }
mkdir -p "$CLAIMS" "$LOGS" "$CACHE/cache/blob/blobs" "$CACHE/build-metadata" "$LOCAL/apptainer-tmp"
export APPTAINER_TMPDIR=$LOCAL/apptainer-tmp
unset APPTAINER_DISABLE_CACHE

build_one() {
  local id="$1" image="$2" private tmp rc
  private=$(mktemp -d "$CACHE/build-metadata/build-XXXXXX")
  mkdir -p "$private/cache/blob"
  ln -s "$CACHE/cache/blob/blobs" "$private/cache/blob/blobs"
  tmp="$OUT/.$id.tmp.sif"
  rm -f "$tmp"
  APPTAINER_CACHEDIR="$private" "$APPTAINER" build "$tmp" "$image" > "$LOGS/$id.log" 2>&1
  rc=$?
  rm -rf "$private"
  if [[ $rc -eq 0 ]]; then mv -f "$tmp" "$OUT/$id.sif"; else rm -f "$tmp"; fi
  return $rc
}

worker() {
  local id image
  while read -r id image; do
    [[ -e "$OUT/$id.sif" ]] && continue
    mkdir "$CLAIMS/$id" 2>/dev/null || continue
    if build_one "$id" "$image"; then echo "$(date +%T) built $id"
    else echo "$(date +%T) FAILED $id"; fi
  done < "$IMAGES"
}

for _ in $(seq "$WORKERS"); do worker & done
wait
echo "$(date +%T) $(hostname -s) done"
