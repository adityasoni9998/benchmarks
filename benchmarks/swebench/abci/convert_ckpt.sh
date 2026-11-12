#!/usr/bin/env bash
# Convert a miles/tinker Megatron torch_dist policy checkpoint to Hugging Face safetensors.
# Runs convert_tinker_ckpt.py (which reuses miles' tools/convert_torch_dist_to_hf.py) in miles'
# SIF with the training PBS scripts' container options. CPU only. The checkpoint is only read;
# the output directory must not exist. verify_hf_ckpt.py then checks it against the base model.
#
# usage: convert_ckpt.sh <iter dir, e.g. .../weights/000160/iter_0000001> <output dir>
# env:   MILES_ENV (default ~/miles-env), MILES_REPO (default ~/miles),
#        MILES_SIF (default $MILES_ENV/images/miles-sglang-a359142-cu13.sif),
#        BASE_HF (default $MILES_ENV/models/Qwen/Qwen3-4B-Instruct-2507)
set -euo pipefail
input_dir="$1"
output_dir="$2"
[[ -f "$input_dir/.metadata" ]] || { echo "not a torch_dist iteration dir: $input_dir" >&2; exit 1; }
[[ ! -e "$output_dir" ]] || { echo "output exists, refusing to overwrite: $output_dir" >&2; exit 1; }
MILES_ENV="${MILES_ENV:-$HOME/miles-env}"
MILES_REPO="${MILES_REPO:-$HOME/miles}"
SIF="${MILES_SIF:-$MILES_ENV/images/miles-sglang-a359142-cu13.sif}"
BASE_HF="${BASE_HF:-$MILES_ENV/models/Qwen/Qwen3-4B-Instruct-2507}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NODE_LOCAL="${PBS_LOCALDIR:-${ABCI_SCRATCH:?set PBS_LOCALDIR or ABCI_SCRATCH}}/miles-convert-$(basename "$output_dir")"
out_parent="$(dirname "$output_dir")"
mkdir -p "$NODE_LOCAL/tmp" "$out_parent"
chmod 1777 "$NODE_LOCAL/tmp"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load singularity-ce/4.4.1 2>/dev/null || true
exec /usr/bin/singularity exec --cleanenv --containall --no-mount cwd \
  --bind "$NODE_LOCAL/tmp:/tmp" --home "$MILES_ENV/rootfs:/root" \
  --bind "$MILES_REPO:/root/miles" --bind "$MILES_ENV:$MILES_ENV" \
  --bind "$out_parent:$out_parent" --bind "$SCRIPT_DIR:$SCRIPT_DIR" --pwd /root/miles \
  --env TMPDIR=/tmp --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env PYTHONNOUSERSITE=1 \
  --env "PYTHONPATH=/root/miles:/root/Megatron-LM:$MILES_ENV/extra-site" \
  --env "HF_HOME=$MILES_ENV/hf-home" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  "$SIF" python "$SCRIPT_DIR/convert_tinker_ckpt.py" "$input_dir" "$output_dir" "$BASE_HF"
