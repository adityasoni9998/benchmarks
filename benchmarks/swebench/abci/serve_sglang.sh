#!/usr/bin/env bash
# Serve an HF checkpoint with SGLang from miles' SIF (OpenAI-compatible API), for SWE-bench evals.
# Container options follow miles' ABCI training PBS scripts; engine settings follow its rollout
# engines: one GPU per engine (--dp-size 8 on a full node), context 66000, mem fraction 0.9,
# chunked prefill 4096, at most 128 running requests per engine, qwen25 tool-call parser.
# Sampling defaults come from the checkpoint's generation_config.json; the LLM config sends its own.
# Tool-call parsing follows hyperion training's renderer (sglang_patch/sitecustomize.py, loaded
# into every server process via PYTHONPATH; parity-checked by sglang_patch/parity/), and calls to
# tools the agent lacks are forwarded so the agent reports them (SGLANG_FORWARD_UNKNOWN_TOOLS=1).
# Must run inside a PBS allocation (PBS_LOCALDIR is the job's node-local scratch).
#
# usage: serve_sglang.sh <hf model dir> <served model name> [port]
# env:   MILES_ENV (default ~/miles-env), MILES_SIF (default $MILES_ENV/images/miles-sglang-a359142-cu13.sif)
set -euo pipefail
model_dir="$1"
served_name="$2"
port="${3:-30000}"
MILES_ENV="${MILES_ENV:-$HOME/miles-env}"
SIF="${MILES_SIF:-$MILES_ENV/images/miles-sglang-a359142-cu13.sif}"
NODE_LOCAL="${PBS_LOCALDIR:?run inside a PBS allocation}/sglang-$served_name"
mkdir -p "$NODE_LOCAL/tmp"
chmod 1777 "$NODE_LOCAL/tmp"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load singularity-ce/4.4.1 2>/dev/null || true
export NVIDIA_VISIBLE_DEVICES=all
PATCH_DIR="$(cd "$(dirname "$0")" && pwd)/sglang_patch"
binds=(--bind "$NODE_LOCAL/tmp:/tmp" --home "$MILES_ENV/rootfs:/root" --bind "$MILES_ENV:$MILES_ENV"
       --bind "$PATCH_DIR:/opt/hyperion_sglang_patch:ro")
case "$model_dir" in "$MILES_ENV"/*) ;; *) binds+=(--bind "$model_dir:$model_dir:ro") ;; esac
exec /usr/bin/singularity exec --nv --cleanenv --containall --no-mount cwd "${binds[@]}" \
  --env TMPDIR=/tmp --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env PYTHONNOUSERSITE=1 \
  --env "HF_HOME=$MILES_ENV/hf-home" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env XDG_CACHE_HOME=/tmp/xdg --env TRITON_CACHE_DIR=/tmp/triton --env TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
  --env FLASHINFER_WORKSPACE_DIR=/tmp/flashinfer --env TORCH_EXTENSIONS_DIR=/tmp/torch-ext \
  --env CUDA_CACHE_PATH=/tmp/nv-cache --env DG_JIT_CACHE_DIR=/tmp/deepgemm \
  --env PYTHONPATH=/opt/hyperion_sglang_patch --env SGLANG_FORWARD_UNKNOWN_TOOLS=1 \
  "$SIF" python -m sglang.launch_server \
    --model-path "$model_dir" --served-model-name "$served_name" \
    --host 0.0.0.0 --port "$port" \
    --dp-size 8 --tp-size 1 \
    --context-length 66000 --mem-fraction-static 0.9 \
    --chunked-prefill-size 4096 --max-running-requests 128 \
    --tool-call-parser qwen25
