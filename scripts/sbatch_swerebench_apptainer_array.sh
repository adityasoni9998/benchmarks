#!/usr/bin/env bash
#SBATCH --job-name=swere-appt
#SBATCH --partition=array
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=/data/user_data/adityabs/apptainer_cache/slurm_logs/%x_%A_%a.out
#SBATCH --error=/data/user_data/adityabs/apptainer_cache/slurm_logs/%x_%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/adityabs/benchmarks_build_apptainerswe}"
OUT_ROOT="${OUT_ROOT:-/data/user_data/adityabs/apptainer_cache}"
SIF_DIR="${SIF_DIR:-$OUT_ROOT}"
WORKERS_PER_TASK="${WORKERS_PER_TASK:-8}"
MKSQUASHFS_PROCESSORS="${MKSQUASHFS_PROCESSORS:-4}"
IMAGE_LIMIT="${IMAGE_LIMIT:-0}"
N_LIMIT="${N_LIMIT:-0}"
DATASET="${DATASET:-nebius/SWE-rebench}"
SPLIT="${SPLIT:-filtered}"

task_id="${SLURM_ARRAY_TASK_ID:-0}"
job_id="${SLURM_JOB_ID:-manual}"

local_root="/tmp/${USER}/swere_apptainer_${job_id}_${task_id}"
cache_dir="${APPTAINER_CACHE_DIR:-$local_root/cache}"
tmp_dir="${APPTAINER_TMP_DIR:-$local_root/tmp}"
local_sif_dir="${LOCAL_SIF_DIR:-$local_root/sif}"
def_dir="$OUT_ROOT/defs/${job_id}_${task_id}"
build_log_dir="$OUT_ROOT/logs/${job_id}_${task_id}"
manifest="$OUT_ROOT/manifests/${job_id}_${task_id}.jsonl"
monitor_dir="$OUT_ROOT/monitor/${job_id}_${task_id}"

mkdir -p "$OUT_ROOT/slurm_logs" "$cache_dir" "$tmp_dir" "$local_sif_dir" "$def_dir" "$build_log_dir" "$(dirname "$manifest")" "$monitor_dir"

export OPENHANDS_APPTAINER_UV_CONCURRENT_DOWNLOADS="${OPENHANDS_APPTAINER_UV_CONCURRENT_DOWNLOADS:-2}"
export OPENHANDS_APPTAINER_UV_CONCURRENT_BUILDS="${OPENHANDS_APPTAINER_UV_CONCURRENT_BUILDS:-1}"
export OPENHANDS_APPTAINER_UV_CONCURRENT_INSTALLS="${OPENHANDS_APPTAINER_UV_CONCURRENT_INSTALLS:-1}"

"$REPO_DIR/scripts/monitor_command.sh" "$monitor_dir" \
  python "$REPO_DIR/scripts/swerebench_sharded_build.py" \
    --repo-dir "$REPO_DIR" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --n-limit "$N_LIMIT" \
    --image-limit "$IMAGE_LIMIT" \
    --workers "$WORKERS_PER_TASK" \
    --sif-dir "$local_sif_dir" \
    --final-sif-dir "$SIF_DIR" \
    --apptainer-cache-dir "$cache_dir" \
    --apptainer-tmp-dir "$tmp_dir" \
    --definition-dir "$def_dir" \
    --log-dir "$build_log_dir" \
    --manifest "$manifest" \
    --mksquashfs-processors "$MKSQUASHFS_PROCESSORS"
