#!/usr/bin/env bash
#SBATCH --job-name=swere-build2
#SBATCH --partition=array
#SBATCH --array=0-7%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --mem=128GB
#SBATCH --time=03:00:00
#SBATCH --output=/data/user_data/adityabs/apptainer_cache/slurm_logs/%x_%A_%a.out
#SBATCH --error=/data/user_data/adityabs/apptainer_cache/slurm_logs/%x_%A_%a.err
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/adityabs/benchmarks_build_apptainerswe}"
OUT_ROOT="${OUT_ROOT:-/data/user_data/adityabs/apptainer_cache}"
SHARD_DIR="${SHARD_DIR:-$OUT_ROOT/swerebench_build2_shards_32}"
DATASET="${DATASET:-nebius/SWE-rebench}"
SPLIT="${SPLIT:-filtered}"
WORKERS_PER_TASK="${WORKERS_PER_TASK:-16}"

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
select_file="$SHARD_DIR/shard_$(printf '%03d' "$task_id").txt"

if [[ ! -s "$select_file" ]]; then
  echo "Missing shard select file: $select_file" >&2
  exit 2
fi

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$OUT_ROOT}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp/${USER}/apptainer_tmp_${job_id}_${task_id}}"
export OPENHANDS_APPTAINER_BUILD_ROOT="$APPTAINER_CACHEDIR"

mkdir -p \
  "$APPTAINER_CACHEDIR" \
  "$APPTAINER_TMPDIR" \
  "$OPENHANDS_APPTAINER_BUILD_ROOT" \
  "$OUT_ROOT/slurm_logs" \
  "$OUT_ROOT/manifests"

cd "$REPO_DIR"

python benchmarks/swerebench/apptainer_build2.py \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --select "$select_file" \
  --max-workers "$WORKERS_PER_TASK" \
  --manifest "$OUT_ROOT/manifests/${job_id}_${task_id}.jsonl"
