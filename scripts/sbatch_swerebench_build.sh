#!/usr/bin/env bash
#SBATCH --job-name=hariom
#SBATCH --partition=cpu
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/data/user_data/adityabs/apptainer_cache/slurm_logs/%x_%j.out
#SBATCH --error=/data/user_data/adityabs/apptainer_cache/slurm_logs/%x_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/adityabs/benchmarks_build_apptainerswe}"
OUT_ROOT="${OUT_ROOT:-/data/user_data/adityabs/apptainer_cache}"
SHARD_DIR="${SHARD_DIR:-$OUT_ROOT/swerebench_build2_shards_16}"
DATASET="${DATASET:-nebius/SWE-rebench}"
SPLIT="${SPLIT:-filtered}"
WORKERS_PER_TASK="${WORKERS_PER_TASK:-8}"

mkdir -p "$OUT_ROOT/slurm_logs" "$OUT_ROOT/manifests"

cd "$REPO_DIR"

srun --wait=0 --ntasks=16 --ntasks-per-node=1 bash -lc '
set -euo pipefail

shard=$(printf "%03d" "$SLURM_PROCID")
select_file="'"$SHARD_DIR"'/shard_${shard}.txt"

if [[ ! -s "$select_file" ]]; then
  echo "Missing shard select file: $select_file" >&2
  exit 2
fi

export APPTAINER_CACHEDIR="'"$OUT_ROOT"'"
export APPTAINER_TMPDIR="/tmp/${USER}/apptainer_tmp_swere_b2_${shard}_${SLURM_JOB_ID}"
export OPENHANDS_APPTAINER_BUILD_ROOT="$APPTAINER_CACHEDIR"

mkdir -p \
  "$APPTAINER_CACHEDIR" \
  "$APPTAINER_TMPDIR" \
  "'"$OUT_ROOT"'/manifests"
cd /home/adityabs/benchmarks_build_apptainerswe
source .venv/bin/activate
python benchmarks/swerebench/apptainer_build2.py \
  --dataset "'"$DATASET"'" \
  --split "'"$SPLIT"'" \
  --select "$select_file" \
  --max-workers "'"$WORKERS_PER_TASK"'" \
  --manifest "'"$OUT_ROOT"'/manifests/array10_shard_${shard}_${SLURM_JOB_ID}.jsonl"
'
