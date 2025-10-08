#!/bin/bash
#SBATCH --job-name=hariom
#SBATCH --partition=general
#SBATCH --output=/home/adityabs/swt_training/ai-hpc-llm-ptest/slurm_eval.out
#SBATCH --error=/home/adityabs/swt_training/ai-hpc-llm-ptest/slurm_eval.log
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A100_40GB:4
#SBATCH --time=12:06:00
#SBATCH --mem=200G
#SBATCH --mail-user=adityabs@andrew.cmu.edu   # Your email address
#SBATCH --mail-type=BEGIN                    # Send email when the job starts
#SBATCH --mail-type=END                      # Send email when the job ends
#SBATCH --mail-type=FAIL                     # Send email if the job fails

#SBATCH --exclude=babel-9-11,babel-9-7
source ~/.bashrc
source /home/adityabs/benchmarks/.venv/bin/activate
sleep 1000000000