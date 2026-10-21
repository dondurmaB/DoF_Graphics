#!/bin/bash
# Dataset generation on the student-lab RTX 5000 Ada cluster: one generate.py worker on one GPU.
#
#   sbatch renderer/mitsuba/hpc/ws_generate.slurm <dataset root>          (from ~/dof_dev on the login node)
#
# The QoS allows 2 jobs per user and ws-ia jobs last at most 1 day, so a job requeues itself 10 minutes before
# its limit (same job id, no new submission) and its worker resumes the chunk it held; finished poses are
# skipped. When the queue is empty the worker exits and the job ends. The RTX 5000 Ada is about 12x faster
# than the CIAI A100s for this (RT cores), measured 2026-10-20.
#SBATCH --job-name=dof-gen
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=/home/rui.gao/dof_dev/logs/dof-gen-%j.out

set -uo pipefail
OUT=${1:?dataset root}
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dof-graphics
export OPTIX_CACHE_PATH=/tmp/optix_cache_$USER   # node-local
cd ~/dof_dev
echo "[$(date '+%F %T')] start on $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

requeue() {
  echo "[$(date '+%F %T')] time limit near: requeueing $SLURM_JOB_ID"
  kill "$worker" 2>/dev/null
  scontrol requeue "$SLURM_JOB_ID"
}
trap requeue USR1

python -u renderer/mitsuba/generate.py work --out "$OUT" &
worker=$!
wait "$worker"   # returns early when USR1 arrives; the trap then requeues
python renderer/mitsuba/generate.py status --out "$OUT"
