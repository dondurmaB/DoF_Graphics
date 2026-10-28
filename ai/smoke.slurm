#!/bin/bash
# Smoke training on one CIAI A100 (cscc-gpu-p). From the code dir on the cluster:
#
#   sbatch ai/smoke.slurm bench   [extra smoke_train.py args]
#   sbatch ai/smoke.slurm overfit --steps 1500
#   sbatch ai/smoke.slurm train   --steps 2000 --exclude runs/bad_samples.txt
#
# Data is read straight from /nfs-stor; runs/ and logs/ live next to the code on /nfs-stor.
#SBATCH --job-name=dof-smoke
#SBATCH --partition=cscc-gpu-p
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-05   # 2026-10-28: hands out GPU index 3 but exposes only GPUs 0-2 ("No CUDA GPUs are available")
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G   # small enough to fit next to other jobs on a partly used node
#SBATCH --time=04:00:00
#SBATCH --output=/nfs-stor/rui.gao/Researches/dof/dof_dev/logs/dof-smoke-%j.out

set -euo pipefail
MODE=${1:?bench|overfit|train}
shift
CODE=${DOF_CODE:-/nfs-stor/rui.gao/Researches/dof/dof_dev}
PY=${DOF_PY:-/nfs-stor/rui.gao/envs/dof-graphics/bin/python}
DATA=${DOF_DATA:-/nfs-stor/rui.gao/Researches/dof/dof_dataset/train10k}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # loader workers decode in parallel processes
cd "$CODE"
echo "$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"; nvidia-smi --query-gpu=index,name,memory.used,compute_mode --format=csv,noheader
"$PY" -u ai/smoke_train.py "$MODE" "$DATA" --out "runs/smoke-$MODE-$SLURM_JOB_ID" "$@"
