#!/bin/bash
# Run any ai/ script on one CIAI A100 (cscc-gpu-p). From the code dir on the cluster:
#
#   sbatch ai/run.slurm ai/preview.py runs/smoke-train-347294/last.pt --exclude runs/bad_samples.txt --out runs/preview-K
#   sbatch ai/run.slurm ai/classic_eval.py --exclude runs/bad_samples.txt --out runs/classic
#
# The dataset root is inserted as the first positional argument (after the checkpoint for preview.py).
#SBATCH --job-name=dof-run
#SBATCH --partition=cscc-gpu-p
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-05   # 2026-10-28: hands out GPU index 3 but exposes only GPUs 0-2 ("No CUDA GPUs are available")
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs-stor/rui.gao/Researches/dof/dof_dev/logs/dof-run-%j.out

set -euo pipefail
SCRIPT=${1:?script path}
shift
CODE=${DOF_CODE:-/nfs-stor/rui.gao/Researches/dof/dof_dev}
PY=${DOF_PY:-/nfs-stor/rui.gao/envs/dof-graphics/bin/python}
DATA=${DOF_DATA:-/nfs-stor/rui.gao/Researches/dof/dof_dataset/train10k}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$CODE"
echo "$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"; nvidia-smi --query-gpu=index,name,memory.used,compute_mode --format=csv,noheader
case "$SCRIPT" in
  *preview.py) CK=${1:?checkpoint}; shift; "$PY" -u "$SCRIPT" "$CK" "$DATA" "$@" ;;
  *) "$PY" -u "$SCRIPT" "$DATA" "$@" ;;
esac
