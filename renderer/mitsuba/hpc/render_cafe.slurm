#!/bin/bash
# Render the Mitsuba cafe on one GPU node.
#
#   sbatch renderer/mitsuba/hpc/render_cafe.slurm
#   sbatch --export=ALL,RENDER_ARGS="--focus-target books --f-number 1.4" renderer/mitsuba/hpc/render_cafe.slurm
#
# Submit from the repository root on login-student-lab (MBZUAI). Only submit
# from the login node; the work itself runs on the allocated GPU node.
# Measured on ws-l2-008: 1920x1080 at 1024 spp is ~10 s per beauty pass.

#SBATCH --job-name=cafe-mitsuba
#SBATCH --partition=ws-ia          # workstations: 1x RTX 5000 Ada 32 GB, 1-day limit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
# logs/ must exist before submitting: SLURM opens --output before this runs.

# --- environment (cluster-specific) ----------------------------------------
# The nodes ship NVIDIA driver 570; the mitsuba wheel needs no CUDA module.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dof-graphics
# ---------------------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR:-.}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true

OUT="output/mitsuba/cafe-${SLURM_JOB_ID:-local}"
python renderer/mitsuba/render.py \
    --variant cuda_ad_rgb \
    --res 1920x1080 \
    --spp 4096 \
    --spp-per-pass 256 \
    --out "$OUT" \
    ${RENDER_ARGS:-}

echo "done: $OUT"
