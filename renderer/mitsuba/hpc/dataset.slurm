#!/bin/bash
# Generate the stage-3 depth-of-field dataset as a SLURM array.
#
#   mkdir -p logs
#   python renderer/mitsuba/gl_context.py          # on a GPU node, FIRST
#   sbatch --array=0-63 renderer/mitsuba/hpc/dataset.slurm
#
# Override anything through --export, for example a smaller trial run:
#   sbatch --array=0-7 --export=ALL,PRESET=draft,OUT=data/dof-draft \
#          renderer/mitsuba/hpc/dataset.slurm
#
# Each task renders every sample whose index is congruent to its array ID modulo
# the array size, so any subset that finishes still covers the whole
# configuration space rather than a contiguous block of one café.
#
# Re-running the same array is safe and is the intended way to recover from a
# timeout: a sample with a sample.json is skipped unless OVERWRITE=1.

#SBATCH --job-name=dof-dataset
#SBATCH --partition=ws-ia          # workstations: 1x RTX 5000 Ada 32 GB, 1-day limit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
# logs/ must exist before submitting: SLURM opens --output before this script runs.

# --- environment (cluster-specific) ----------------------------------------
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dof-graphics
# ---------------------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR:-.}"

OUT="${OUT:-data/dof-v1}"
PRESET="${PRESET:-train}"
SCENES="${SCENES:-12}"
SHARDS="${SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true

# The gather needs an OpenGL context from the vendor driver. Under SLURM,
# CUDA_VISIBLE_DEVICES masks GPUs for CUDA but does not always mask them for
# EGL, so the EGL device is selected explicitly from the same allocation.
export DOF_GL_BACKEND=egl
export DOF_EGL_DEVICE="${DOF_EGL_DEVICE:-${CUDA_VISIBLE_DEVICES%%,*}}"
export DOF_EGL_DEVICE="${DOF_EGL_DEVICE:-0}"

# Fail the task now, with a clear message, rather than after Mitsuba has spent
# minutes building a café only to discover it cannot run the baseline gather.
echo "--- GL context check (EGL device ${DOF_EGL_DEVICE}) ---"
python renderer/mitsuba/gl_context.py

echo "--- shard ${SHARD}/${SHARDS} -> ${OUT} ---"
python dataset/build_dataset.py \
    --out "$OUT" \
    --preset "$PRESET" \
    --scenes "$SCENES" \
    --shard "$SHARD" \
    --shards "$SHARDS" \
    --variant cuda_ad_rgb \
    --gl-backend egl \
    --egl-device "$DOF_EGL_DEVICE" \
    ${OVERWRITE:+--overwrite} \
    ${EXTRA_ARGS:-}

echo "done: shard ${SHARD} of ${OUT}"
