#!/bin/bash
# Dataset generation on the CIAI A100 cluster: one worker per GPU pulling chunks from generate.py's queue.
#
#   sbatch renderer/mitsuba/hpc/ciai_generate.slurm <dataset root>       (from the code dir on the cluster)
#
# Our account (cscc-users) may use the cscc-gpu-p partition (QoS cscc-gpu-qos: 14 nodes x 4 A100-40GB, jobs up
# to 48 h; the `long` partition rejects it). Several of these jobs can share one queue; a job that ends (time
# limit, preemption) leaves at most one claimed chunk per GPU behind (`generate.py release`).
# Code, conda env and data live on /nfs-stor: /home on this cluster is nearly full.
#SBATCH --job-name=dof-gen
#SBATCH --partition=cscc-gpu-p
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs-stor/rui.gao/dof_dev/logs/dof-gen-%j.out

set -euo pipefail
OUT=${1:?dataset root}
CODE=${DOF_CODE:-/nfs-stor/rui.gao/dof_dev}
PY=${DOF_PY:-/nfs-stor/rui.gao/envs/dof-graphics/bin/python}
export OPTIX_CACHE_PATH=/tmp/optix_cache_$USER   # node-local
cd "$CODE"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
IFS=, read -ra GPUS <<<"${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
for g in "${GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=$g "$PY" -u renderer/mitsuba/generate.py work --out "$OUT" \
    >"$CODE/logs/dof-gen-$SLURM_JOB_ID-gpu$g.log" 2>&1 &
done
wait
"$PY" renderer/mitsuba/generate.py status --out "$OUT"
