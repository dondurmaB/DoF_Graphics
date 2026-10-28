#!/bin/bash
# Several smoke_train.py runs in one job, one per GPU (round-robin if there are more runs than GPUs, so small
# models can share a GPU). The CIAI GPU QoS allows 2 running jobs but 4 GPUs per user, so packing runs into
# multi-GPU jobs is what gets more than 2 runs going at once. From the code dir on the cluster:
#
#   sbatch ai/multi.slurm "train --arch unet --width 8 --film --gamma 20" "train --arch unet --width 24 --film --gamma 20"
#
# Each quoted spec is "<mode> [smoke_train.py args]"; run i writes runs/smoke-<mode>-<jobid>-<i> and
# logs/dof-multi-<jobid>-<i>.out.
#SBATCH --job-name=dof-multi
#SBATCH --partition=cscc-gpu-p
#SBATCH --gres=gpu:2
#SBATCH --exclude=gpu-05   # 2026-10-28: hands out GPU index 3 but exposes only GPUs 0-2
#SBATCH --cpus-per-task=32
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs-stor/rui.gao/Researches/dof/dof_dev/logs/dof-multi-%j.out

set -euo pipefail
CODE=${DOF_CODE:-/nfs-stor/rui.gao/Researches/dof/dof_dev}
PY=${DOF_PY:-/nfs-stor/rui.gao/envs/dof-graphics/bin/python}
DATA=${DOF_DATA:-/nfs-stor/rui.gao/Researches/dof/dof_dataset/train10k}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$CODE"
echo "$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
i=0
for spec in "$@"; do
  read -ra a <<< "$spec"
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  echo "run $i on GPU $g: $spec"
  CUDA_VISIBLE_DEVICES=$g "$PY" -u ai/smoke_train.py "${a[0]}" "$DATA" --out "runs/smoke-${a[0]}-$SLURM_JOB_ID-$i" \
    "${a[@]:1}" > "logs/dof-multi-$SLURM_JOB_ID-$i.out" 2>&1 &
  i=$((i + 1))
done
wait
echo "all $i runs finished"
