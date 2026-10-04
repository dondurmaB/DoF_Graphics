#!/bin/bash
# Run the interactive Mitsuba viewer on a GPU node and reach it from a laptop.
#
#   sbatch renderer/mitsuba/hpc/viewer.slurm          # from the repo root, login node
#   grep "tunnel:" logs/cafe-viewer-<jobid>.out       # prints the two commands to run locally
#
# The viewer listens on the node's network interface, so it requires the random
# token in the printed URL; nobody else on the cluster can drive it. Stop it
# with `scancel <jobid>` when done, so the GPU goes back to the pool.

#SBATCH --job-name=cafe-viewer
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dof-graphics
cd "${SLURM_SUBMIT_DIR:-.}"

PORT=${PORT:-8765}
NODE=$(hostname -s)
python -u renderer/mitsuba/viewer.py --variant cuda_ad_rgb --host 0.0.0.0 --port "$PORT" 2>&1 \
  | while IFS= read -r line; do
      echo "$line"
      case "$line" in
        *"viewer on http://localhost:"*)
          url=${line#*viewer on }; url=${url%% *}
          echo "tunnel: ssh -N -L $PORT:$NODE:$PORT login-student-lab.mbzu.ae"
          echo "tunnel: open $url" ;;
      esac
    done
