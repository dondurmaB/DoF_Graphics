#!/bin/bash
# Run any renderer script on the cluster GPU and bring the previews back. Never renders locally.
#
#   renderer/mitsuba/hpc/remote_run.sh <label> <script.py> [args...]      (run from anywhere inside the repo)
#
#   renderer/mitsuba/hpc/remote_run.sh desk-close renderer/mitsuba/render.py \
#       --scene desk --view close --res 960x540 --spp 256 --out output/dev/desk-close
#   renderer/mitsuba/hpc/remote_run.sh desk-check renderer/mitsuba/check_scene.py desk --keep output/dev/desk-check
#
# What it does:
#   1. rsyncs renderer/mitsuba to ~/dof_dev on the cluster (your git clone there is left alone; no --delete)
#   2. picks the allocation to run in: your `interactive` job, else a `dof-render` one (see _remote_lib.sh)
#   3. runs the command as a detached `srun --overlap` step inside it (survives VPN drops), one GPU job at a
#      time (flock); waits for it and prints the log tail
#   4. copies the --out / --keep directory back (PNG and JSON; EXR=1 adds .exr/.npy)
# Output paths must be relative to the repo root. Env: HPC_HOST, HPC_DEV_DIR, LOG_LINES (default 40), EXR=1.

set -euo pipefail
[ $# -ge 2 ] || { sed -n 2,17p "$0"; exit 2; }
. "$(dirname "$0")/_remote_lib.sh"
LABEL=$1; shift
cd "$(repo_root)"

OUT=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[i]}" in --out|--keep) OUT=${args[i+1]:-} ;; esac
done
case "$OUT" in /*|*..*) echo "output path must be relative to the repo root, got: $OUT" >&2; exit 2 ;; esac

sync_code
srv=$(ensure_server) && [ -n "$srv" ] || { echo "no allocation to run in" >&2; exit 1; }
read -r JOB NODE <<<"$srv"
CMD=$(printf '%q ' "$@")
"${SSH[@]}" "cat >| $REMOTE/logs/$LABEL.sh" <<EOF
#!/bin/bash
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dof-graphics
export OPTIX_CACHE_PATH=/tmp/optix_cache_\$USER   # node-local: two nodes sharing the NFS home corrupt one cache
cd $REMOTE
echo "node \$(hostname -s), GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
exec flock /tmp/dof_gpu.lock python -u $CMD
EOF

# The step is detached on the login node and writes its log and exit code to the NFS home, so a VPN drop
# only interrupts the polling below, never the render. Rerunning a label while it is still running is refused.
LOG=$REMOTE/logs/$LABEL.log; EXITF=$REMOTE/logs/$LABEL.exit
"${SSH[@]}" "squeue -u $USER_NAME -s -h -o '%j' | grep -qx '$LABEL'" && { echo "$LABEL is already running" >&2; exit 1; }
"${SSH[@]}" "rm -f $LOG $EXITF; setsid nohup bash -c 'srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=${CPUS:-4} -J $LABEL bash $REMOTE/logs/$LABEL.sh >$LOG 2>&1; echo \$? >$EXITF' </dev/null >/dev/null 2>&1 &"
echo "running $LABEL on $NODE (allocation $JOB); log at $LOG ..."
status=
until [ -n "$status" ]; do
  sleep 15
  status=$("${SSH[@]}" "cat $EXITF 2>/dev/null" 2>/dev/null) || status=   # ssh down: keep waiting
done
until "${SSH[@]}" "cat $LOG" 2>/dev/null | grep -v '^  \[' | tail -"${LOG_LINES:-40}"; do sleep 15; done

if [ -n "$OUT" ]; then
  mkdir -p "$OUT"
  filter=(--exclude '*.exr' --exclude '*.npy'); [ "${EXR:-0}" = 1 ] && filter=()
  for _ in $(seq 40); do   # retries ride out a VPN drop (10 min)
    rsync -az ${filter[@]+"${filter[@]}"} -e "ssh -o BatchMode=yes" "$HOST:$REMOTE/$OUT/" "$OUT/" && break
    sleep 15
  done
  echo "results in $OUT (exit $status)"
fi
exit $status
