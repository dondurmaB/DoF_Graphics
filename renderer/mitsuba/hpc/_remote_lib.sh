#!/bin/bash
# Shared helpers for remote_run.sh and open_viewer.sh (sourced, not run).
#
# The cluster caps jobs and CPUs per user (QOSMaxJobsPerUserLimit, QOSMaxCpuPerUserLimit), so renders do not get
# their own jobs. They run as `srun --overlap` steps inside an existing allocation, in this order of preference:
#   1. HPC_JOB=<jobid> if set
#   2. your running job named `interactive` (the standard workflow: the node your tmux session holds)
#   3. a `dof-render` GPU allocation, submitted with sbatch from the login node if none is running
# Steps never send input to tmux panes and never cancel jobs they did not start.

HOST=${HPC_HOST:-rui.gao@login-student-lab.mbzu.ae}
USER_NAME=${HOST%@*}
REMOTE=${HPC_DEV_DIR:-/home/rui.gao/dof_dev}
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST")
SERVER_NAME=dof-render

repo_root() { cd "$(git rev-parse --show-toplevel)" && pwd; }

sync_code() {
  "${SSH[@]}" "mkdir -p $REMOTE/renderer/mitsuba $REMOTE/logs $REMOTE/output"
  rsync -az --exclude __pycache__ --exclude generated -e "ssh -o BatchMode=yes" \
    renderer/mitsuba/ "$HOST:$REMOTE/renderer/mitsuba/"
}

# Prints "<jobid> <node>" of the allocation to run in, submitting one first if there is none.
ensure_server() {
  local lock=/tmp/dof-render-submit.lock waited=0 line id state node
  if [ -n "${HPC_JOB:-}" ]; then
    "${SSH[@]}" "squeue -j $HPC_JOB -h -t RUNNING -o '%i %N'"; return
  fi
  line=$("${SSH[@]}" "squeue -u $USER_NAME -n interactive -h -t RUNNING -o '%i %N'" | head -1)
  if [ -n "$line" ]; then echo "$line"; return; fi
  until mkdir "$lock" 2>/dev/null; do sleep 1; done
  trap 'rmdir "$lock" 2>/dev/null' RETURN
  line=$("${SSH[@]}" "squeue -u $USER_NAME -n $SERVER_NAME -h -o '%i %T %N'" | head -1)
  if [ -z "$line" ]; then
    "${SSH[@]}" "cat >| $REMOTE/logs/$SERVER_NAME.sh" <<EOF
#!/bin/bash
#SBATCH --job-name=$SERVER_NAME
#SBATCH --partition=${HPC_PARTITION:-ws-ia}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=${HPC_SERVER_TIME:-08:00:00}
#SBATCH --output=$REMOTE/logs/$SERVER_NAME-%j.out
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
sleep infinity
EOF
    id=$("${SSH[@]}" "sbatch --parsable $REMOTE/logs/$SERVER_NAME.sh")
    echo "submitted GPU allocation $SERVER_NAME ($id); waiting for it to start..." >&2
    line="$id PENDING"
  fi
  read -r id state node <<<"$line"
  while [ "$state" != RUNNING ]; do
    sleep 10; waited=$((waited + 10))
    [ $waited -le "${HPC_WAIT:-1800}" ] || { echo "allocation $id still $state after ${HPC_WAIT:-1800}s" >&2; return 1; }
    read -r id state node <<<"$("${SSH[@]}" "squeue -j $id -h -o '%i %T %N'")"
    [ $((waited % 60)) -ne 0 ] || echo "  allocation $id: $state ($waited s)" >&2
  done
  echo "$id $node"
}
