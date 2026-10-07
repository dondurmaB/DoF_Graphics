#!/bin/bash
# Open the interactive path-traced viewer for a scene on the cluster GPU and tunnel it to this machine.
#
#   renderer/mitsuba/hpc/open_viewer.sh <scene> [env] [port]      prints http://localhost:<port>/?t=<token>
#   renderer/mitsuba/hpc/open_viewer.sh --stop <scene>            stops that viewer (the GPU allocation stays)
#   renderer/mitsuba/hpc/open_viewer.sh --reconnect               reopens tunnels to all running viewers, prints links
#
# The viewer is an `srun --overlap` step inside your allocation (see _remote_lib.sh; the cluster caps jobs per user), listens with a random per-run token, and is reached through `ssh -L`. Each scene gets its own port
# (default 8800 + checksum(scene) % 100) so several viewers can run side by side. If the link stops working,
# run this again: it restarts the tunnel and the viewer. Nothing renders locally.

set -euo pipefail
. "$(dirname "$0")/_remote_lib.sh"
cd "$(repo_root)"

stop_steps() {  # cancel running viewer steps of a scene
  local job=$1 scene=$2 step
  for step in $("${SSH[@]}" "squeue -s -j $job -h -o '%i %j'" | awk -v n="viewer-$scene" '$2 == n {print $1}'); do
    "${SSH[@]}" "scancel $step" || true
  done
}

tunnel() {  # (re)open the local forward for a port; replaces an old ssh tunnel on that port
  local port=$1 node=$2 pid
  for pid in $(lsof -ti "tcp:$port" -sTCP:LISTEN 2>/dev/null || true); do
    ps -p "$pid" -o comm= | grep -q '^ssh$' && kill "$pid"
  done
  ssh -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
    -L "$port:$node:$port" "$HOST"
}

if [ "${1:-}" = "--reconnect" ]; then  # reopen tunnels for every running viewer, print all links
  srv=$(ensure_server) && [ -n "$srv" ] || { echo "no allocation" >&2; exit 1; }
  read -r JOB NODE <<<"$srv"
  for scene in $("${SSH[@]}" "squeue -s -j $JOB -h -o '%j'" | sed -n 's/^viewer-//p' | sort); do
    URL=$("${SSH[@]}" "grep -o 'http://localhost:[0-9]*/?t=[A-Za-z0-9_-]*' $REMOTE/logs/viewer-$scene.out | tail -1")
    PORT=$(sed 's|http://localhost:\([0-9]*\)/.*|\1|' <<<"$URL")
    tunnel "$PORT" "$NODE"
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 "$URL" || true)
    printf '%-14s %s   [HTTP %s]\n' "$scene" "$URL" "$code"
  done
  exit 0
fi

if [ "${1:-}" = "--stop" ]; then
  SCENE=${2:?scene}
  JOB=$(ensure_server | cut -d' ' -f1)
  [ -n "$JOB" ] && stop_steps "$JOB" "$SCENE" && echo "stopped viewer-$SCENE"
  exit 0
fi

SCENE=${1:?usage: open_viewer.sh <scene> [env] [port]}
ENVNAME=${2:-clear}
PORT=${3:-$((8800 + $(printf %s "$SCENE" | cksum | cut -d' ' -f1) % 100))}

sync_code
srv=$(ensure_server) && [ -n "$srv" ] || { echo "no allocation to run in" >&2; exit 1; }
read -r JOB NODE <<<"$srv"
stop_steps "$JOB" "$SCENE"
LOG=$REMOTE/logs/viewer-$SCENE.out
"${SSH[@]}" "rm -f $LOG; setsid nohup srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=4 -J viewer-$SCENE \
  bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate dof-graphics && cd $REMOTE && \
  exec python -u renderer/mitsuba/viewer.py --scene $SCENE --env $ENVNAME --variant cuda_ad_rgb --host 0.0.0.0 --port $PORT \
  --out output/viewer/$SCENE' >$LOG 2>&1 < /dev/null &"

echo "starting viewer for $SCENE on $NODE (first start builds the scene and compiles kernels) ..."
URL=""
for _ in $(seq 1 90); do
  sleep 4
  URL=$("${SSH[@]}" "grep -o 'http://localhost:$PORT/?t=[A-Za-z0-9_-]*' $LOG 2>/dev/null | head -1" || true)
  [ -n "$URL" ] && break
  "${SSH[@]}" "grep -qi 'traceback\|error' $LOG 2>/dev/null" && { "${SSH[@]}" "tail -15 $LOG"; exit 1; }
done
[ -n "$URL" ] || { echo "viewer did not come up; log:"; "${SSH[@]}" "tail -15 $LOG"; exit 1; }

tunnel "$PORT" "$NODE"
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 "$URL" || true)
echo "$SCENE ($ENVNAME): $URL   [HTTP $code]"
