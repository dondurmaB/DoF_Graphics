#!/bin/bash
# Stream v4 top-up: render an explicit list of units on one ws-ia GPU, adapted from ~/dof/tools/stream_v4_batch.slurm.
# Copy to ~/dof/tools/ (new file only); the code checkout stays clean at EXPECT_COMMIT.
#
#   FP="FP_CLEAR=<16 hex>,FP_OVERCAST=<16 hex>,FP_SNOW=<16 hex>"      # from the fingerprint job
#   sbatch --time=01:45:00 --export=ALL,TAG=a,UNITS=overcast:1208:7,EST_OVERCAST=292,EST_SNOW=154,$FP ~/dof/tools/stream_topup.slurm
#   UNITS=snow:1199:2,snow:1215:7 sbatch --time=01:45:00 --export=ALL,TAG=b,EST_OVERCAST=292,EST_SNOW=154,$FP ~/dof/tools/stream_topup.slurm
# A UNITS list with more than one unit goes in the environment, not in --export (sbatch splits --export on commas).
#
# Why not stream_v4_batch.slurm: there STOP_AT/KILL_AT were wall-clock times fixed at submission, so a job that sat
# in the queue (254374 started 65 min late) had almost no clock left, and slot 0's last unit was trimmed to fit its
# clock. Here both are counted from this job's own start (STOP_MIN, KILL_MIN), and the work is a unit list
# UNITS=env:start:count,... instead of a slot's index range.
# Each job stops at the first of: the end of its unit list, the storage cap (du before every unit), the clock (a unit
# is trimmed to the poses that fit before STOP_AT; a running unit is killed at KILL_AT), or 3 sampler failures in a row.
# Guards: clean code at EXPECT_COMMIT; fingerprints of all three envs equal FP_CLEAR / FP_OVERCAST / FP_SNOW
# (required). Per-env estimates EST_<ENV> are wall clock per pose measured in 254373/254374 (load included).
# Coverage check over exactly the listed indices; unfinished pose directories are moved aside, never deleted.

#SBATCH --job-name=dof-stream-topup
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/beibarys.myrzash/dof/logs/%x-%j.out

set -uo pipefail
SCENE=stream
UNITS=${UNITS:?UNITS=env:start:count,...}
FP_CLEAR=${FP_CLEAR:?}; FP_OVERCAST=${FP_OVERCAST:?}; FP_SNOW=${FP_SNOW:?}
declare -A EST=([clear]=${EST_CLEAR:-316} [overcast]=${EST_OVERCAST:-292} [snow]=${EST_SNOW:-154})
OVERHEAD=${OVERHEAD:-60}          # seconds per sampler call on top of EST (margin; the measured rates include the load)
STOP_MIN=${STOP_MIN:-75}; KILL_MIN=${KILL_MIN:-85}
CAP_BYTES=${CAP_BYTES:-16000000000}; POSE_BYTES=${POSE_BYTES:-332000000}
LENSES=${LENSES:-8}
T_START=$(date +%s); STOP_AT=$(( T_START + STOP_MIN * 60 )); KILL_AT=$(( T_START + KILL_MIN * 60 ))
# never past this job's own SLURM end time either
END_SLURM=$(date -d "$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %e 2>/dev/null)" +%s 2>/dev/null || echo 0)
if (( END_SLURM > 0 )); then
  (( STOP_AT > END_SLURM - 900 )) && STOP_AT=$(( END_SLURM - 900 ))
  (( KILL_AT > END_SLURM - 600 )) && KILL_AT=$(( END_SLURM - 600 ))
fi
B=${B_ROOT:-$HOME/dof}
CODE=$B/code; RD=$CODE/renderer/mitsuba; TOOLS=$B/tools
PY=${PY_OVERRIDE:-$HOME/.conda/envs/dof-graphics/bin/python}
ASSETS=$B/generated; OUT=$B/dataset/mitsuba
JOB=${SLURM_JOB_ID:-local}; LD=$B/logs/batch_$JOB${TAG:+_$TAG}; mkdir -p "$LD"
EXPECT_COMMIT=${EXPECT_COMMIT:-87fff67}
declare -A EXPECT=([clear]=$FP_CLEAR [overcast]=$FP_OVERCAST [snow]=$FP_SNOW)
export DOF_WEB_ASSETS=$B/web_assets
export OPTIX_CACHE_PATH=/tmp/optix_cache_$USER
export PATH=$HOME/.conda/envs/dof-graphics/bin:$PATH
mkdir -p "$OPTIX_CACHE_PATH"
log() { echo "$(date '+%F %T') $*"; }
quiet() { grep -v -E 'jitc|catalog size|^\s+\[(probe|gbuffer|sharp|dof)\] [0-9]+/'; }
homeuse() { echo "$(date +%s),$(du -sbx $HOME 2>/dev/null | cut -f1)" >> "$LD/home.csv"; }
CSV=$LD/units.csv; echo "slot,env,start_index,count,start,end,seconds,rc,written,failed,skipped,stopped" > "$CSV"
IFS=, read -r -a LIST <<< "$UNITS"
for u in "${LIST[@]}"; do
  [[ $u =~ ^(clear|overcast|snow):[0-9]+:[0-9]+$ ]] || { log "ABORT: bad unit '$u' in UNITS"; exit 2; }
done

log "job $JOB ${TAG:+($TAG) }on $(hostname): units $UNITS, est clear ${EST[clear]} / overcast ${EST[overcast]} / snow ${EST[snow]} s/pose" \
    "+ $OVERHEAD s per call, cap $CAP_BYTES bytes ($POSE_BYTES per pose), stop $(date -d @$STOP_AT '+%H:%M') (+$STOP_MIN min)," \
    "kill $(date -d @$KILL_AT '+%H:%M') (+$KILL_MIN min)"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader || true

# ---- guards
head=$(git -C "$CODE" rev-parse --short HEAD); dirty=$(git -C "$CODE" status --porcelain --untracked-files=no | wc -l)
[ "$head" = "$EXPECT_COMMIT" ] && [ "$dirty" = 0 ] || { log "ABORT: code is $head with $dirty dirty files, not clean $EXPECT_COMMIT"; exit 3; }
for env in clear overcast snow; do
  out=$("$PY" -u -I "$TOOLS/fingerprint_guard.py" "$RD" "$ASSETS" "$OUT" $SCENE "$env" 2>&1); grc=$?
  out=$(echo "$out" | quiet)
  fp=$(echo "$out" | grep -o "current fingerprint [0-9a-f]*" | awk '{print $3}')
  log "fingerprint $env: ${fp:-none} (expected ${EXPECT[$env]}), guard rc=$grc"
  [ "$fp" = "${EXPECT[$env]}" ] && [ "$grc" = 0 ] || { echo "$out" | tail -5; log "ABORT: fingerprint guard failed for $env; nothing rendered"; exit 4; }
done

homeuse
( while true; do
    printf "%s,%s,%s\n" "$(date +%s)" "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" \
      "$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | tr -d ' ' | tr '\n' ';')"
    sleep 30
  done ) > "$LD/gpu.csv" 2>/dev/null &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT

# ---- units
written=0; fails=0; stop_reason=""; mine=()
for u in "${LIST[@]}"; do
  IFS=: read -r env start count <<< "$u"
  est=${EST[$env]}
  now=$(date +%s); fit=$(( (STOP_AT - now - OVERHEAD) / est ))
  (( fit <= 0 )) && { stop_reason="clock: no time for another pose before $(date -d @$STOP_AT '+%H:%M')"; break; }
  used=$(du -sb "$OUT" 2>/dev/null | cut -f1)
  room=$(( (CAP_BYTES - used) / POSE_BYTES ))
  (( room <= 0 )) && { stop_reason="storage cap: dataset at $used of $CAP_BYTES bytes"; break; }
  n=$count; (( n > fit )) && n=$fit; (( n > room )) && n=$room
  (( n < count )) && log "$env $start: trimmed to $n of $count poses (fit $fit, room $room)"
  plog=$LD/${env}_$(printf %06d $start).log; t0=$(date +%s)
  timeout --signal=TERM --kill-after=30 $(( KILL_AT - t0 )) \
    "$PY" -u -I "$RD/sampler.py" --scene $SCENE --env "$env" --start "$start" --count "$n" --lenses-per-pose "$LENSES" --variant cuda_ad_rgb \
      --out "$OUT" --assets "$ASSETS" --manifest-name "manifest.job$JOB.jsonl" > "$plog" 2>&1
  rc=$?; t1=$(date +%s)
  w=$(grep -c -E '^\[[0-9]+\] p[0-9]+_[0-9a-f]+ written' "$plog"); s=$(grep -c 'skipped (done)' "$plog")
  f=$(grep -o -E 'failed [0-9]+' "$plog" | tail -1 | awk '{print $2}'); f=${f:-$(( n - w ))}
  for i in $(seq "$start" $(( start + n - 1 ))); do mine+=("$env:$i"); done
  written=$(( written + w ))
  stopped=0; (( rc == 124 || rc == 137 )) && stopped=1
  echo "${TAG:-topup},$env,$start,$n,$t0,$t1,$(( t1 - t0 )),$rc,$w,$f,$s,$stopped" >> "$CSV"
  homeuse
  log "$env $start..$(( start + n - 1 )): rc=$rc written $w failed $f skipped $s in $(( t1 - t0 )) s (total $written)"
  (( stopped )) && { stop_reason="clock: unit killed at $(date -d @$KILL_AT '+%H:%M')"; break; }
  if (( rc != 0 )); then
    fails=$(( fails + 1 )); grep -a -E 'Error|Traceback|FAILED' "$plog" | tail -2
    (( fails >= 3 )) && { stop_reason="3 sampler failures in a row (last rc=$rc)"; break; }
  else
    fails=0
  fi
done
stop_reason=${stop_reason:-"unit list done"}
log "stopped: $stop_reason"
kill $MON 2>/dev/null

# ---- coverage: every listed index must have a finished pose in its unit's env
miss=()
for u in "${LIST[@]}"; do
  IFS=: read -r env start count <<< "$u"
  for (( i = start; i < start + count; i++ )); do
    ok=0; for p in "$OUT"/$SCENE/s1/"$env"/p$(printf %06d "$i")_*; do [ -e "$p/.done" ] && ok=1; done
    (( ok )) || miss+=("$env:$i")
  done
done
(( ${#miss[@]} )) && log "COVERAGE MISSING: ${miss[*]}" || log "COVERAGE OK: $UNITS all finished"

# ---- move this job's unfinished pose directories (killed mid-render: no .done) out of the dataset, then scan and report
PART=$B/dataset/_unfinished/job$JOB
for item in "${mine[@]}"; do
  env=${item%%:*}; i=${item##*:}
  for p in "$OUT"/$SCENE/s1/"$env"/p$(printf %06d "$i")_*; do
    [ -d "$p" ] && [ ! -e "$p/.done" ] && { mkdir -p "$PART/$env"; log "moving unfinished $p to $PART/$env"; mv -- "$p" "$PART/$env/"; }
  done
done
"$PY" -I "$TOOLS/nonfinite_scan.py" "$OUT" $SCENE --out "$LD/nonfinite" 2>&1 | grep -v jitc | tail -15
SCENE=$SCENE ENVS="clear overcast snow" LENSES=$LENSES "$PY" -I "$TOOLS/hpc_batch_report_scene.py" "$OUT" "$LD" "$JOB" "${TAG:-topup}" "$stop_reason" 2>&1 | tail -40
log "done"
