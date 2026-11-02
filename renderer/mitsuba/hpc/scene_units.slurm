#!/bin/bash
# Batch of an explicit unit list for any scene on one ws-ia GPU: stream_topup.slurm made scene-agnostic (SCENE, ENVS,
# CODE, ASSETS, EXPECT_COMMIT and the per-env fingerprints and estimates are all given). Copy to ~/dof/tools/ (new file).
#
#   FP="FP_CLEAR=<16 hex>,FP_OVERCAST=<16 hex>,FP_SAND=<16 hex>"     # from two agreeing fingerprint jobs
#   COMMON="ALL,SCENE=dry_riverbed,CODE=$HOME/dof/code_scene4,ASSETS=<assets root>,EXPECT_COMMIT=<short sha>,LENSES=6"
#   ENVS="clear overcast sand" UNITS=clear:0:7,overcast:0:7,sand:0:6 \
#     sbatch --time=<STOP_MIN + 30 min> --export=$COMMON,TAG=a,STOP_MIN=<min>,KILL_MIN=<min>,EST_CLEAR=..,EST_OVERCAST=..,EST_SAND=..,$FP ~/dof/tools/scene_units.slurm
# ENVS and a UNITS list with more than one unit go in the environment, not in --export (sbatch splits --export on
# commas and the env list has spaces).
#
# Stop and kill times are counted from this job's own start (STOP_MIN, KILL_MIN), never from submission. Each job stops
# at the first of: the end of its unit list, the storage cap (du before every unit), the clock (a unit is trimmed to
# the poses that fit before STOP_AT; a running unit is killed at KILL_AT), or 3 sampler failures in a row.
# Guards: clean code at EXPECT_COMMIT; the fingerprint of every env in ENVS equals FP_<ENV> (required).
# Coverage check over exactly the listed indices; finished poses with fewer than LENSES lenses are listed; unfinished
# pose directories are moved aside, never deleted.

#SBATCH --job-name=dof-units
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/beibarys.myrzash/dof/logs/%x-%j.out

set -uo pipefail
SCENE=${SCENE:?}; ENVS=${ENVS:?ENVS=\"env env ...\"}; read -r -a ENV_LIST <<< "$ENVS"
UNITS=${UNITS:?UNITS=env:start:count,...}
EXPECT_COMMIT=${EXPECT_COMMIT:?}
declare -A EXPECT EST
for env in "${ENV_LIST[@]}"; do
  fv=FP_${env^^}; ev=EST_${env^^}
  EXPECT[$env]=${!fv:?$fv required}; EST[$env]=${!ev:?$ev required (smoke rate + 30%)}
done
OVERHEAD=${OVERHEAD:-60}          # seconds per sampler call on top of EST (scene load, kernel recording)
STOP_MIN=${STOP_MIN:?}; KILL_MIN=${KILL_MIN:?}
CAP_BYTES=${CAP_BYTES:?}; POSE_BYTES=${POSE_BYTES:?}
LENSES=${LENSES:-6}
T_START=$(date +%s); STOP_AT=$(( T_START + STOP_MIN * 60 )); KILL_AT=$(( T_START + KILL_MIN * 60 ))
# never past this job's own SLURM end time either
END_SLURM=$(date -d "$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %e 2>/dev/null)" +%s 2>/dev/null || echo 0)
if (( END_SLURM > 0 )); then
  (( STOP_AT > END_SLURM - 900 )) && STOP_AT=$(( END_SLURM - 900 ))
  (( KILL_AT > END_SLURM - 600 )) && KILL_AT=$(( END_SLURM - 600 ))
fi
B=${B_ROOT:-$HOME/dof}
CODE=${CODE:-$B/code}; RD=$CODE/renderer/mitsuba; TOOLS=$B/tools
PY=${PY_OVERRIDE:-$HOME/.conda/envs/dof-graphics/bin/python}
ASSETS=${ASSETS:-$B/generated}; OUT=${OUT:-$B/dataset/mitsuba}
JOB=${SLURM_JOB_ID:-local}; LD=$B/logs/batch_$JOB${TAG:+_$TAG}; mkdir -p "$LD"
export DOF_WEB_ASSETS=$B/web_assets
export OPTIX_CACHE_PATH=/tmp/optix_cache_$USER
export PATH=$HOME/.conda/envs/dof-graphics/bin:$PATH
mkdir -p "$OPTIX_CACHE_PATH"
log() { echo "$(date '+%F %T') $*"; }
quiet() { grep -v -E 'jitc|catalog size|^\s+\[(probe|gbuffer|sharp|dof)\] [0-9]+/'; }
homeuse() { echo "$(date +%s),$(du -sbx $HOME 2>/dev/null | cut -f1)" >> "$LD/home.csv"; }
CSV=$LD/units.csv; echo "slot,env,start_index,count,start,end,seconds,rc,written,failed,skipped,stopped" > "$CSV"
IFS=, read -r -a LIST <<< "$UNITS"
ENV_RE=$(IFS='|'; echo "${ENV_LIST[*]}")
for u in "${LIST[@]}"; do
  [[ $u =~ ^($ENV_RE):[0-9]+:[0-9]+$ ]] || { log "ABORT: bad unit '$u' in UNITS (envs: $ENVS)"; exit 2; }
done

log "job $JOB ${TAG:+($TAG) }on $(hostname): $SCENE units $UNITS, est $(for e in "${ENV_LIST[@]}"; do printf '%s %s ' $e ${EST[$e]}; done)s/pose" \
    "+ $OVERHEAD s per call, cap $CAP_BYTES bytes ($POSE_BYTES per pose), stop $(date -d @$STOP_AT '+%H:%M') (+$STOP_MIN min)," \
    "kill $(date -d @$KILL_AT '+%H:%M') (+$KILL_MIN min)"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader || true

# ---- guards
head=$(git -C "$CODE" rev-parse --short HEAD); dirty=$(git -C "$CODE" status --porcelain --untracked-files=no | wc -l)
[ "$head" = "$EXPECT_COMMIT" ] && [ "$dirty" = 0 ] || { log "ABORT: $CODE is $head with $dirty dirty files, not clean $EXPECT_COMMIT"; exit 3; }
for env in "${ENV_LIST[@]}"; do
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
  echo "${TAG:-units},$env,$start,$n,$t0,$t1,$(( t1 - t0 )),$rc,$w,$f,$s,$stopped" >> "$CSV"
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

# ---- coverage: every listed index must have a finished pose in its unit's env; finished poses must hold LENSES lenses
miss=(); short=()
for u in "${LIST[@]}"; do
  IFS=: read -r env start count <<< "$u"
  for (( i = start; i < start + count; i++ )); do
    ok=0
    for p in "$OUT"/$SCENE/s1/"$env"/p$(printf %06d "$i")_*; do
      [ -e "$p/.done" ] || continue
      ok=1; nl=$(ls -d "$p"/lens_*/ 2>/dev/null | while read -r l; do [ -e "$l/dof.exr" ] && [ -e "$l/metadata.json" ] && echo; done | wc -l)
      (( nl < LENSES )) && short+=("$env:$i($nl)")
    done
    (( ok )) || miss+=("$env:$i")
  done
done
(( ${#miss[@]} )) && log "COVERAGE MISSING: ${miss[*]}" || log "COVERAGE OK: $UNITS all finished"
(( ${#short[@]} )) && log "SHORT POSES (lenses < $LENSES): ${short[*]}" || log "LENS COUNT OK: every finished pose has $LENSES lenses"

# ---- move this job's unfinished pose directories (killed mid-render: no .done) out of the dataset, then scan and report
PART=$B/dataset/_unfinished/job$JOB
for item in "${mine[@]}"; do
  env=${item%%:*}; i=${item##*:}
  for p in "$OUT"/$SCENE/s1/"$env"/p$(printf %06d "$i")_*; do
    [ -d "$p" ] && [ ! -e "$p/.done" ] && { mkdir -p "$PART/$env"; log "moving unfinished $p to $PART/$env"; mv -- "$p" "$PART/$env/"; }
  done
done
"$PY" -I "$TOOLS/nonfinite_scan.py" "$OUT" $SCENE --out "$LD/nonfinite" 2>&1 | grep -v jitc | tail -15
SCENE=$SCENE ENVS="$ENVS" LENSES=$LENSES "$PY" -I "$TOOLS/hpc_batch_report_scene.py" "$OUT" "$LD" "$JOB" "${TAG:-units}" "$stop_reason" 2>&1 | tail -40
log "done"
