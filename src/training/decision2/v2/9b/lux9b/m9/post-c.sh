#!/usr/bin/env bash
# 9B M9 post-training chain on node C for one arm: wait until every seed has a terminal marker and the GPU's training
# chain has released it, then build the arm artifact (soup.sh: LoRA merges on this GPU, soup on CPU). A failed step
# stops the chain (never rerun).
# Stage 3 (amendment 3): KIB / KIBX wait for three seeds and soup full checkpoints on CPU only (no GPU, no lease).
#
# usage: M9_NODE=c post-c.sh launch <mirror-dir> <ARM> <gpu>      (gpu "-" for KIB / KIBX)
#        M9_NODE=c post-c.sh run <mirror-dir> <ARM> <gpu>
set -u
MODE=$1 SRC=$2 ARM=$3 GPU=$4
NODE=${M9_NODE:?set M9_NODE=c}
M=/data/dev2/runs/9b/m9
ST=$M/status
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/9b/lux9b/m9
case $ARM in KIB | KIBX) SEEDS="1 2 3" CPU=1 ;; *) SEEDS="1 2" CPU=0 ;; esac
mkdir -p "$M/chains" "$M/logs"
if [ "$MODE" = launch ]; then
  mkdir "$M/chains/post-c-$ARM.lock" 2> /dev/null || { echo "post chain $ARM already launched"; exit 0; }
  M9_NODE=$NODE setsid nohup bash "$0" run "$SRC" "$ARM" "$GPU" > "$M/logs/post-c-$ARM.log" 2>&1 < /dev/null &
  echo $! > "$M/chains/post-c-$ARM.pid"
  echo "$(date -u +%FT%TZ) M9 post chain $ARM launched on node C GPU$GPU from $SRC (pid $(cat "$M/chains/post-c-$ARM.pid"))" \
    | tee -a "$M/OPERATIONS.log"
  exit 0
fi
log() { echo "$(date -u +%FT%TZ) post-c-$ARM $*" | tee -a "$M/OPERATIONS.log"; }
terminal() { [ -f "$ST/m9-$ARM-s$1.DONE" ] || [ -f "$ST/m9-$ARM-s$1.FAILED" ] || [ -f "$ST/m9-$ARM-s$1.STOPPED" ]; }
all_terminal() {
  local s
  for s in $SEEDS; do terminal "$s" || return 1; done
}
n=0
log "waiting for the seeds of $ARM"
until all_terminal; do
  n=$((n + 1))
  [ $((n % 30)) = 0 ] && log "still waiting for the seeds of $ARM"
  sleep 60
done
if [ "$CPU" = 1 ]; then
  M9_NODE=$NODE bash "$OPS/soup.sh" "$SRC" "$ARM" -
  rc=$?
  log "soup chain finished (exit $rc)"
  exit $rc
fi
lease=/data/dev2/leases/gpu$GPU.lock/owner
while grep -qs "^status=busy" "$lease" && grep -qs "training" "$lease"; do sleep 60; done
grep -qs '^track=9b-m9' "$lease" || { log "GPU$GPU lease is not 9b-m9's; no merge"; exit 1; }
printf 'track=9b-m9\nstatus=busy\npurpose=9B M9 %s soup (LoRA merges)\nstart_utc=%s\nexpected_end_utc=%s\n' \
  "$ARM" "$(date -u +%FT%TZ)" "$(date -u -d '+45 min' +%FT%TZ)" > "$lease"
M9_NODE=$NODE bash "$OPS/soup.sh" "$SRC" "$ARM" "$GPU"
rc=$?
printf 'track=9b-m9\nstatus=idle\npurpose=9B M9 (%s soup finished, exit %s)\nstart_utc=%s\n' "$ARM" "$rc" \
  "$(date -u +%FT%TZ)" > "$lease"
log "soup chain finished (exit $rc)"
exit $rc
