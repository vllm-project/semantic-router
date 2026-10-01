#!/usr/bin/env bash
# Decoder M10 post-training chain for one arm on one M10 GPU: wait until the arm's three seeds have a terminal marker
# (and the GPU's training chain has finished), build the arm soup (m10-soup.sh: LoRA merges on this GPU, soup on CPU),
# then read every development panel of the soup (m10-lines.sh read). A failed step stops the chain (never rerun).
#
# usage: M10_NODE=e|f m10-post.sh launch <mirror-dir> <ARM> <gpu>
#        M10_NODE=e|f m10-post.sh run <mirror-dir> <ARM> <gpu>
set -u
MODE=$1 SRC=$2 ARM=$3 GPU=$4
NODE=${M10_NODE:?set M10_NODE=e or f}
M=/data/dev2/runs/dec/m10
ST=$M/status
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m10
BASE=/data/dev2/models/Qwen--Qwen3.5-4B-Base/1001bb4d826a52d1f399e183466143f4da7b741b
NOX=/data/dev2/models/Decision-1.0-Nox-4B/cde2a68dbaa557ea65dc458104d410a0802ee259
mkdir -p "$M/chains" "$M/logs"
if [ "$MODE" = launch ]; then
  mkdir "$M/chains/post-$ARM.lock" 2> /dev/null || { echo "post chain $ARM already launched"; exit 0; }
  M10_NODE=$NODE setsid nohup bash "$0" run "$SRC" "$ARM" "$GPU" > "$M/logs/post-$ARM.log" 2>&1 < /dev/null &
  echo $! > "$M/chains/post-$ARM.pid"
  echo "$(date -u +%FT%TZ) M10 post chain $ARM launched on GPU$GPU from $SRC (pid $(cat "$M/chains/post-$ARM.pid"))" \
    | tee -a "$M/OPERATIONS.log"
  exit 0
fi
log() { echo "$(date -u +%FT%TZ) post-$ARM $*" | tee -a "$M/OPERATIONS.log"; }
terminal() { [ -f "$ST/m10-$ARM-s$1.DONE" ] || [ -f "$ST/m10-$ARM-s$1.FAILED" ] || [ -f "$ST/m10-$ARM-s$1.STOPPED" ]; }
n=0
until terminal 1 && terminal 2 && terminal 3; do
  [ $((n % 30)) = 0 ] && log "waiting for the seeds of $ARM"
  n=$((n + 1))
  sleep 60
done
while grep -qs "^status=busy" "/data/dev2/leases/gpu$GPU.lock/owner" && grep -qs "training" "/data/dev2/leases/gpu$GPU.lock/owner"; do
  sleep 60
done
printf 'track=dec-m10\nstatus=busy\npurpose=decoder M10 %s soup (merges)\nstart_utc=%s\nexpected_end_utc=%s\n' \
  "$ARM" "$(date -u +%FT%TZ)" "$(date -u -d '+45 min' +%FT%TZ)" > "/data/dev2/leases/gpu$GPU.lock/owner"
M10_NODE=$NODE bash "$OPS/m10-soup.sh" "$SRC" "$ARM" "$GPU" || { log "soup failed; no readout"; exit 1; }
case $ARM in NT2) source=$NOX ;; *) source=$BASE ;; esac
M10_NODE=$NODE bash "$OPS/m10-lines.sh" read "$SRC" "$GPU" "4b-$ARM" "$(cat "$M/soup/$ARM/DONE")" "$source"
log "readouts of 4b-$ARM finished"
