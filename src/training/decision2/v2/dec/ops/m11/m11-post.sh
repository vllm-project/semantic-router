#!/usr/bin/env bash
# Decoder M11 reference and post-training chains on one M11 GPU, each holding the GPU's chain flock (so it starts only
# when no training chain holds that GPU, and readouts on a GPU never overlap):
#   refs (node E GPU3): per tier, C0 (DEV2.0-<t>'s weights, the M8s start copies) read as <t>-C0-e, then the untrained
#        base through the label-token readout (a LoRA zero-step checkpoint, as M10's 4b-BASE) read as <t>-BASE-e;
#   arm <ARM>: wait until the arm's three seeds have a terminal marker, read C0 on this node as <t>-C0-<node> if not
#        read yet (every point is gated against C0 read on its own node), build the arm soup (m11-soup.sh) and read it
#        as <ARM>.
# A failed step stops the chain (never rerun).
#
# usage: M11_NODE=e|f m11-post.sh launch <mirror-dir> <gpu> refs|<ARM>
#        M11_NODE=e|f m11-post.sh run <mirror-dir> <gpu> refs|<ARM>
set -u
MODE=$1 SRC=$2 GPU=$3 WHAT=$4
NODE=${M11_NODE:?set M11_NODE=e or f}
M=/data/dev2/runs/dec/m11
ST=$M/status
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m11
mkdir -p "$M/chains" "$M/logs"
if [ "$MODE" = launch ]; then
  mkdir "$M/chains/post-$WHAT.lock" 2> /dev/null || { echo "post chain $WHAT already launched"; exit 0; }
  M11_NODE=$NODE setsid nohup flock "$M/chains/gpu$GPU.flock" bash "$0" run "$SRC" "$GPU" "$WHAT" \
    > "$M/logs/post-$WHAT.log" 2>&1 < /dev/null &
  echo $! > "$M/chains/post-$WHAT.pid"
  echo "$(date -u +%FT%TZ) M11 post chain $WHAT launched on node ${NODE^^} GPU$GPU from $SRC (pid $(cat "$M/chains/post-$WHAT.pid"))" \
    | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }
log() { echo "$(date -u +%FT%TZ) post-$WHAT $*" | tee -a "$M/OPERATIONS.log"; }
H=/data/dev2/models
declare -A BASE=([2b]=$H/Qwen--Qwen3.5-2B-Base/b1485b2fa6dfa1287294f269f5fb618e03d52d7c
  [08b]=$H/Qwen--Qwen3.5-0.8B-Base/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68)
declare -A REV=([2b]=b1485b2fa6dfa1287294f269f5fb618e03d52d7c [08b]=dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68)
declare -A ONE=([2b]=$H/Decision-1.0-Sol-2B/ce0c018a28de16d6639b1cd203b761bf643b89e6
  [08b]=$H/Decision-1.0-Eos-0.8B/363c4a5e56afc115b1c78c837633956d0bbb63ab)
declare -A C0=([2b]=$H/DEV2.0-2B/a53cf66a0d9d492a84b6617b61e7ce35fcd03af0
  [08b]=$H/DEV2.0-0.8B/bede7938a8c209c09f27400b79eed57948d6b75e)
declare -A TRAIN=([2b]=/runs/m11/inputs/m4-v2m-ret-r2/train.jsonl [08b]=/runs/m11/inputs/m6-e8f-r2clean/train.jsonl)
lease() {  # <status> <purpose> <minutes>
  printf 'track=dec-m11\nstatus=%s\npurpose=decoder M11 %s\nstart_utc=%s\nexpected_end_utc=%s\n' \
    "$1" "$2" "$(date -u +%FT%TZ)" "$(date -u -d "+$3 min" +%FT%TZ)" > "/data/dev2/leases/gpu$GPU.lock/owner"
}
read_point() {  # <point> <checkpoint> <source>
  M11_NODE=$NODE bash "$OPS/m11-lines.sh" read "$SRC" "$GPU" "$1" "$2" "$3"
  log "readouts of $1 finished"
}
c0() {  # <tier>
  [ -f "$M/lines/$1-C0-$NODE/m10-probes.launch.json" ] && return 0
  read_point "$1-C0-$NODE" "${C0[$1]}" "${ONE[$1]}"
}
base_zero() {  # <tier>: the label-token LoRA zero-step checkpoint from the base (no update; LoRA B = 0)
  local t=$1 out ntlen
  out=$M/arms/pre/m11-$t-BASE-zero
  [ -d "$out/checkpoint-0000000" ] && return 0
  [ -e "$out.launch.json" ] && { log "$t-BASE zero-step failed earlier; not rerun"; return 1; }
  ntlen=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["nt_max_length"])' "$M/data/READY-$t.json") \
    || { log "$t-BASE: no READY-$t.json"; return 1; }
  lease busy "$t-BASE zero-step (no update)" 30
  M11_NODE=$NODE M11_CACHE=$t-read bash "$OPS/m11-launch.sh" "$t-BASE-zero" "$SRC" "$out" --gpu "$GPU" -- \
    -m v2.dec.train_dec --model-path "/models/${BASE[$t]#"$H"/}" --init base --revision "${REV[$t]}" \
    --train "${TRAIN[$t]}" --select /data/select.jsonl --cal /data/cal.jsonl --output /out --arm "m11-$t-BASE" \
    --train-mode lora --lora-rank 128 --lora-alpha 256 --lora-dropout 0.05 --lora-lr 1e-4 --readout label_token \
    --max-length "$ntlen" --seed 20260926 --zero-step-only || { log "$t-BASE zero-step FAILED"; return 1; }
  log "$t-BASE zero-step done"
}

if [ "$WHAT" = refs ]; then
  for t in 2b 08b; do
    c0 "$t"
    base_zero "$t" && read_point "$t-BASE-$NODE" "$M/arms/pre/m11-$t-BASE-zero/checkpoint-0000000" "${BASE[$t]}"
  done
  lease idle "reference readouts finished" 30
  log "refs finished"
  exit 0
fi

ARM=$WHAT t=${WHAT%%-*}
terminal() { [ -f "$ST/m11-$ARM-s$1.DONE" ] || [ -f "$ST/m11-$ARM-s$1.FAILED" ] || [ -f "$ST/m11-$ARM-s$1.STOPPED" ]; }
n=0
until terminal 1 && terminal 2 && terminal 3; do
  [ $((n % 30)) = 0 ] && log "waiting for the seeds of $ARM"
  n=$((n + 1))
  sleep 60
done
lease busy "$ARM post chain (C0 / soup / readouts)" 120
c0 "$t"
M11_NODE=$NODE bash "$OPS/m11-soup.sh" "$SRC" "$ARM" "$GPU" || { log "soup failed; no readout"; lease idle "post $ARM failed" 30; exit 1; }
case $ARM in *-LH) source=${BASE[$t]} ;; *) source=${ONE[$t]} ;; esac
read_point "$ARM" "$(cat "$M/soup/$ARM/DONE")" "$source"
lease idle "post $ARM finished" 30
