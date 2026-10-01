#!/usr/bin/env bash
# Decoder M9 training chains (prereg dec-m9-prereg-2026-10-01.md), one per GPU on node A; each holds its GPU's flock
# for its whole run and writes the GPU's lease `owner` file (the lender's previous entry is kept once as
# owner.before-dec-m9). Seeds are interleaved so each arm's seeds use both GPUs:
#   g6 (node A GPU6): H9 s1 -> C9 s2 -> H9 s3
#   g7 (node A GPU7): C9 s1 -> H9 s2 -> C9 s3
# Before a seed starts, its TRAIN and teacher files are re-hashed against READY files (<dir>/READY holding the SHA-256
# from the committed data lock). Each seed is one drive_arm.sh run (zero-step, one-step, preflight_dec gate, full run,
# postrun) under run root m9/arms, image dbe5f32b. After seed 1 the chain reads HT-DEV v2 of its seed's BEST
# (m9-lines.sh early) and runs the early rule once both reads exist (m9-lines.sh e1 -> m9/early/E1.json); no seed 2-3
# starts before E1 is decided, unless a seed 1 (or its read) failed, which leaves the rule unapplied.
# Stop rules: a failed preflight stops the arm (no rerun, no replacement seed); E1 "stop" stops both arms; an arm stops
# at its GPU-h cap; no seed starts once the milestone has used 14 GPU-h; an arm with fewer than two finished seeds has
# no soup.
# Markers (under /data/dev2/runs/dec/m9/status):
#   m9-<ARM>-s<i>.DONE | .FAILED | .STOPPED   per seed
#   <ARM>.DONE | .FAILED                       per arm (DONE after its soup: soup/<ARM>/DONE)
# usage: m9-chains.sh launch <mirror-dir> g6|g7
#        m9-chains.sh run <mirror-dir> g6|g7   (the chain body, started by launch under flock)
set -u
MODE=$1 SRC=$2 CH=$3
M=/data/dev2/runs/dec/m9
C=$M/chains ST=$M/status
S=/data/dev2/src/$SRC/src/training/decision2
OPS=$S/v2/dec/ops/m9
D=$S/v2/dec/drive_arm.sh
mkdir -p "$C" "$ST" "$M/logs" "$M/soup" "$M/arms" "$M/early"
case $CH in
  g6) GPU=6 ITEMS="H9:1 C9:2 H9:3" ;;
  g7) GPU=7 ITEMS="C9:1 H9:2 C9:3" ;;
  # amendment 2: the GPU6 chain replaced in flight (H9-s1 adopted through chains/adopt-m9-H9-s1.pid)
  g6b) GPU=6 ITEMS="H9:1 C9:2 H9:3" ;;
  *) echo "unknown chain $CH" >&2; exit 2 ;;
esac

if [ "$MODE" = launch ]; then
  mkdir "$C/launch-$CH.lock" 2> /dev/null || { echo "M9 chain $CH already launched"; exit 0; }
  setsid nohup flock "$C/gpu$GPU.flock" bash "$0" run "$SRC" "$CH" > "$M/logs/chain-$CH.log" 2>&1 < /dev/null &
  echo $! > "$C/chain-$CH.pid"
  echo "$(date -u +%FT%TZ) M9 chain $CH launched from $SRC (pid $(cat "$C/chain-$CH.pid"))" | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }

NOX=/hf/models--llm-semantic-router--Decision-1.0-Nox-4B/snapshots/cde2a68dbaa557ea65dc458104d410a0802ee259
COMMON=(--train-mode full --batching tokens --max-batch-tokens 32768 --max-batch-rows 64 --update-rows 64 --teacher-partial)
SEEDS=(20260926 20260927 20260928)
CAP=${M9_CAP:-4.8} BUDGET_STOP=14  # amendment 2: the H9 chain runs with M9_CAP=5.8
EST=1.45  # expected GPU-h per seed until a seed of the arm has finished (M7: 1.35 at 43.5M tokens)
RUNENV=(DEC_NODE=a DEC_IMAGE_A=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
  DEC_DATA_DIR=/data/dev2/runs/dec/m3/data-sel700-cal698)
LEASE_DIR=/data/dev2/leases/gpu$GPU.lock
log() { echo "$(date -u +%FT%TZ) chain-$CH $*" | tee -a "$M/OPERATIONS.log"; }
gpuh() { python3 "$OPS/m9_gpuh.py" "$@"; }
gt() { python3 -c "import sys;sys.exit(0 if float(sys.argv[1])>float(sys.argv[2]) else 1)" "$1" "$2"; }
add() { python3 -c "import sys;print(float(sys.argv[1])+float(sys.argv[2]))" "$1" "$2"; }
lease() {  # <status> <purpose> <minutes>
  mkdir -p "$LEASE_DIR"
  [ -e "$LEASE_DIR/owner" ] && [ ! -e "$LEASE_DIR/owner.before-dec-m9" ] && cp -p "$LEASE_DIR/owner" "$LEASE_DIR/owner.before-dec-m9"
  printf 'track=dec\nstatus=%s\npurpose=decoder M9 %s (node A GPU6-7 lent by 9B, COORDINATION 2026-10-01 10:30)\nstart_utc=%s\nexpected_end_utc=%s\n' \
    "$1" "$2" "$(date -u +%FT%TZ)" "$(date -u -d "+$3 min" +%FT%TZ)" > "$LEASE_DIR/owner"
}
verify() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$(awk '{print $1; exit}' "$2")" ]; }
wait_file() {  # <file> [<log every N minutes>]
  local n=0
  until [ -f "$1" ]; do
    [ $((n % ${2:-30})) = 0 ] && log "waiting for $1"
    n=$((n + 1))
    sleep 60
  done
}
outcome() {  # <run name>
  local r=$1 f=$M/arms/OPERATIONS.log
  if grep -qE "^[^ ]+ $r (zero-step FAILED|one-step FAILED|gate job FAILED|preflight FAIL)" "$f" 2> /dev/null; then
    echo preflight
  elif grep -qE "^[^ ]+ $r full run FAILED" "$f" 2> /dev/null; then
    echo full
  elif grep -qE "^[^ ]+ $r postrun complete" "$f" 2> /dev/null; then
    echo complete
  elif grep -qE "^[^ ]+ $r full run complete" "$f" 2> /dev/null; then
    echo nopost
  else
    echo unknown
  fi
}
terminal() { [ -f "$ST/m9-$1-s$2.DONE" ] || [ -f "$ST/m9-$1-s$2.FAILED" ] || [ -f "$ST/m9-$1-s$2.STOPPED" ]; }

soup_if_complete() {  # <ARM>: after the arm's last seed (on either chain), build the soup once
  local g=$1 s n=0
  for s in 1 2 3; do terminal "$g" "$s" || return 0; done
  for s in 1 2 3; do [ -f "$ST/m9-$g-s$s.DONE" ] && n=$((n + 1)); done
  if [ $n -lt 2 ]; then
    [ -f "$ST/$g.FAILED" ] || echo "fewer than two finished seeds; no soup" > "$ST/$g.FAILED"
    log "$g: fewer than two finished seeds; no soup"
    return 0
  fi
  lease busy "$g soup build (CPU)" 20
  bash "$OPS/m9-soup.sh" "$SRC" "$g"
  if [ -f "$M/soup/$g/DONE" ] && [ ! -f "$ST/$g.DONE" ]; then
    printf 'soup=%s\ngpu_h=%s\n' "$M/soup/$g/DONE" "$(gpuh arm "$g")" > "$ST/$g.DONE"
    log "$g DONE (soup $M/soup/$g/build/$g-soup)"
  elif [ -f "$M/soup/$g/FAILED" ] && [ ! -f "$ST/$g.FAILED" ]; then
    echo "soup failed" > "$ST/$g.FAILED"
    log "$g soup FAILED"
  fi
}

early_read() {  # <ARM>: HT-DEV v2 of the seed-1 BEST, then the rule if both reads exist
  local g=$1
  [ -f "$ST/m9-$g-s1.DONE" ] || return 0
  lease busy "early-rule read $g-s1 (HT-DEV v2)" 20
  if ! env M9_GPU=$GPU bash "$OPS/m9-lines.sh" early "$g" >> "$M/logs/early-$g.log" 2>&1; then
    echo "early read failed (see logs/early-$g.log)" > "$M/early/$g-read.FAILED"
    log "early read of $g-s1 FAILED: E1 is not applied"
    return 0
  fi
  log "early read of $g-s1 done"
  bash "$OPS/m9-lines.sh" e1 >> "$M/logs/early-e1.log" 2>&1 || log "E1 rule step failed (see logs/early-e1.log)"
}

e1_gate() {  # prints continue | stop | na once the early rule is decided or cannot apply
  local n=0 g
  lease busy "waiting for the early rule (E1)" 30
  while :; do
    if [ -f "$M/early/E1.json" ]; then
      python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "$M/early/E1.json"
      return 0
    fi
    for g in H9 C9; do
      if [ -f "$ST/m9-$g-s1.FAILED" ] || [ -f "$ST/m9-$g-s1.STOPPED" ] || [ -f "$M/early/$g-read.FAILED" ]; then
        echo na
        return 0
      fi
    done
    [ $((n % 15)) = 0 ] && log "waiting for E1 (both seed-1 reads)" >&2
    n=$((n + 1))
    # the other chain may have finished its read before this one: retry the rule
    [ $((n % 5)) = 0 ] && bash "$OPS/m9-lines.sh" e1 >> "$M/logs/early-e1.log" 2>&1
    sleep 60
  done
}

finish_item() {  # <ARM> <seed index> <seed>: markers after a seed's drive_arm.sh run, then the seed-1 early read
  local g=$1 i=$2 seed=$3 r=m9-$1-s$2 out note
  gpuh write
  out=$(outcome "$r")
  note="seed $seed; GPU-h $(gpuh seed "$g" "$i"); $(date -u +%FT%TZ)"
  if [ -f "$ST/$r.capstop" ]; then
    if [ "$out" = complete ] || [ "$out" = nopost ]; then
      echo "full run complete; postrun $([ "$out" = complete ] && echo complete || echo 'stopped at the cap'); $note" > "$ST/$r.DONE"
    else
      echo "stopped at the arm cap $CAP GPU-h (stage result: $out); $note" > "$ST/$r.STOPPED"
    fi
    log "$r reached the cap ($out)"
  else
    case $out in
      preflight)
        echo "preflight failed; $note" > "$ST/$r.FAILED"
        log "$r preflight failed: arm $g stops (no rerun, no replacement seed)"
        ;;
      complete | nopost)
        echo "full run complete; postrun $([ "$out" = complete ] && echo complete || echo FAILED); $note" > "$ST/$r.DONE"
        log "$r finished ($out)"
        ;;
      *)
        echo "full run failed ($out); $note" > "$ST/$r.FAILED"
        log "$r failed ($out); the arm continues with its next seed"
        ;;
    esac
  fi
  [ "$i" = 1 ] && early_read "$g"
  return 0
}

item() {  # <ARM> <seed index>
  local g=$1 i=$2 seed r t0 wd used est total fin=() decision apid
  seed=${SEEDS[$((i - 1))]} r=m9-$g-s$i
  terminal "$g" "$i" && return 0
  [ -f "$ST/$g.DONE" ] || [ -f "$ST/$g.FAILED" ] && { log "$r skipped: $g already has an arm marker"; return 0; }
  if [ -f "$C/adopt-$r.pid" ]; then  # amendment 2: a run started by the replaced chain, finished here
    apid=$(cat "$C/adopt-$r.pid")
    lease busy "$r (adopted drive_arm.sh pid $apid)" 60
    log "$r: adopting the running drive_arm.sh (pid $apid)"
    while kill -0 "$apid" 2> /dev/null; do sleep 60; done
    finish_item "$g" "$i" "$seed"
    return 0
  fi
  local train=$M/data/4b/mix/m9-4b-$g/train.jsonl teach=$M/teacher/m9-4b-$g/teacher.jsonl
  lease busy "$r waiting for the data lock (READY)" 60
  wait_file "$(dirname "$train")/READY"
  wait_file "$(dirname "$teach")/READY"
  if ! verify "$train" "$(dirname "$train")/READY" || ! verify "$teach" "$(dirname "$teach")/READY"; then
    echo "TRAIN or teacher hash differs from the lock; not started" > "$ST/$r.STOPPED"
    log "$r not started: TRAIN or teacher hash differs from the lock"
    return 0
  fi
  if [ "$i" -gt 1 ]; then
    decision=$(e1_gate)
    log "$r: early rule $decision"
    if [ "$decision" = stop ]; then
      echo "not started: the early rule E1 stopped H9 and C9 (early/E1.json)" > "$ST/$r.STOPPED"
      log "$r not started: E1 stop"
      return 0
    fi
  fi
  if ls "$ST"/m9-"$g"-s?.FAILED > /dev/null 2>&1 && grep -qs "preflight failed" "$ST"/m9-"$g"-s?.FAILED; then
    echo "not started: a preflight of arm $g failed" > "$ST/$r.STOPPED"
    log "$r not started: arm $g stopped by a failed preflight"
    return 0
  fi
  used=$(gpuh arm "$g")
  total=$(gpuh total)
  est=$EST
  for s in 1 2 3; do [ -f "$ST/m9-$g-s$s.DONE" ] && fin+=("$(gpuh seed "$g" "$s")"); done
  [ ${#fin[@]} -gt 0 ] && est=$(python3 -c "import sys;v=[float(x) for x in sys.argv[1:]];print(sum(v)/len(v))" "${fin[@]}")
  if gt "$(add "$used" "$est")" "$CAP"; then
    echo "cap: used $used + expected $est > cap $CAP GPU-h; not started" > "$ST/$r.STOPPED"
    log "$r not started: arm used $used + expected $est GPU-h > cap $CAP"
    return 0
  fi
  if gt "$total" "$BUDGET_STOP" || [ "$total" = "$BUDGET_STOP" ]; then
    echo "budget: milestone used $total GPU-h >= $BUDGET_STOP; not started" > "$ST/$r.STOPPED"
    log "$r not started: milestone used $total GPU-h >= $BUDGET_STOP"
    return 0
  fi
  lease busy "$r (training; arm cap $CAP GPU-h, arm used $used, milestone used $total)" "$(python3 -c "print(int(float('$est')*60)+10)")"
  log "start $r on node A GPU$GPU (seed $seed; arm used $used, expected $est, cap $CAP; milestone used $total)"
  t0=$(date -u +%s)
  (
    while sleep 60; do
      if gt "$(add "$used" "$(python3 -c "print(($(date -u +%s) - $t0) / 3600)")")" "$CAP"; then
        touch "$ST/$r.capstop"
        log "$r reached the arm cap $CAP GPU-h; stopping its containers"
        docker ps --format '{{.Names}}' | grep -E "^dec-(m9-$g-s$i-|m9-arms-full-m9-$g-s$i-)" | xargs -r docker stop > /dev/null
      fi
    done
  ) &
  wd=$!
  env "${RUNENV[@]}" bash "$D" "$r" "$GPU" "$SRC" m9/arms "$NOX" -- \
    --train "/runs/${train#/data/dev2/runs/dec/}" --teacher "/runs/${teach#/data/dev2/runs/dec/}" \
    --teacher-kl-weight 1.0 "${COMMON[@]}" --backbone-lr 5e-6 --head-lr 5e-5 --seed "$seed"
  kill "$wd" 2> /dev/null
  wait "$wd" 2> /dev/null
  finish_item "$g" "$i" "$seed"
}

log "chain $CH started on node A GPU$GPU from $SRC: $ITEMS"
for it in $ITEMS; do
  item "${it%%:*}" "${it#*:}"
  soup_if_complete H9
  soup_if_complete C9
done
printf 'track=dec\nstatus=idle (decoder M9 chain %s finished)\nend_utc=%s\n' "$CH" "$(date -u +%FT%TZ)" > "$LEASE_DIR/owner"
gpuh write
log "chain $CH finished"
