#!/usr/bin/env bash
# Decoder M7 training chains (prereg dec-m7-prereg-2026-09-30.md), one per GPU; each holds its GPU's flock for its
# whole run and writes the GPU's lease `owner` file (never other tracks' entries). The M6 chain (ops/m6/m6-chains.sh)
# with the M7 arms, seed subsets and a soup built by whichever chain finishes an arm's last seed.
#   b3 (node B GPU3): N7H s1-s3 -> N7P s1, s3
#   b4 (node B GPU4): N7C s1-s3 -> N7P s2
#   a5 (node A GPU5): S7H s1-s3 -> S7C s1-s3 -> S7P s1-s3
# Before an arm's seed starts, its TRAIN and teacher files are re-hashed against READY files (<dir>/READY holding the
# SHA-256 from the committed data lock); P arms wait for their READY (the PN1 revision rule). Each seed is one
# drive_arm.sh run (zero-step, one-step, preflight_dec gate, full run, postrun) under run root m7/arms. Stop rules: a
# failed preflight stops the arm (no rerun, no replacement seed; a seed of the same arm queued on the other chain is
# not started either); an arm stops at its GPU-h cap; an arm with fewer than two finished seeds has no soup.
# Markers (under /data/dev2/runs/dec/m7/status on the arm's node):
#   m7-<ARM>-s<i>.DONE | .FAILED | .STOPPED   per seed
#   <ARM>.DONE | .FAILED                       per arm (DONE after its soup: soup/<ARM>/DONE)
# usage: m7-chains.sh launch <mirror-dir> b3|b4|a5
#        m7-chains.sh run <mirror-dir> b3|b4|a5   (the chain body, started by launch under flock)
set -u
MODE=$1 SRC=$2 CH=$3
M=/data/dev2/runs/dec/m7
C=$M/chains ST=$M/status
S=/data/dev2/src/$SRC/src/training/decision2
OPS=$S/v2/dec/ops/m7
D=$S/v2/dec/drive_arm.sh
mkdir -p "$C" "$ST" "$M/logs" "$M/soup" "$M/arms"
case $CH in
  b3) NODE=b GPU=3 ITEMS="N7H N7P:1,3" ;;
  b4) NODE=b GPU=4 ITEMS="N7C N7P:2" ;;
  a5) NODE=a GPU=5 ITEMS="S7H S7C S7P" ;;
  *) echo "unknown chain $CH" >&2; exit 2 ;;
esac

if [ "$MODE" = launch ]; then
  mkdir "$C/launch-$CH.lock" 2>/dev/null || { echo "M7 chain $CH already launched"; exit 0; }
  setsid nohup flock "$C/gpu$GPU.flock" bash "$0" run "$SRC" "$CH" > "$M/logs/chain-$CH.log" 2>&1 < /dev/null &
  echo $! > "$C/chain-$CH.pid"
  echo "$(date -u +%FT%TZ) M7 chain $CH launched from $SRC (pid $(cat "$C/chain-$CH.pid"))" | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }

NOX=/hf/models--llm-semantic-router--Decision-1.0-Nox-4B/snapshots/cde2a68dbaa557ea65dc458104d410a0802ee259
SOLM=/hf/models--llm-semantic-router--Decision-1.0-Sol-2B/snapshots/ce0c018a28de16d6639b1cd203b761bf643b89e6
COMMON=(--train-mode full --batching tokens --max-batch-tokens 32768 --max-batch-rows 64 --update-rows 64 --teacher-partial)
declare -A START=([N7H]=$NOX [N7P]=$NOX [N7C]=$NOX [S7H]=$SOLM [S7P]=$SOLM [S7C]=$SOLM)
# TRAIN / teacher directories under m7/ (lock record); the P arms use the PN1-r2 rebuild (prereg PN1 revision rule).
declare -A TRAINDIR=([N7H]=data/4b/mix/m7-4b-H [N7C]=data/4b/mix/m7-4b-C [N7P]=data/4b-r2/mix/m7-4b-P
  [S7H]=data/2b/mix/m7-2b-H [S7C]=data/2b/mix/m7-2b-C [S7P]=data/2b-r2/mix/m7-2b-P)
declare -A TEACHDIR=([N7H]=teacher/m7-4b-H [N7C]=teacher/m7-4b-C [N7P]=teacher/m7-4b-P-r2
  [S7H]=teacher/m7-2b-H [S7C]=teacher/m7-2b-C [S7P]=teacher/m7-2b-P-r2)
declare -A KL=([N7H]=1.0 [N7P]=1.0 [N7C]=1.0 [S7H]=0.5 [S7P]=0.5 [S7C]=0.5)
SEEDS=(20260926 20260927 20260928)
declare -A CAP=([N7H]=4.5 [N7P]=4.5 [N7C]=4.5 [S7H]=2.8 [S7P]=2.8 [S7C]=2.8)
# Expected GPU-h per seed (preflights + full run + postrun) from M6's per-token cost; after a finished seed of the
# arm the observed mean replaces it.
declare -A EST=([N7H]=1.35 [N7P]=1.35 [N7C]=1.35 [S7H]=0.75 [S7P]=0.75 [S7C]=0.75)
DATA_ENV=(DEC_DATA_DIR=/data/dev2/runs/dec/m3/data-sel700-cal698)  # N4XF's and S2T's SELECT700 / CAL698
log() { echo "$(date -u +%FT%TZ) chain-$CH $*" | tee -a "$M/OPERATIONS.log"; }
gpuh() { python3 "$OPS/m7-gpuh.py" "$@"; }
lease() {  # <status> <purpose> <minutes>
  printf 'track=dec\nstatus=%s\npurpose=decoder M7 %s\nstart_utc=%s\nexpected_end_utc=%s\n' "$1" "$2" \
    "$(date -u +%FT%TZ)" "$(date -u -d "+$3 min" +%FT%TZ)" > "/data/dev2/leases/gpu$GPU.lock/owner"
}
verify() {  # <file> <READY file>
  [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$(awk '{print $1; exit}' "$2")" ]
}
wait_ready() {
  local n=0
  until [ -f "$1" ]; do
    [ $((n % 30)) = 0 ] && log "waiting for $1"
    n=$((n + 1))
    sleep 60
  done
}
outcome() {  # <run name>
  local r=$1 f=$M/arms/OPERATIONS.log
  if grep -qE "^[^ ]+ $r (zero-step FAILED|one-step FAILED|gate job FAILED|preflight FAIL)" "$f" 2>/dev/null; then
    echo preflight
  elif grep -qE "^[^ ]+ $r full run FAILED" "$f" 2>/dev/null; then
    echo full
  elif grep -qE "^[^ ]+ $r postrun complete" "$f" 2>/dev/null; then
    echo complete
  elif grep -qE "^[^ ]+ $r full run complete" "$f" 2>/dev/null; then
    echo nopost
  else
    echo unknown
  fi
}
terminal() { [ -f "$ST/m7-$1-s$2.DONE" ] || [ -f "$ST/m7-$1-s$2.FAILED" ] || [ -f "$ST/m7-$1-s$2.STOPPED" ]; }

soup_if_complete() {  # <ARM>: after the arm's last seed (on either chain), build the soup once
  local g=$1 s n=0
  for s in 1 2 3; do terminal "$g" "$s" || return 0; done
  for s in 1 2 3; do [ -f "$ST/m7-$g-s$s.DONE" ] && n=$((n + 1)); done
  if [ $n -lt 2 ]; then
    [ -f "$ST/$g.FAILED" ] || echo "fewer than two finished seeds; dropped" > "$ST/$g.FAILED"
    log "$g dropped: fewer than two finished seeds"
    return 0
  fi
  lease busy "$g soup build (CPU)" 20
  bash "$OPS/m7-soup.sh" "$SRC" "$g" "$NODE"
  if [ -f "$M/soup/$g/DONE" ] && [ ! -f "$ST/$g.DONE" ]; then
    printf 'soup=%s\ngpu_h=%s\n' "$M/soup/$g/DONE" "$(gpuh arm "$g")" > "$ST/$g.DONE"
    log "$g DONE (soup $M/soup/$g/build/$g-soup)"
  elif [ -f "$M/soup/$g/FAILED" ] && [ ! -f "$ST/$g.FAILED" ]; then
    echo "soup failed" > "$ST/$g.FAILED"
    log "$g soup FAILED"
  fi
}

arm() {  # <ARM>[:<seed index list>]
  local g=${1%%:*} which=1,2,3 i seed r t0 wd used est out note fin=()
  [[ $1 == *:* ]] && which=${1#*:}
  [ -f "$ST/$g.DONE" ] || [ -f "$ST/$g.FAILED" ] && { log "$g already has a marker; skipped"; return 0; }
  local train=$M/${TRAINDIR[$g]}/train.jsonl teach=$M/${TEACHDIR[$g]}/teacher.jsonl
  lease busy "$g waiting for its data lock (READY)" 60
  wait_ready "$(dirname "$train")/READY"
  wait_ready "$(dirname "$teach")/READY"
  if ! verify "$train" "$(dirname "$train")/READY" || ! verify "$teach" "$(dirname "$teach")/READY"; then
    echo "TRAIN or teacher hash differs from the lock" > "$ST/$g.FAILED"
    log "$g stopped: TRAIN or teacher hash differs from the lock"
    return 0
  fi
  log "$g inputs verified: $(sha256sum "$train" | cut -c1-12) / $(sha256sum "$teach" | cut -c1-12)"
  for i in ${which//,/ }; do
    seed=${SEEDS[$((i - 1))]} r=m7-$g-s$i
    terminal "$g" "$i" && continue
    if ls "$ST"/m7-"$g"-s?.FAILED > /dev/null 2>&1 && grep -qs "preflight failed" "$ST"/m7-"$g"-s?.FAILED; then
      echo "not started: a preflight of arm $g failed" > "$ST/$r.STOPPED"
      log "$r not started: arm $g stopped by a failed preflight"
      continue
    fi
    used=$(gpuh arm "$g")
    est=${EST[$g]}
    for s in 1 2 3; do
      [ -f "$ST/m7-$g-s$s.DONE" ] && fin+=("$(gpuh seed "$g" "$s")")
    done
    [ ${#fin[@]} -gt 0 ] && est=$(python3 -c "import sys;v=[float(x) for x in sys.argv[1:]];print(sum(v)/len(v))" "${fin[@]}")
    fin=()
    if python3 -c "import sys;sys.exit(0 if float(sys.argv[1])+float(sys.argv[2])>float(sys.argv[3]) else 1)" "$used" "$est" "${CAP[$g]}"; then
      echo "cap: used $used + expected $est > cap ${CAP[$g]} GPU-h; not started" > "$ST/$r.STOPPED"
      log "$r not started: used $used + expected $est GPU-h > cap ${CAP[$g]}"
      continue
    fi
    lease busy "$r (training; arm cap ${CAP[$g]} GPU-h, used $used)" "$(python3 -c "print(int(float('$est')*60)+10)")"
    log "start $r on node ${NODE^^} GPU$GPU (seed $seed; arm GPU-h used $used, expected $est, cap ${CAP[$g]})"
    t0=$(date -u +%s)
    (
      while sleep 60; do
        if python3 -c "import sys;sys.exit(0 if float(sys.argv[1])+(float(sys.argv[2]))/3600>float(sys.argv[3]) else 1)" \
          "$used" "$(($(date -u +%s) - t0))" "${CAP[$g]}"; then
          touch "$ST/$r.capstop"
          log "$r reached the arm cap ${CAP[$g]} GPU-h; stopping its containers"
          docker ps --format '{{.Names}}' | grep -E "^dec-(m7-$g-s$i-|m7-arms-full-m7-$g-s$i-)" | xargs -r docker stop > /dev/null
        fi
      done
    ) &
    wd=$!
    env DEC_NODE=$NODE "${DATA_ENV[@]}" bash "$D" "$r" "$GPU" "$SRC" m7/arms "${START[$g]}" -- \
      --train "/runs/${train#/data/dev2/runs/dec/}" --teacher "/runs/${teach#/data/dev2/runs/dec/}" \
      --teacher-kl-weight "${KL[$g]}" "${COMMON[@]}" --backbone-lr 5e-6 --head-lr 5e-5 --seed "$seed"
    kill "$wd" 2> /dev/null
    wait "$wd" 2> /dev/null
    gpuh write "$NODE"
    out=$(outcome "$r")
    note="seed $seed; GPU-h $(gpuh seed "$g" "$i"); $(date -u +%FT%TZ)"
    if [ -f "$ST/$r.capstop" ]; then
      if [ "$out" = complete ] || [ "$out" = nopost ]; then
        echo "full run complete; postrun $([ "$out" = complete ] && echo complete || echo 'stopped at the cap'); $note" > "$ST/$r.DONE"
      else
        echo "stopped at the arm cap ${CAP[$g]} GPU-h (stage result: $out); $note" > "$ST/$r.STOPPED"
      fi
      log "$r reached the cap ($out)"
      continue
    fi
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
  done
  soup_if_complete "$g"
  gpuh write "$NODE"
}

log "chain $CH started on node ${NODE^^} GPU$GPU from $SRC: $ITEMS"
for item in $ITEMS; do arm "$item"; done
printf 'track=dec\nstatus=idle (decoder M7 chain %s finished)\nend_utc=%s\n' "$CH" "$(date -u +%FT%TZ)" > "/data/dev2/leases/gpu$GPU.lock/owner"
gpuh write "$NODE"
log "chain $CH finished"
