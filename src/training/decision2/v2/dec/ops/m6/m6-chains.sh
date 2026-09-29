#!/usr/bin/env bash
# Decoder M6 training chains (prereg dec-m6-prereg-2026-09-29.md), one per GPU; each holds its GPU's flock for its
# whole run and writes the GPU's lease `owner` file (never owner.data or other tracks' entries).
#   b3 (node B GPU3): N6D s1 -> s3 -> soup
#   b4 (node B GPU4): own-Sol labels -> S6X s1 -> s3 -> soup -> N6A s1 -> s3 -> soup -> S6D s1 -> s3 -> soup
#   a5 (node A GPU5): own-Eos labels -> E6K s1 -> s3 -> soup
# Before an arm starts, its TRAIN and teacher files are re-hashed against READY files (<dir>/READY holding the
# SHA-256 from the committed data lock; for c7d51219 the hash is fixed here). Each seed is one drive_arm.sh
# run (zero-step, one-step, preflight_dec gate, full run, postrun) under run root m6/arms. Stop rules: a failed
# preflight stops the arm (no rerun, no replacement seed); an arm stops at its GPU-h cap (a seed is not started
# when used + expected cost > cap, and a watchdog stops a running seed's containers when used + elapsed > cap);
# an arm with fewer than two finished seeds has no soup. Failed arms are never rerun.
# Markers (poll these), all under /data/dev2/runs/dec/m6/status on the arm's node:
#   <ARM>-s<i>.DONE | .FAILED | .STOPPED   per seed (DONE = full run complete; the file notes the postrun status)
#   <ARM>.DONE | .FAILED                   per arm (DONE after its soup: soup/<ARM>/DONE)
#   label-sol.LABELED / label-eos.LABELED  labels made and lock-checked (teacher READY follows the lock commit)
# usage: m6-chains.sh launch <mirror-dir> b3|b4|a5
#        m6-chains.sh run <mirror-dir> b3|b4|a5   (the chain body, started by launch under flock)
set -u
MODE=$1 SRC=$2 CH=$3
M=/data/dev2/runs/dec/m6
C=$M/chains ST=$M/status
S=/data/dev2/src/$SRC/src/training/decision2
OPS=$S/v2/dec/ops/m6
D=$S/v2/dec/drive_arm.sh
mkdir -p "$C" "$ST" "$M/logs" "$M/soup" "$M/arms"
case $CH in
  b3) NODE=b GPU=3 ITEMS="N6D" ;;
  b4) NODE=b GPU=4 ITEMS="label:sol S6X N6A S6D" ;;
  a5) NODE=a GPU=5 ITEMS="label:eos E6K" ;;
  *) echo "unknown chain $CH" >&2; exit 2 ;;
esac

if [ "$MODE" = launch ]; then
  if [ "$NODE" = b ]; then
    if docker ps --format '{{.Names}}' | grep -q '^dev2-data-pn1-' \
      || grep -qs '^status=running' /data/dev2/leases/gpu3.lock/owner.data /data/dev2/leases/gpu4.lock/owner.data; then
      echo "research & data's PN1 lend is still running on node B GPU3-4; not launched" >&2
      exit 1
    fi
  fi
  mkdir "$C/launch-$CH.lock" 2>/dev/null || { echo "M6 chain $CH already launched"; exit 0; }
  setsid nohup flock "$C/gpu$GPU.flock" bash "$0" run "$SRC" "$CH" > "$M/logs/chain-$CH.log" 2>&1 < /dev/null &
  echo $! > "$C/chain-$CH.pid"
  echo "$(date -u +%FT%TZ) M6 chain $CH launched from $SRC (pid $(cat "$C/chain-$CH.pid"))" | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }

NOX=/hf/models--llm-semantic-router--Decision-1.0-Nox-4B/snapshots/cde2a68dbaa557ea65dc458104d410a0802ee259
SOLM=/hf/models--llm-semantic-router--Decision-1.0-Sol-2B/snapshots/ce0c018a28de16d6639b1cd203b761bf643b89e6
EOSM=/hf/models--llm-semantic-router--Decision-1.0-Eos-0.8B/snapshots/363c4a5e56afc115b1c78c837633956d0bbb63ab
N4XF_SHA=c7d51219e9f0fdd107b914df5c7e9f81f2b53496bb43a94796857b8ea3fcfa60
COMMON=(--train-mode full --batching tokens --max-batch-tokens 32768 --max-batch-rows 64 --update-rows 64)
declare -A START=([N6D]=$NOX [N6A]=$NOX [S6X]=$SOLM [S6D]=$SOLM [E6K]=$EOSM)
declare -A TRAIN=([N6D]=m6/data/m6-xl-full-59m [N6A]=m4/data/m4-xl-full-29m [S6X]=m4/data/m4-xl-full-29m
  [S6D]=m6/data/m6-xl-full-59m [E6K]=m6/data/m6-e8f-r2clean)
declare -A TEACH=([N6D]=lux-all-59m [N6A]=lux-all-29m [S6X]=sol-29m [S6D]=sol-59m [E6K]=eos-e8f-r2clean)
declare -A KL=([N6D]=1.0 [N6A]=1.0 [S6X]=0.5 [S6D]=0.5 [E6K]=0.5)
declare -A LR=([N6D]="5e-6 5e-5" [N6A]="5e-6 5e-5" [S6X]="5e-6 5e-5" [S6D]="5e-6 5e-5" [E6K]="1e-5 1e-4")
declare -A SEEDS=([N6D]="20260926 20260927 20260928" [N6A]="20260929 20260930 20260931"
  [S6X]="20260926 20260927 20260928" [S6D]="20260926 20260927 20260928" [E6K]="20260926 20260927 20260928")
declare -A CAP=([N6D]=6.5 [N6A]=3.5 [S6X]=2.0 [S6D]=3.5 [E6K]=4.5)
# Expected GPU-h per seed (preflights + full run + postrun), from the incumbents' recipes; after a seed the
# observed mean replaces it.
declare -A EST=([N6D]=2.0 [N6A]=1.0 [S6X]=0.5 [S6D]=1.0 [E6K]=1.25)
declare -A LABELS_FOR=([S6X]=sol [S6D]=sol [E6K]=eos)
if [ "$NODE" = a ]; then
  DATA_ENV=()  # drive_arm's node-A default SELECT / CAL dir: hash-identical to E8F's (lock part 1)
else
  DATA_ENV=(DEC_DATA_DIR=/data/dev2/runs/dec/m3/data-sel700-cal698)  # N4XF's and S2T's SELECT700 / CAL698
fi
log() { echo "$(date -u +%FT%TZ) chain-$CH $*" | tee -a "$M/OPERATIONS.log"; }
gpuh() { python3 "$OPS/m6-gpuh.py" "$@"; }
lease() {  # <status> <purpose> <minutes>
  printf 'track=dec\nstatus=%s\npurpose=decoder M6 %s\nstart_utc=%s\nexpected_end_utc=%s\n' "$1" "$2" \
    "$(date -u +%FT%TZ)" "$(date -u -d "+$3 min" +%FT%TZ)" > "/data/dev2/leases/gpu$GPU.lock/owner"
}
verify() {  # <file> <READY file or literal sha256>: 0 when the file's SHA-256 equals the expected one
  local file=$1 want=$2
  if [ -f "$want" ]; then want=$(awk '{print $1; exit}' "$want"); fi
  [ "$(sha256sum "$file" | cut -d' ' -f1)" = "$want" ]
}
wait_ready() {  # <READY file>: waits (the part-2 lock for label teachers)
  local n=0
  until [ -f "$1" ]; do
    [ $((n % 30)) = 0 ] && log "waiting for $1"
    n=$((n + 1))
    sleep 60
  done
}
outcome() {  # <run name>: the drive_arm stage result from the arm log
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

label() {  # <sol|eos>
  local k=$1
  [ -f "$ST/label-$k.LABELED" ] && return 0
  [ -f "$ST/label-$k.FAILED" ] && return 1
  lease busy "own-$k 1.0 teacher labels (then the part-2 lock)" 40
  log "start own-$k labels on GPU$GPU"
  if bash "$OPS/m6-label.sh" "$SRC" "$k"; then
    log "own-$k labels done and lock-checked"
  else
    log "own-$k labels FAILED; arms using them stop"
    date -u +%FT%TZ > "$ST/label-$k.FAILED"
    gpuh write "$NODE"
    return 1
  fi
  gpuh write "$NODE"
}

arm() {  # <ARM>
  local g=$1 i seed r t0 wd used est out fin=() note bl hl
  [ -f "$ST/$g.DONE" ] || [ -f "$ST/$g.FAILED" ] && { log "$g already has a marker; skipped"; return 0; }
  if [ -n "${LABELS_FOR[$g]:-}" ] && [ -f "$ST/label-${LABELS_FOR[$g]}.FAILED" ]; then
    echo "teacher labels failed" > "$ST/$g.FAILED"
    log "$g stopped: its teacher labels failed"
    return 0
  fi
  local train=/data/dev2/runs/dec/${TRAIN[$g]}/train.jsonl teach=$M/teacher/${TEACH[$g]}/teacher.jsonl want_train
  want_train=$N4XF_SHA
  [ "${TRAIN[$g]}" = m4/data/m4-xl-full-29m ] || { want_train=/data/dev2/runs/dec/${TRAIN[$g]}/READY; wait_ready "$want_train"; }
  lease busy "$g waiting for its teacher READY (part-2 lock)" 60
  wait_ready "$M/teacher/${TEACH[$g]}/READY"
  if ! verify "$train" "$want_train" || ! verify "$teach" "$M/teacher/${TEACH[$g]}/READY"; then
    echo "TRAIN or teacher hash differs from the lock" > "$ST/$g.FAILED"
    log "$g stopped: TRAIN or teacher hash differs from the lock"
    return 0
  fi
  log "$g inputs verified: $(sha256sum "$train" | cut -c1-12) / $(sha256sum "$teach" | cut -c1-12)"
  read -r bl hl <<< "${LR[$g]}"
  i=0
  for seed in ${SEEDS[$g]}; do
    i=$((i + 1))
    r=m6-$g-s$i
    [ -f "$ST/$r.DONE" ] || [ -f "$ST/$r.FAILED" ] || [ -f "$ST/$r.STOPPED" ] && continue
    used=$(gpuh arm "$g")
    est=${EST[$g]}
    if [ ${#fin[@]} -gt 0 ]; then
      local obs=()
      for s in "${fin[@]}"; do obs+=("$(gpuh seed "$g" "$s")"); done
      est=$(python3 -c "import sys;v=[float(x) for x in sys.argv[1:]];print(sum(v)/len(v))" "${obs[@]}")
    fi
    if python3 -c "import sys;sys.exit(0 if float(sys.argv[1])+float(sys.argv[2])>float(sys.argv[3]) else 1)" "$used" "$est" "${CAP[$g]}"; then
      echo "cap: used $used + expected $est > cap ${CAP[$g]} GPU-h; not started" > "$ST/$r.STOPPED"
      log "$r not started: used $used + expected $est GPU-h > cap ${CAP[$g]}; arm stops at its cap"
      break
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
          docker ps --format '{{.Names}}' | grep -E "^dec-(m6-$g-s$i-|m6-arms-full-m6-$g-s$i-)" | xargs -r docker stop > /dev/null
        fi
      done
    ) &
    wd=$!
    env DEC_NODE=$NODE "${DATA_ENV[@]}" bash "$D" "$r" "$GPU" "$SRC" m6/arms "${START[$g]}" -- \
      --train "/runs/${TRAIN[$g]}/train.jsonl" --teacher "/runs/m6/teacher/${TEACH[$g]}/teacher.jsonl" \
      --teacher-kl-weight "${KL[$g]}" "${COMMON[@]}" --backbone-lr "$bl" --head-lr "$hl" --seed "$seed"
    kill "$wd" 2> /dev/null
    wait "$wd" 2> /dev/null
    gpuh write "$NODE"
    out=$(outcome "$r")
    note="seed $seed; GPU-h $(gpuh seed "$g" "$i"); $(date -u +%FT%TZ)"
    if [ -f "$ST/$r.capstop" ]; then
      if [ "$out" = complete ] || [ "$out" = nopost ]; then
        echo "full run complete; postrun $([ "$out" = complete ] && echo complete || echo 'stopped at the cap'); $note" > "$ST/$r.DONE"
        fin+=("$i")
      else
        echo "stopped at the arm cap ${CAP[$g]} GPU-h (stage result: $out); $note" > "$ST/$r.STOPPED"
      fi
      log "$r reached the cap ($out); arm stops"
      break
    fi
    case $out in
      preflight)
        echo "preflight failed; $note" > "$ST/$r.FAILED"
        log "$r preflight failed: arm $g stops (no rerun, no replacement seed)"
        break
        ;;
      complete | nopost)
        echo "full run complete; postrun $([ "$out" = complete ] && echo complete || echo FAILED); $note" > "$ST/$r.DONE"
        fin+=("$i")
        log "$r finished ($out)"
        ;;
      *)
        echo "full run failed ($out); $note" > "$ST/$r.FAILED"
        log "$r failed ($out); the arm continues with its next seed"
        ;;
    esac
  done
  if [ ${#fin[@]} -ge 2 ]; then
    lease busy "$g soup build (CPU)" 20
    if bash "$OPS/m6-soup.sh" "$SRC" "$g" "$NODE"; then
      printf 'seeds=%s\nsoup=%s\ngpu_h=%s\n' "${fin[*]}" "$M/soup/$g/DONE" "$(gpuh arm "$g")" > "$ST/$g.DONE"
      log "$g DONE (seeds ${fin[*]}; soup $M/soup/$g/build/$g-soup)"
    else
      echo "soup failed; seeds ${fin[*]}" > "$ST/$g.FAILED"
      log "$g soup FAILED"
    fi
  else
    echo "fewer than two finished seeds (${fin[*]:-none}); dropped" > "$ST/$g.FAILED"
    log "$g dropped: fewer than two finished seeds"
  fi
  gpuh write "$NODE"
}

log "chain $CH started on node ${NODE^^} GPU$GPU from $SRC: $ITEMS"
for item in $ITEMS; do
  case $item in
    label:*) label "${item#label:}" ;;
    *) arm "$item" ;;
  esac
done
printf 'track=dec\nstatus=idle (decoder M6 chain %s finished)\nend_utc=%s\n' "$CH" "$(date -u +%FT%TZ)" > "/data/dev2/leases/gpu$GPU.lock/owner"
gpuh write "$NODE"
log "chain $CH finished"
