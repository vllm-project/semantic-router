#!/usr/bin/env bash
# Decoder M8-small GPU chains on node B (prereg dec-m8s-prereg-2026-09-30.md), one per GPU, each under its GPU's
# flock with this track's shared lease entry (owner.dec-m8s). Items, in order:
#   L:parity | L:<k>:<K>     A20r parity gate / label shard k of K (m8s-label.sh)
#   W:parity                 wait until the parity gate has a result
#   T:<tier>:<arm>:<i>[:pf]  top-up seed i (m8s-run.sh) after its READY files match; D seeds 2-3 wait for the early
#                            verdict (status/early-<tier>-<arm>.PASS; .STOP marks them STOPPED); budget and arm caps
#   E:<tier>:<arm>           HT-DEV v2 collection of <tier>-<arm>-s1 (16K, T = 1) for the early rule
#   R:<tier>:<arm>           the early rule for a D arm (m8s_rules.py early) once D-s1 and C-s1 are collected
#   S:<tier>:<arm>           the arm soup (v2.dec.soup, CPU) once every seed is terminal (>= 2 finished)
# usage: m8s-chains.sh launch|run <chain: g2 g6 g7>   (amendment 1: node B GPU2, GPU6, GPU7)
set -u
. "$(dirname "$0")/m8s-lib.sh"
MODE=$1 CH=$2
C=$M/chains ST=$M/status
mkdir -p "$C"
case $CH in
  g2) GPU=2 ITEMS="T:2b:C:1:pf E:2b:C T:08b:C:1:pf E:08b:C T:2b:C:2 T:2b:C:3 S:2b:C T:08b:C:2 T:08b:C:3 S:08b:C" ;;
  g6) GPU=6 ITEMS="L:parity L:0:2 T:2b:D1:1:pf E:2b:D1 R:2b:D1 T:08b:D1:1:pf E:08b:D1 R:08b:D1 T:2b:D1:2 T:2b:D1:3 S:2b:D1 T:08b:D1:2 T:08b:D1:3 S:08b:D1" ;;
  g7) GPU=7 ITEMS="W:parity L:1:2 T:2b:D2:1:pf E:2b:D2 R:2b:D2 T:08b:D2:1:pf E:08b:D2 R:08b:D2 T:2b:D2:2 T:2b:D2:3 S:2b:D2 T:08b:D2:2 T:08b:D2:3 S:08b:D2" ;;
  *) echo "unknown chain $CH" >&2; exit 2 ;;
esac
if [ "$MODE" = launch ]; then
  mkdir "$C/launch-$CH.lock" 2>/dev/null || { echo "M8s chain $CH already launched"; exit 0; }
  setsid nohup flock "$C/gpu$GPU.flock" bash "$0" run "$CH" > "$M/logs/chain-$CH.log" 2>&1 < /dev/null &
  echo $! > "$C/chain-$CH.pid"
  log "chain $CH launched from $SRC (pid $(cat "$C/chain-$CH.pid")): $ITEMS"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }
declare -A SOURCE10=(
  [2b]=/hf/models--llm-semantic-router--Decision-1.0-Sol-2B/snapshots/ce0c018a28de16d6639b1cd203b761bf643b89e6
  [08b]=/hf/models--llm-semantic-router--Decision-1.0-Eos-0.8B/snapshots/363c4a5e56afc115b1c78c837633956d0bbb63ab)
declare -A EST=([2b]=0.2 [08b]=0.12)

wait_file() {  # <file> <what>
  local n=0
  until [ -f "$1" ]; do
    [ $((n % 30)) = 0 ] && log "chain $CH waiting for $2"
    n=$((n + 1))
    sleep 60
  done
}
ready() {  # <file>: its READY holds its current sha256
  local r
  r=$(dirname "$1")/READY
  wait_file "$r" "READY of $1"
  [ "$(sha "$1")" = "$(awk '{print $1; exit}' "$r")" ]
}
parity_failed() { [ -f "$M/labels/parity.json" ] && ! grep -qs '"status": "PASS"' "$M/labels/parity.json"; }
terminal() { [ -f "$ST/$1.DONE" ] || [ -f "$ST/$1.FAILED" ] || [ -f "$ST/$1.STOPPED" ]; }
stopped() { echo "$2; $(date -u +%FT%TZ)" > "$ST/$1.STOPPED"; log "$1 STOPPED: $2"; }
arm_failed_preflight() { grep -qs "preflight failed" "$ST/$1-s1.FAILED"; }

item_T() {  # tier arm i [pf]
  local t=$1 arm=$2 i=$3 pf=${4:-} run=$1-$2-s$3 used est f
  terminal "$run" && return 0
  if [ "$i" != 1 ] && arm_failed_preflight "$t-$arm"; then stopped "$run" "a preflight of arm $t-$arm failed"; return 0; fi
  if [ "$i" != 1 ] && [ -f "$ST/$t-$arm-s1.FAILED" ] && [ "$arm" != C ]; then stopped "$run" "seed 1 failed; no early verdict"; return 0; fi
  if [ "$arm" != C ]; then
    parity_failed && { stopped "$run" "A20r parity gate FAILED"; return 0; }
    if [ "$i" != 1 ]; then
      until [ -f "$ST/early-$t-$arm.PASS" ] || [ -f "$ST/early-$t-$arm.STOP" ]; do
        wait_file "$ST/early-$t-$arm.json" "the early verdict of $t-$arm"
        [ -f "$ST/early-$t-$arm.PASS" ] || [ -f "$ST/early-$t-$arm.STOP" ] || sleep 60
      done
      [ -f "$ST/early-$t-$arm.STOP" ] && { stopped "$run" "early rule STOP (status/early-$t-$arm.json)"; return 0; }
    fi
  fi
  ready "$M/data/$t/topup/train.jsonl" || { stopped "$run" "TRAIN hash differs from READY"; return 0; }
  f=$M/teacher/$t-$arm/teacher.jsonl
  if [ "$arm" != C ] || [ -n "${CKL[$t]}" ]; then
    ready "$f" || { stopped "$run" "teacher hash differs from READY"; return 0; }
  fi
  used=$(gpuh_item "arm:$t-$arm") est=${EST[$t]}
  if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)" "$(gpuh_total)" "$TRAIN_STOP_GPUH"; then
    stopped "$run" "milestone total reached $TRAIN_STOP_GPUH GPU-h; no new training seed"; return 0
  fi
  if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) + float(sys.argv[2]) > float(sys.argv[3]) else 1)" "$used" "$est" "${CAP[$t-$arm]}"; then
    stopped "$run" "arm cap: used $used + expected $est > ${CAP[$t-$arm]} GPU-h"; return 0
  fi
  bash "$OPS/m8s-run.sh" "$GPU" "$t" "$arm" "$i" ${pf:+--preflight}
}

item_E() {  # tier arm
  local t=$1 arm=$2 p=$M/early/$1-$2-s1 ck rc
  [ -f "$p/ht-dev2/ht-dev2.predictions.jsonl" ] && return 0
  [ -f "$ST/$t-$arm-s1.DONE" ] || { log "E $t-$arm: seed 1 not DONE; skipped"; return 0; }
  [ -e "$p/ht-dev2.launch.json" ] && { log "E $t-$arm: earlier collection failed; not rerun"; return 0; }
  ck=$(sed -n 's/^checkpoint=//p' "$ST/$t-$arm-s1.DONE")
  [ "$(sha "$R/panels/ht-dev2.prompts.jsonl")" = "$HT_PROMPTS_SHA" ] || { log "E: HT-DEV v2 prompts differ"; return 0; }
  dec_env "$GPU" && wait_gpu "$GPU" 60
  mkdir -p "$p"
  lease "$GPU" busy "early HT-DEV v2 $t-$arm-s1"
  bash "$LAUNCH" "m8s-early-$t-$arm-ht-dev2" "$SRC" "$p/ht-dev2" -- -m v2.dec.infer_dec --checkpoint "$(incontainer "$ck")" \
    --source-path "${SOURCE10[$t]}" --input /panels/ht-dev2.prompts.jsonl --output /out/ht-dev2.predictions.jsonl \
    --model-id "decision2-dec-m8s-$t-$arm-s1" --model-revision "$t-$arm-s1" --max-length 16384
  rc=$?
  gpu_seconds "$p/ht-dev2.launch.json" "early" "early HT-DEV v2 $t-$arm-s1"
  lease "$GPU" idle "early HT-DEV v2 $t-$arm-s1 (exit $rc)"
  log "E $t-$arm-s1 HT-DEV v2 collected (exit $rc)"
}

item_R() {  # tier arm
  local t=$1 arm=$2
  [ -f "$ST/early-$t-$arm.json" ] && return 0
  [ -f "$ST/$t-$arm-s1.DONE" ] || return 0
  [ -f "$ST/$t-C-s1.FAILED" ] && { echo '{"status": "STOP", "reasons": ["control seed 1 failed"]}' > "$ST/early-$t-$arm.json"; touch "$ST/early-$t-$arm.STOP"; return 0; }
  wait_file "$M/early/$t-C-s1/ht-dev2/ht-dev2.predictions.jsonl" "C-s1 HT-DEV v2 of $t"
  wait_file "$M/early/$t-$arm-s1/ht-dev2/ht-dev2.predictions.jsonl" "$t-$arm-s1 HT-DEV v2"
  [ "$(sha "$HT_GOLD")" = "$HT_GOLD_SHA" ] || { log "R: HT-DEV v2 gold differs"; return 1; }
  (cd "$S" && PYTHONPATH=$S python3 -B "$OPS/m8s_rules.py" early --root "$M" --tier "$t" --arm "$arm" --gold "$HT_GOLD") \
    >> "$M/logs/early.log" 2>&1 || log "R $t-$arm: early rule FAILED to run (see logs/early.log)"
  log "R $t-$arm: $(tail -1 "$M/logs/early.log")"
}

item_S() {  # tier arm
  local t=$1 arm=$2 g=$1-$2 s n=0 members=() out=$M/soup/$1-$2
  [ -f "$out/DONE" ] || [ -f "$out/FAILED" ] && return 0
  for s in 1 2 3; do terminal "$g-s$s" || { log "S $g: seed $s not terminal; soup deferred"; return 0; }; done
  for s in 1 2 3; do
    if [ -f "$ST/$g-s$s.DONE" ]; then
      n=$((n + 1))
      members+=(--member "$(incontainer "$(sed -n 's/^checkpoint=//p' "$ST/$g-s$s.DONE")")")
    fi
  done
  mkdir -p "$out"
  [ $n -lt 2 ] && { echo "fewer than two finished seeds ($n); arm dropped" > "$out/FAILED"; log "S $g dropped ($n seeds)"; return 0; }
  if bash "$LAUNCH" "m8s-soup-$g" "$SRC" "$out/build" --cpu -- -m v2.dec.soup "${members[@]}" --output "/out/$g-soup"; then
    printf 'soup=%s\nseeds=%s\n' "$out/build/$g-soup" "$n" > "$out/DONE"
    log "S $g soup DONE ($n seeds)"
  else
    echo "soup build failed (see $out/build.stderr.log)" > "$out/FAILED"
    log "S $g soup FAILED"
  fi
}

log "chain $CH started on node B GPU$GPU from $SRC: $ITEMS"
for item in $ITEMS; do
  IFS=: read -r kind a b c d <<< "$item"
  case $kind in
    L) if [ "$a" = parity ]; then bash "$OPS/m8s-label.sh" parity "$GPU"; else
         wait_file "$M/labels/parity.json" "the parity gate"
         parity_failed || { wait_file "$M/data/2b/topup/READY" "2b TRAIN READY"; wait_file "$M/data/08b/topup/READY" "08b TRAIN READY"
                            bash "$OPS/m8s-label.sh" label "$GPU" "$a" "$b"; }
       fi ;;
    W) wait_file "$M/labels/parity.json" "the parity gate" ;;
    T) item_T "$a" "$b" "$c" "$d" ;;
    E) item_E "$a" "$b" ;;
    R) item_R "$a" "$b" ;;
    S) item_S "$a" "$b" ;;
  esac
done
lease "$GPU" idle "chain $CH finished"
log "chain $CH finished (milestone GPU-h $(gpuh_total))"
