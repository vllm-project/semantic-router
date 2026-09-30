#!/usr/bin/env bash
# Decoder M8 member chains (prereg dec-m8-prereg-2026-09-30.md, "Arms", "Early stop"), node B, one per GPU; each holds
# its GPU's flock for its whole run and writes the GPU's lease `owner` file (never other tracks' entries).
#   b3 (GPU3): C:1 D1:1 D1:2 D2:2
#   b4 (GPU4): C:2 C:3 D2:1 D2:3 D1:3
# Member k of an arm continues the k-th released N4XF soup member (v2.dec.train_dec --init decision2) on the top-up
# slice with the arm's teacher file, seed 20260930 + k, and one checkpoint at the final update. Member 1 runs the
# drive_arm.sh preflights (zero-step, one-step, preflight_dec gate); members 2-3 wait for member 1's gate PASS and run
# the full continuation only. Inputs are re-hashed against READY files written after the data-lock commit:
#   m8/data/READY          slice TRAIN + C teacher
#   m8/teacher-a20r/READY  D1 + D2 teachers (A20r)
# Early stop: after D1-m1 / D2-m1 (and C-m1) finish, m8_rules.py early writes m8/early/<ARM>.json; members 2-3 of that
# arm run only if it says continue. Stop rules: a failed preflight stops the arm (no rerun, no replacement member); an
# arm stops at its GPU-h cap; at 20 GPU-h milestone total no new member starts; an arm with fewer than two finished
# members has no soup. The chain that finishes an arm's last member builds its soup (m8/soup/<ARM>/build/<ARM>-soup).
# Markers under m8/status: m8-<ARM>-m<k>.DONE | .FAILED | .STOPPED, m8-<ARM>-m1.GATE_PASS, <ARM>.DONE | .FAILED.
# usage: m8-chains.sh launch|run <mirror-dir> b3|b4
#        m8-chains.sh status <mirror-dir>
set -u
MODE=$1 SRC=$2 CH=${3:-}
M=/data/dev2/runs/dec/m8
C=$M/chains ST=$M/status A=$M/arms
S=/data/dev2/src/$SRC/src/training/decision2
OPS=$S/v2/dec/ops/m8
LAUNCH=$S/v2/dec/launch.sh
mkdir -p "$C" "$ST" "$M/logs" "$M/soup" "$A" "$M/early"
declare -A START=([1]=/runs/m4/arms/full/m4-N4XF-s1/checkpoint-0000690 [2]=/runs/m4/arms/full/m4-N4XF-s2/checkpoint-0000676
  [3]=/runs/m4/arms/full/m4-N4XF-s3/checkpoint-0000492)
TRAIN=$M/data/build/slice/train.jsonl
declare -A TEACH=([C]=$M/data/build/teacher/C/teacher.jsonl [D1]=$M/teacher-a20r/D1/teacher.jsonl
  [D2]=$M/teacher-a20r/D2/teacher.jsonl)
declare -A READYF=([C]=$M/data/READY [D1]=$M/teacher-a20r/READY [D2]=$M/teacher-a20r/READY)
ARM_CAP=1.4 TOTAL_STOP=20
EST1=0.40 EST=0.32

if [ "$MODE" = status ]; then
  ls "$ST" 2>/dev/null | sort
  for f in "$M"/early/*.json; do [ -f "$f" ] && echo "$(basename "$f"): $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["decision"], d["x"]["select_family_macro"], d["c"]["select_family_macro"])' "$f")"; done
  python3 "$OPS/m8_gpuh.py" total b 2>/dev/null
  exit 0
fi
case $CH in
  b3) GPU=3 ITEMS="C:1 D1:1 D1:2 D2:2" ;;
  b4) GPU=4 ITEMS="C:2 C:3 D2:1 D2:3 D1:3" ;;
  *) echo "unknown chain $CH" >&2; exit 2 ;;
esac
if [ "$MODE" = launch ]; then
  mkdir "$C/launch-$CH.lock" 2>/dev/null || { echo "M8 chain $CH already launched"; exit 0; }
  setsid nohup flock "$C/gpu$GPU.flock" bash "$0" run "$SRC" "$CH" > "$M/logs/chain-$CH.log" 2>&1 < /dev/null &
  echo $! > "$C/chain-$CH.pid"
  echo "$(date -u +%FT%TZ) M8 chain $CH launched from $SRC (pid $(cat "$C/chain-$CH.pid"))" | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }

declare -A RENDER=([3]=/dev/dri/renderD153 [4]=/dev/dri/renderD161)
export DEC_IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
export DEC_RENDER=${RENDER[$GPU]} DEC_GPU_LABEL="node B GPU$GPU" DEC_DATA=/data/dev2/runs/dec/m3/data-sel700-cal698
COMMON=(--init decision2 --train-mode full --batching tokens --max-batch-tokens 32768 --max-batch-rows 64 --update-rows 64
  --teacher-partial --teacher-kl-weight 1.0 --brier-weight 0.5 --backbone-lr 5e-6 --head-lr 5e-5 --weight-decay 0.01
  --warmup-ratio 0.05 --epochs 1 --max-length 8192 --checkpoint-schedule every --save-every 1000000)
log() { echo "$(date -u +%FT%TZ) chain-$CH $*" | tee -a "$M/OPERATIONS.log"; }
gpuh() { python3 "$OPS/m8_gpuh.py" "$@"; }
lease() {  # <status> <purpose> <minutes>
  printf 'track=dec\nstatus=%s\npurpose=decoder M8 %s\nstart_utc=%s\nexpected_end_utc=%s\n' "$1" "$2" \
    "$(date -u +%FT%TZ)" "$(date -u -d "+$3 min" +%FT%TZ)" > "/data/dev2/leases/gpu$GPU.lock/owner"
}
verify() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$(awk -v f="$(basename "$(dirname "$1")")/$(basename "$1")" '$2 == f {print $1}' "$2")" ]; }
wait_for() {  # <file> <what>
  local n=0
  until [ -f "$1" ]; do
    [ $((n % 30)) = 0 ] && log "waiting for $2 ($1)"
    n=$((n + 1))
    sleep 60
  done
}
terminal() { [ -f "$ST/m8-$1-m$2.DONE" ] || [ -f "$ST/m8-$1-m$2.FAILED" ] || [ -f "$ST/m8-$1-m$2.STOPPED" ]; }
final_ckpt() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$A/full/$1/LATEST.json"; }
stage() {  # <job> <out dir> <python args...>: one launch.sh job; 0 on success
  local job=$1 out=$2
  shift 2
  bash "$LAUNCH" "$job" "$SRC" "$out" -- "$@"
}

soup_if_complete() {  # <ARM>: after the arm's last member (on either chain), build its soup once
  local g=$1 k n=0 members=() names=()
  for k in 1 2 3; do terminal "$g" "$k" || return 0; done
  mkdir "$M/soup/$g.started" 2>/dev/null || return 0
  for k in 1 2 3; do
    if [ -f "$ST/m8-$g-m$k.DONE" ]; then
      n=$((n + 1))
      members+=(--member "/runs/m8/arms/full/m8-$g-m$k/$(final_ckpt "m8-$g-m$k")")
      names+=("m8-$g-m$k/$(final_ckpt "m8-$g-m$k")")
    fi
  done
  if [ $n -lt 2 ]; then
    echo "fewer than two finished members (${names[*]:-none}); no soup" > "$ST/$g.FAILED"
    log "$g: fewer than two finished members; no soup"
    return 0
  fi
  lease busy "$g soup build (CPU)" 20
  if bash "$LAUNCH" "m8-$g-soup" "$SRC" "$M/soup/$g/build" --cpu -- -m v2.dec.soup "${members[@]}" --output "/out/$g-soup"; then
    (cd "$M/soup/$g/build/$g-soup" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > "$M/soup/$g/$g-soup.sha256"
    printf 'soup=%s\nmembers=%s\nsha256_list_sha256=%s\nbuilt_utc=%s\n' "$M/soup/$g/build/$g-soup" "${names[*]}" \
      "$(sha256sum "$M/soup/$g/$g-soup.sha256" | cut -d' ' -f1)" "$(date -u +%FT%TZ)" > "$ST/$g.DONE"
    log "$g soup built from ${names[*]}"
  else
    echo "soup build failed" > "$ST/$g.FAILED"
    log "$g soup FAILED (see $M/soup/$g/build.stderr.log)"
  fi
}

early_rule() {  # <ARM>: after its member 1 and C-m1
  local g=$1
  [ -f "$M/early/$g.json" ] && return 0
  local n=0
  until terminal C 1; do
    [ $((n % 30)) = 0 ] && log "waiting for C-m1 (early rule for $g)"
    n=$((n + 1))
    sleep 60
  done
  if [ ! -f "$ST/m8-C-m1.DONE" ]; then
    printf '{"arm": "%s", "decision": "stop", "reason": "control member 1 did not finish"}\n' "$g" > "$M/early/$g.json"
    log "early rule $g: stop (control member 1 did not finish)"
    return 0
  fi
  python3 -B "$OPS/m8_rules.py" early --arm "$g" --x-run "$A/full/m8-$g-m1" --c-run "$A/full/m8-C-m1" \
    --output "$M/early/$g.json" >> "$M/logs/early-$g.log" 2>&1 || { log "early rule for $g FAILED to evaluate"; return 1; }
  log "early rule $g: $(tail -1 "$M/logs/early-$g.log")"
}

member() {  # <ARM> <k>
  local g=$1 k=$2 r=m8-$1-m$2 seed=$((20260930 + $2)) used est out rc t0 wd
  terminal "$g" "$k" && { log "$r already has a marker; skipped"; return 0; }
  wait_for "${READYF[$g]}" "$g data lock"
  if ! verify "$TRAIN" "$M/data/READY" || ! verify "${TEACH[$g]}" "${READYF[$g]}"; then
    echo "TRAIN or teacher hash differs from the lock" > "$ST/$r.FAILED"
    log "$r stopped: TRAIN or teacher hash differs from the lock"
    return 0
  fi
  if [ "$k" != 1 ]; then
    until [ -f "$ST/m8-$g-m1.GATE_PASS" ] || terminal "$g" 1; do sleep 60; done
    if [ ! -f "$ST/m8-$g-m1.GATE_PASS" ]; then
      echo "not started: member 1 of $g has no preflight PASS" > "$ST/$r.STOPPED"
      log "$r not started: member 1 of $g has no preflight PASS"
      return 0
    fi
    if [ "$g" != C ]; then
      wait_for "$M/early/$g.json" "early rule of $g"
      if ! python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["decision"] == "continue" else 1)' "$M/early/$g.json"; then
        echo "stopped by the early rule ($M/early/$g.json)" > "$ST/$r.STOPPED"
        log "$r not started: arm $g stopped by its early rule"
        return 0
      fi
    fi
  fi
  used=$(gpuh arm "$g")
  est=$([ "$k" = 1 ] && echo $EST1 || echo $EST)
  if python3 -c "import sys;sys.exit(0 if float(sys.argv[1])+float(sys.argv[2])>float(sys.argv[3]) else 1)" "$used" "$est" "$ARM_CAP"; then
    echo "cap: used $used + expected $est > arm cap $ARM_CAP GPU-h; not started" > "$ST/$r.STOPPED"
    log "$r not started: arm cap"
    return 0
  fi
  if python3 -c "import sys;sys.exit(0 if float(sys.argv[1])>=float(sys.argv[2]) else 1)" "$(gpuh total b)" "$TOTAL_STOP"; then
    echo "milestone total reached $TOTAL_STOP GPU-h; not started" > "$ST/$r.STOPPED"
    log "$r not started: milestone total >= $TOTAL_STOP GPU-h"
    return 0
  fi
  lease busy "$r (continuation; arm cap $ARM_CAP GPU-h, used $used)" 40
  log "start $r on node B GPU$GPU (start ${START[$k]}, seed $seed; arm GPU-h used $used)"
  local args=(--model-path "${START[$k]}" --train "/runs/${TRAIN#/data/dev2/runs/dec/}"
    --teacher "/runs/${TEACH[$g]#/data/dev2/runs/dec/}" --select /data/select.jsonl --cal /data/cal.jsonl --output /out
    --arm "$r" --seed "$seed" "${COMMON[@]}")
  mkdir -p "$A/pre" "$A/gates" "$A/full"
  if [ "$k" = 1 ]; then
    if ! stage "$r-zero" "$A/pre/$r-zero" -m v2.dec.train_dec "${args[@]}" --zero-step-only \
      || ! stage "$r-one" "$A/pre/$r-one" -m v2.dec.train_dec "${args[@]}" --max-steps 1 \
      || ! stage "$r-gate" "$A/gates/$r" -m v2.dec.preflight_dec --source-path "${START[$k]}" --select /data/select.jsonl \
        --zero-run "/runs/m8/arms/pre/$r-zero" --one-run "/runs/m8/arms/pre/$r-one" --output /out/preflight-receipt.json \
      || [ "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$A/gates/$r/preflight-receipt.json" 2>/dev/null)" != PASS ]; then
      gpuh write b
      echo "preflight failed ($(date -u +%FT%TZ)); arm $g stops (no rerun, no replacement member)" > "$ST/$r.FAILED"
      for kk in 2 3; do terminal "$g" "$kk" || echo "not started: preflight of $g member 1 failed" > "$ST/m8-$g-m$kk.STOPPED"; done
      log "$r preflight FAILED: arm $g stops"
      return 0
    fi
    date -u +%FT%TZ > "$ST/$r.GATE_PASS"
    log "$r preflight PASS"
  fi
  t0=$(date -u +%s)
  ( while sleep 60; do
      if python3 -c "import sys;sys.exit(0 if float(sys.argv[1])+float(sys.argv[2])/3600>float(sys.argv[3]) else 1)" \
        "$used" "$(($(date -u +%s) - t0))" "$ARM_CAP"; then
        touch "$ST/$r.capstop"
        log "$r reached the arm cap; stopping its container"
        docker stop "dec-$r-full" > /dev/null 2>&1
      fi
    done ) &
  wd=$!
  stage "$r-full" "$A/full/$r" -m v2.dec.train_dec "${args[@]}"
  rc=$?
  kill "$wd" 2> /dev/null
  wait "$wd" 2> /dev/null
  gpuh write b
  if [ $rc = 0 ] && [ -f "$A/full/$r/COMPLETE.json" ]; then
    printf 'seed=%s\ncheckpoint=%s\ngpu_h_arm=%s\nend_utc=%s\n' "$seed" "$(final_ckpt "$r")" "$(gpuh arm "$g")" \
      "$(date -u +%FT%TZ)" > "$ST/$r.DONE"
    log "$r finished ($(final_ckpt "$r"))"
  elif [ -f "$ST/$r.capstop" ]; then
    echo "stopped at the arm cap ($(date -u +%FT%TZ))" > "$ST/$r.STOPPED"
    log "$r stopped at the arm cap"
  else
    echo "full run failed (exit $rc, $(date -u +%FT%TZ))" > "$ST/$r.FAILED"
    log "$r full run FAILED (exit $rc)"
  fi
  if [ "$k" = 1 ] && [ "$g" != C ] && [ -f "$ST/$r.DONE" ]; then early_rule "$g"; fi
  if [ "$k" = 1 ] && [ "$g" != C ] && [ ! -f "$ST/$r.DONE" ]; then
    printf '{"arm": "%s", "decision": "stop", "reason": "member 1 did not finish"}\n' "$g" > "$M/early/$g.json"
  fi
  soup_if_complete "$g"
}

log "chain $CH started on node B GPU$GPU from $SRC: $ITEMS"
for item in $ITEMS; do member "${item%%:*}" "${item#*:}"; done
printf 'track=dec\nstatus=idle (decoder M8 chain %s finished)\nend_utc=%s\n' "$CH" "$(date -u +%FT%TZ)" > "/data/dev2/leases/gpu$GPU.lock/owner"
gpuh write b
log "chain $CH finished"
