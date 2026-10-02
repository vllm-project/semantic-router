#!/usr/bin/env bash
# Index sweep (COORDINATION 2026-10-02 10:00, Index-first rule of 09:55) node-side chain over one node's queue, with
# the IX1 harness (v2/eval/ix1: image host2, kit 87d4650b, the 86-request parity gate, dual scoring):
#   1. the parity gate of every queued model (in parallel, one per distinct listed GPU); a failed gate drops the model;
#   2. per model, in queue order: the full run over the panel's shards with the model's own frozen parity cache
#      (shard k on the k-th GPU slot; a GPU listed twice runs its later shard in the next wave), every shard exit 0,
#      then score.sh (merge, port + kit, compare; the scorer gate must pass); a wave an earlier chain already
#      launched on the same panel is only waited for (a restarted chain picks up a running model);
#   3. per scored model, in the background (CPU): family_delta vs the reference run and two paired bootstraps
#      (2,000 replicates, seed 20261002, cases resampled within benchmarks through the board weights):
#        runs/NAME/paired-boot-full-vs-ref.json      the full headline (the Index-first release-gate evidence)
#        runs/NAME/paired-boot-transfer-vs-ref.json  --exclude HoVer When2Call iSarcasmEval GSM8K BPoMP (private)
#   4. the listed GPUs' IX1 lease files written for queued models -> status released.
# Index values stay in the private runs / logs; this script logs none.
#
# usage: chain.sh MIRROR PANEL "GPUS" NAME=SIZE=REFDIR ...
#   MIRROR  /data/dev2/src/<sha>-src_training_decision2 (an exact mirror whose launch.sh lists every NAME)
#   PANEL   panel directory under ix1/ (panel-3 | panel-7 | panel-8); its shard count equals the number of GPU slots
#   GPUS    GPU slots, e.g. "1 2 3 4 5 6 7" or "4 5 6 7 4 5 6 7"
#   REFDIR  a run directory with merged/results.jsonl and merged/compare.json (the tier's current-release run)
set -uo pipefail
M=$1 PANEL=$2 GPUS=$3
shift 3
R=/data/dev2/private/eval/index021/ix1
S=$M/src/training/decision2
L=$S/v2/eval/ix1/launch.sh
P=$R/$PANEL
umask 077
mkdir -p "$R/logs"
LOG=$R/logs/isweep-chain-$PANEL.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
read -r -a slots <<< "$GPUS"
n=${#slots[@]}
[ -f "$S/v2/eval/ix1/launch.sh" ] && [ -f "$P/shard-0-of-$n.jsonl.gz" ] || { log "no mirror or no $n-way shards"; exit 1; }
declare -A seen=()
wave_of=()
distinct=()
waves=0
for k in "${!slots[@]}"; do
  g=${slots[$k]}
  [ -n "${seen[$g]:-}" ] || distinct+=("$g")
  w=${seen[$g]:-0}
  wave_of[k]=$w
  seen[$g]=$((w + 1))
  (( w + 1 > waves )) && waves=$((w + 1))
done
names=()
declare -A size=() ref=()
for spec in "$@"; do
  IFS='=' read -r name sz refdir <<< "$spec"
  [ -f "$refdir/merged/results.jsonl" ] && [ -f "$refdir/merged/compare.json" ] || { log "$name: no reference run $refdir"; exit 1; }
  names+=("$name") size[$name]=$sz ref[$name]=$refdir
done
log "start: mirror $(basename "$M"), $PANEL, GPUs '$GPUS' ($waves wave(s)), queue ${names[*]}"

passed() { python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["pass"] else 1)' "$1" 2>/dev/null; }
idle() {  # GPU...: every listed GPU at <= 5% use and <= 2 GiB VRAM (the launcher's own busy rule), lease free or IX1's
  local g f
  for g in "$@"; do
    f=/data/dev2/leases/gpu$g.lock/owner
    [ ! -s "$f" ] || grep -qx 'track=eval-ix1' "$f" || return 1
  done
  rocm-smi --showuse --showmeminfo vram --json | python3 -c '
import json, sys
d = json.load(sys.stdin)
sys.exit(0 if all(float(d["card" + g]["GPU use (%)"]) <= 5 and int(d["card" + g]["VRAM Total Used Memory (B)"]) <= 2 * 2**30
                  for g in sys.argv[1:]) else 1)' "$@"
}
wait_idle() {  # GPU...: wait up to 3 h for them (e.g. an earlier chain's shards still running)
  local t=0
  until idle "$@"; do
    (( t == 0 )) && log "waiting for GPU(s) $* to be idle"
    (( t < 10800 )) || return 1
    sleep 30
    t=$((t + 30))
  done
}
queue=()
i=0
while (( i < ${#names[@]} )); do
  batch=()
  for g in "${distinct[@]}"; do
    (( i < ${#names[@]} )) || break
    m=${names[$i]}
    i=$((i + 1))
    if passed "$R/parity/$m/parity.json"; then log "parity $m already passed"; continue; fi
    [ -e "$R/parity/$m" ] && { log "parity $m ran before without a pass; dropped"; continue; }
    wait_idle "$g" || { log "GPU $g stayed busy; parity $m not run"; continue; }
    (cd "$S" && bash "$L" parity --src "$M" --model "$m" --gpu "$g" --run "$R/parity/$m" \
      --rows "$P/compat-86.gold-free.jsonl.gz" > "$R/logs/isweep-parity-$m.log" 2>&1
     echo $? > "$R/logs/isweep-parity-$m.exit") &
    batch+=("$m")
    sleep 20
  done
  wait
  for m in "${batch[@]}"; do log "parity $m exit $(cat "$R/logs/isweep-parity-$m.exit")"; done
done
for m in "${names[@]}"; do
  if passed "$R/parity/$m/parity.json"; then queue+=("$m"); else log "parity gate failed for $m; not run"; fi
done

boots=()
for m in "${queue[@]}"; do
  run=$R/runs/$m
  if [ -f "$run/merged/compare.json" ]; then
    log "$m already scored"
  else
    ok=1
    for ((w = 0; w < waves; w++)); do
      only=""
      for k in "${!slots[@]}"; do [ "${wave_of[k]}" = "$w" ] && only+="$k "; done
      only=${only% }
      started=1
      for k in $only; do [ -f "$run/shard-$k/launched" ] || [ -f "$run/shard-$k/end_epoch" ] || started=0; done
      if (( started )); then
        log "run $m wave $w shards '$only' already started (an earlier chain); waiting for them"
      else
        gs=()
        for k in $only; do gs+=("${slots[$k]}"); done
        wait_idle "${gs[@]}" || { log "GPU(s) ${gs[*]} stayed busy; $m not run"; ok=0; break; }
        log "run $m wave $w shards '$only'"
        (cd "$S" && bash "$L" run --src "$M" --model "$m" --gpus "$GPUS" --only "$only" --run "$run" --rows-dir "$P" \
          --cache "$R/parity/$m/cache-frozen" >> "$R/logs/isweep-run-$m.log" 2>&1) || { log "launch $m wave $w failed"; ok=0; break; }
      fi
      for k in $only; do
        while [ ! -f "$run/shard-$k/end_epoch" ]; do sleep 30; done
        e=$(cat "$run/shard-$k/exit_code")
        [ "$e" = 0 ] || { log "$m shard $k exit $e"; ok=0; }
      done
      (( ok )) || break
    done
    (( ok )) || { log "$m stopped (see shard logs); next model"; continue; }
    log "shards done $m"
    bash "$S/v2/eval/ix1/score.sh" --src "$M" --model "$m" --size "${size[$m]}" --panel "$P" > "$R/logs/isweep-score-$m.log" 2>&1
    log "score $m exit $?"
  fi
  python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["scorers"]["pass"] else 1)' "$run/merged/compare.json" \
    || { log "$m: scorer gate failed or missing; no bootstrap"; continue; }
  b=${ref[$m]}
  (
    PYTHONPATH=$S python3 -m v2.eval.ix1.family_delta --base "ref=$b/merged/compare.json" \
      --new "$m=$run/merged/compare.json" --out "$run/family-delta-vs-ref.json" > /dev/null 2>> "$R/logs/isweep-boot-$m.log"
    echo "{\"reference_run\": \"$b\"}" > "$run/isweep-ref.json"
    cd "$R/.." || exit 1
    for v in full transfer; do
      x=()
      [ "$v" = transfer ] && x=(--exclude HoVer When2Call iSarcasmEval GSM8K BPoMP)
      [ -f "$run/paired-boot-$v-vs-ref.json" ] && continue
      CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' PYTHONPATH=$S:$PWD/kit-19ad28ec nice -n 10 \
        venv/bin/python -m v2.eval.ix1.paired_boot --suite-dir suite-0.2 --base "$b/merged/results.jsonl" \
        --new "$run/merged/results.jsonl" --external "$R/../external/index021-frontier-gap-2026-10-01.json" \
        --replicates 2000 --seed 20261002 --workers 20 "${x[@]}" --out "$run/paired-boot-$v-vs-ref.json" \
        >> "$R/logs/isweep-boot-$m.log" 2>&1 &
    done
    wait
  ) &
  boots+=($!)
  log "bootstraps of $m started"
done

for g in "${distinct[@]}"; do
  f=/data/dev2/leases/gpu$g.lock/owner
  grep -qx 'track=eval-ix1' "$f" 2>/dev/null || continue
  for m in "${names[@]}"; do
    if grep -q "^purpose=IX1 .* $m\$" "$f"; then
      printf 'track=eval-ix1\nstatus=released (Index sweep queue on %s done)\nlast_job_end_utc=%s\n' "$PANEL" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$f"
      log "gpu$g lease released"
      break
    fi
  done
done
for p in "${boots[@]}"; do wait "$p"; done
for m in "${queue[@]}"; do
  for v in full transfer; do
    if [ -f "$R/runs/$m/paired-boot-$v-vs-ref.json" ]; then log "bootstrap $v $m done"; else log "bootstrap $v $m MISSING"; fi
  done
done
log "all done"
