#!/usr/bin/env bash
# Decoder M16 readout chain on one M16 GPU (prereg dec-m16-prereg-2026-10-01.md, "GPUs, order, budget"), holding the
# GPU's chain flock for the whole chain. Items, in order:
#   mlx:<REF>        the tier reference's MLX-DEV readout (08b-C0-a, 2b-C0-b or 4b-LH-b; its eight panels are M14's
#                    same-node readouts, staged by m16-prep.sh)
#   <ARM>:<pct>      point <ARM>-a<pct> (pct 25 / 50 / 75): the CPU build W = (1 - a)·R + a·X (m16_interp.py build;
#                    the pair's lineage file m16/lineage/<ARM>/<ARM>.json must say pass), then the eight panels, MLX-DEV and
#                    the MLX-DEV compare against the tier reference
# A failed build or readout stops that point (marker m16/status/<point>.FAILED; never rerun); the chain goes on. No
# item starts when this node's M16 GPU-hours exceed 13.5 (the 27 GPU-h start limit split over the two nodes).
#
# usage: M16_NODE=a|b m16-chain.sh launch|run <mirror-dir> <gpu> <item> [...]
set -u
MODE=$1 SRC=$2 GPU=$3
shift 3
NODE=${M16_NODE:?set M16_NODE=a or b}
M=/data/dev2/runs/dec/m16
ST=$M/status
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m16
HF=/data/dev2/hf-cache
mkdir -p "$M/chains" "$M/logs" "$ST" "$M/points"
case $NODE:$GPU in a:3 | a:4 | a:5 | b:2 | b:3 | b:4) ;; *) echo "GPU$GPU is not an M16 GPU on node $NODE" >&2; exit 2 ;; esac
if [ "$MODE" = launch ]; then
  NAME=gpu$GPU-$(date -u +%H%M%S)
  M16_NODE=$NODE setsid nohup flock "$M/chains/gpu$GPU.flock" bash "$0" run "$SRC" "$GPU" "$@" \
    > "$M/logs/chain-$NAME.log" 2>&1 < /dev/null &
  echo "$(date -u +%FT%TZ) M16 chain $NAME launched on node ${NODE^^} GPU$GPU from $SRC: $* (pid $!)" \
    | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }
log() { echo "$(date -u +%FT%TZ) chain-${NODE}$GPU $*" | tee -a "$M/OPERATIONS.log"; }
source_of() {
  case $1 in
    2b) echo "$HF/models--llm-semantic-router--Decision-1.0-Sol-2B/snapshots/ce0c018a28de16d6639b1cd203b761bf643b89e6" ;;
    08b) echo "$HF/models--llm-semantic-router--Decision-1.0-Eos-0.8B/snapshots/363c4a5e56afc115b1c78c837633956d0bbb63ab" ;;
    4b) echo "$HF/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b" ;;
  esac
}
R14=/data/dev2/runs/dec/m14/inputs/refs
ref_of() {  # <tier> -> reference name
  case $1 in 08b) echo 08b-C0-a ;; 2b) echo 2b-C0-b ;; 4b) echo 4b-LH-b ;; esac
}
ref_ck() {  # <REF> -> release checkpoint (host)
  case $1 in
    08b-C0-a) echo "$R14/DEV2.0-0.8B/bede7938a8c209c09f27400b79eed57948d6b75e" ;;
    2b-C0-b) echo "$R14/DEV2.0-2B/a53cf66a0d9d492a84b6617b61e7ce35fcd03af0" ;;
    4b-LH-b) echo "$R14/LH-soup" ;;
  esac
}
arm_ck() {  # <ARM> -> arm soup (host)
  case $1 in
    08b-RAUP | 2b-RAUP | 4b-LHA10UP) echo "/data/dev2/runs/dec/m14/soup/$1/build/$1-soup" ;;
    08b-RASD | 08b-RA | 2b-RASD | 2b-RA | 4b-LHA10SD) echo "$M/inputs/arms/$1" ;;
  esac
}
host2c() { echo "/runs/${1#/data/dev2/runs/dec/}"; }
gpuh() { python3 -B "$OPS/m16_gpuh.py" total --node "$NODE"; }

for item in "$@"; do
  over=$(python3 -c 'import sys; print(int(float(sys.argv[1]) > 13.5))' "$(gpuh)")
  [ "$over" = 0 ] || { log "node GPU-hours above 13.5; $item and later items not started"; break; }
  case $item in
    mlx:*)
      ref=${item#mlx:} t=${ref%%-*}
      [ -n "$(ref_ck "$ref")" ] || { log "unknown reference $ref"; continue; }
      M16_NODE=$NODE bash "$OPS/m16-lines.sh" mlx "$SRC" "$GPU" "$ref" "$(ref_ck "$ref")" "$(source_of "$t")" \
        || log "$ref MLX-DEV failed"
      continue
      ;;
  esac
  arm=${item%%:*} pct=${item#*:} t=${item%%-*}
  point=$arm-a$pct
  case $pct in 25 | 50 | 75) ;; *) log "$item: alpha must be 25, 50 or 75"; continue ;; esac
  ref=$(ref_of "$t") X=$(arm_ck "$arm")
  [ -n "$ref" ] && [ -n "$X" ] || { log "$item: unknown arm"; continue; }
  [ -f "$ST/$point.FAILED" ] && { log "$point failed earlier; not rerun"; continue; }
  if ! grep -qs '"pass": true' "$M/lineage/$arm/$arm.json"; then
    log "$point: the lineage check of $arm did not pass; skipped"
    echo "lineage" > "$ST/$point.FAILED"
    continue
  fi
  B=$M/points/$point
  if [ ! -f "$B/DONE" ]; then
    alpha=$(python3 -c 'import sys; print(int(sys.argv[1]) / 100)' "$pct")
    if M16_NODE=$NODE bash "$OPS/m16-launch.sh" "build-$point" "$SRC" "$B/build" --cpu -- \
      v2/dec/ops/m16/m16_interp.py build --release "$(host2c "$(ref_ck "$ref")")" --arm "$(host2c "$X")" \
      --alpha "$alpha" --output "/out/$point"; then
      echo "$B/build/$point" > "$B/DONE"
      log "$point built: $(tail -c 400 "$B/build.stdout.log" | tr '\n' ' ')"
    else
      echo "build" > "$ST/$point.FAILED"
      log "$point build FAILED (see $B/build.stderr.log); point stopped"
      continue
    fi
  fi
  ck=$(cat "$B/DONE")
  if ! M16_NODE=$NODE bash "$OPS/m16-lines.sh" read "$SRC" "$GPU" "$point" "$ck" "$(source_of "$t")"; then
    echo "readout" > "$ST/$point.FAILED"
    log "$point readout FAILED; point stopped"
    continue
  fi
  if ! M16_NODE=$NODE bash "$OPS/m16-lines.sh" mlx "$SRC" "$GPU" "$point" "$ck" "$(source_of "$t")"; then
    echo "mlx readout" > "$ST/$point.FAILED"
    log "$point MLX-DEV readout FAILED; point stopped"
    continue
  fi
  n=0
  until [ -f "$M/lines/$ref/mlxdev/mlxdev-predictions.jsonl" ] || [ $n -ge 120 ]; do sleep 60; n=$((n + 1)); done
  M16_NODE=$NODE bash "$OPS/m16-lines.sh" mlxcmp "$SRC" "$point" "$ref" || log "$point MLX-DEV compare failed"
  echo "read" > "$ST/$point.READ"
  log "$point finished"
done
log "chain finished: $*"
