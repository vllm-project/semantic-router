#!/usr/bin/env bash
# Decoder M4 formal helpers for node A (sourced; the M3 helpers with M4 paths). Every GPU job waits until
# node A GPU5's lease is back with the decoder track (release engineering may borrow it); the eval runner
# itself refuses another track's lease and a busy GPU.
F=/data/dev2/runs/dec/formal/m4
H=/data/dev2/hf-cache
mkdir -p "$F"
flog() { echo "$(date -u +%FT%TZ) $*" >> "$F/OPERATIONS.log"; }
wait_gpu5() {
  local o=/data/dev2/leases/gpu5.lock/owner
  while [ -f "$o" ] && ! grep -qx "track=dec" "$o"; do sleep 60; done
}
# collect <mirror> <name> <model-dir> <revision> <adapter-spec> <purpose> [--extra K=V ...] [--mount P ...]
collect() {
  local src=$1 name=$2 model=$3 rev=$4 spec=$5 purpose=$6
  shift 6
  local S=/data/dev2/src/$src/src/training/decision2 tc=$F/$name-triton extras=() mounts=()
  while [ $# -gt 0 ]; do
    case $1 in
      --extra) extras+=(--extra "$2"); shift 2 ;;
      --mount) mounts+=(--mount "$2"); shift 2 ;;
      *) shift ;;
    esac
  done
  mkdir -p "$tc"
  wait_gpu5
  local common=(--gpu 5 --track dec --src "$src" --model-dir "$model" --mount "$H" "${mounts[@]}"
    --env TRITON_CACHE_AUTOTUNING=1 --env "TRITON_CACHE_DIR=$tc" --mount-rw "$tc" --purpose "$purpose")
  local args=(--adapter-spec "$spec" --model-path "$model" --revision "$rev" "${extras[@]}")
  bash "$S/v2/eval/run_same_panel.sh" "${common[@]}" --run-dir "$F/$name-smoke" -- "${args[@]}" --max-items 20 \
    > "$F/$name-smoke.collect.log" 2>&1 || { flog "$name smoke FAILED"; return 1; }
  sleep 20
  if bash "$S/v2/eval/run_same_panel.sh" "${common[@]}" --run-dir "$F/$name" -- "${args[@]}" > "$F/$name.collect.log" 2>&1; then
    flog "$name collected"
  else
    flog "$name collection FAILED"
    return 1
  fi
  sleep 20
}
# mlx <mirror> <name> <model-dir> <revision> <adapter-spec> [--extra K=V ...] [--mount P ...]
mlx() {
  local src=$1 name=$2 model=$3 rev=$4 spec=$5
  shift 5
  local S=/data/dev2/src/$src/src/training/decision2 tc=$F/$name-triton extras=() mounts=()
  while [ $# -gt 0 ]; do
    case $1 in
      --extra) extras+=(--extra "$2"); shift 2 ;;
      --mount) mounts+=(--mount "$2"); shift 2 ;;
      *) shift ;;
    esac
  done
  mkdir -p "$tc"
  wait_gpu5
  bash "$S/v2/eval/run_same_panel.sh" --gpu 5 --track dec --src "$src" --model-dir "$model" --mount "$H" "${mounts[@]}" \
    --env TRITON_CACHE_AUTOTUNING=1 --env "TRITON_CACHE_DIR=$tc" --mount-rw "$tc" --purpose "M4 mlx-diag $name" \
    --run-dir "$F/$name-mlx" -- --adapter-spec "$spec" --model-path "$model" --revision "$rev" "${extras[@]}" --panels mlx-diag \
    > "$F/$name-mlx.collect.log" 2>&1 || { flog "$name mlx FAILED"; return 1; }
  if (cd "$S" && PYTHONPATH=$S python3 -m v2.eval.multilingual_panel score --panel /data/dev2/private/panels/mlx-diag-v1 \
    --predictions "$F/$name-mlx/output/mlx-diag.predictions.jsonl" --output "$F/$name-mlx/mlx-diag.score.json" > "$F/$name-mlx.score.log" 2>&1); then
    flog "$name mlx scored"
  else
    flog "$name mlx scoring FAILED"
  fi
}
# score <mirror> <name> <label> <tier> <family> <count-dir> <comparator-run> <comparator-name> [<comparator-run> <name> ...]
score() {
  local src=$1 name=$2 label=$3 tier=$4 fam=$5 count=$6
  shift 6
  local S=/data/dev2/src/$src/src/training/decision2 run=$F/$name
  if (cd "$S" && export PYTHONPATH=$S \
    && { [ -f "$run/SEAL.json" ] || python3 -m v2.eval.same_panel seal --run-dir "$run" > "$run.seal.log" 2>&1; } \
    && { [ -f "$run/REPORT.json" ] || python3 -m v2.eval.same_panel report --run-dir "$run" --label "$label" --tier "$tier" \
      --family "$fam" --count-safetensors "$count" > "$run.report.log" 2>&1; }); then
    while [ $# -ge 2 ]; do
      (cd "$S" && PYTHONPATH=$S python3 -m v2.eval.same_panel compare --run-dir "$run" --comparator-run-dir "$1" \
        --left-name "$label" --right-name "$2" > "$run.compare-$2.log" 2>&1) || flog "$name compare $2 FAILED"
      shift 2
    done
    flog "$name scored"
  else
    flog "$name scoring FAILED"
  fi
}
