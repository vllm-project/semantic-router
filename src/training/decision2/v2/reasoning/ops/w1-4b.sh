#!/usr/bin/env bash
# Reasoning wave 1 (4B), node F side (prereg records/rsn-w1-4b-prereg-2026-10-04.md).
#
# usage: w1-4b.sh <mirror-dir> launch [ARM-SEED ...]   seeds as detached rsn-arm.sh jobs (default: all six)
#        w1-4b.sh <mirror-dir> soup <NAME> <member>...   uniform FP32 soup (v2.dec.soup, CPU 0-47) into
#                                                       /data/dev2/runs/reasoning/soup/<NAME>; members: run:<RUN>
#                                                       (the run's final checkpoint), soup:<NAME>, release
#        w1-4b.sh <mirror-dir> devread <NAME> <gpu>     dev readout of soup NAME (or "release") on <gpu>
# Seeds: R4-TF-s1 GPU2, R4-TF-s2 GPU3, R4-TFM-s1 GPU4, R4-TFM-s2 GPU5, R4-F0-s1 GPU6, R4-F0-s2 GPU7; CPUs 16 each.
set -euo pipefail
src=$1 mode=$2
shift 2
OPS=/data/dev2/src/$src/src/training/decision2/v2/reasoning/ops
R=/data/dev2/runs/reasoning
START=/af/4b/soup/4b-LRHxALL/build/4b-LRHxALL
START_HOST=/data/dev2/runs/af/4b/soup/4b-LRHxALL/build/4b-LRHxALL
DATA=/rsn/data/4b-v1
declare -A GPU=([R4-TF-s1]=2 [R4-TF-s2]=3 [R4-TFM-s1]=4 [R4-TFM-s2]=5 [R4-F0-s1]=6 [R4-F0-s2]=7)
declare -A CPU=([R4-TF-s1]=0-15 [R4-TF-s2]=16-31 [R4-TFM-s1]=32-47 [R4-TFM-s2]=48-63 [R4-F0-s1]=64-79 [R4-F0-s2]=80-95)
declare -A SEED=([s1]=20261004 [s2]=20261005)
case $mode in
  launch)
    runs=("$@")
    [[ ${#runs[@]} -gt 0 ]] || runs=(R4-TF-s1 R4-TF-s2 R4-TFM-s1 R4-TFM-s2 R4-F0-s1 R4-F0-s2)
    for run in "${runs[@]}"; do
      arm=${run%-s*} seed=${SEED[${run##*-}]}
      lower=$(tr '[:upper:]' '[:lower:]' <<< "${arm#R4-}")
      mkdir -p "$R/logs"
      setsid nohup bash "$OPS/rsn-arm.sh" "$run" "${GPU[$run]}" "${CPU[$run]}" "$src" 4b "$START" -- \
        --init decision2 --train-mode full --backbone-lr 5e-6 --head-lr 5e-5 --warmup-ratio 0.05 --epochs 1 \
        --weight-decay 0.01 --brier-weight 0.5 --batching tokens --max-batch-tokens 32768 --max-batch-rows 64 \
        --update-rows 64 --max-length 8192 --checkpoint-schedule even8 --selection matrix-v1 \
        --teacher "$DATA/teacher-self.jsonl" --teacher-kl-weight 1.0 --teacher-partial \
        --train "$DATA/train-$lower.jsonl" --example-weights "$DATA/weights-$lower.jsonl" --seed "$seed" \
        > "$R/logs/$run.log" 2>&1 < /dev/null &
      echo "launched $run on GPU${GPU[$run]} (CPUs ${CPU[$run]}, seed $seed)"
    done ;;
  soup)
    name=$1
    shift
    out=$R/soup/$name
    [[ -e $out ]] && { echo "$out exists" >&2; exit 2; }
    members=()
    for spec in "$@"; do
      case $spec in
        run:*) r=${spec#run:}
          last=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"] == "complete"
print("checkpoint-%07d" % d["planned_updates"])' "$R/arms/full/$r/COMPLETE.json")
          members+=(--member "/rsn/arms/full/$r/$last") ;;
        soup:*) members+=(--member "/rsn/soup/${spec#soup:}/build") ;;
        release) members+=(--member "$START") ;;
        *) echo "bad member $spec" >&2; exit 2 ;;
      esac
    done
    mkdir -p "$R/soup"
    bash "$OPS/rsn-launch.sh" "soup-$name" "$src" "$out" --cpu --cpus 0-47 --image 4b -- -m v2.dec.soup "${members[@]}" \
      --output /out/build
    python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])["model_sha256"])' \
      "$out.stdout.log" > "$out/MODEL_SHA256"
    echo "soup $name: $(cat "$out/MODEL_SHA256")" ;;
  interp)
    name=$1 tuned=$2 alpha=$3
    out=$R/soup/$name
    [[ -e $out ]] && { echo "$out exists" >&2; exit 2; }
    bash "$OPS/rsn-launch.sh" "interp-$name" "$src" "$out" --cpu --cpus "${RSN_SOUP_CPUS:-0-47}" --image 4b -- \
      -m v2.reasoning.interpolate --tuned "/rsn/soup/$tuned/build" --start "$START" --alpha "$alpha" --output /out/build
    python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])["model_sha256"])' \
      "$out.stdout.log" > "$out/MODEL_SHA256"
    echo "interp $name: $(cat "$out/MODEL_SHA256")" ;;
  devread)
    name=$1 gpu=$2
    ck=/rsn/soup/$name/build
    [[ $name == release ]] && ck=$START
    out=$R/devread/w1-$name
    bash "$OPS/rsn-launch.sh" "devread-$name" "$src" "$out" --gpu "$gpu" --cpus "$((gpu * 16 - 32))-$((gpu * 16 - 17))" \
      --image 4b -- -m v2.reasoning.devread --checkpoint "$ck" --rows "$DATA/rpdev-final.jsonl" "$DATA/rpdev-nodes.jsonl" \
      /af/4b/inputs/dec/m10/inputs/sel700-cal698/select.jsonl --out /out/devread.json
    python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[2], d["file_macro_accuracy"])' \
      "$out/devread.json" "$name" ;;
  *) echo "unknown mode $mode" >&2; exit 2 ;;
esac
