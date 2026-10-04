#!/usr/bin/env bash
# Reasoning waves for the 9B and 2B tiers on node D (the 4B wave-1 recipe, records/rsn-w1-9b2b-prereg-2026-10-04.md).
#
# usage: wave.sh <mirror-dir> <9b|2b> launch [RUN ...]          seeds as detached rsn-arm.sh jobs (default: all)
#        wave.sh <mirror-dir> <9b|2b> soup <NAME> <member>...   uniform FP32 soup into soup/<NAME>; members
#                                                               run:<RUN> (final checkpoint), soup:<NAME>, release
#        wave.sh <mirror-dir> <9b|2b> devread <NAME> <gpu> <cpus>
# Runs: R9-TF-s1 GPU0, R9-TF-s2 GPU1, R9-F0-s1 GPU2, R9-F0-s2 GPU3; R2-TF-s1 GPU4, R2-TF-s2 GPU5, R2-F0-s1 GPU6,
# R2-F0-s2 GPU7. CPUs: GPU0-3 jobs take 128-159 in blocks of 8, GPU4-7 jobs 0-31 in blocks of 8.
set -euo pipefail
src=$1 size=$2 mode=$3
shift 3
OPS=/data/dev2/src/$src/src/training/decision2/v2/reasoning/ops
R=/data/dev2/runs/reasoning
export RSN_SEL=/rsn/inputs/sel700-cal698
case $size in
  9b) START=/rsn/starts/9b-KIB4-a40 IMAGE=9b DATA=/rsn/data/9b-v1 P=R9
      declare -A GPU=([R9-TF-s1]=0 [R9-TF-s2]=1 [R9-F0-s1]=2 [R9-F0-s2]=3)
      declare -A CPU=([R9-TF-s1]=128-135 [R9-TF-s2]=136-143 [R9-F0-s1]=144-151 [R9-F0-s2]=152-159) ;;
  2b) START=/rsn/starts/2b-RASDML IMAGE=4b DATA=/rsn/data/2b-v1 P=R2
      declare -A GPU=([R2-TF-s1]=4 [R2-TF-s2]=5 [R2-F0-s1]=6 [R2-F0-s2]=7)
      declare -A CPU=([R2-TF-s1]=0-7 [R2-TF-s2]=8-15 [R2-F0-s1]=16-23 [R2-F0-s2]=24-31) ;;
  *) echo "unknown size $size" >&2; exit 2 ;;
esac
declare -A SEED=([s1]=20261004 [s2]=20261005)
case $mode in
  launch)
    runs=("$@")
    [[ ${#runs[@]} -gt 0 ]] || runs=("$P-TF-s1" "$P-TF-s2" "$P-F0-s1" "$P-F0-s2")
    mkdir -p "$R/logs"
    for run in "${runs[@]}"; do
      arm=${run%-s*} seed=${SEED[${run##*-}]}
      lower=$(tr '[:upper:]' '[:lower:]' <<< "${arm#"$P"-}")
      setsid nohup bash "$OPS/rsn-arm.sh" "$run" "${GPU[$run]}" "${CPU[$run]}" "$src" "$IMAGE" "$START" -- \
        --init decision2 --train-mode full --backbone-lr 5e-6 --head-lr 5e-5 --warmup-ratio 0.05 --epochs 1 \
        --weight-decay 0.01 --brier-weight 0.5 --batching tokens --max-batch-tokens 32768 --max-batch-rows 64 \
        --update-rows 64 --max-length 8192 --checkpoint-schedule even8 --selection matrix-v1 \
        --teacher "$DATA/teacher-self.jsonl" --teacher-kl-weight 1.0 --teacher-partial \
        --train "$DATA/train-$lower.jsonl" --example-weights "$DATA/weights-$lower.jsonl" --seed "$seed" \
        > "$R/logs/$run.log" 2>&1 < /dev/null &
      echo "launched $run on GPU${GPU[$run]} (CPUs ${CPU[$run]}, seed $seed)"
      sleep 20
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
    bash "$OPS/rsn-launch.sh" "soup-$name" "$src" "$out" --cpu --cpus "${RSN_SOUP_CPUS:-128-159}" --image "$IMAGE" -- \
      -m v2.dec.soup "${members[@]}" --output /out/build
    python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])["model_sha256"])' \
      "$out.stdout.log" > "$out/MODEL_SHA256"
    echo "soup $name: $(cat "$out/MODEL_SHA256")" ;;
  interp)
    name=$1 tuned=$2 alpha=$3
    out=$R/soup/$name
    [[ -e $out ]] && { echo "$out exists" >&2; exit 2; }
    bash "$OPS/rsn-launch.sh" "interp-$name" "$src" "$out" --cpu --cpus "${RSN_SOUP_CPUS:-128-159}" --image "$IMAGE" -- \
      -m v2.reasoning.interpolate --tuned "/rsn/soup/$tuned/build" --start "$START" --alpha "$alpha" --output /out/build
    python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])["model_sha256"])' \
      "$out.stdout.log" > "$out/MODEL_SHA256"
    echo "interp $name: $(cat "$out/MODEL_SHA256")" ;;
  devread)
    name=$1 gpu=$2 cpus=$3
    ck=/rsn/soup/$name/build
    [[ $name == release ]] && ck=$START
    out=$R/devread/$P-$name
    bash "$OPS/rsn-launch.sh" "devread-$P-$name" "$src" "$out" --gpu "$gpu" --cpus "$cpus" --image "$IMAGE" -- \
      -m v2.reasoning.devread --checkpoint "$ck" --rows "$DATA/rpdev-final.jsonl" "$DATA/rpdev-nodes.jsonl" \
      "$RSN_SEL/select.jsonl" --out /out/devread.json
    python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[2], d["file_macro_accuracy"])' \
      "$out/devread.json" "$name" ;;
  *) echo "unknown mode $mode" >&2; exit 2 ;;
esac
