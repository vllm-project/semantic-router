#!/usr/bin/env bash
# Node B GPU7 (research & data): positional-key R2 re-derivation and own-Lux wave 4 (M3a).
#
# Usage: pk1_nodeB.sh R2_PROMPTS WAVE4_PROMPTS OUT_DIR
#
# R2: the six Milestone 1 own-1.0 teachers with the same mirror (0a42f3dd), image, snapshots and
# command line as the Milestone 1 R2 run (no persisted autotune cache, as then), one process each.
# Wave 4: the Milestone 2 own-Lux wave launcher (/data/dev2/logs/data/teach.sh: mirror 5f5cd80a,
# decision20-lux-runtime, node-B Lux autotune cache), unchanged. Appends one timing line per run to
# OUT_DIR/timing.jsonl.
set -uo pipefail
r2_in="$1"; w4_in="$2"; out="$3"
SRC=/data/dev2/src/0a42f3dd9399c32935688cfd74b86ccfb1115b16/src/training/decision2
KL=/data/decision20-20260926/envs/kai-lex/bin/python
M=/data/dev2/private/models
mkdir -p "$out"
run() { # name backend repo revision module python
  local name=$1 backend=$2 repo=$3 rev=$4 module=$5 py=$6 s rc
  local -a ob=()
  [[ "$module" == inference.run ]] && ob=(--over-budget-invalid)
  s=$(date -u +%s)
  docker run --rm --network none --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
    -e ROCR_VISIBLE_DEVICES=7 -e HIP_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0 -e PYTHONDONTWRITEBYTECODE=1 \
    -v /data:/data -w "$SRC" --entrypoint "$py" decision20-lux-runtime:latest \
    -m "$module" --backend "$backend" --model-path "$M/$repo@$rev" --model-revision "$rev" --input "$r2_in" \
    --output "$out/$name.pk1.jsonl" --device cuda:0 "${ob[@]}" > "$out/$name.pk1.log" 2>&1
  rc=$?
  echo "{\"teacher\":\"$name\",\"input\":\"$r2_in\",\"rc\":$rc,\"wall_s\":$(( $(date -u +%s) - s )),\"end_utc\":\"$(date -u +%FT%TZ)\"}" >> "$out/timing.jsonl"
}
run kai kai Decision-1.0-Kai-0.6B 7185f514f54b8f93c55998b1e8f9c5cc67f0d029 inference.kai_lex "$KL"
run lex lex Decision-1.0-Lex-0.6B ee8e74d912fca8328a353c11d174b44da3f91781 inference.kai_lex "$KL"
run eos eos Decision-1.0-Eos-0.8B 3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd inference.run python3
run sol sol Decision-1.0-Sol-2B 0665a41108e8f0b33a9515c98311c45947b99399 inference.run python3
run nox nox Decision-1.0-Nox-4B 0bb833504965c0eabdb9630b7bbd385cb2fe5cd4 inference.run python3
run lux lux Decision-1.0-Lux-9B bd45a30aee8c84032791c245c70f86dee5389cc8 inference.run python3
s=$(date -u +%s)
/data/dev2/logs/data/teach.sh lux "$w4_in" "$out/lux.wave4.jsonl"
echo "{\"teacher\":\"lux-wave4\",\"input\":\"$w4_in\",\"wall_s\":$(( $(date -u +%s) - s )),\"end_utc\":\"$(date -u +%FT%TZ)\"}" >> "$out/timing.jsonl"
echo PK1_NODEB_DONE >> "$out/timing.jsonl"
