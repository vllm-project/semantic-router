#!/usr/bin/env bash
# M3b amendment 3 §4, node A (CPU): own-Lux prompt waves for the XL control-only rows.
#
# Usage: luxxl_control_nodeA.sh MIRROR_DIR
#
# Run from the exact mirror of a pushed commit. Control-only ids = union of the four cx-xl-*
# lux1 missing lists minus the mx-xl-full / mx-xl-short missing lists, sorted by id
# (v2.data.m3.luxxl control-ids); waves lux-xl-c-w1 (60,000) and lux-xl-c-w2 (the rest) by
# v2.data.m3.xl_prompts from the XL pools; then the target guard of waves w1-w5 on each wave
# (SELECT700, CAL700, CAL698, every v1 / v2 / A7 AHO and SHO slice, every gold-free panel).
# Outputs go to /data/dev2/runs/data/m3b/lux-xl/ and are never overwritten.
set -euo pipefail
src="$1"
code="$src/src/training/decision2"
B=/data/dev2/runs/data/m3b
W=$B/lux-xl
X=$B/upload-xl/m3/mixtures/xl
H=/data/dev2/runs/data/m3a/hf
F=/data/dev2/private/data/arms-v2/final
G=/data/dev2/private/panels/goldfree
umask 077
cd "$code"

protected=(--protected-rows "$H/select.jsonl" --protected-rows "$H/cal.jsonl" --protected-rows "$H/m2/cal/CAL698/cal.jsonl")
for f in "$F"/*.aho.jsonl "$F"/*.sho.jsonl "$H"/v2/arms/*/aho.jsonl "$B"/hf/v2/a7/arms/*/aho.jsonl; do
  protected+=(--protected-rows "$f")
done
for f in "$G"/*.prompts.jsonl; do protected+=(--protected-prompts "$f"); done

python3 -m v2.data.m3.luxxl control-ids --missing-dir "$X" --out "$W/control-only.missing.jsonl"
python3 -m v2.data.m3.xl_prompts --pools "$B/xl-pools.json" --missing "$W/control-only.missing.jsonl" \
  --wave-size 60000 --prefix "$W/lux-xl-c" > "$W/lux-xl-c.waves.json"
cat "$W/lux-xl-c.waves.json"
for k in 1 2; do
  python3 -m v2.data.m3.guard --prompts "$W/lux-xl-c-w$k.prompts.jsonl" --rows "$W/lux-xl-c-w$k.rows.jsonl" \
    "${protected[@]}" --receipt "$W/lux-xl-c-w$k.guard.json" > /dev/null
  python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(sys.argv[2], r["pass"], r["prompts"], r["prompts_sha256"])' \
    "$W/lux-xl-c-w$k.guard.json" "lux-xl-c-w$k"
done
