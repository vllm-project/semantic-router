#!/usr/bin/env bash
# Node A GPU2-4: AutoJev-27B target waves after a passed qualification (M3a preregistration v2 §5).
#
# Usage: produce_nodeA.sh MIRROR_DIR WORK_DIR QUALIFICATION_JSON CACHE
#
# WORK_DIR/prompts/{aj-m,aj-a0s,aj-sl}.prompts.jsonl and WORK_DIR/rows/{aj-m,aj-a0s,aj-sl}.rows.jsonl
# must exist. For each wave in order: guard, three shards on GPU2/3/4, a 128-prompt production repeat
# check on the wave's check GPU, conversion with per-row attestation, and a private HF upload under
# m3/teachers/autojev27/. Events go to WORK_DIR/produce.events.jsonl; a failing step stops the queue.
set -uo pipefail
src="$1"; work="$2"; qual="$3"; cache="$4"
code="$src/src/training/decision2"
job="$code/v2/data/m3/autojev_job.sh"
H=/data/dev2/runs/data/m3a/hf
G=/data/dev2/private/panels/goldfree
F=/data/dev2/private/data/arms-v2/final
REPO=llm-semantic-router/decision-2.0-training-data
export HF_HUB_CACHE=/data/dev2/hf-cache
event() { echo "{\"event\":\"$1\",\"wave\":\"$2\",\"rc\":${3:-0},\"utc\":\"$(date -u +%FT%TZ)\"}" >> "$work/produce.events.jsonl"; }
cd "$code" || exit 1
python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["pass"] else 1)' "$qual" || { event qualification_not_passed all 1; exit 1; }

protected=(--protected-rows "$H/select.jsonl" --protected-rows "$H/cal.jsonl" --protected-rows "$H/m2/cal/CAL698/cal.jsonl")
for f in "$F"/*.aho.jsonl "$F"/*.sho.jsonl "$H"/v2/arms/*/aho.jsonl; do protected+=(--protected-rows "$f"); done
for f in "$G"/*.prompts.jsonl; do protected+=(--protected-prompts "$f"); done

wave() { # name check_index upload_subdir target_stem
  local name=$1 check=$2 sub=$3 stem=$4 p="$work/prompts/$1.prompts.jsonl" r="$work/rows/$1.rows.jsonl"
  local d="$work/$name" k pids=() rc=0
  mkdir -p "$d/in" "$d/out" "$d/check-in" "$d/check-out"
  python3 -m v2.data.m3.guard --prompts "$p" --rows "$r" "${protected[@]}" --receipt "$d/guard.json" > "$d/guard.stdout" 2>&1 \
    || { event guard_failed "$name" 1; return 1; }
  event guard_passed "$name"
  python3 -m v2.data.m3.shards split --prompts "$p" --shards 3 --out-prefix "$d/in/$name" > "$d/split.json" || return 1
  for k in 0 1 2; do
    bash "$job" --gpu $((2 + k)) --src "$src" --input "$d/in/$name.shard$k.prompts.jsonl" \
      --output "$d/out/$name.shard$k.jsonl" --cache "$cache" --label "m3a2-$name-s$k" & pids+=($!)
  done
  for k in 0 1 2; do wait "${pids[$k]}" || rc=1; done
  event shards_done "$name" $rc
  [[ $rc -eq 0 ]] || return 1
  python3 -m v2.data.m3.qualify prodrepeat-select --prompts "$p" --shards 3 --exclude-shard "$check" \
    --out "$d/check-in/$name.repeat.prompts.jsonl" > /dev/null || return 1
  bash "$job" --gpu $((2 + check)) --src "$src" --input "$d/check-in/$name.repeat.prompts.jsonl" \
    --output "$d/check-out/$name.repeat.jsonl" --cache "$cache" --label "m3a2-$name-repeat" || { event repeat_run_failed "$name" 1; return 1; }
  python3 -m v2.data.m3.qualify prodrepeat-check --qualification "$qual" --repeat "$d/check-out/$name.repeat.jsonl" \
    --manifest "$d/check-out/$name.repeat.jsonl.manifest.json" \
    --shard-output "$d/out/$name.shard0.jsonl" --shard-output "$d/out/$name.shard1.jsonl" --shard-output "$d/out/$name.shard2.jsonl" \
    --out "$d/repeat-check.json" > /dev/null || { event repeat_check_failed "$name" 1; return 1; }
  event repeat_check_passed "$name"
  local up="$work/upload-$name/$sub"
  mkdir -p "$up"
  shards=()
  for k in 0 1 2; do shards+=(--shard "$d/out/$name.shard$k.jsonl=$d/out/$name.shard$k.jsonl.manifest.json"); done
  python3 -m v2.data.m3.teacher_targets --wave "$name" --rows "$r" --prompts "$p" --qualification "$qual" \
    "${shards[@]}" --guard "$d/guard.json" --out "$up/$stem.targets.jsonl" --report "$d/report.raw.json" \
    --attestation "$up/$stem.attestation.jsonl" > "$d/convert.stdout" 2>&1 || { event convert_failed "$name" 1; return 1; }
  python3 - "$d/report.raw.json" "$d/repeat-check.json" "$d/guard.json" "$up/$stem.report.json" <<'EOF'
import json, sys
report = json.load(open(sys.argv[1]))
report["production_repeat_check"] = json.load(open(sys.argv[2]))
guard = json.load(open(sys.argv[3]))
guard["protected_files"] = len(guard["protected_files"])
report["guard"] = guard
json.dump(report, open(sys.argv[4], "x"), indent=1, sort_keys=True)
EOF
  cp "$code/v2/data/records/hf-autojev27-provenance.md" "$work/upload-$name/PROVENANCE.md"
  [[ "$name" == aj-m ]] && python3 -c 'import json,sys; json.dump(json.load(open(sys.argv[1])), open(sys.argv[2], "x"), indent=1, sort_keys=True)' "$qual" "$work/upload-$name/qualification-v2.report.json"
  if grep -rlq "/data/" "$work/upload-$name"; then event path_leak "$name" 1; return 1; fi
  hf upload "$REPO" "$work/upload-$name" m3/teachers/autojev27 --repo-type dataset \
    --commit-message "Add AutoJev-27B soft targets wave $name (node A, M3a qualification v2)" > "$d/upload.log" 2>&1 \
    || { event upload_failed "$name" 1; return 1; }
  event "published:$(grep -o 'commit/[0-9a-f]*' "$d/upload.log" | tail -1 | cut -d/ -f2)" "$name"
}
wave aj-m 0 rp-v2 aj-m || exit 1
wave aj-a0s 1 pk1 A0s-train || exit 1
wave aj-sl 2 rp-v2 aj-sl || exit 1
event all_done all
