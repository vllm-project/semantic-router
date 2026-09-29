#!/usr/bin/env bash
# usage: aj_wave.sh SHA SPLIT_NAME
# Node A GPU6 + GPU7: the Milestone 6 AutoJev-27B wave on the S rows without production targets,
# exactly as research & data's produce_nodeA.sh ran a wave, on two shards: target guard (SELECT /
# CAL / CAL698, every AHO / SHO slice, every gold-free eval panel), two collector shards in parallel
# (aj_job.sh; GPU6 shard 0, GPU7 shard 1) on one copy of the qualified frozen autotune cache (its
# autotune digest must equal the qualification's), a 128-prompt production repeat check of shard-1
# prompts on GPU6, then conversion with per-row attestation (v2.data.m3.teacher_targets). Inputs:
# /data/dev2/runs/9b/m6/data/SPLIT_NAME/build/wave.{rows,prompts}.jsonl; outputs in
# /data/dev2/runs/9b/m6/aj-wave (targets, attestation, report.json, events.jsonl). Nothing is
# uploaded. Writes both GPU leases (track=9b-m6) while it runs.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
sha=$1; split_name=$2
SRC=$(mirror_dir "$sha")
code=$(code_dir "$sha")
L=$code/v2/9b/lux9b/m6
W=$M6/aj-wave
R3=$DATA/dev2/runs/data/m3a2
QUAL=$R3/qualification.json
FROZEN_CACHE=$R3/triton-cache-autojev-nodeA
CACHE=$M6/aj-cache
H=$DATA/dev2/runs/data/m3a/hf
F=$DATA/dev2/private/data/arms-v2/final
G=$DATA/dev2/private/panels/goldfree
P=$M6/data/$split_name/build/wave.prompts.jsonl
ROWS=$M6/data/$split_name/build/wave.rows.jsonl
NAME=aj-9b-m6
[ -s "$P" ] && [ -s "$ROWS" ] || { echo "no wave inputs under $M6/data/$split_name/build" >&2; exit 2; }
[ ! -e "$W" ] || { echo "$W exists" >&2; exit 66; }
mkdir -p "$W"/{in,out,check-in,check-out}
event() { printf '{"event":"%s","rc":%s,"utc":"%s"}\n' "$1" "${2:-0}" "$(date -u +%FT%TZ)" >> "$W/events.jsonl"; }
lease() {
  local g
  for g in 6 7; do
    mkdir -p "$LEASES/gpu$g.lock"
    printf "track=%s\nstatus=%s\npurpose=9B Milestone 6: AutoJev-27B wave on the human-rated rows\nstep=%s\nupdated_utc=%s\n" \
      "$TRACK" "$1" "$2" "$(date -u +%FT%TZ)" > "$LEASES/gpu$g.lock/owner"
  done
}
digest() { (cd "$1" && find . -type f -name '*.autotune.json' -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -d' ' -f1); }
want=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["autotune_frozen"]["sha256"])' "$QUAL")
[ "$(digest "$FROZEN_CACHE")" = "$want" ] || { echo "frozen AutoJev cache digest differs from the qualification" >&2; exit 2; }
if [ ! -d "$CACHE" ]; then
  cp -a "$FROZEN_CACHE" "$CACHE.pending" && mv "$CACHE.pending" "$CACHE"
  printf '{"source": "%s", "copied_utc": "%s", "autotune_sha256": "%s", "tree_sha256": "%s"}\n' "$FROZEN_CACHE" \
    "$(date -u +%FT%TZ)" "$(digest "$CACHE")" "$(tree_sha "$CACHE")" > "$M6/aj-cache.copy.json"
fi
[ "$(digest "$CACHE")" = "$want" ] || { echo "M6 AutoJev cache copy digest differs" >&2; exit 2; }
lease reserved start
cd "$code" || exit 2
export PYTHONPATH=$code PYTHONDONTWRITEBYTECODE=1
protected=(--protected-rows "$H/select.jsonl" --protected-rows "$H/cal.jsonl" --protected-rows "$H/m2/cal/CAL698/cal.jsonl")
for f in "$F"/*.aho.jsonl "$F"/*.sho.jsonl "$H"/v2/arms/*/aho.jsonl; do protected+=(--protected-rows "$f"); done
for f in "$G"/*.prompts.jsonl; do protected+=(--protected-prompts "$f"); done
python3 -m v2.data.m3.guard --prompts "$P" --rows "$ROWS" "${protected[@]}" --receipt "$W/guard.json" > "$W/guard.stdout" 2>&1 \
  || { event guard_failed 1; lease idle guard_failed; exit 1; }
event guard_passed
python3 -m v2.data.m3.shards split --prompts "$P" --shards 2 --out-prefix "$W/in/$NAME" > "$W/split.json" \
  || { event split_failed 1; lease idle split_failed; exit 1; }
lease running shards
pids=(); rc=0
for k in 0 1; do
  bash "$L/aj_job.sh" --gpu $((6 + k)) --src "$SRC" --input "$W/in/$NAME.shard$k.prompts.jsonl" \
    --output "$W/out/$NAME.shard$k.jsonl" --cache "$CACHE" --label "$NAME-s$k" & pids+=($!)
done
for k in 0 1; do wait "${pids[$k]}" || rc=1; done
event shards_done $rc
[ "$rc" = 0 ] || { lease idle shards_failed; exit 1; }
python3 -m v2.data.m3.qualify prodrepeat-select --prompts "$P" --shards 2 --exclude-shard 0 \
  --out "$W/check-in/$NAME.repeat.prompts.jsonl" > /dev/null || { event repeat_select_failed 1; lease idle repeat; exit 1; }
bash "$L/aj_job.sh" --gpu 6 --src "$SRC" --input "$W/check-in/$NAME.repeat.prompts.jsonl" \
  --output "$W/check-out/$NAME.repeat.jsonl" --cache "$CACHE" --label "$NAME-repeat" \
  || { event repeat_run_failed 1; lease idle repeat; exit 1; }
python3 -m v2.data.m3.qualify prodrepeat-check --qualification "$QUAL" --repeat "$W/check-out/$NAME.repeat.jsonl" \
  --manifest "$W/check-out/$NAME.repeat.jsonl.manifest.json" \
  --shard-output "$W/out/$NAME.shard0.jsonl" --shard-output "$W/out/$NAME.shard1.jsonl" \
  --out "$W/repeat-check.json" > /dev/null || { event repeat_check_failed 1; lease idle repeat; exit 1; }
event repeat_check_passed
lease idle converting
python3 -m v2.data.m3.teacher_targets --wave "$NAME" --rows "$ROWS" --prompts "$P" --qualification "$QUAL" \
  --shard "$W/out/$NAME.shard0.jsonl=$W/out/$NAME.shard0.jsonl.manifest.json" \
  --shard "$W/out/$NAME.shard1.jsonl=$W/out/$NAME.shard1.jsonl.manifest.json" \
  --guard "$W/guard.json" --out "$W/$NAME.targets.jsonl" --report "$W/report.raw.json" \
  --attestation "$W/$NAME.attestation.jsonl" > "$W/convert.stdout" 2>&1 || { event convert_failed 1; exit 1; }
python3 - "$W/report.raw.json" "$W/repeat-check.json" "$W/guard.json" "$W/report.json" <<'EOF'
import json, sys
report = json.load(open(sys.argv[1]))
report["production_repeat_check"] = json.load(open(sys.argv[2]))
guard = json.load(open(sys.argv[3]))
guard["protected_files"] = len(guard["protected_files"])
report["guard"] = guard
json.dump(report, open(sys.argv[4], "x"), indent=1, sort_keys=True)
EOF
event converted
python3 - "$W" <<'EOF'
import glob, json, sys
jobs = [json.load(open(p)) for p in sorted(glob.glob(sys.argv[1] + "/*/*.manifest.json"))]
print(json.dumps({"jobs": len(jobs), "gpu_hours": round(sum(j["gpu_hours"] for j in jobs), 4),
                  "autotune_after": sorted({j["autotune_after"]["sha256"] for j in jobs})}))
EOF
echo "done aj wave"
