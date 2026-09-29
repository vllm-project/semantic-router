#!/usr/bin/env bash
# Own-Lux wave h-w1 (prereg m3b-lux-h-prereg-2026-09-29.md), node A (CPU): the prompt file for the
# H7 / H8 rows of the XL r2 recipes that lack own-Lux targets, its 256-prompt repeat subset and the
# target guard on both.
#
# Usage: luxxl_gap_nodeA.sh MIRROR_DIR
#
# Run from the exact mirror of a pushed commit. Stops before any output if the node-A missing
# lists of mx-xl-full-r2 / mx-xl-short-r2 differ from the private revision 10053613, or if a
# TRAIN file differs from the r2 manifest. The prompt set is the union of both lists (full
# first), built by v2.data.m3.xl_prompts with the H7 / H8 TRAIN files as the only pools; it must
# be one wave of 25,664 H7 / H8 TRAIN rows. The guard is the one of waves w1-w5 / c-w1 / c-w2
# plus the H7 / H8 AHO slices; the H7 / H8 SHO slices are never read. Outputs go to
# /data/dev2/runs/data/m3b/lux-xl/ and are never overwritten.
set -euo pipefail
set -o noclobber
src="$1"
code="$src/src/training/decision2"
B=/data/dev2/runs/data/m3b
W=$B/lux-xl
X=$B/xl-r2-a4/build
GAP=$B/gap/c2/final
H=/data/dev2/runs/data/m3a/hf
F=/data/dev2/private/data/arms-v2/final
G=/data/dev2/private/panels/goldfree
REPO=llm-semantic-router/decision-2.0-training-data
R2_REVISION=100536133e192c54ec57c2599a5e4706f6d334ff
EXPECTED=25664
export HF_HUB_CACHE=/data/dev2/hf-cache
umask 077
cd "$code"

tmp=$(mktemp -d -p "$W")
hf download "$REPO" m3/mixtures/xl-r2/mx-xl-full-r2.lux1.missing.jsonl \
  m3/mixtures/xl-r2/mx-xl-short-r2.lux1.missing.jsonl --repo-type dataset \
  --revision "$R2_REVISION" --local-dir "$tmp" > /dev/null
for r in mx-xl-full-r2 mx-xl-short-r2; do
  cmp -s "$tmp/m3/mixtures/xl-r2/$r.lux1.missing.jsonl" "$X/$r.lux1.missing.jsonl" \
    || { echo "$r: node-A missing list differs from revision $R2_REVISION" >&2; exit 1; }
done
rm -rf "$tmp"
python3 - "$X/mx-xl-r2.manifest.json" "$GAP" <<'EOF'
import hashlib, json, sys
manifest = json.load(open(sys.argv[1]))
for arm, spec in sorted(manifest["gap_arms"].items()):
    path = f"{sys.argv[2]}/{arm.lower()}.train.jsonl"
    if hashlib.sha256(open(path, "rb").read()).hexdigest() != spec["rows_sha256"]:
        sys.exit(f"{arm}: TRAIN file differs from the r2 manifest")
EOF

printf '{"H7": {"rows": ["%s"]}, "H8": {"rows": ["%s"]}}\n' \
  "$GAP/h7.train.jsonl" "$GAP/h8.train.jsonl" > "$W/lux-xl-h.pools.json"
python3 -m v2.data.m3.xl_prompts --pools "$W/lux-xl-h.pools.json" \
  --missing "$X/mx-xl-full-r2.lux1.missing.jsonl" --missing "$X/mx-xl-short-r2.lux1.missing.jsonl" \
  --wave-size 60000 --prefix "$W/lux-xl-h" > "$W/lux-xl-h.waves.json"
python3 - "$W" "$X" "$GAP" "$EXPECTED" <<'EOF' > "$W/lux-xl-h.check.json"
import collections, hashlib, json, sys
w, x, gap, expected = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
def ids(path):
    return [json.loads(line)["id"] for line in open(path) if line.strip()]
waves = json.load(open(f"{w}/lux-xl-h.waves.json"))
if list(waves) != [f"{w}/lux-xl-h-w1"]:
    sys.exit(f"expected one wave, got {len(waves)}")
prompts = ids(f"{w}/lux-xl-h-w1.prompts.jsonl")
full = ids(f"{x}/mx-xl-full-r2.lux1.missing.jsonl")
short = ids(f"{x}/mx-xl-short-r2.lux1.missing.jsonl")
arm = {}
for name in ("h7", "h8"):
    for line in open(f"{gap}/{name}.train.jsonl"):
        row = json.loads(line)
        if row["split"] == "train":
            arm[row["id"]] = name.upper()
native = {}
for line in open(f"{x}/mx-xl-full-r2.ids.jsonl"):
    row = json.loads(line)
    native[row["id"]] = row["native"]
problems = []
if len(prompts) != expected or len(set(prompts)) != expected:
    problems.append("count")
if set(prompts) != set(full) | set(short) or not set(short) <= set(full):
    problems.append("not_the_missing_union")
if any(i not in arm for i in prompts):
    problems.append("not_an_h7_h8_train_row")
if problems:
    sys.exit(f"prompt set check failed: {problems}")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
print(json.dumps({
    "wave": "lux-xl-h-w1",
    "prompts": len(prompts),
    "by_arm": dict(sorted(collections.Counter(arm[i] for i in prompts).items())),
    "missing_full_r2": len(full),
    "missing_short_r2": len(short),
    "short_subset_of_full": True,
    "native_tokens": sum(native[i] for i in prompts),
    "prompts_sha256": sha(f"{w}/lux-xl-h-w1.prompts.jsonl"),
    "rows_sha256": sha(f"{w}/lux-xl-h-w1.rows.jsonl"),
}, sort_keys=True))
EOF
cat "$W/lux-xl-h.check.json"
python3 -m v2.data.m3.luxxl repeat-prompts --prompts "$W/lux-xl-h-w1.prompts.jsonl" \
  --out "$W/lux-xl-h-w1-r256.prompts.jsonl"

protected=(--protected-rows "$H/select.jsonl" --protected-rows "$H/cal.jsonl" --protected-rows "$H/m2/cal/CAL698/cal.jsonl")
for f in "$F"/*.aho.jsonl "$F"/*.sho.jsonl "$H"/v2/arms/*/aho.jsonl "$B"/hf/v2/a7/arms/*/aho.jsonl \
  "$GAP/h7.aho.jsonl" "$GAP/h8.aho.jsonl"; do
  protected+=(--protected-rows "$f")
done
for f in "$G"/*.prompts.jsonl; do protected+=(--protected-prompts "$f"); done
for p in lux-xl-h-w1 lux-xl-h-w1-r256; do
  python3 -m v2.data.m3.guard --prompts "$W/$p.prompts.jsonl" --rows "$W/lux-xl-h-w1.rows.jsonl" \
    "${protected[@]}" --receipt "$W/$p.guard.json" > /dev/null
  python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(sys.argv[2], r["pass"], r["prompts"], r["prompts_sha256"], r["shared_state_informational"], len(r["protected_files"]))' \
    "$W/$p.guard.json" "$p"
done
