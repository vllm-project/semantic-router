#!/usr/bin/env bash
# usage: teacher.sh SHA GPU NAME (parity N | shard K)
# One DEV2.0-27B (A20r) collection on one lent GPU with the scored runtime of its post-key run
# M4-A20r-soup/formal: v2.27b.typed_collect_kernel from that run's mirror (TEACHER_CODE) in the 27B
# kernel image (IMAGE27), the scored checkpoint (identity 2e074511...), the Qwen3.8-27B base, the
# package's T = 1 calibration.json, 32,768 tokens, HIP_FORCE_DEV_KERNARG=1 and a private copy of the
# run's frozen autotune cache (tree hashes before / after in NAME/cache.json). Output
# /data/dev2/runs/9b/m8/NAME/predictions.jsonl (+ .manifest.json, .runtime.json).
#   parity N  the first N typed FINAL gold-free prompts (the collector's --max-items smoke), then
#             the exact-parity check against the scored run's stored predictions (parity.json;
#             same answers, probabilities within 1e-4); a FAIL stops the milestone.
#   shard K   the teacher prompts m8/data/m8-prompts/build/prompts-K.jsonl (the K-mix top-up rows).
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
sha=$1; gpu=$2; name=$3; mode=$4; arg=$5
L=$(code_dir "$sha")/v2/9b/lux9b/m8
TS=$(code_dir "$TEACHER_CODE")
[ -d "$TS" ] || { echo "teacher runtime mirror $TS missing" >&2; exit 2; }
for p in "$TEACHER_CKPT/decision_config.json" "$TEACHER_CAL" "$TEACHER_BASE"; do
  [ -e "$p" ] || { echo "missing $p" >&2; exit 2; }
done
case "$mode" in
  parity) input=$DATA/dev2/private/panels/goldfree/typed-final.prompts.jsonl; extra=(--max-items "$arg")
          [[ "$arg" =~ ^[0-9]+$ ]] || { echo "parity needs N" >&2; exit 2; } ;;
  shard) input=$M8/data/m8-prompts/build/prompts-$arg.jsonl; extra=()
         [ -s "$input" ] || { echo "no prompt shard $input" >&2; exit 2; } ;;
  *) echo "mode must be parity or shard" >&2; exit 2 ;;
esac
tc=$M8/tcache/$name
[ ! -e "$tc" ] || { echo "$tc exists" >&2; exit 66; }
mkdir -p "$M8/tcache"
cp -a "$TEACHER_CACHE" "$tc.pending" && mv "$tc.pending" "$tc" || exit 2
before=$(tree_sha "$tc"); src_tree=$(tree_sha "$TEACHER_CACHE")
indir=$(dirname "$input")
set +e
"$L/run_gpu.sh" "$gpu" "$M8/$name" "M8 A20r teacher $mode $arg" 60 -- \
  -e PYTHONPATH=/code -e TRITON_CACHE_AUTOTUNING=1 -e TRITON_CACHE_DIR=/triton -e HIP_FORCE_DEV_KERNARG=1 \
  --mount type=bind,src="$tc",dst=/triton \
  --mount type=bind,src="$TS",dst=/code,readonly \
  --mount type=bind,src="$TEACHER_CKPT",dst="$TEACHER_CKPT",readonly \
  --mount type=bind,src="$TEACHER_BASE",dst="$TEACHER_BASE",readonly \
  --mount type=bind,src="$(dirname "$TEACHER_CAL")",dst="$(dirname "$TEACHER_CAL")",readonly \
  --mount type=bind,src="$indir",dst=/in,readonly \
  --mount type=bind,src="$M8/$name",dst=/out \
  -w /code "$IMAGE27" python3 -m v2.27b.typed_collect_kernel \
    --checkpoint "$TEACHER_CKPT" --source-path "$TEACHER_BASE" --model-id llm-semantic-router/DEV2.0-27B \
    --model-revision "$TEACHER_REV" --max-length 32768 --calibration "$TEACHER_CAL" \
    --input "/in/$(basename "$input")" --output /out/predictions.jsonl "${extra[@]}"
status=$?
set -e
printf '{"source": "%s", "source_tree_sha256": "%s", "before_tree_sha256": "%s", "after_tree_sha256": "%s", "input": "%s", "input_sha256": "%s", "teacher_code": "%s", "image": "%s"}\n' \
  "$TEACHER_CACHE" "$src_tree" "$before" "$(tree_sha "$tc")" "$input" "$(sha256sum "$input" | cut -c1-64)" \
  "$TEACHER_CODE" "$IMAGE27" > "$M8/$name/cache.json"
[ "$status" = 0 ] || exit "$status"
if [ "$mode" = parity ] && [ "${DRY_RUN:-0}" != 1 ]; then
  python3 - "$M8/$name/predictions.jsonl" "$TEACHER_PARITY" "$M8/$name/parity.json" <<'EOF' || exit 1
import json, sys
got_path, ref_path, out = sys.argv[1:]
got = [json.loads(l) for l in open(got_path) if l.strip()]
ref = {}
with open(ref_path) as f:
    for l in f:
        r = json.loads(l)
        ref[r["id"]] = r
        if len(ref) >= len(got):
            break
diffs, worst, n = [], 0.0, 0
for g in got:
    r = ref.get(g["id"])
    if r is None:
        diffs.append(f"{g['id']}: not in the stored run"); continue
    for q, a in g["answers"].items():
        b = r["answers"].get(q)
        n += 1
        if b is None or a.get("type") != b.get("type") or ("error" in a) != ("error" in b):
            diffs.append(f"{g['id']}/{q}: type or validity differs"); continue
        if "error" in a:
            continue
        if a["type"] == "noul":
            d = abs(a["noul"] - b["noul"]); same = (a["noul"] > 0.5) == (b["noul"] > 0.5)
        else:
            pa, pb = a["probabilities"], b["probabilities"]
            d = max(abs(pa[k] - pb[k]) for k in pa) if set(pa) == set(pb) else 1.0
            same = max(pa, key=pa.get) == max(pb, key=pb.get) and a.get("choice") == b.get("choice")
        worst = max(worst, d)
        if not same or d > 1e-4:
            diffs.append(f"{g['id']}/{q}: answer differs or |dp| {d:.2e} > 1e-4")
res = {"status": "PASS" if not diffs and got else "FAIL", "prompts": len(got), "questions": n,
       "max_abs_probability_diff": worst, "differences": diffs[:50], "stored": ref_path, "tolerance": 1e-4}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("status", "prompts", "questions", "max_abs_probability_diff")}))
sys.exit(0 if res["status"] == "PASS" else 1)
EOF
fi
echo "done $name"
