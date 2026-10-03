#!/usr/bin/env bash
# Board submission of one IX1 run (node side, CPU): merge the stored run with its complement run,
# score the merged file with kit 87d4650b, write the public run directory, score that file again
# from the public root (relative paths only) and check it. The private merge stays under
# $BASE/submit/merged/<model>; the public run directory is $BASE/submit/public/runs/<name>.
#
# Usage: submit.sh --src DIR --model NAME --name RUN_NAME --panel DIR [--complement DIR]
#
# Checks: the public file re-scores to the same scores.json, index.json and benchmark-summary.json
# as the private merged file, and its Index (index, raw index, areas, the 38 benchmarks) equals the
# stored IX1 run's kit scoring.
set -euo pipefail
BASE=/data/dev2/private/eval/index021
KIT="$BASE/kit-87d4650b"
KIT_REVISION="87d4650b42b377c0291a89c1f1a879f9b31082bf"
src="" model="" name="" panel="" complement=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) src="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --name) name="$2"; shift 2 ;;
    --panel) panel="$2"; shift 2 ;;
    --complement) complement="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -f "$src/.dev2-mirror.json" && -n "$model" && "$name" =~ ^[a-z0-9.-]+$ && -f "$panel/panel.json" ]] \
  || { sed -n '2,11p' "$0" >&2; exit 2; }
[[ "$(git -C "$KIT" rev-parse HEAD)" == "$KIT_REVISION" && -z "$(git -C "$KIT" status --porcelain)" ]] \
  || { echo "kit is not a clean checkout of $KIT_REVISION" >&2; exit 1; }
S="$src/src/training/decision2"
stored="$BASE/ix1/runs/$model"
complement="${complement:-$BASE/submit/runs/$model}"
merged="$BASE/submit/merged/$model"
root="$BASE/submit/public"
pub="$root/runs/$name"
umask 077
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
kit() { PYTHONPATH="$KIT" "$BASE/venv/bin/python" -m decision_index "$@"; }
tools() { PYTHONPATH="$S:$KIT" "$BASE/venv/bin/python" -m v2.eval.ix1.submission "$@"; }

[[ ! -e "$merged" && ! -e "$pub" ]] || { echo "$merged or $pub exists" >&2; exit 1; }
tools merge --suite-dir "$BASE/suite-0.2" --stored "$stored" --complement "$complement" --out "$merged"
kit score --edition 0.2.1 --suite-dir "$BASE/suite-0.2" --engine "$name" \
  --results "$merged/results.jsonl" --out "$merged/kit" > "$merged/kit.log"
tools public --merged "$merged" --stored "$stored" --complement "$complement" \
  --panel "$panel/panel.json" --out "$pub"
(cd "$root" && kit score --edition 0.2.1 --suite-dir "$BASE/suite-0.2" --engine "$name" \
  --results "runs/$name/results.jsonl.gz" --out "runs/$name" > "runs/$name/score.log")
tools compare --a "$merged/kit" --b "$pub" | tee "$merged/compare-public.txt"
tools compare --a "$stored/merged/kit" --b "$pub" --index-only | tee "$merged/compare-stored-index.txt"
if grep -r -l -E '/data/|/root/|/home/|/mnt/' "$pub" --include='*.json' --include='*.log'; then
  echo "node paths in public files" >&2; exit 1
fi
python3 - "$pub/scores.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
print(json.dumps({k: s[k] for k in ("engine", "edition", "completed", "complete", "counts", "decision_index", "raw_index", "scores", "latency_ms")}))
if not s["complete"]:
    raise SystemExit("the public run is not complete")
EOF
