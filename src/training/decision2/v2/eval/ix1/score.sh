#!/usr/bin/env bash
# IX1 scoring (node side, CPU): merge the shard results, score them with the port and with kit
# 87d4650b, then compare (scorer gate, external report, frontier peer). Outputs stay private.
#
# Usage: score.sh --src DIR --model NAME --size SIZE --panel DIR [--allow-errors]
#                 [--suffix S (--force FILE | --results FILE)]
#
# --force FILE (a JSON list of run IDs) rescores with those rows counted as wrong (status
# "unsupported"), the declared contamination forcing rule; --results FILE scores a derived results
# file (e.g. calibrated predictions); outputs go to merged<S>/.
set -euo pipefail
BASE=/data/dev2/private/eval/index021
R=$BASE/ix1
src="" model="" size="" panel="" allow="" suffix="" force="" derived=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --results) derived="$2"; shift 2 ;;
    --src) src="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --size) size="$2"; shift 2 ;;
    --panel) panel="$2"; shift 2 ;;
    --allow-errors) allow="--allow-errors"; shift ;;
    --suffix) suffix="$2"; shift 2 ;;
    --force) force="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -f "$src/.dev2-mirror.json" && -n "$model" && -n "$size" && -f "$panel/panel.json" ]] \
  || { sed -n '2,9p' "$0" >&2; exit 2; }
S="$src/src/training/decision2"
run="$R/runs/$model"
out="$run/merged$suffix"
umask 077
export CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
if [[ -n "$derived" ]]; then
  [[ -n "$suffix" ]] || { echo "--results needs --suffix" >&2; exit 2; }
  mkdir -p "$out"
  cp "$derived" "$out/results.jsonl"
elif [[ -z "$force" ]]; then
  PYTHONPATH="$S" python3 -m v2.eval.ix1.merge --panel "$panel/panel.json" --run "$run" --out "$out" $allow
else
  mkdir -p "$out"
  python3 - "$run/merged/results.jsonl" "$force" "$out/results.jsonl" <<'EOF'
import json, sys
source, force, target = sys.argv[1:]
forced = set(json.load(open(force)))
with open(source) as rows, open(target, "x") as out:
    for line in rows:
        r = json.loads(line)
        if r["run_id"] in forced:
            r = {**r, "status": "unsupported", "error": "contamination_override"}
            r.pop("response", None)
        out.write(json.dumps(r, separators=(",", ":")) + "\n")
EOF
fi
cd "$BASE"
PYTHONPATH="$S:$BASE/kit-19ad28ec" venv/bin/python -m external_index021 score --suite-dir suite-0.2 \
  --results "$out/results.jsonl" --out "$out/port.json" > "$out/port.log"
PYTHONPATH="$BASE/kit-87d4650b" venv/bin/python -m decision_index score --edition 0.2.1 --suite-dir suite-0.2 \
  --results "$out/results.jsonl" --out "$out/kit" > "$out/kit.log"
PYTHONPATH="$S" python3 -m v2.eval.ix1.compare --port "$out/port.json" --kit "$out/kit/index.json" \
  --size "$size" --external "$R/../external/index021-frontier-gap-2026-10-01.json" --out "$out/compare.json"
