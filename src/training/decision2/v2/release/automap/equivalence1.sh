#!/usr/bin/env bash
# Show that two remote-code versions answer identically: the same staged repository with
# the parity-tested files and with the files about to be uploaded, on the CPU in the
# clean venv, over the first N public-231 prompts; every response must be equal.
#
# Usage: equivalence1.sh <tested-staged-dir> <upload-stage-files-dir> <work-dir> <N> [threads]
set -euo pipefail
tested="$1"; upload="$2"; work="$3"; count="$4"; threads="${5:-32}"
here="$(cd "$(dirname "$0")" && pwd)"
fresh=/data/dev2/tools/envs/fresh-cpu/bin/python
goldfree=/data/dev2/private/dev1-automap/private/panels/goldfree
[[ -e "$work" ]] && { echo "exists: $work" >&2; exit 1; }
mkdir -p "$work"
cp -a "$tested" "$work/upload"
cp --remove-destination "$upload"/* "$work/upload/"
for side in tested upload; do
  directory="$tested"; [[ "$side" == upload ]] && directory="$work/upload"
  docker run --rm --network none -v /data/dev2:/data/dev2 -e HF_HUB_OFFLINE=1 \
    -e HF_MODULES_CACHE="$work/modules-$side" --entrypoint "$fresh" decision20-train-fast:host2 \
    -B "$here/parity1.py" --model "$directory" --device cpu --threads "$threads" \
    --panel "public231:$goldfree/public231.prompts.jsonl:$goldfree/public231.prompts.jsonl:$count" \
    --output "$work/$side.json" --answers "$work/$side.answers.jsonl" > "$work/$side.log" 2>&1 || true
done
python3 - "$work" <<'PY'
import hashlib, json, sys
from pathlib import Path
work = Path(sys.argv[1])
sides = {s: (work / f"{s}.answers.jsonl").read_text() for s in ("tested", "upload")}
codes = {s: json.loads((work / f"{s}.json").read_text())["remote_code_sha256"] for s in sides}
print(json.dumps({
    "responses": len(sides["tested"].splitlines()),
    "identical": sides["tested"] == sides["upload"] and bool(sides["tested"]),
    "tested_answers_sha256": hashlib.sha256(sides["tested"].encode()).hexdigest(),
    "changed_code_files": sorted(k for k in codes["upload"] if codes["tested"].get(k) != codes["upload"][k]),
}))
PY
