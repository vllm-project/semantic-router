#!/usr/bin/env bash
# usage: ship_cal.sh RUNNER_SHA NAME DEV_READOUT
# CPU, node A, host python3: the coordinator's 23:15 calibration rule for a Milestone 7 finalist.
# CAL698 per-type temperatures are ADOPTED only if they do not worsen aggregate calibration on
# the development panels (typed DEV Brier and 10-bin ECE, CSS pilot median task Brier-sum and
# 15-bin ECE); otherwise the package ships T = 1. Same tool and call as the DEV2.0-8B release
# (release records devcal/cmd.sh.txt): v2.release.dev_calibration from the mirror RUNNER_SHA on
# the finalist's M7 readout. DEV_READOUT is an M7 readout name or a host prefix P with
# P-dev/dev.predictions.jsonl, P-css-pilot/css-pilot.predictions.jsonl and P-cal/calibration.json
# (the temperatures those predictions carry, which are also the candidate). Writes
# /data/dev2/runs/9b/m7/NAME-shipcal/{work/,devcal.json,SHIP.txt}; prints ship=CAL698 or ship=T1.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
rsha=$1; name=$2; prefix=$3
S=$(code_dir "$rsha")
[ -f "$S/v2/release/dev_calibration.py" ] || { echo "no v2.release.dev_calibration in $S" >&2; exit 2; }
case "$prefix" in /*) ;; *) prefix=$M7/$prefix ;; esac
dev=$prefix-dev/dev.predictions.jsonl
css=$prefix-css-pilot/css-pilot.predictions.jsonl
cal=$prefix-cal/calibration.json
for f in "$dev" "$css" "$cal"; do [ -s "$f" ] || { echo "missing $f" >&2; exit 2; }; done
python3 - "$cal" "$dev" "$css" <<'EOF'
import hashlib, json, sys
cal, *preds = sys.argv[1:]
want = hashlib.sha256(open(cal, "rb").read()).hexdigest()
for p in preds:
    got = json.load(open(p + ".manifest.json"))["calibration"]["file_sha256"]
    assert got == want, (p, got, want)
print("predictions carry calibration", want[:12])
EOF
out=$M7/$name-shipcal
[ ! -e "$out" ] || { echo "$out exists" >&2; exit 66; }
mkdir -p "$out"
(cd "$S" && PYTHONPATH="$S" dry python3 -B -m v2.release.dev_calibration --label "$name" \
  --panel-root "$DATA/dev2/private/panels" --typed-dev "$dev" --css-pilot "$css" \
  --source-calibration "$cal" --candidate "$cal" --work "$out/work" --output "$out/devcal.json")
[ "${DRY_RUN:-0}" != 1 ] || exit 0
ship=$(python3 -c 'import json,sys; print("CAL698" if json.load(open(sys.argv[1]))["adopt"] else "T1")' "$out/devcal.json")
printf 'ship=%s\ndevcal_sha256=%s\n' "$ship" "$(sha256sum "$out/devcal.json" | cut -c1-64)" > "$out/SHIP.txt"
echo "ship=$ship"
