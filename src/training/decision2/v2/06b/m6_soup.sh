#!/usr/bin/env bash
# One Milestone 6 soup on one leased GPU: soup.py (export + reload parity) on the image Python,
# then the typed DEV + CSS pilot readout of the soup export.
# Usage (node A): m6_soup.sh <gpu> <mirror-dir-name> <soup-name> <spec>
# <spec> is a soup spec under /data/dev2/runs/06b (written by m6_soupspec.py) whose arm is <soup-name>.
set -euo pipefail
gpu="$1"
sha="$2"
name="$3"
spec="$4"
D=/data/dev2/src/$sha/src/training/decision2/v2/06b
A=/data/dev2/runs/06b/m1/arms
owner=/data/dev2/leases/gpu$gpu.lock/owner
case "$spec" in
  /data/dev2/runs/06b/*) ;;
  *)
    echo "the soup spec must be under /data/dev2/runs/06b" >&2
    exit 2
    ;;
esac
arm=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['arm'])" "$spec")
[ "$arm" = "$name" ] || {
  echo "spec arm $arm differs from $name" >&2
  exit 2
}
status() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" "$1" 2> /dev/null || echo MISSING; }
lease() {
  grep -qs '^track=06b-encoder$' "$owner" || exit 2
  printf 'track=06b-encoder\nstatus=%s\npurpose=%s\ncurrent_run=%s\nupdated_utc=%s\n' "$1" "$2" "$name" "$(date -u +%FT%TZ)" > "$owner"
}
export DEV2_PYTHON=python3
lease active "M6 soup and readout"
bash "$D/run_container.sh" "$gpu" "$sha" "$name" -- -m v2.06b.soup \
  --spec "/runs/${spec#/data/dev2/runs/06b/}" --output "/runs/m1/arms/$name/full"
[ "$(status "$A/$name/full/COMPLETE.json")" = COMPLETE ] || {
  echo "$name: export or reload parity failed"
  lease idle "0.6B soup $name failed parity; no job running"
  exit 5
}
DEV2_SPEC_JSON="$spec" bash "$D/m1_arm.sh" "$gpu" "$sha" "$name" readout
lease idle "0.6B soup $name done; no job running"
echo "$name done"
