#!/usr/bin/env bash
# One Milestone 5 arm on one leased GPU: the preflight of seed 1, then the three seeds
# trained concurrently on that GPU, the typed DEV + CSS pilot readout of every BEST export,
# and the uniform soup of the three same-init seeds with its readout.
# Usage (node A): m5_arm.sh <gpu> <mirror-dir-name> <arm>   (specs records/arms/<arm>-s1..s3.json)
set -uo pipefail
gpu="$1"
sha="$2"
arm="$3"
D=/data/dev2/src/$sha/src/training/decision2/v2/06b
A=/data/dev2/runs/06b/m1/arms
specs=/data/dev2/runs/06b/m5/specs
logs=/data/dev2/logs/06b
mkdir -p "$specs" "$logs"
status() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" "$1" 2> /dev/null || echo MISSING; }

bash "$D/m1_arm.sh" "$gpu" "$sha" "$arm-s1" preflight || {
  echo "$arm-s1 preflight did not pass: arm stopped"
  exit 3
}
pids=()
for s in s1 s2 s3; do
  bash "$D/m1_arm.sh" "$gpu" "$sha" "$arm-$s" full > "$logs/$arm-$s.driver.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || true; done
complete=0
for s in s1 s2 s3; do
  result=$(status "$A/$arm-$s/full/COMPLETE.json")
  echo "$arm-$s full: $result"
  if [ "$result" = COMPLETE ]; then
    bash "$D/m1_arm.sh" "$gpu" "$sha" "$arm-$s" readout && complete=$((complete + 1))
  fi
done
if [ "$complete" -ne 3 ]; then
  echo "$arm: $complete of 3 seeds complete, no soup"
  exit 4
fi
python3 - "$A" "$arm" "$specs/$arm-soup.json" << 'PY'
import json
import sys

runs, arm, out = sys.argv[1:]
ingredients = []
for s in ("s1", "s2", "s3"):
    best = json.load(open(f"{runs}/{arm}-{s}/full/BEST.json"))
    ingredients.append(
        {
            "arm": f"{arm}-{s}",
            "run": f"/runs/m1/arms/{arm}-{s}/full",
            "state_sha256": best["state_sha256"],
        }
    )
spec = {
    "arm": f"{arm}-soup",
    "family": "qwen-causal",
    "recipe": f"/src/src/training/decision2/v2/06b/records/arms/{arm}-s1.json",
    "ingredients": ingredients,
    "allowed_differences": ["arm", "seed"],
    "export": True,
}
with open(out, "x") as stream:
    json.dump(spec, stream, indent=2)
    stream.write("\n")
PY
bash "$D/run_container.sh" "$gpu" "$sha" "$arm-soup" -- -m v2.06b.soup \
  --spec "/runs/m5/specs/$arm-soup.json" --output "/runs/m1/arms/$arm-soup/full"
[ "$(status "$A/$arm-soup/full/COMPLETE.json")" = COMPLETE ] || {
  echo "$arm-soup: export or reload parity failed"
  exit 5
}
DEV2_SPEC_JSON="$specs/$arm-soup.json" bash "$D/m1_arm.sh" "$gpu" "$sha" "$arm-soup" readout
echo "$arm done"
