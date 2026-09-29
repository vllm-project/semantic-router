#!/usr/bin/env bash
# M6 baselines of the M7(b) matched contrasts: uncorrected CHK probabilities + devcheck of the
# m6-mx seeds and soup on GPU0 and the m6-cx seeds and soup on GPU1 (m7_chk.sh, sequential per
# GPU, recorded co-tenants of the running M7(b) queues). m6-mxcx-soup reuses its M7(a) CHK
# probabilities (/data/dev2/runs/06b/m7/a/probs) for a host-only devcheck.
# Usage (node A): m7_baselines.sh <mirror-dir-name>   -> /data/dev2/runs/06b/m7/baselines/<arm>/
set -uo pipefail
sha="$1"
R=/data/dev2/runs/06b
B=$R/m7/baselines
S=/data/dev2/src/$sha/src/training/decision2
mkdir -p "$B"
lane() {
  local gpu=$1 arm
  shift
  for arm in "$@"; do
    bash "$S/v2/06b/m7_chk.sh" "$gpu" "$sha" "$arm" "$B/$arm" cotenant || echo "$(date -u +%FT%TZ) $arm failed"
  done
}
lane 0 m6-mx-s1 m6-mx-s2 m6-mx-s3 m6-mx-soup &
p0=$!
lane 1 m6-cx-s1 m6-cx-s2 m6-cx-s3 m6-cx-soup &
p1=$!
mkdir -p "$B/m6-mxcx-soup"
[ -f "$B/m6-mxcx-soup/devcheck.json" ] || PYTHONPATH="$S" PYTHONDONTWRITEBYTECODE=1 python3 -m v2.06b.m7_scorebias devcheck \
  --aho-dir "$R/m7/a/aho" --probs "$R/m7/a/probs/PROBS.json" --label m6-mxcx-soup --output "$B/m6-mxcx-soup/devcheck.json"
wait "$p0"
wait "$p1"
echo "$(date -u +%FT%TZ) baselines done"
