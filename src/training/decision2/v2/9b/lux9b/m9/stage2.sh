#!/usr/bin/env bash
# 9B M9 stage-2 inputs on node C (amendment 2), host python3 from an exact mirror: the release-safe IB1-r3 / IB2 files
# (downloaded with the host HF CLI at their pinned dataset revisions) are hash-checked, then lux9b/m9_data.py builds
#   data/l9ib   x60 + IB1 + IB2
#   data/l9ibx  x60 + IB1 + IB2 without the in-distribution families isarc, w2c, hover, gsm2
# and data/READY2.json records both builds' TRAIN / teacher hashes (the stage-2 chains re-hash against it).
#
# usage: stage2.sh fetch | build        (fetch also runs on node A for the IB DEV diagnostics)
set -euo pipefail
M=/data/dev2/runs/9b/m9
D=$M/inputs/ib
OWN=$(cd "$(dirname "$0")/../../../.." && pwd)
IB1_REV=31b200a34759d6b2ca197737603dfaa253c471cd
IB2_REV=c5dbdd0a88efe58059c6ece8ae2b181f9132619f
IB1=$D/ib1-31b200a3/m6/ib1
IB2=$D/ib2-c5dbdd0a/m6/ib2
declare -A WANT=(
  ["$IB1/ib1.train.jsonl"]=1e1b08f3d37f9051ffe2e4b99fd7be673f315d05cc74a27a8d350eae2bb706c5
  ["$IB1/ib1.dev.jsonl"]=3f56aa418e90e58f3fbaa50bf2f9f4f0405eeb9dd2f0c51ffca6f602711c693f
  ["$IB2/ib2.train.jsonl"]=ee137efa8bbf86e5c62574f3b8fbd6204063e095a8da514f32600ba3d51d1cfa
  ["$IB2/ib2.dev.jsonl"]=ab009fb12f9563c3ef4a846f5455dd5233f2a2c5fafe6ac8a9c74349532eb923
)
log() { echo "$(date -u +%FT%TZ) stage2 $*" | tee -a "$M/OPERATIONS.log"; }
check() {
  local f
  for f in "${!WANT[@]}"; do
    [ "$(sha256sum < "$f" | cut -d' ' -f1)" = "${WANT[$f]}" ] || { log "hash differs: $f"; return 1; }
  done
}
case ${1:-} in
  fetch)
    mkdir -p "$D"
    export HF_HUB_CACHE=/data/dev2/hf-cache
    [ -f "$IB1/ib1.train.jsonl" ] || hf download llm-semantic-router/decision-2.0-training-data --repo-type dataset \
      --revision "$IB1_REV" --include "m6/ib1/*" --local-dir "$D/ib1-31b200a3" > "$D/hf-ib1.log" 2>&1
    [ -f "$IB2/ib2.train.jsonl" ] || hf download llm-semantic-router/decision-2.0-training-data --repo-type dataset \
      --revision "$IB2_REV" --include "m6/ib2/*" --local-dir "$D/ib2-c5dbdd0a" > "$D/hf-ib2.log" 2>&1
    check
    log "IB1-r3 @$IB1_REV and IB2 @$IB2_REV fetched; TRAIN / DEV hashes match the release records"
    ;;
  build)
    check
    python3 "$OWN/v2/9b/lux9b/m9_data.py" --x60-dir "$M/data/x60" --ib1 "$IB1/ib1.train.jsonl" \
      --ib2 "$IB2/ib2.train.jsonl" --output "$M/data/l9ib" > "$M/data/l9ib.build.log"
    python3 "$OWN/v2/9b/lux9b/m9_data.py" --x60-dir "$M/data/x60" --ib1 "$IB1/ib1.train.jsonl" \
      --ib2 "$IB2/ib2.train.jsonl" --exclude-family isarc --exclude-family w2c --exclude-family hover \
      --exclude-family gsm2 --output "$M/data/l9ibx" > "$M/data/l9ibx.build.log"
    python3 - "$M" << 'EOF'
import json, sys
m = sys.argv[1]
lock = {"schema": "lux9b-m9-ready2/1"}
for name in ("l9ib", "l9ibx"):
    man = json.load(open(f"{m}/data/{name}/manifest.json"))
    lock[name] = {k: man[k] for k in ("train_sha256", "teacher_sha256", "rows", "ib_rows_kept", "ib_native_tokens_kept")}
with open(f"{m}/data/READY2.json", "x") as f:
    json.dump(lock, f, indent=2)
print(json.dumps(lock))
EOF
    log "stage-2 TRAIN built: data/READY2.json"
    ;;
  *) sed -n '2,9p' "$0"; exit 2 ;;
esac
