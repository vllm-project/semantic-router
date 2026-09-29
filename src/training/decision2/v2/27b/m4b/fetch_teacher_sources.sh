#!/usr/bin/env bash
# M4b: fetch the one teacher source node B lacks (AutoJev-27B AJ-SL, 177 of the A6h rows) into the node's HF
# cache, read-only from the private training-data dataset at its pinned revision, and verify every teacher
# source named by teacher-sources-m4b.json (node B host; download only, nothing is uploaded).
# Usage: fetch_teacher_sources.sh MIRROR_SHA
set -euo pipefail
echo "m4b fetch_teacher_sources $* start $(date -u +%FT%TZ)"

MIRROR=$1
S=/data/dev2/src/$MIRROR/src/training/decision2
[ -d "$S" ] || S=/data/dev2/src/$MIRROR-src_training_decision2/src/training/decision2
[ -f "$S/v2/27b/m4b/teacher-sources-m4b.json" ] || { echo "no m4b code in mirror $MIRROR" >&2; exit 2; }
REPO=llm-semantic-router/decision-2.0-training-data
REV=780d27439c5a07650a37b9a0ce3369dae3359baf
FILE=m3/teachers/autojev27/rp-v2/aj-sl.targets.jsonl
SHA=1254cb4703a10810acdef695a8218866fe7a664f5901074e8e8dffe68486aa75
export HF_HUB_CACHE=/data/dev2/hf-cache TMPDIR=/data/dev2/tmp HF_HUB_DISABLE_TELEMETRY=1
DEST=$HF_HUB_CACHE/datasets--${REPO/\//--}/snapshots/$REV/$FILE

if [ ! -f "$DEST" ]; then
  hf download "$REPO" --repo-type dataset --revision "$REV" --include "$FILE" > /dev/null
fi
[ "$(sha256sum "$DEST" | cut -d' ' -f1)" = "$SHA" ] || { echo "$DEST does not hash to $SHA" >&2; exit 1; }
echo "AJ-SL verified: $DEST"

python3 - "$S/v2/27b/m4b/teacher-sources-m4b.json" <<'EOF'
import hashlib, json, sys
spec = json.load(open(sys.argv[1]))
items = [spec["train"], spec["mixtures"]]
for teacher in spec["teachers"].values():
    items += teacher["files"] + ([teacher["provenance"]] if teacher.get("provenance") else [])
bad = []
for item in items:
    digest = hashlib.sha256()
    with open(item["path"], "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != item["sha256"]:
        bad.append(item["path"])
print(json.dumps({"sources": len(items), "mismatched": bad}))
raise SystemExit(1 if bad else 0)
EOF
echo "m4b fetch_teacher_sources complete"
