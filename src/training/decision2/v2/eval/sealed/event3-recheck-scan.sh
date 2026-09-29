#!/usr/bin/env bash
# JevArena-C1 event 3: confirmatory overlap scan (custodian step; run on node A right before the event).
# Usage: event3-recheck-scan.sh <mirror-dir-name>
# It reads the protected C1 source rows in place under the sealed directory (read-only, logged in C1's
# ACCESS.log), scans the event-3 recheck training manifest with the event-2 configuration inside the
# pinned image (CPU only, --network none, no GPU devices), and writes SCAN-VERDICT.json, which
# event3.sh requires (PASS, same manifest and protected rows) before it reads the key. FAIL = any
# OVERLAP, any non-CLEAN id that was CLEAN at event 2, or a recurring id with higher containment.
set -euo pipefail
umask 077
SRC=${1:?usage: event3-recheck-scan.sh <mirror-dir-name>}
[ -f "/data/dev2/src/$SRC/.dev2-mirror.json" ] || { echo "no verified mirror $SRC" >&2; exit 1; }
S="/data/dev2/src/$SRC/src/training/decision2"
IMG=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
C1=/data/dev2/private/sealed/c1
PROT=$C1/v1_1/protected-all-splits.jsonl
PROT_SHA=36797f509bd96c3cb703df37cc48114c9bbdf0d2241802e56262139f4bef0a1a
W=/data/dev2/runs/eval/m4/c1-event3-recheck
MANIFEST=$W/training-manifest.json
MANIFEST_SHA=e37e73f9c1519362bda350ec7d475ed6acf7e47ed1f83dc084b3a5a93247acfc
BASE=/data/dev2/runs/eval/m4/c1-event2-recheck2/overlap-hits.jsonl
BASE_SHA=077f1b53148c1fa4bc1a84d8caebba156c5da800a8d86e8989148deeef6984dd
O="$W/scan-$(date -u +%Y%m%dT%H%M%SZ)"
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$W/OPERATIONS.log"; }
sha() { sha256sum <"$1" | cut -c1-64; }

[ "$(docker image inspect --format '{{.Id}}' "$IMG")" = "$IMG" ] || { log "scan refused: image $IMG missing"; exit 1; }
[ "$(sha "$MANIFEST")" = "$MANIFEST_SHA" ] || { log "scan refused: training manifest differs from $MANIFEST_SHA"; exit 1; }
[ "$(sha "$BASE")" = "$BASE_SHA" ] || { log "scan refused: event-2 hits differ from $BASE_SHA"; exit 1; }
[ "$(sha "$PROT")" = "$PROT_SHA" ] || { log "scan refused: protected rows differ from $PROT_SHA"; exit 1; }
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) RECHECK3 protected rows $PROT_SHA read in place (read-only) for the event-3 confirmatory overlap scan, manifest $MANIFEST_SHA, mirror $SRC" >>"$C1/ACCESS.log"
mkdir -p "$O"
log "scan start: protected $PROT_SHA, manifest $MANIFEST_SHA, mirror $SRC, output $O (image $IMG, --network none, no GPU)"
docker run --rm --network none -v "$S:$S:ro" -v "$C1/v1_1:$C1/v1_1:ro" -v "$W:$W" \
  -e PYTHONPATH="$S" -e PYTHONDONTWRITEBYTECODE=1 -w "$S" --entrypoint python3 "$IMG" \
  -m v2.eval.sealed.overlap scan --protected "$PROT" --manifest "$MANIFEST" \
  --workers 48 --exact-min-tokens 8 --output "$O/overlap-receipt.json" --hits "$O/overlap-hits.jsonl" \
  >"$O/scan.log" 2>&1 || { log "scan FAILED to run (see $O/scan.log); no verdict"; exit 1; }
set +e
(cd "$S" && PYTHONPATH="$S" PYTHONDONTWRITEBYTECODE=1 python3 -m v2.eval.sealed.scanverdict compare \
  --hits "$O/overlap-hits.jsonl" --baseline "$BASE" --receipt "$O/overlap-receipt.json" \
  --manifest "$MANIFEST" --protected-sha "$PROT_SHA" --output "$O/SCAN-VERDICT.json") | tee -a "$W/OPERATIONS.log"
rc=${PIPESTATUS[0]}
set -e
cp "$O/SCAN-VERDICT.json" "$W/SCAN-VERDICT.json"
log "scan verdict $([ "$rc" = 0 ] && echo PASS || echo FAIL): $(sha "$W/SCAN-VERDICT.json") (receipt $(sha "$O/overlap-receipt.json"), hits $(sha "$O/overlap-hits.jsonl"))"
exit "$rc"
