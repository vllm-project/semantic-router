#!/usr/bin/env bash
# JevArena-C1 rescan of an item set (custodian step, before its scoring event; CPU only).
# Usage:
#   c1-rescan.sh hf-delta --src MIRROR --item-set v1.2                                      (node A)
#   c1-rescan.sh scan  --src MIRROR --item-set v1.2 --node NAME --protected PATH [--extra-root LABEL=DIR]...
#   c1-rescan.sh judge --src MIRROR --item-set v1.2 --scan NODE=DIR [--scan NODE=DIR]...   (node A)
# `hf-delta` downloads every version of every private training-data file that landed after the
# event-2 revision (the spec's hf_delta) under /data/dev2/private, which the node-A scan covers.
# `scan` runs on each node over that node's own files: it builds the coverage manifest of
# c1-rescan-coverage.json (every text-bearing file under the existing roots, minus the excludes,
# archives extracted, compressed files decompressed, suffix-less blobs sniffed, hot files copied),
# then scans it against the protected rows (pinned SHA-256; read in place, logged) in the pinned
# image with --network none and no GPU device. `judge` extracts every node's hits against the pinned
# event-2 baseline and writes the merged verdict for the item set's pinned retired list to
# c1-rescan-<set>/SCAN-VERDICT.json, which event3.sh requires by its SHA-256 before it reads the key.
set -euo pipefail
umask 077
CMD=${1:?usage: c1-rescan.sh scan|judge --src MIRROR --item-set SET ...}
shift
SRC="" SET="" NODE="" PROT=""
EXTRA=() SCANS=()
while [ $# -gt 0 ]; do
  case $1 in
  --src) SRC=$2; shift 2 ;;
  --item-set) SET=$2; shift 2 ;;
  --node) NODE=$2; shift 2 ;;
  --protected) PROT=$2; shift 2 ;;
  --extra-root) EXTRA+=("$2"); shift 2 ;;
  --scan) SCANS+=("$2"); shift 2 ;;
  *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$SET" =~ ^v[0-9]+\.[0-9]+$ ]] || { echo "--item-set vN.M is required" >&2; exit 2; }
[ -f "/data/dev2/src/$SRC/.dev2-mirror.json" ] || { echo "no verified mirror $SRC" >&2; exit 1; }
S="/data/dev2/src/$SRC/src/training/decision2"
SPEC="$S/v2/eval/sealed/c1-rescan-coverage.json"
W="/data/dev2/runs/eval/m4/c1-rescan-${SET//./_}"
mkdir -p "$W"
spec() { python3 -c 'import json, sys; v = json.load(open(sys.argv[1]))
for k in sys.argv[2].split("/"): v = v[k]
print("\n".join(v) if isinstance(v, list) else v)' "$SPEC" "$1"; }
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [$CMD${NODE:+ $NODE}] $*" | tee -a "$W/OPERATIONS.log"; }
sha() { sha256sum <"$1" | cut -c1-64; }
IMG=$(spec image)
PROT_SHA=$(spec protected_sha256)
RETIRED=$(spec "item_sets/$SET/retired")
RETIRED_SHA=$(spec "item_sets/$SET/retired_sha256")
BASE=$(spec baseline/hits)
BASE_SHA=$(spec baseline/sha256)
[[ "$RETIRED_SHA" =~ ^[0-9a-f]{64}$ ]] || { log "refused: item set $SET is not registered"; exit 1; }

if [ "$CMD" = scan ]; then
  [[ "$NODE" =~ ^[A-Za-z0-9-]+$ ]] || { echo "--node NAME is required" >&2; exit 2; }
  [ -f "$PROT" ] || { echo "--protected PATH is required" >&2; exit 2; }
  [ "$(docker image inspect --format '{{.Id}}' "$IMG")" = "$IMG" ] || { log "refused: image $IMG missing"; exit 1; }
  [ "$(sha "$PROT")" = "$PROT_SHA" ] || { log "refused: protected rows differ from $PROT_SHA"; exit 1; }
  O="$W/$NODE-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir "$O"
  ARGS=() MOUNTS=()
  mapfile -t ROOTS < <(spec roots)
  mapfile -t GLOBS < <(spec root_globs)
  for pattern in "${GLOBS[@]}"; do
    for dir in $pattern; do ROOTS+=("$dir"); done
  done
  for dir in "${ROOTS[@]}"; do
    if [ -d "$dir" ]; then
      label=$(basename "$(dirname "$dir")")-$(basename "$dir")
      ARGS+=(--root "${label#-}=$dir")
      MOUNTS+=(-v "$dir:$dir:ro")
    else
      log "root $dir absent on this node"
    fi
  done
  for item in "${EXTRA[@]}"; do
    ARGS+=(--root "$item")
    MOUNTS+=(-v "${item#*=}:${item#*=}:ro")
  done
  mapfile -t EXCLUDES < <(spec excludes)
  for pattern in "${EXCLUDES[@]}"; do ARGS+=(--exclude "$pattern"); done
  # The sealed directory is hidden from both containers; only the protected rows' own dir is mounted.
  MOUNTS+=(--mount "type=tmpfs,destination=/data/dev2/private/sealed")
  log "coverage start: ${#ROOTS[@]} spec roots, ${#EXTRA[@]} extra, output $O"
  docker run --rm --network none -v "$S:$S:ro" "${MOUNTS[@]}" -v "$O:$O" \
    -e PYTHONPATH="$S" -e PYTHONDONTWRITEBYTECODE=1 -w "$S" --entrypoint python3 "$IMG" \
    -m v2.eval.sealed.coverage build --node "$NODE" "${ARGS[@]}" --work "$O/work" \
    --output "$O/training-manifest.json" --receipt "$O/coverage-receipt.json" \
    --hot-minutes "$(spec hot_minutes)" --workers 32 >"$O/coverage.log" 2>&1 ||
    { log "coverage FAILED (see $O/coverage.log); no scan"; exit 1; }
  log "coverage done: manifest $(sha "$O/training-manifest.json"), receipt $(sha "$O/coverage-receipt.json")"
  if [ "$PROT" = /data/dev2/private/sealed/c1/v1_1/protected-all-splits.jsonl ]; then
    ACCESS=/data/dev2/private/sealed/c1/ACCESS.log
  else
    ACCESS="$(dirname "$PROT")/ACCESS.log"
  fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) RESCAN-${SET} protected rows $PROT_SHA read in place (read-only) on $NODE for the $SET rescan, manifest $(sha "$O/training-manifest.json"), mirror $SRC" >>"$ACCESS"
  mapfile -t SCAN_ARGS < <(spec scan)
  log "scan start: protected $PROT_SHA (image $IMG, --network none, no GPU)"
  docker run --rm --network none -v "$S:$S:ro" "${MOUNTS[@]}" -v "$(dirname "$PROT"):$(dirname "$PROT"):ro" -v "$O:$O" \
    -e PYTHONPATH="$S" -e PYTHONDONTWRITEBYTECODE=1 -w "$S" --entrypoint python3 "$IMG" \
    -m v2.eval.sealed.overlap scan --protected "$PROT" --manifest "$O/training-manifest.json" \
    "${SCAN_ARGS[@]}" --output "$O/overlap-receipt.json" --hits "$O/overlap-hits.jsonl" \
    >"$O/scan.log" 2>&1 || { log "scan FAILED to run (see $O/scan.log); no hits"; exit 1; }
  log "scan done: receipt $(sha "$O/overlap-receipt.json"), hits $(sha "$O/overlap-hits.jsonl")"
  exit 0
fi

if [ "$CMD" = hf-delta ]; then
  H="/data/dev2/private/c1-rescan-hf-delta/${SET//./_}-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$H"
  log "hf-delta start: $(spec hf_delta/repo) since $(spec hf_delta/base), output $H"
  (cd "$S" && PYTHONPATH="$S" python3 -m v2.eval.sealed.coverage hf-delta --repo "$(spec hf_delta/repo)" \
    --repo-type "$(spec hf_delta/repo_type)" --base "$(spec hf_delta/base)" \
    --token-file /root/.cache/huggingface/token --output "$H/files" --receipt "$H/hf-delta-receipt.json") \
    >"$H.log" 2>&1 || { log "hf-delta FAILED (see $H.log)"; exit 1; }
  log "hf-delta done: $H/files, receipt $(sha "$H/hf-delta-receipt.json")"
  exit 0
fi

[ "$CMD" = judge ] || { echo "unknown command $CMD" >&2; exit 2; }
[ "${#SCANS[@]}" -gt 0 ] || { echo "--scan NODE=DIR is required" >&2; exit 2; }
[ "$(sha "$BASE")" = "$BASE_SHA" ] || { log "refused: event-2 hits differ from $BASE_SHA"; exit 1; }
[ "$(sha "$RETIRED")" = "$RETIRED_SHA" ] || { log "refused: the $SET retired list differs from $RETIRED_SHA"; exit 1; }
J="$W/judge-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$J"
JARGS=()
for item in "${SCANS[@]}"; do
  node=${item%%=*} dir=${item#*=}
  (cd "$S" && PYTHONPATH="$S" python3 -m v2.eval.sealed.scanverdict extract --hits "$dir/overlap-hits.jsonl" \
    --receipt "$dir/overlap-receipt.json" --baseline "$BASE" --node "$node" --output "$J/EXTRACT-$node.json") |
    tee -a "$W/OPERATIONS.log"
  JARGS+=(--extract "$J/EXTRACT-$node.json" --coverage "$node=$dir/coverage-receipt.json")
done
set +e
(cd "$S" && PYTHONPATH="$S" python3 -m v2.eval.sealed.scanverdict judge "${JARGS[@]}" --baseline "$BASE" \
  --retired "$RETIRED" --retired-sha "$RETIRED_SHA" --protected-sha "$PROT_SHA" --output "$J/SCAN-VERDICT.json") |
  tee -a "$W/OPERATIONS.log"
rc=${PIPESTATUS[0]}
set -e
cp "$J/SCAN-VERDICT.json" "$W/SCAN-VERDICT.json"
log "verdict $([ "$rc" = 0 ] && echo PASS || echo FAIL) for $SET: $(sha "$W/SCAN-VERDICT.json")"
exit "$rc"
