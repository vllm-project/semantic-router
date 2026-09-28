#!/bin/bash
# M3b XL recipe revision r2 on node A (CPU only): rescreen of the r1 rows, then the r2 build.
#
#   xl_r2_nodeA.sh CODE RUN rows|scan|scan-union
#
# CODE = src/training/decision2 of an exact mirror; RUN = output root (rescreen/). Rules:
# records/m3b-prereg-amendment-3-2026-09-28.md §2-§3 and amendment 2 §4. Optional
# environment: PARALLEL scans at a time (default 4) with WORKERS processes each (default 16).
set -euo pipefail
CODE=$1
RUN=$2
STEP=$3
B=/data/dev2/runs/data/m3b
R1=$B/upload-xl/m3/mixtures/xl
R1_MANIFEST_SHA=04a07531bea4f3ba39e5d7719faeab433676464cccff8581772a6b2ea512d655
POOLS=$B/xl-pools.json
POOLS_SHA=7c3391dc65810f9951b8056c082c2f5d5cef03fe1eecf5739a4f010b5546d86b
PIQ=$B/gap/c1/pi/manifest.quarantining.json
PIQ_SHA=48fd2537e63b1069a3569557bf9212983384c8c800c7e229084d339f4e0e954d
S=$RUN/rescreen/scan
cd "$CODE"

expect() { # expect FILE SHA256
  [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ] || {
    echo "SHA-256 mismatch: $1" >&2
    exit 1
  }
}

case $STEP in
rows)
  expect "$POOLS" "$POOLS_SHA"
  mkdir -p "$RUN" && chmod 700 "$RUN"
  nice -n 10 python3 -m v2.data.m3.xl_r2 rows --pools "$POOLS" --r1-dir "$R1" \
    --r1-manifest-sha256 "$R1_MANIFEST_SHA" --out-dir "$RUN/rescreen" --workers 16
  ;;
scan)
  # One v2.data.overlap scan per pool against PI-v4's quarantining roles only, with the
  # methods and thresholds of every arm's audit; largest candidate files first.
  expect "$PIQ" "$PIQ_SHA"
  mkdir -p "$S" && chmod 700 "$S"
  export PIQ S
  echo "{\"event\": \"start\", \"utc\": \"$(date -u +%FT%TZ)\"}" >> "$S/wall.jsonl"
  # shellcheck disable=SC2016
  ls -S "$RUN"/rescreen/rows/*.jsonl | xargs -P "${PARALLEL:-4}" -I{} bash -c '
    N=$(basename "$1" .jsonl)
    [ -f "$S/$N.private.json" ] && exit 0
    T0=$(date +%s)
    nice -n 10 python3 -m v2.data.overlap --candidates "$1" --protected-inventory "$PIQ" \
      --private-receipt "$S/$N.private.json" --public-receipt "$S/$N.public.json" \
      --workers "$2" > "$S/$N.stdout" 2> "$S/$N.stderr"
    echo "{\"pool\": \"$N\", \"start\": $T0, \"end\": $(date +%s)}" >> "$S/times.jsonl"
  ' _ {} "${WORKERS:-16}"
  echo "{\"event\": \"end\", \"utc\": \"$(date -u +%FT%TZ)\"}" >> "$S/wall.jsonl"
  ;;
scan-union)
  # The same scan with the whole union as one candidate set, so boilerplate counts every
  # pool's groups; the rescreen flags the union of both passes.
  expect "$PIQ" "$PIQ_SHA"
  U=$RUN/rescreen/scan-union
  mkdir -p "$U" && chmod 700 "$U"
  CANDS=()
  for F in "$RUN"/rescreen/rows/*.jsonl; do CANDS+=(--candidates "$F"); done
  echo "{\"event\": \"start\", \"utc\": \"$(date -u +%FT%TZ)\"}" >> "$U/wall.jsonl"
  [ -f "$U/union.private.json" ] || nice -n 10 python3 -m v2.data.overlap "${CANDS[@]}" \
    --protected-inventory "$PIQ" --private-receipt "$U/union.private.json" \
    --public-receipt "$U/union.public.json" --workers "${WORKERS:-64}" > "$U/union.stdout" 2> "$U/union.stderr"
  echo "{\"event\": \"end\", \"utc\": \"$(date -u +%FT%TZ)\"}" >> "$U/wall.jsonl"
  ;;
*)
  echo "unknown step $STEP" >&2
  exit 2
  ;;
esac
