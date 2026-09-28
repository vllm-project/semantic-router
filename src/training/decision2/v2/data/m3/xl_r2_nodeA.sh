#!/bin/bash
# M3b XL recipe revision r2 on node A (CPU only): rescreen of the r1 rows, then the r2 build.
#
#   xl_r2_nodeA.sh CODE RUN rows|scan|scan-union|rescreen|targets|build|check|upload|readback
#
# CODE = src/training/decision2 of an exact mirror; RUN = output root (rescreen/, targets/,
# build/). Rules: records/m3b-prereg-amendment-3-2026-09-28.md §2-§3 and amendment 2 §4.
# Optional environment: PARALLEL scans at a time (default 4) with WORKERS processes each
# (default 16; scan-union 64); TREE (the code tree hash, required by build).
set -euo pipefail
CODE=$1
RUN=$2
STEP=$3
B=/data/dev2/runs/data/m3b
REPO=llm-semantic-router/decision-2.0-training-data
R1=$B/upload-xl/m3/mixtures/xl
R1_REV=ba848147b0efdd3e2b9531f99f8930d7a9f364aa
R1_MANIFEST_SHA=04a07531bea4f3ba39e5d7719faeab433676464cccff8581772a6b2ea512d655
R1_ARGS=(--r1-dir "$R1" --r1-manifest-sha256 "$R1_MANIFEST_SHA")
POOLS=$B/xl-pools.json
POOLS_SHA=7c3391dc65810f9951b8056c082c2f5d5cef03fe1eecf5739a4f010b5546d86b
PIQ=$B/gap/c1/pi/manifest.quarantining.json
PIQ_SHA=48fd2537e63b1069a3569557bf9212983384c8c800c7e229084d339f4e0e954d
F=$B/gap/c2/final
GAP_REV=09f73967bc21b2b1e27160397272b7f66a1ef3af
GAP=(--gap "H7=$F/h7.train.jsonl,$F/h7.train.tokens.jsonl"
  --gap "H8=$F/h8.train.jsonl,$F/h8.train.tokens.jsonl")
LUX_XL_REV=0ce4ca604cff506edbb69117f62df975e6dc0e6a
T=$RUN/targets/m3/teachers/lux1/xl
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
rescreen)
  nice -n 10 python3 -m v2.data.m3.xl_r2 rescreen --rescreen-dir "$RUN/rescreen" "${R1_ARGS[@]}" \
    --inventory-sha256 "$PIQ_SHA" --private "$RUN/rescreen/rescreen.private.json" \
    --public "$RUN/rescreen/rescreen.public.json"
  ;;
targets)
  # The published own-Lux XL waves w1-w5 at the w5 revision (w5 is a plain git blob).
  mkdir -p "$RUN/targets" && chmod 700 "$RUN/targets"
  HF_HUB_CACHE=/data/dev2/hf-cache hf download "$REPO" --repo-type dataset --revision "$LUX_XL_REV" \
    --include 'm3/teachers/lux1/xl/w*.targets.jsonl' --local-dir "$RUN/targets" > "$RUN/targets/download.log" 2>&1
  while read -r K SHA; do expect "$T/w$K.targets.jsonl" "$SHA"; done <<EOF
1 4a852b74794801e061facf42911f2a26434af8083a773268a481a1a481aebc7f
2 a659c3e59774387f6f9f008a55182c374e321685f57ea3dcc0fe3c5fe5a1f2f2
3 6d141c96b54a9548b4572c04c4fbd30e413b168a43d583f3086000413a207c73
4 b8ae13749243a7e59c191bf1e6f81d3fff0ea5667a507b086426ae6568343a22
EOF
  BLOB=$(python3 -c "import hashlib, sys; d = open(sys.argv[1], 'rb').read(); \
print(hashlib.sha1(b'blob %d\0' % len(d) + d).hexdigest())" "$T/w5.targets.jsonl")
  [ "$BLOB" = 640da791fd91324247a3a3922f2bb9fb7fc28c2c ] || {
    echo "w5 blob mismatch" >&2
    exit 1
  }
  sha256sum "$T"/w*.targets.jsonl > "$RUN/targets/sha256.txt"
  ;;
build)
  expect "$F/h7.train.jsonl" 7c4133b05664de492bb80b40df5eef0f028257f10327a492564bea5f62813a32
  expect "$F/h8.train.jsonl" 1f19e5ab84e6b80182e99cd8e2b7efc79ec17275048e6daf7987afc335e7d361
  expect "$F/h7.train.tokens.jsonl" 5c48a2c6bf635edbfa6036453302bd9a92d40e102090a35287be2370f5271c59
  expect "$F/h8.train.tokens.jsonl" 7c10a12790b01466409991ea361b45519c5e4ea216fa0780feba89a6275ac150
  COMMIT=$(basename "${CODE%/src/training/decision2}" | cut -d- -f1)
  # The target files r1 counted (its coverage is reproduced first), then the XL waves.
  H=/data/dev2/runs/data/m3a/hf/m2/teachers/lux1/rp-v2
  J=/data/dev2/runs/data/m3a2
  LUX_R1=$B/a0s-strict/lux1.jsonl,$H/wave1.targets.jsonl,$H/wave2.targets.jsonl,$H/wave3.targets.jsonl
  LUX_R1=$LUX_R1,/data/dev2/runs/data/m3a/pk1/lux-wave4/wave4.targets.jsonl
  AJ_R1=$B/a0s-strict/autojev27.jsonl,$J/upload-aj-m/rp-v2/aj-m.targets.jsonl,$J/upload-aj-sl/rp-v2/aj-sl.targets.jsonl
  LUX_XL=$T/w1.targets.jsonl,$T/w2.targets.jsonl,$T/w3.targets.jsonl,$T/w4.targets.jsonl,$T/w5.targets.jsonl
  PENDING=()
  for W in c-w1 c-w2; do
    if [ -f "$B/lux-xl/lux-xl-$W.rows.jsonl" ]; then
      PENDING+=(--pending "lux1=lux-xl-$W=$B/lux-xl/lux-xl-$W.rows.jsonl")
    fi
  done
  nice -n 10 python3 -m v2.data.m3.xl_r2 build --pools "$POOLS" "${GAP[@]}" "${R1_ARGS[@]}" \
    --r1-revision "$R1_REV" --gap-revision "$GAP_REV" --rescreen "$RUN/rescreen/rescreen.private.json" \
    --out-dir "$RUN/build" --targets "lux1=$LUX_R1" --targets "autojev27=$AJ_R1" \
    --extra-targets "lux1=$LUX_XL" "${PENDING[@]}" --code-commit "$COMMIT" --code-tree "${TREE:?}" \
    > "$RUN/build.stdout"
  ;;
check)
  nice -n 10 python3 -m v2.data.m3.xl_r2 check --pools "$POOLS" --out-dir "$RUN/build" "${R1_ARGS[@]}" \
    --rescreen "$RUN/rescreen/rescreen.private.json" --rows-dir "$RUN/rescreen/rows" "${GAP[@]}" \
    --report "$RUN/build/check.json"
  ;;
upload)
  # m3/mixtures/xl-r2/: ids, missing lists, manifest, checks, public receipts and README.
  [ ! -e "$RUN/upload" ] || {
    echo "upload folder exists" >&2
    exit 1
  }
  X=$RUN/upload/m3/mixtures/xl-r2
  mkdir -p "$X/rescreen" && chmod -R 700 "$RUN/upload"
  cp "$RUN"/build/*.jsonl "$RUN/build/mx-xl-r2.manifest.json" "$RUN/build/check.json" \
    "$RUN/rescreen/rescreen.public.json" "$X/"
  for P in "$S"/*.public.json; do cp "$P" "$X/rescreen/$(basename "$P" .public.json).overlap.public.json"; done
  cp "$RUN/rescreen/scan-union/union.public.json" "$X/rescreen/union.overlap.public.json"
  cp "$CODE/v2/data/records/hf-xl-r2-readme.md" "$X/README.md"
  if grep -rlE '(^|[^[:alnum:]_.-])/(data|home|tmp)/' "$X"; then
    echo "path leak" >&2
    exit 1
  fi
  (cd "$RUN/upload" && find m3 -type f | LC_ALL=C sort | xargs sha256sum) > "$RUN/upload.sha256"
  HF_HUB_CACHE=/data/dev2/hf-cache hf upload "$REPO" "$X" m3/mixtures/xl-r2 --repo-type dataset \
    --commit-message "M3b XL recipe revision r2: rescreened r1 + gap arms H7/H8" > "$RUN/upload.log" 2>&1
  grep -oE 'commit/[0-9a-f]{40}' "$RUN/upload.log" | head -1 | cut -d/ -f2 > "$RUN/upload.revision"
  ;;
readback)
  REV=$(cat "$RUN/upload.revision")
  HF_HUB_CACHE=/data/dev2/hf-cache hf download "$REPO" --repo-type dataset --revision "$REV" \
    --include 'm3/mixtures/xl-r2/*' --local-dir "$RUN/readback" > "$RUN/readback.log" 2>&1
  (cd "$RUN/readback" && find m3 -type f | LC_ALL=C sort | xargs sha256sum) > "$RUN/readback.sha256"
  diff "$RUN/upload.sha256" "$RUN/readback.sha256"
  echo "readback equal at $REV: $(wc -l < "$RUN/readback.sha256") files"
  ;;
*)
  echo "unknown step $STEP" >&2
  exit 2
  ;;
esac
