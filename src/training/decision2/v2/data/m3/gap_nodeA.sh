#!/bin/bash
# M3b gap sources on node A (CPU only): NQ extraction, H7/H8 builds, audits, finalize.
#
#   gap_nodeA.sh CODE RUN extract|build|audit|finalize [ARM]
#
# CODE = src/training/decision2 of an exact mirror; RUN = output root (build/,
# audits/<arm>/, final/). Sources: /data/dev2/private/sources (m3b/ for the new
# downloads); the rules are in records/m3b-gap-sources-2026-09-28.md.
set -euo pipefail
CODE=$1
RUN=$2
STEP=$3
ARM=${4:-}
IMG=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
SRC=/data/dev2/private/sources
TOK=/data/dev2/hf-cache/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
TOKENIZERS=/data/dev2/private/data/arms-v1/tokenizers.json
PI=$RUN/pi/manifest.json
EXISTING=$RUN/existing.json
V1_ROWS=/data/dev2/private/data/arms-v2/v1-rows.json
DOCKER=(docker run --rm --network none -v /data:/data -w "$CODE" --entrypoint python3 "$IMG")
REPORT_ONLY=(rights_clean_train v1_train_a1 v1_train_a2 v1_train_a3 v1_train_a6g v1_train_a5 v1_train_a6h
  v1_train_a4v2h v1_train_a4v2r v2_aho_E11 v2_aho_G2 v2_aho_G4h v2_aho_G4r v2_aho_G6 v2_aho_H1 v2_aho_H3
  v2_aho_H5 v2_aho_H6)
cd "$CODE"

tokens() {  # tokens OUT ROWS...
  local out=$1
  shift
  local args=()
  for rows in "$@"; do args+=(--rows "$rows"); done
  "${DOCKER[@]}" -m v2.data.m2.row_tokens --tokenizers "$TOKENIZERS" --native qwen3.5-0.8b-base@dc7cdfe2 \
    --raw kai-0.6b@7185f514 "${args[@]}" --out "$out"
}

case $STEP in
extract)
  mkdir -p "$SRC/m3b/nq-extract-64"
  "${DOCKER[@]}" -m v2.data.m3.src_gap extract-nq --shards "$SRC/m3b/nq/default" \
    --out "$SRC/m3b/nq-extract-64/nq-train.jsonl" --report "$SRC/m3b/nq-extract-64/nq-train.report.json" --workers 64
  ;;
build)
  mkdir -p "$RUN/build"
  "${DOCKER[@]}" -m v2.data.m3.src_gap build --arm "$ARM" --sources "$SRC" --tokenizer "$TOK" --existing "$EXISTING" \
    --v1-rows "$V1_ROWS" --out-dir "$RUN/build" --workers 24
  ;;
audit)
  A=$RUN/audits/$ARM
  mkdir -p "$A" && chmod 700 "$A"
  [ -f "$A/build.tokens.jsonl" ] || tokens "$A/build.tokens.jsonl" "$RUN/build/$ARM".{train,aho,sho}.jsonl > "$A/tokens.stdout"
  for S in train aho sho; do
    [ -f "$A/$ARM.$S.b.jsonl" ] || python3 -m v2.data.m3.src_gap budget --rows "$RUN/build/$ARM.$S.jsonl" \
      --tokens "$A/build.tokens.jsonl" --out "$A/$ARM.$S.b.jsonl" --report "$A/$ARM.$S.budget.json" > /dev/null
  done
  [ -f "$A/overlap.private.json" ] || python3 -m v2.data.overlap --candidates "$A/$ARM.train.b.jsonl" \
    --candidates "$A/$ARM.aho.b.jsonl" --candidates "$A/$ARM.sho.b.jsonl" --protected-inventory "$PI" \
    --private-receipt "$A/overlap.private.json" --public-receipt "$A/overlap.public.json" --workers 48 \
    > "$A/overlap.stdout" 2> "$A/overlap.stderr"
  RO=()
  for R in "${REPORT_ONLY[@]}"; do RO+=(--report-only-role "$R"); done
  for S in train aho sho; do
    [ -f "$A/$ARM.$S.q.jsonl" ] || python3 -m v2.data.apply_quarantine --rows "$A/$ARM.$S.b.jsonl" \
      --overlap-receipt "$A/overlap.private.json" "${RO[@]}" --out "$A/$ARM.$S.q.jsonl" \
      --report "$A/$ARM.$S.quarantine.json" > /dev/null
  done
  [ -d "$A/cells" ] || python3 -m v2.data.m2.audit_cells --rows "$A/$ARM.train.q.jsonl" --out-dir "$A/cells" > "$A/cells.json"
  ls "$A"/cells/*.jsonl | xargs -P 16 -I{} sh -c \
    '[ -f {}.shortcut.json ] || python3 -m v2.data.shortcut --rows {} --receipt {}.shortcut.json --workers 8 > {}.shortcut.stdout 2>&1 || true'
  [ -f "$A/length-baseline.json" ] || python3 -m v2.data.m3.src_gap length-baseline --cells "$A/cells" \
    --out "$A/length-baseline.json" > /dev/null
  ;;
finalize)
  A=$RUN/audits/$ARM
  F=$RUN/final
  mkdir -p "$F" && chmod 700 "$F"
  DROP_ARGS=$(python3 -c "
import glob, json, os
failed = set()
for path in glob.glob('$A/cells/*.jsonl.shortcut.json'):
    receipt = json.load(open(path))
    if receipt['verdict'] != 'PASS':
        failed.add(os.path.basename(path).split('__', 1)[1].split('.jsonl')[0])
print(' '.join(f'--drop-family {name}' for name in sorted(failed)))
")
  read -r -a DROP <<< "$DROP_ARGS"
  for S in train aho sho; do
    python3 -m v2.data.apply_quarantine --rows "$A/$ARM.$S.q.jsonl" "${DROP[@]}" --out "$A/$ARM.$S.g.jsonl" \
      --report "$A/$ARM.$S.gates.json" > /dev/null
  done
  cp "$A/$ARM.train.g.jsonl" "$F/$ARM.train.jsonl"
  for S in aho sho; do
    python3 -m v2.data.m2.dedup_heldout --train "$F/$ARM.train.jsonl" --heldout "$A/$ARM.$S.g.jsonl" \
      --out "$F/$ARM.$S.jsonl" --report "$F/$ARM.$S.dedup.json" > /dev/null
  done
  tokens "$F/$ARM.tokens.jsonl" "$F/$ARM".{train,aho,sho}.jsonl > "$F/$ARM.tokens.stdout"
  PARTS=(--partition "$ARM/train=$F/$ARM.train.jsonl" --partition "$ARM/aho=$F/$ARM.aho.jsonl"
    --partition "$ARM/sho=$F/$ARM.sho.jsonl")
  mapfile -t EXISTING_PATHS < <(python3 -c "import json, sys; print('\n'.join(json.load(open(sys.argv[1]))))" "$EXISTING")
  for P in "${EXISTING_PATHS[@]}"; do
    NAME=$(echo "$P" | sed -e 's#/data/dev2/##' -e 's#/#_#g')
    PARTS+=(--partition "existing/$NAME=$P")
  done
  python3 -m v2.data.freeze isolation "${PARTS[@]}" --report "$F/$ARM.isolation.json" > "$F/$ARM.isolation.stdout" 2>&1 || true
  ;;
*)
  echo "unknown step $STEP" >&2
  exit 2
  ;;
esac
