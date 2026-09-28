#!/bin/bash
# M3b gap sources on node A (CPU only): NQ extraction, PI-v4, H7/H8 builds, audits, finalize, freeze.
#
#   gap_nodeA.sh CODE RUN extract|protected|a7k|build|audit|finalize|stats|freeze [ARM]
#
# CODE = src/training/decision2 of an exact mirror; RUN = output root (pi/, build/,
# audits/<arm>/, final/). Sources: /data/dev2/private/sources (m3b/ for the new
# downloads); the rules are in records/m3b-gap-sources-2026-09-28.md and
# records/m3b-prereg-amendment-2-2026-09-28.md. Optional environment: NQ_EXTRACT
# (extract output directory, default the sources copy the build reads), FINAL
# (finalize/stats/freeze directory under RUN, default final), EMBED_RECEIPT (a private
# embedding-scan receipt whose quarantined groups finalize removes) and EMBED_PUBLIC
# (its public receipt, summarized by stats).
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
A7=/data/dev2/runs/data/m3b/hf/v2/a7/arms
V2=/data/dev2/private/data/arms-v2/final
F=$RUN/${FINAL:-final}
DOCKER=(docker run --rm --network none -v /data:/data -w "$CODE" --entrypoint python3 "$IMG")
REPORT_ONLY=(rights_clean_train v1_train_a1 v1_train_a2 v1_train_a3 v1_train_a6g v1_train_a5 v1_train_a6h
  v1_train_a4v2h v1_train_a4v2r v2_aho_E11 v2_aho_G2 v2_aho_G4h v2_aho_G4r v2_aho_G6 v2_aho_H1 v2_aho_H3
  v2_aho_H5 v2_aho_H6 a7_train_A7g a7_train_A7h a7_train_A7i a7_train_A7k a7_train_A7m a7_train_A7o
  a7_train_A7p a7_train_A7q a7_train_A7r a7_train_A7s a7_train_A7x)
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
  OUT=${NQ_EXTRACT:-$SRC/m3b/nq-extract-64}
  mkdir -p "$OUT"
  "${DOCKER[@]}" -m v2.data.m3.src_gap extract-nq --shards "$SRC/m3b/nq/default" \
    --out "$OUT/nq-train.jsonl" --report "$OUT/nq-train.report.json" --workers 64
  ;;
protected)
  # PI-v4: PI-v3 plus the A7 held-out slices (quarantining), A7 TRAIN and the v2 AHO slices
  # (report-only); every origin must match its registry SHA-256.
  ADD=()
  while read -r ROLE FILE SHA; do
    ADD+=(--add "$ROLE=$FILE" --expect "$ROLE=$SHA")
  done <<EOF
a7_aho_A7g $A7/A7g/aho.jsonl 25f31a2edad5a6768104a5de5bab993bc41388cc97dd0a254b750737baba2580
a7_aho_A7h $A7/A7h/aho.jsonl db7bba1a2f568bec430a069a6a48eb50639b59c1fe5783d9b6ff1fe060432b84
a7_aho_A7i $A7/A7i/aho.jsonl 78f580f62b7784a12fb2841774f85d5e6b4a7bc9aa22847c90fc939cbefea2d6
a7_aho_A7k $A7/A7k/aho.jsonl 0be22b8c6b0e243bc84fb52217d1e9ddf06c0ca5aff246912d9c5c5dbbf847d8
a7_aho_A7m $A7/A7m/aho.jsonl 1aa5d174b1f7c6981739b82f5fb3c743af6e274aa557e1d2404674dbec77fb1c
a7_aho_A7o $A7/A7o/aho.jsonl 00986778604daf87906b29eef9d4e48622868fdc9a9fcfad1987eb0f0897a4a0
a7_aho_A7p $A7/A7p/aho.jsonl 19ab727c9ec76c9ed262813c01f0e651a3fa23ea24a0666adb4827e17e6db740
a7_aho_A7q $A7/A7q/aho.jsonl f04557e25bcbeb6bf31db07cf19b7f2ca3caa62c7dc55251f90509eaac2e7250
a7_aho_A7r $A7/A7r/aho.jsonl d9aff24b30eae99834e815a6ec9fa4601faea5e651232a56edb680f74dda01e5
a7_aho_A7s $A7/A7s/aho.jsonl 86d3bc73ccf730af74b4b63fa1235f3d8ecaee44a8f4bdc182ee4711d3113fa3
a7_aho_A7x $A7/A7x/aho.jsonl 5cba8f8c46a2f1453292251842928941d3ec0aee9a885c18c996289772208de7
a7_train_A7g $A7/A7g/train.jsonl cc87e0b7a441774aaacbd34b59425294bf8fd5f1b678f92c9a15fd008dcdfddd
a7_train_A7h $A7/A7h/train.jsonl 86474b739d8297ebb98ca6ea818c7f6c3cef153a1d778108e6a6723336812718
a7_train_A7i $A7/A7i/train.jsonl 0b8284b1e08229698e9516be7104fe2b6f7dad79ec25eddc3cd3c1ad48e72e12
a7_train_A7k $A7/A7k/train.jsonl 8aab25e53e6f3c52e91f0bb45647ec448ba7b408491098285162abddb42577b9
a7_train_A7m $A7/A7m/train.jsonl dd9b1ef029f87e9f7c33165002b5dc74e26776ae28edabe503ef85fd7df9b692
a7_train_A7o $A7/A7o/train.jsonl dbba002225188fb4dca8f45d15a4801c41a8964d313ee49eba335e2af6355625
a7_train_A7p $A7/A7p/train.jsonl cc9217e6128da1910784cb1839127a3edf7e5b90d5c795cd104d78f35f2f137e
a7_train_A7q $A7/A7q/train.jsonl c65aac85a7399ebde36039d9de7b8945d8570909e61a3f4a69a7ad5001ac7bfd
a7_train_A7r $A7/A7r/train.jsonl 0fc421e4937146ae85d66b9c3a039fd2e5b1f344b9aad4bad77d9ea3983a534d
a7_train_A7s $A7/A7s/train.jsonl f7eef149493d83e93cca1d5128132477412bbd99239832939fd77a80fb43e6ec
a7_train_A7x $A7/A7x/train.jsonl 296ae2f8c897dafa6e757e5b73c5819bc0218d5cc4e188291da372de8c13b275
v2_aho_E11 $V2/E11.aho.jsonl 833faf89e5b7cb3f5463798470caf51c9d3d618e5cd5c5334c3baec8717d07b7
v2_aho_G2 $V2/G2.aho.jsonl aa6254cbe1dd23562fe26661c6f24f1f828a8df5bd5552e3988399ba7792287c
v2_aho_G4h $V2/G4h.aho.jsonl a1d8fc571213fa4c0602e44b0030fafd87de0e0d7306763a4632899fb734977a
v2_aho_G4r $V2/G4r.aho.jsonl 7ec218308f8c88daaed7524888242572fc8e4faa3c16eab93b3ead3e23336b2d
v2_aho_G6 $V2/G6.aho.jsonl 4a78dade0d33b72d4f14ed73667542d7d8fc9aff5d7309a926cd1d693ac02923
v2_aho_H1 $V2/H1.aho.jsonl 79d12b6ae34b017461dd546e7408bb6cc73ae590aa41a4d6d3453d656632834a
v2_aho_H3 $V2/H3.aho.jsonl 764975c4d054bf7d2ad0fff8ac3ece42351d932cb5342cd6897fe4162639d9cd
v2_aho_H5 $V2/H5.aho.jsonl 4001fefb18195ca7544c76874d44f87829f29ad5dca28a383b5f0c543ee9a755
v2_aho_H6 $V2/H6.aho.jsonl e7223148fd47a0a8291d11c6f90933f1bada861b327bf9f78ef3d04ae80b48ba
EOF
  RO=()
  for R in "${REPORT_ONLY[@]}"; do RO+=(--report-only "$R"); done
  python3 -m v2.data.m3.src_gap protected --base /data/dev2/private/data/pi-v3/manifest.json \
    --base-sha256 fc09b2bd1fecf823b3fbd6218d1bee76109677474608d19f17e9c2aff080b7f2 "${ADD[@]}" "${RO[@]}" \
    --out-dir "$RUN/pi"
  ;;
a7k)
  python3 -m v2.data.m3.src_gap pair-check --pairs "$A7/A7k/train.jsonl" --pairs "$A7/A7k/aho.jsonl" \
    --against "$V2/H6.aho.jsonl" --against "$V2/H6.sho.jsonl" --out "$RUN/a7k-pairs.json"
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
  mkdir -p "$F" && chmod 700 "$F"
  for CELL in "$A"/cells/*.jsonl; do
    [ -f "$CELL.shortcut.json" ] || { echo "no shortcut receipt for $CELL" >&2; exit 1; }
  done
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
  EMBED=()
  [ -z "${EMBED_RECEIPT:-}" ] || EMBED=(--embed-receipt "$EMBED_RECEIPT")
  for S in train aho sho; do
    python3 -m v2.data.apply_quarantine --rows "$A/$ARM.$S.q.jsonl" "${DROP[@]}" "${EMBED[@]}" \
      --out "$F/$ARM.$S.g.jsonl" --report "$F/$ARM.$S.gates.json" > /dev/null
  done
  cp "$F/$ARM.train.g.jsonl" "$F/$ARM.train.jsonl"
  for S in aho sho; do
    python3 -m v2.data.m2.dedup_heldout --train "$F/$ARM.train.jsonl" --heldout "$F/$ARM.$S.g.jsonl" \
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
stats)
  EP=()
  [ -z "${EMBED_PUBLIC:-}" ] || EP=(--embed-public "$EMBED_PUBLIC")
  python3 -m v2.data.m3.src_gap stats --arm "$ARM" --final "$F" --audits "$RUN/audits/$ARM" "${EP[@]}" \
    --out "$F/$ARM.stats.json"
  ;;
freeze)
  for S in train aho; do
    "${DOCKER[@]}" -m v2.data.freeze freeze --rows "$F/$ARM.$S.jsonl" --arm-id "${ARM^^}" --role "$S" \
      --license-registry "$CODE/v2/data/records/license-registry-m3b.json" --tokenizers "$TOKENIZERS" \
      --out-manifest "$F/$ARM.$S.manifest.json" > "$F/$ARM.$S.freeze.stdout"
  done
  ;;
*)
  echo "unknown step $STEP" >&2
  exit 2
  ;;
esac
