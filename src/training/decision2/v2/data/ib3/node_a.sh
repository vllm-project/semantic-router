#!/bin/bash
# IB3 build, audits, reviews and publication on node A (prereg records/ib3-prereg-2026-10-01.md §2-§5).
# CPU only, in the pinned image with --network none; the mirror and every input are mounted read-only, only the
# step's output directory is writable. Every stage refuses to overwrite. Adapted from v2/data/ib2/node_a.sh.
#
#   node_a.sh <commit> build          candidates from the pinned raw files
#   node_a.sh <commit> scans          G0 and G0u (reference, scan, positive controls); PI-ib3 inventory; overlap vs
#                                     PI-v4 (full, quarantining) and PI-ib3; DEV-vs-TRAIN self-scan; G1; G3 names
#   node_a.sh <commit> pass1          quarantine lists and finalize pass 1
#   node_a.sh <commit> rescan <n>     re-scan pass <n> (G0, G0u, overlap, quarantine, C1 names) into rescan<n>/
#   node_a.sh <commit> pass <n>       finalize pass <n> (n >= 2) with every list up to rescan<n-1>; G4 on it when
#                                     n is IB3_SAMPLE_PASS
#   node_a.sh <commit> screen         stage S sample and packets (G4-failing families left out)
#   node_a.sh <commit> screen-score   stage S verdict from screen/answers/s1.*.jsonl
#   node_a.sh <commit> review         stage R sample and packets (screened rows / groups and dropped families out)
#   node_a.sh <commit> splits         R3 packet from review/answers/{r1,r2}.*.jsonl
#   node_a.sh <commit> score          stage R verdict (review/answers/r3.jsonl if any)
#   node_a.sh <commit> final          finalize, freeze, isolation (G7), tokens and stats (G5, G8)
#   node_a.sh <commit> leak           ids of final rows with a leak-guard finding; a new `final` drops them
#   node_a.sh <commit> hf-assemble    m6/ib3 upload tree (registry.json inside) and the leak guard
#   node_a.sh <commit> hf-upload      private upload with the HF CLI, pinned revision, read-back check
set -euo pipefail
umask 077
SHA=$1
STAGE=$2
MIRROR=/data/dev2/src/${SHA}-src_training_decision2
CODE=$MIRROR/src/training/decision2
H=/data/dev2/private/data/ib3
RAW=$H/raw
RUN=${IB3_RUN:-d1}
R=$H/$RUN
CAND=$R/cand
T=/data/dev2/tmp/ib3-$RUN
IMG=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
SUITE=/data/dev2/private/eval/index021/suite-0.2
PI3=/data/dev2/private/data/pi-v3
PI4=/data/dev2/runs/data/m3b/gap/c1/pi
P=/data/dev2/private/panels/goldfree
HFD=/data/dev2/hf-cache/datasets--llm-semantic-router--decision-2.0-training-data
PN1DEV=$HFD/snapshots/27b1d2f130292268b43a618584bebab5d4e4a6b5/m4/pn1/dev
CAL698=$HFD/snapshots/ed87a03ab80ca5b9560780bba51a83a77ff47d14/m2/cal/CAL698/cal.jsonl
HR2DEV=$HFD/snapshots/afc3bc1e1d6849058f6fafdfbc3dfe007d067400/m5/hr2/hr2.dev.jsonl
HS1DEV=/data/dev2/private/data/hs1/21bdb5e90e2b/build/hs1.dev.jsonl
IB1DEV=/data/dev2/private/data/ib1/b2/final/out/ib1.dev.jsonl
IB1DEV3=/data/dev2/private/data/ib1/r3/final/out/ib1.dev.jsonl
IB2DEV=/data/dev2/private/data/ib2/c3/final/out/ib2.dev.jsonl
TOKJ=/data/dev2/private/data/arms-v1/tokenizers.json
Q06=/data/dev2/hf-cache/models--Qwen--Qwen3-0.6B-Base
Q08=/data/dev2/hf-cache/models--Qwen--Qwen3.5-0.8B-Base
KAI=/data/dev2/private/models/Decision-1.0-Kai-0.6B@7185f514f54b8f93c55998b1e8f9c5cc67f0d029/native/tokenizer
LIC=$CODE/v2/data/records/license-registry-ib3.json
TRAIN=$CAND/ib3.train.cand.jsonl
DEV=$CAND/ib3.dev.cand.jsonl
WORKERS=${IB3_WORKERS:-48}
SAMPLE_PASS=${IB3_SAMPLE_PASS:-2}

[ -d "$CODE" ] || { echo "no mirror $MIRROR" >&2; exit 1; }
for d in "$H" "$R" "$T" "$R/logs"; do
  [ -d "$d" ] || mkdir -m 700 "$d"
done
BASE=(--rm --network none --entrypoint python3 -w "$CODE" --cpu-shares 256
  -e PYTHONPATH=. -e TMPDIR="$T" -e CUDA_VISIBLE_DEVICES= -e HIP_VISIBLE_DEVICES=
  --label dev2.track=data-ib3 --label "dev2.commit=$SHA"
  -v "$MIRROR:$MIRROR:ro" -v "$T:$T:rw")

fresh() {
  for d in "$@"; do
    [ ! -e "$R/$d" ] || { echo "$R/$d exists" >&2; exit 1; }
    mkdir -m 700 "$R/$d"
  done
}

# run <name> <docker args ...>: one container, output logged; returns the container's exit code.
run() {
  local name=$1 rc=0
  shift
  echo "$(date -u +%FT%TZ) start $name" >> "$R/logs/steps.log"
  docker run --name "ib3-$RUN-$name" "${BASE[@]}" "$@" > "$R/logs/$name.log" 2>&1 || rc=$?
  echo "$(date -u +%FT%TZ) done $name exit=$rc" >> "$R/logs/steps.log"
  return "$rc"
}

build() {
  [ ! -e "$CAND" ] || { echo "$CAND exists" >&2; exit 1; }
  mkdir -m 700 "$CAND"
  run build -v "$RAW:$RAW:ro" -v "$CAND:$CAND:rw" "$IMG" -m v2.data.ib3.build build --raw "$RAW" --out "$CAND"
}

# overlap <dir> <train> <dev> <inventory dir>: the four overlap scans of one pair of files, in parallel.
overlap() {
  local X=$1 tr=$2 dv=$3 I=$4
  local M=(-v "$(dirname "$tr"):$(dirname "$tr"):ro" -v "$X:$X:rw")
  local cand=(--candidates "$tr" --candidates "$dv")
  local tag
  tag=$(basename "$X")
  run "$tag-piv4" "${M[@]}" -v "$PI3:$PI3:ro" -v "$PI4:$PI4:ro" "$IMG" -m v2.data.overlap "${cand[@]}" \
    --protected-inventory "$PI4/manifest.json" --private-receipt "$X/piv4.private.json" \
    --public-receipt "$X/piv4.public.json" --workers "$WORKERS" &
  run "$tag-piv4q" "${M[@]}" -v "$PI3:$PI3:ro" -v "$PI4:$PI4:ro" "$IMG" -m v2.data.overlap "${cand[@]}" \
    --protected-inventory "$PI4/manifest.quarantining.json" --private-receipt "$X/piv4q.private.json" \
    --public-receipt "$X/piv4q.public.json" --workers "$WORKERS" &
  run "$tag-piib3" "${M[@]}" -v "$I:$I:ro" "$IMG" -m v2.data.overlap "${cand[@]}" \
    --protected-inventory "$I/manifest.json" --private-receipt "$X/piib3.private.json" \
    --public-receipt "$X/piib3.public.json" --workers "$WORKERS" &
  run "$tag-self" "${M[@]}" "$IMG" -m v2.data.overlap --self-scan --candidates "$dv" \
    --candidates "$tr" --private-receipt "$X/self.private.json" \
    --public-receipt "$X/self.public.json" --workers "$WORKERS" &
}

# quarantine <scan dir> <lists dir> <train> <dev>: the quarantine lists of one scan directory.
quarantine() {
  local X=$1 Q=$2 tr=$3 dv=$4
  local M=(-v "$(dirname "$tr"):$(dirname "$tr"):ro" -v "$X:$X:rw")
  [ "$(dirname "$Q")" = "$X" ] || M+=(-v "$(dirname "$Q"):$(dirname "$Q"):rw")
  run "$(basename "$X")-quarantine" "${M[@]}" -v "$PI4:$PI4:ro" \
    "$IMG" -m v2.data.hr2.audit quarantine --candidates "$tr" --candidates "$dv" \
    --quarantining "$X/piv4q.private.json" --quarantining "$X/piib3.private.json" --full "$X/piv4.private.json" \
    --quarantining-manifest "$PI4/manifest.quarantining.json" --self-scan "$X/self.private.json" \
    --out-dir "$Q"
}

scans() {
  fresh g0 g0u overlap names
  local G=$R/g0 U=$R/g0u O=$R/overlap
  run g0u-reference -v "$SUITE:$SUITE:ro" -v "$U:$U:rw" "$IMG" -m v2.data.ib3.index_guard url-reference \
    --suite "$SUITE/selected-rows.jsonl.gz" --suite "$SUITE/added-rows.jsonl.gz" --out-dir "$U/ref"
  run g0u-scan -v "$CAND:$CAND:ro" -v "$U:$U:rw" "$IMG" -m v2.data.ib3.index_guard url-scan --reference "$U/ref" \
    --candidates "$TRAIN" --candidates "$DEV" --out-dir "$U/scan" &
  run g0u-controls -v "$SUITE:$SUITE:ro" -v "$U:$U:rw" "$IMG" -m v2.data.ib3.index_guard url-controls \
    --suite "$SUITE/selected-rows.jsonl.gz" --suite "$SUITE/added-rows.jsonl.gz" --reference "$U/ref" \
    --out "$U/controls.json" &
  run g0-reference -v "$SUITE:$SUITE:ro" -v "$G:$G:rw" "$IMG" -m v2.data.ib3.index_guard reference \
    --suite "$SUITE/selected-rows.jsonl.gz" --suite "$SUITE/added-rows.jsonl.gz" --out-dir "$G/ref" \
    --workers "$WORKERS"
  run g0-scan -v "$CAND:$CAND:ro" -v "$G:$G:rw" "$IMG" -m v2.data.ib3.index_guard scan --reference "$G/ref" \
    --candidates "$TRAIN" --candidates "$DEV" --out-dir "$G/scan" --workers "$WORKERS" &
  run g0-controls -v "$SUITE:$SUITE:ro" -v "$G:$G:rw" "$IMG" -m v2.data.ib3.index_guard controls \
    --suite "$SUITE/selected-rows.jsonl.gz" --suite "$SUITE/added-rows.jsonl.gz" --reference "$G/ref" \
    --out "$G/controls.json" &
  cat > "$O/pi-ib3.spec.json" <<EOF
[
 {"role": "ht_dev_goldfree", "origin": "$P/ht-dev.prompts.jsonl", "sha256": "30b0bd3569da9dd183f142e606ba6dcb598de7e4d3d9c168f87f91852178fd65", "project": true},
 {"role": "ht_dev2_goldfree", "origin": "$P/ht-dev2.prompts.jsonl", "sha256": "90cd409a1e091a623362c0e5b227d13b7301bf13fe266f7905e09233cf815f74", "project": true},
 {"role": "score5_dev_goldfree", "origin": "$P/score5-dev.prompts.jsonl", "sha256": "a01551c280473c9251ebbf8aed7bf9cfba3e12927def8959e4dd1283c072a86a", "project": true},
 {"role": "score5t_dev_goldfree", "origin": "$P/score5t-dev.prompts.jsonl", "sha256": "8e35bfffc2c3b8e4252d3c39ec1c65054250a3ddf80159d219a16b41bca1d93c", "project": true},
 {"role": "hs1_dev_goldfree", "origin": "$P/hs1-dev.prompts.jsonl", "sha256": "49f192a700242efe46265c4377a3cedb44dd635e5c5d23db1fc2d1e6fac3f072", "project": true},
 {"role": "pn1_dev_goldfree", "origin": "$PN1DEV/pn1.dev.prompts.jsonl", "sha256": "79dbf9996ad09caa622c4a9ea2f28075ddf80a5ae1ba9db8cd633573621bc14a", "project": true},
 {"role": "cal698", "origin": "$CAL698", "sha256": "19cc1a8c4ebe6fd13031079f6b7d131046ce503f435f894e3ea672f91c2ed41f", "project": true},
 {"role": "hr2_dev", "origin": "$HR2DEV", "sha256": "697c31428ae821d7c0238a64ced8b997cc5671be47307f88a273d082bc5aaebe", "project": true},
 {"role": "ib1_dev", "origin": "$IB1DEV", "sha256": "b412ab9312e834d45a1aef910b8921ebd1a25b5b0ddc5d226e6230a8b2a031cd", "project": true},
 {"role": "ib1_dev_r3", "origin": "$IB1DEV3", "sha256": "3f56aa418e90e58f3fbaa50bf2f9f4f0405eeb9dd2f0c51ffca6f602711c693f", "project": true},
 {"role": "ib2_dev", "origin": "$IB2DEV", "sha256": "ab009fb12f9563c3ef4a846f5455dd5233f2a2c5fafe6ac8a9c74349532eb923", "project": true}
]
EOF
  run pi-ib3 -v "$P:$P:ro" -v "$HFD:$HFD:ro" -v "$IB1DEV:$IB1DEV:ro" -v "$IB1DEV3:$IB1DEV3:ro" \
    -v "$IB2DEV:$IB2DEV:ro" -v "$O:$O:rw" "$IMG" \
    -m v2.data.build_protected_inventory --spec "$O/pi-ib3.spec.json" --out-dir "$O/pi-ib3"
  overlap "$O" "$TRAIN" "$DEV" "$O/pi-ib3"
  run names -v "$R/names:$R/names:rw" "$IMG" -m v2.data.ib3.audit names --out "$R/names/g1.json"
  run c1-names -v "$CAND:$CAND:ro" -v "$RAW:$RAW:ro" -v "$R/names:$R/names:rw" "$IMG" -m v2.eval.sealed.independence \
    names --terms v2/eval/sealed/c1-source-terms.json --root "$RAW" --root "$CAND" \
    --output "$R/names/c1-names.json"
  wait
}

pass1() {
  fresh quarantine pass1
  quarantine "$R/overlap" "$R/quarantine/lists" "$TRAIN" "$DEV"
  run pass1 -v "$CAND:$CAND:ro" -v "$R:$R:ro" -v "$R/pass1:$R/pass1:rw" \
    "$IMG" -m v2.data.ib3.build finalize --cand "$CAND" --out "$R/pass1/out" \
    --drop-groups "$R/quarantine/lists/drop-groups.txt" --drop-groups "$R/g0/scan/drop-groups.txt" \
    --drop-groups "$R/g0u/scan/drop-groups.txt" --drop-dev-groups "$R/quarantine/lists/drop-dev-groups.txt"
}

# Drop arguments of every finalize after pass 1: the pass-1 lists and rescan<1..$1>.
pass_drops() {
  local Q=$R/quarantine/lists n
  printf -- '%s\n' --drop-groups "$Q/drop-groups.txt" --drop-groups "$R/g0/scan/drop-groups.txt" \
    --drop-groups "$R/g0u/scan/drop-groups.txt" --drop-dev-groups "$Q/drop-dev-groups.txt"
  for ((n = 1; n <= $1; n++)); do
    printf -- '%s\n' --drop-groups "$R/rescan$n/lists/drop-groups.txt" \
      --drop-groups "$R/rescan$n/g0/drop-groups.txt" --drop-groups "$R/rescan$n/g0u/drop-groups.txt" \
      --drop-dev-groups "$R/rescan$n/lists/drop-dev-groups.txt"
  done
}

# rescan <n|final>: the finalized files of pass <n> (or final/) against the Index rows and the held-out panels.
rescan() {
  local tag=$1 src
  [ "$tag" = final ] && src=$R/final/out || src=$R/pass$tag/out
  fresh "rescan$tag"
  local X=$R/rescan$tag
  run "rs$tag-g0" -v "$src:$src:ro" -v "$X:$X:rw" -v "$R/g0:$R/g0:ro" "$IMG" -m v2.data.ib3.index_guard scan \
    --reference "$R/g0/ref" --candidates "$src/ib3.train.jsonl" --candidates "$src/ib3.dev.jsonl" \
    --out-dir "$X/g0" --workers "$WORKERS" &
  run "rs$tag-g0u" -v "$src:$src:ro" -v "$X:$X:rw" -v "$R/g0u:$R/g0u:ro" "$IMG" -m v2.data.ib3.index_guard \
    url-scan --reference "$R/g0u/ref" --candidates "$src/ib3.train.jsonl" --candidates "$src/ib3.dev.jsonl" \
    --out-dir "$X/g0u" &
  overlap "$X" "$src/ib3.train.jsonl" "$src/ib3.dev.jsonl" "$R/overlap/pi-ib3"
  run "rs$tag-c1" -v "$src:$src:ro" -v "$X:$X:rw" "$IMG" -m v2.eval.sealed.independence names \
    --terms v2/eval/sealed/c1-source-terms.json --root "$src" --output "$X/c1-names.json" &
  wait
  quarantine "$X" "$X/lists" "$src/ib3.train.jsonl" "$src/ib3.dev.jsonl"
  echo "rescan$tag new drops: g0=$(grep -c . "$X/g0/drop-groups.txt" || true)" \
    "g0u=$(grep -c . "$X/g0u/drop-groups.txt" || true)" \
    "groups=$(grep -c . "$X/lists/drop-groups.txt" || true)" \
    "dev_groups=$(grep -c . "$X/lists/drop-dev-groups.txt" || true)" | tee "$X/summary.txt"
}

# shortcut <pass>: per-family G4 audits of that pass's TRAIN file; failing families go to shortcut/shortcut-fail.txt.
shortcut() {
  fresh shortcut
  local S=$R/shortcut
  run g4rows -v "$R/$1:$R/$1:ro" -v "$S:$S:rw" "$IMG" -m v2.data.ib3.audit g4rows \
    --rows "$R/$1/out/ib3.train.jsonl" --out-dir "$S/rows"
  local f name
  for f in "$S"/rows/*.jsonl; do
    name=$(basename "$f" .jsonl)
    run "sc-$name" -v "$S:$S:rw" "$IMG" -m v2.data.shortcut --rows "$f" \
      --receipt "$S/$name.json" --workers 4 &
  done
  wait
  python3 - "$S" > "$S/shortcut-fail.txt" <<'EOF'
import json, pathlib, sys
for path in sorted(pathlib.Path(sys.argv[1]).glob("*.json")):
    if json.loads(path.read_text()).get("verdict") == "FAIL":
        print(path.stem)
EOF
}

passn() {
  local n=$1
  fresh "pass$n"
  local -a drops
  mapfile -t drops < <(pass_drops $((n - 1)))
  run "pass$n" -v "$CAND:$CAND:ro" -v "$R:$R:ro" -v "$R/pass$n:$R/pass$n:rw" "$IMG" \
    -m v2.data.ib3.build finalize --cand "$CAND" --out "$R/pass$n/out" "${drops[@]}"
  [ "$n" != "$SAMPLE_PASS" ] || shortcut "pass$n"
}

screen() {
  fresh screen
  local Pd=$R/pass$SAMPLE_PASS
  run screen-sample -v "$Pd:$Pd:ro" -v "$R/shortcut:$R/shortcut:ro" -v "$R/screen:$R/screen:rw" "$IMG" \
    -m v2.data.ib3.review screen-sample --train "$Pd/out/ib3.train.jsonl" \
    --drop-families "$R/shortcut/shortcut-fail.txt" --out-dir "$R/screen/sample"
  mkdir -m 700 "$R/screen/answers"
}

answers() {
  local dir=$1 who=$2 f
  for f in "$dir"/"$who".*.jsonl; do
    printf -- '--%s\n%s\n' "${3:-$who}" "$f"
  done
}

screen_score() {
  local S=$R/screen
  local -a args
  mapfile -t args < <(answers "$S/answers" s1 answers)
  run screen-score -v "$S:$S:rw" "$IMG" -m v2.data.ib3.review screen-score --key "$S/sample/key.jsonl" "${args[@]}" \
    --out "$S/screen.public.json" --private "$S/screen.private.json" \
    --drop-families-out "$S/drop-families.txt" --drop-ids-out "$S/drop-ids.txt"
  cat "$R/shortcut/shortcut-fail.txt" "$S/drop-families.txt" | sort -u > "$S/families-out.txt"
}

review() {
  fresh review
  local Pd=$R/pass$SAMPLE_PASS
  run sample -v "$Pd:$Pd:ro" -v "$R/screen:$R/screen:ro" -v "$R/review:$R/review:rw" "$IMG" \
    -m v2.data.ib3.review sample --train "$Pd/out/ib3.train.jsonl" \
    --screen-key "$R/screen/sample/key.jsonl" --drop-families "$R/screen/families-out.txt" \
    --out-dir "$R/review/sample"
  mkdir -m 700 "$R/review/answers"
}

splits() {
  local V=$R/review
  local -a args
  mapfile -t args < <(answers "$V/answers" r1; answers "$V/answers" r2)
  run splits -v "$V:$V:rw" "$IMG" -m v2.data.ib3.review splits --key "$V/sample/key.jsonl" "${args[@]}" \
    --packets "$V/sample/packet.r1.1.jsonl" --packets "$V/sample/packet.r1.2.jsonl" \
    --out "$V/answers/r3.packet.jsonl"
}

score() {
  local V=$R/review
  local -a args
  mapfile -t args < <(answers "$V/answers" r1; answers "$V/answers" r2)
  if [ -s "$V/answers/r3.jsonl" ]; then
    args+=(--r3 "$V/answers/r3.jsonl")
  fi
  run score -v "$V:$V:rw" "$IMG" -m v2.data.ib3.review score --sample "$V/sample/sample.json" \
    --key "$V/sample/key.jsonl" "${args[@]}" --out "$V/answers/review.public.json" \
    --private "$V/answers/review.private.json"
  python3 - "$V/answers" "$R/screen/drop-ids.txt" > "$R/drop-ids.txt" <<'EOF'
import json, pathlib, sys
answers, screen = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
ids = {item["id"] for item in json.loads((answers / "review.private.json").read_text())["errors"]}
ids |= {line.strip() for line in screen.read_text().split("\n") if line.strip()}
print("\n".join(sorted(ids)))
EOF
}

final() {
  fresh final freeze
  local F=$R/final/out Z=$R/freeze
  local -a drops=() leak=() lists=()
  mapfile -t drops < "$R/screen/families-out.txt"
  mapfile -t lists < <(pass_drops $((SAMPLE_PASS - 1)))
  [ ! -e "$R/rescanfinal" ] || lists+=(--drop-groups "$R/rescanfinal/lists/drop-groups.txt"
    --drop-groups "$R/rescanfinal/g0/drop-groups.txt" --drop-groups "$R/rescanfinal/g0u/drop-groups.txt"
    --drop-dev-groups "$R/rescanfinal/lists/drop-dev-groups.txt")
  if [ -e "$R/leak/drop-ids.txt" ]; then
    leak=(--drop-leak-ids "$R/leak/drop-ids.txt")
  fi
  run final -v "$CAND:$CAND:ro" -v "$R:$R:ro" -v "$R/final:$R/final:rw" "$IMG" -m v2.data.ib3.build finalize \
    --cand "$CAND" --out "$F" "${lists[@]}" --drop-ids "$R/drop-ids.txt" "${leak[@]}" --drop-families "${drops[@]}"
  local TOKM=(-v "$TOKJ:$TOKJ:ro" -v "$Q06:$Q06:ro" -v "$Q08:$Q08:ro" -v "$KAI:$KAI:ro")
  run freeze-train "${TOKM[@]}" -v "$R/final:$R/final:ro" -v "$Z:$Z:rw" "$IMG" -m v2.data.freeze freeze \
    --rows "$F/ib3.train.jsonl" --arm-id IB3 --role train --license-registry "$LIC" --tokenizers "$TOKJ" \
    --out-manifest "$Z/ib3.train.manifest.json" &
  run freeze-dev "${TOKM[@]}" -v "$R/final:$R/final:ro" -v "$Z:$Z:rw" "$IMG" -m v2.data.freeze freeze \
    --rows "$F/ib3.dev.jsonl" --arm-id IB3 --role aho --license-registry "$LIC" --tokenizers "$TOKJ" \
    --out-manifest "$Z/ib3.dev.manifest.json" &
  run isolation -v "$R/final:$R/final:ro" -v "$HS1DEV:$HS1DEV:ro" -v "$HFD:$HFD:ro" -v "$IB1DEV3:$IB1DEV3:ro" \
    -v "$IB2DEV:$IB2DEV:ro" \
    -v "$Z:$Z:rw" "$IMG" -m v2.data.freeze isolation --partition "train/IB3=$F/ib3.train.jsonl" \
    --partition "aho/IB3=$F/ib3.dev.jsonl" --partition "aho/IB1=$IB1DEV3" --partition "aho/IB2=$IB2DEV" \
    --partition "aho/HS1=$HS1DEV" \
    --partition "aho/PN1=$PN1DEV/pn1.dev.jsonl" --partition "aho/HR2=$HR2DEV" --report "$Z/isolation.json" &
  local part
  for part in train dev; do
    run "tokens-$part" "${TOKM[@]}" -v "$R/final:$R/final:ro" -v "$Z:$Z:rw" "$IMG" -m v2.data.m2.row_tokens \
      --tokenizers "$TOKJ" --native qwen3.5-0.8b-base@dc7cdfe2 --raw kai-0.6b@7185f514 \
      --rows "$F/ib3.$part.jsonl" --out "$Z/ib3.$part.tokens.jsonl" &
  done
  wait
  run stats -v "$R/final:$R/final:ro" -v "$Z:$Z:rw" "$IMG" -m v2.data.ib3.audit stats \
    --train "$F/ib3.train.jsonl" --dev "$F/ib3.dev.jsonl" --tokens "$Z/ib3.train.tokens.jsonl" \
    --tokens "$Z/ib3.dev.tokens.jsonl" --out "$Z/stats.json"
}

# Ids of final rows with a leak-guard finding; the next `final` run (old final/ and freeze/ moved aside) drops them.
leak() {
  fresh leak
  local F=$R/final/out rc=0
  (cd "$F" && bash "$CODE/v2/common/check_no_private.sh" -- ib3.train.jsonl ib3.dev.jsonl) \
    > "$R/leak/findings.txt" 2> "$R/leak/guard.err" || rc=$?
  python3 - "$F" "$R/leak/findings.txt" > "$R/leak/drop-ids.txt" <<'EOF'
import collections, json, pathlib, sys
final, findings = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
wanted = collections.defaultdict(set)
for line in findings.read_text().split("\n"):
    if line.strip():
        name, number, _ = line.split(":", 2)
        wanted[name.removeprefix("./")].add(int(number))
ids = set()
for name, numbers in wanted.items():
    lines = (final / name).read_text(encoding="utf-8").split("\n")
    ids |= {json.loads(lines[n - 1])["id"] for n in numbers}
print("\n".join(sorted(ids)))
EOF
  echo "leak-guard exit=$rc findings=$(grep -c . "$R/leak/findings.txt" || true) rows=$(grep -c . "$R/leak/drop-ids.txt" || true)"
}

hf_assemble() {
  fresh hf
  local spec=$R/hf/spec.json
  # The G0 receipt for upload keeps IB3-side counts only; suite statistics stay in the node copy.
  local name
  for name in g0/scan/index-guard g0u/scan/index-url-guard; do
    python3 - "$R/$name.public.json" > "$R/hf/$(basename "$name").public.json" <<'EOF'
import json, sys
receipt = json.load(open(sys.argv[1]))
receipt["reference"] = "every Decision Index suite row (selected and added rows); statistics kept on the node"
print(json.dumps(receipt, indent=1, sort_keys=True))
EOF
  done
  python3 - "$R" "$CAND" "$CODE/v2/data/records" > "$spec" <<'EOF'
import json, pathlib, sys
run, cand, rec = map(pathlib.Path, sys.argv[1:])
items = [
    (rec / "ib3/hf-readme.md", "README.md"),
    (rec / "ib3/status.json", "status.json"),
    (rec / "license-registry-ib3.json", "license-registry-ib3.json"),
    (run / "final/out/ib3.train.jsonl", "ib3.train.jsonl"),
    (run / "final/out/ib3.dev.jsonl", "ib3.dev.jsonl"),
    (run / "final/out/final.json", "final.json"),
    (cand / "build.json", "build.json"),
    (run / "freeze/ib3.train.tokens.jsonl", "ib3.train.tokens.jsonl"),
    (run / "freeze/ib3.dev.tokens.jsonl", "ib3.dev.tokens.jsonl"),
    (run / "freeze/ib3.train.manifest.json", "train.manifest.json"),
    (run / "freeze/ib3.dev.manifest.json", "dev.manifest.json"),
    (run / "freeze/stats.json", "stats.json"),
    (run / "freeze/isolation.json", "isolation.json"),
    (run / "hf/index-guard.public.json", "audits/index-guard.public.json"),
    (run / "g0/controls.json", "audits/index-guard-controls.json"),
    (run / "hf/index-url-guard.public.json", "audits/index-url-guard.public.json"),
    (run / "g0u/controls.json", "audits/index-url-guard-controls.json"),
    (run / "shortcut/g4rows.json", "audits/shortcut-g4rows.json"),
    (run / "overlap/piv4.public.json", "audits/overlap-piv4.public.json"),
    (run / "overlap/piv4q.public.json", "audits/overlap-piv4q.public.json"),
    (run / "overlap/piib3.public.json", "audits/overlap-piib3.public.json"),
    (run / "overlap/self.public.json", "audits/overlap-dev-vs-train.public.json"),
    (run / "quarantine/lists/quarantine.public.json", "audits/quarantine.public.json"),
    (run / "names/g1.json", "audits/names-g1.json"),
    (run / "names/c1-names.json", "audits/c1-names.json"),
    (run / "screen/screen.public.json", "audits/screen.public.json"),
    (run / "screen/sample/sample.json", "audits/screen-sample.json"),
    (run / "review/answers/review.public.json", "audits/review.public.json"),
    (run / "review/sample/sample.json", "audits/review-sample.json"),
]
for scan in sorted(run.glob("rescan*")):
    items.append((scan / "summary.txt", f"audits/{scan.name}-summary.txt"))
items += [(p, f"audits/shortcut/{p.name}") for p in sorted((run / "shortcut").glob("*.json")) if p.name != "g4rows.json"]
print(json.dumps([{"src": str(src), "dst": "ib3/" + dst} for src, dst in items], indent=1))
EOF
  run hf-assemble -v "$R:$R:ro" -v "$CAND:$CAND:ro" -v "$R/hf:$R/hf:rw" "$IMG" -m v2.data.assemble_hf_upload \
    --spec "$spec" --out-dir "$R/hf/upload/m6"
  local rc=0
  (cd "$R/hf/upload/m6/ib3" && bash "$CODE/v2/common/check_no_private.sh" -- .) \
    > "$R/logs/hf-leak-guard.out" 2> "$R/logs/hf-leak-guard.err" || rc=$?
  echo "leak-guard exit=$rc $(tail -1 "$R/logs/hf-leak-guard.err")"
  find "$R/hf/upload/m6/ib3" -type f -printf '%s\t%P\n' | sort -k2
}

hf_upload() {
  export HF_HUB_CACHE=/data/dev2/hf-cache HF_HUB_DISABLE_TELEMETRY=1
  local repo=llm-semantic-router/decision-2.0-training-data tree=$R/hf/upload/m6/ib3
  local py=/data/dev2/tools/hf-cli/bin/python log=$R/logs/hf-upload.log
  local counts safe msg
  counts=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); t=d["train"]["rows"]; v=d["dev"]["rows"]; print(f"TRAIN {t:,}, DEV {v:,}")' "$R/final/out/final.json")
  safe=$(python3 -c 'import json,sys; print("release-safe" if json.load(open(sys.argv[1]))["release_safe"] else "NOT release-safe")' "$tree/status.json")
  msg="IB3 Index-family breadth block 3 ($safe): m6/ib3 ($counts; build ${SHA:0:12})"
  info() {
    hf datasets info "$repo" --expand private,sha | "$py" -c 'import json,sys; d=json.load(sys.stdin); print(d["private"], d["sha"])'
  }
  [ -f "$tree/registry.json" ] || { echo "no assembled tree" >&2; exit 1; }
  [ ! -e "$R/hf/readback" ] || { echo "$R/hf/readback exists" >&2; exit 1; }
  local before parent after head rev
  read -r before parent < <(info)
  [ "$before" = "True" ] || { echo "dataset is not private; refusing to upload" >&2; exit 1; }
  echo "$(date -u +%FT%TZ) parent=$parent private=$before" | tee -a "$log"
  hf upload "$repo" "$tree" m6/ib3 --repo-type dataset --commit-message "$msg" >> "$log" 2>&1
  read -r after head < <(info)
  [ "$after" = "True" ] || { echo "dataset private flag changed" >&2; exit 1; }
  rev=$("$py" "$CODE/v2/data/hr2/hf_readback.py" pin "$repo" "$parent" "$msg")
  echo "$(date -u +%FT%TZ) head=$head revision=$rev private=$after" | tee -a "$log"
  mkdir -m 700 "$R/hf/readback"
  hf download "$repo" --repo-type dataset --revision "$rev" --include 'm6/ib3/*' --include 'm6/ib3/**' \
    --local-dir "$R/hf/readback" > /dev/null
  cmp "$R/hf/readback/m6/ib3/registry.json" "$tree/registry.json"
  "$py" "$CODE/v2/data/hr2/hf_readback.py" verify "$repo" "$rev" m6/ib3 "$tree" \
    "$R/hf/readback/m6/ib3" | tee "$R/hf/readback.json"
}

case "$STAGE" in
  build | scans | pass1 | screen | review | splits | score | final | leak) "$STAGE" ;;
  rescan) rescan "$3" ;;
  pass) passn "$3" ;;
  screen-score) screen_score ;;
  hf-assemble) hf_assemble ;;
  hf-upload) hf_upload ;;
  *)
    echo "unknown stage $STAGE" >&2
    exit 2
    ;;
esac
echo "$(date -u +%FT%TZ) stage $STAGE finished" >> "$R/logs/steps.log"
