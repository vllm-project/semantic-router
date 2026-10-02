#!/bin/bash
# IB4 build, audits and publication on node A (prereg records/ib4-prereg-2026-10-02.md §2-§5).
# CPU only, in the pinned image with --network none; the mirror and every input are mounted read-only, only the
# step's output directory is writable. Every stage refuses to overwrite. Adapted from v2/data/ib3/node_a.sh.
#
#   node_a.sh <commit> raw            copy the IB1 / IB2 raw files IB4 reuses into the IB4 raw directory (no edits)
#   node_a.sh <commit> build          candidates from the pinned raw files, deduplicated against IB1-r3 / IB2 / IB3-r2
#   node_a.sh <commit> scans          G0 and G0u (reference, scan, positive controls); PI-ib4 inventory; overlap vs
#                                     PI-v4 (full, quarantining) and PI-ib4; DEV-vs-TRAIN self-scan; G1; C1 names
#   node_a.sh <commit> pass1          quarantine lists, finalize pass 1 and G4 on its TRAIN file
#   node_a.sh <commit> final          finalize (G4-failing families out), freeze, isolation (G7), tokens, stats (G5, G8)
#   node_a.sh <commit> rescan         G0, G0u and C1 names again on the final files
#   node_a.sh <commit> ix1            IX1 row-level contamination audit of final TRAIN and DEV vs the full private
#                                     Index panel, planted control 200 / 200
#   node_a.sh <commit> spot           author spot-check sample (IB4_SPOT rows per family) for reading on the node
#   node_a.sh <commit> leak           ids of final rows with a leak-guard finding; a new `final` drops them
#   node_a.sh <commit> hf-assemble    m6/ib4/<phase> upload tree and the leak guard
#   node_a.sh <commit> hf-upload      private upload with the HF CLI, pinned revision, read-back check
#
# IB4_RUN names the phase (p1, p2, ...): run directory, upload folder m6/ib4/<run>; IB4_FAMILIES the families built.
set -euo pipefail
umask 077
SHA=$1
STAGE=$2
MIRROR=/data/dev2/src/${SHA}-src_training_decision2
CODE=$MIRROR/src/training/decision2
H=/data/dev2/private/data/ib4
RAW=$H/raw
RUN=${IB4_RUN:-p1}
R=$H/$RUN
CAND=$R/cand
T=/data/dev2/tmp/ib4-$RUN
IMG=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
SUITE=/data/dev2/private/eval/index021/suite-0.2
PANEL=/data/dev2/private/eval/index021/ix1/panel-3
PI3=/data/dev2/private/data/pi-v3
PI4=/data/dev2/runs/data/m3b/gap/c1/pi
P=/data/dev2/private/panels/goldfree
HFD=/data/dev2/hf-cache/datasets--llm-semantic-router--decision-2.0-training-data
PN1DEV=$HFD/snapshots/27b1d2f130292268b43a618584bebab5d4e4a6b5/m4/pn1/dev
CAL698=$HFD/snapshots/ed87a03ab80ca5b9560780bba51a83a77ff47d14/m2/cal/CAL698/cal.jsonl
HR2DEV=$HFD/snapshots/afc3bc1e1d6849058f6fafdfbc3dfe007d067400/m5/hr2/hr2.dev.jsonl
HS1DEV=/data/dev2/private/data/hs1/21bdb5e90e2b/build/hs1.dev.jsonl
IB1=/data/dev2/private/data/ib1
IB2=/data/dev2/private/data/ib2
IB3=/data/dev2/private/data/ib3
IB1DEV=$IB1/b2/final/out/ib1.dev.jsonl
IB1OUT=$IB1/r3/final/out
IB2OUT=$IB2/c3/final/out
IB3OUT=$IB3/r2/final/out
TOKJ=/data/dev2/private/data/arms-v1/tokenizers.json
Q06=/data/dev2/hf-cache/models--Qwen--Qwen3-0.6B-Base
Q08=/data/dev2/hf-cache/models--Qwen--Qwen3.5-0.8B-Base
KAI=/data/dev2/private/models/Decision-1.0-Kai-0.6B@7185f514f54b8f93c55998b1e8f9c5cc67f0d029/native/tokenizer
LIC=$CODE/v2/data/records/license-registry-ib4.json
read -r -a FAMS <<< "${IB4_FAMILIES:-}"
TRAIN=$CAND/ib4.train.cand.jsonl
DEV=$CAND/ib4.dev.cand.jsonl
WORKERS=${IB4_WORKERS:-48}
SPOT=${IB4_SPOT:-12}

[ -d "$CODE" ] || { echo "no mirror $MIRROR" >&2; exit 1; }
for d in "$H" "$RAW" "$R" "$T" "$R/logs"; do
  [ -d "$d" ] || mkdir -m 700 "$d"
done
BASE=(--rm --network none --entrypoint python3 -w "$CODE" --cpu-shares 256
  -e PYTHONPATH=. -e TMPDIR="$T" -e CUDA_VISIBLE_DEVICES= -e HIP_VISIBLE_DEVICES=
  --label dev2.track=data-ib4 --label "dev2.commit=$SHA"
  -v "$MIRROR:$MIRROR:ro" -v "$T:$T:rw")
PRIORS=(-v "$IB1OUT:$IB1OUT:ro" -v "$IB2OUT:$IB2OUT:ro" -v "$IB3OUT:$IB3OUT:ro")

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
  docker run --name "ib4-$RUN-$name" "${BASE[@]}" "$@" > "$R/logs/$name.log" 2>&1 || rc=$?
  echo "$(date -u +%FT%TZ) done $name exit=$rc" >> "$R/logs/steps.log"
  return "$rc"
}

raw() {
  local rel src
  for rel in iabufarha_iSarcasmEval/train/train.En.csv iabufarha_iSarcasmEval/train/train.Ar.csv \
    pyRis_SEntFiN/SEntFiN.csv nvidia_When2Call/train/when2call_train_pref.jsonl; do
    src=$IB1/raw/$rel
    [ -e "$RAW/$rel" ] || { install -d -m 700 "$(dirname "$RAW/$rel")"; cp -p "$src" "$RAW/$rel"; }
    cmp "$src" "$RAW/$rel"
  done
  rel=glaiveai_glaive-function-calling-v2/glaive-function-calling-v2.json
  [ -e "$RAW/$rel" ] || { install -d -m 700 "$(dirname "$RAW/$rel")"; cp -p "$IB2/raw/$rel" "$RAW/$rel"; }
  cmp "$IB2/raw/$rel" "$RAW/$rel"
  (cd "$RAW" && find . -type f | sort | xargs sha256sum)
}

build() {
  [ ! -e "$CAND" ] || { echo "$CAND exists" >&2; exit 1; }
  mkdir -m 700 "$CAND"
  local -a sel=()
  [ "${#FAMS[@]}" -eq 0 ] || sel=(--families "${FAMS[@]}")
  run build -v "$RAW:$RAW:ro" -v "$CAND:$CAND:rw" "${PRIORS[@]}" "$IMG" -m v2.data.ib4.build build --raw "$RAW" \
    --prior "$IB1OUT/ib1.train.jsonl" --prior "$IB1OUT/ib1.dev.jsonl" --prior "$IB2OUT/ib2.train.jsonl" \
    --prior "$IB2OUT/ib2.dev.jsonl" --prior "$IB3OUT/ib3.train.jsonl" --prior "$IB3OUT/ib3.dev.jsonl" \
    --out "$CAND" "${sel[@]}"
}

# overlap <dir> <train> <dev> <inventory dir>: the four overlap scans of one pair of files, in parallel.
overlap() {
  local X=$1 tr=$2 dv=$3 I=$4
  local M=(-v "$(dirname "$tr"):$(dirname "$tr"):ro" -v "$X:$X:rw")
  local cand=(--candidates "$tr" --candidates "$dv")
  run piv4 "${M[@]}" -v "$PI3:$PI3:ro" -v "$PI4:$PI4:ro" "$IMG" -m v2.data.overlap "${cand[@]}" \
    --protected-inventory "$PI4/manifest.json" --private-receipt "$X/piv4.private.json" \
    --public-receipt "$X/piv4.public.json" --workers "$WORKERS" &
  run piv4q "${M[@]}" -v "$PI3:$PI3:ro" -v "$PI4:$PI4:ro" "$IMG" -m v2.data.overlap "${cand[@]}" \
    --protected-inventory "$PI4/manifest.quarantining.json" --private-receipt "$X/piv4q.private.json" \
    --public-receipt "$X/piv4q.public.json" --workers "$WORKERS" &
  run piib4 "${M[@]}" -v "$I:$I:ro" "$IMG" -m v2.data.overlap "${cand[@]}" \
    --protected-inventory "$I/manifest.json" --private-receipt "$X/piib4.private.json" \
    --public-receipt "$X/piib4.public.json" --workers "$WORKERS" &
  run self "${M[@]}" "$IMG" -m v2.data.overlap --self-scan --candidates "$dv" \
    --candidates "$tr" --private-receipt "$X/self.private.json" \
    --public-receipt "$X/self.public.json" --workers "$WORKERS" &
}

scans() {
  fresh g0 g0u overlap names
  local G=$R/g0 U=$R/g0u O=$R/overlap
  run g0u-reference -v "$SUITE:$SUITE:ro" -v "$U:$U:rw" "$IMG" -m v2.data.ib4.index_guard url-reference \
    --suite "$SUITE/selected-rows.jsonl.gz" --suite "$SUITE/added-rows.jsonl.gz" --out-dir "$U/ref"
  run g0u-scan -v "$CAND:$CAND:ro" -v "$U:$U:rw" "$IMG" -m v2.data.ib4.index_guard url-scan --reference "$U/ref" \
    --candidates "$TRAIN" --candidates "$DEV" --out-dir "$U/scan" &
  run g0u-controls -v "$SUITE:$SUITE:ro" -v "$U:$U:rw" "$IMG" -m v2.data.ib4.index_guard url-controls \
    --suite "$SUITE/selected-rows.jsonl.gz" --suite "$SUITE/added-rows.jsonl.gz" --reference "$U/ref" \
    --out "$U/controls.json" &
  run g0-reference -v "$SUITE:$SUITE:ro" -v "$G:$G:rw" "$IMG" -m v2.data.ib4.index_guard reference \
    --suite "$SUITE/selected-rows.jsonl.gz" --suite "$SUITE/added-rows.jsonl.gz" --out-dir "$G/ref" \
    --workers "$WORKERS"
  run g0-scan -v "$CAND:$CAND:ro" -v "$G:$G:rw" "$IMG" -m v2.data.ib4.index_guard scan --reference "$G/ref" \
    --candidates "$TRAIN" --candidates "$DEV" --out-dir "$G/scan" --workers "$WORKERS" &
  run g0-controls -v "$SUITE:$SUITE:ro" -v "$G:$G:rw" "$IMG" -m v2.data.ib4.index_guard controls \
    --suite "$SUITE/selected-rows.jsonl.gz" --suite "$SUITE/added-rows.jsonl.gz" --reference "$G/ref" \
    --out "$G/controls.json" &
  local ib3dev
  ib3dev=$(sha256sum "$IB3OUT/ib3.dev.jsonl" | cut -d' ' -f1)
  [ "$ib3dev" = 4c6f2ca06e131ed28058f6f53bea8f2015ccfbd2d9494a1b6ff6733a94be51af ] || { echo "IB3-r2 DEV hash" >&2; exit 1; }
  cat > "$O/pi-ib4.spec.json" <<EOF
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
 {"role": "ib1_dev_r3", "origin": "$IB1OUT/ib1.dev.jsonl", "sha256": "3f56aa418e90e58f3fbaa50bf2f9f4f0405eeb9dd2f0c51ffca6f602711c693f", "project": true},
 {"role": "ib2_dev", "origin": "$IB2OUT/ib2.dev.jsonl", "sha256": "ab009fb12f9563c3ef4a846f5455dd5233f2a2c5fafe6ac8a9c74349532eb923", "project": true},
 {"role": "ib3_dev_r2", "origin": "$IB3OUT/ib3.dev.jsonl", "sha256": "$ib3dev", "project": true}
]
EOF
  run pi-ib4 -v "$P:$P:ro" -v "$HFD:$HFD:ro" -v "$IB1DEV:$IB1DEV:ro" "${PRIORS[@]}" -v "$O:$O:rw" "$IMG" \
    -m v2.data.build_protected_inventory --spec "$O/pi-ib4.spec.json" --out-dir "$O/pi-ib4"
  overlap "$O" "$TRAIN" "$DEV" "$O/pi-ib4"
  run names -v "$R/names:$R/names:rw" "$IMG" -m v2.data.ib4.audit names --out "$R/names/g1.json"
  run c1-names -v "$CAND:$CAND:ro" -v "$RAW:$RAW:ro" -v "$R/names:$R/names:rw" "$IMG" -m v2.eval.sealed.independence \
    names --terms v2/eval/sealed/c1-source-terms.json --root "$RAW" --root "$CAND" \
    --output "$R/names/c1-names.json"
  wait
}

shortcut() {
  local S=$R/shortcut
  run g4rows -v "$R/$1:$R/$1:ro" -v "$S:$S:rw" "$IMG" -m v2.data.ib4.audit g4rows \
    --rows "$R/$1/out/ib4.train.jsonl" --out-dir "$S/rows"
  local f name
  for f in "$S"/rows/*.jsonl; do
    name=$(basename "$f" .jsonl)
    run "sc-$name" -v "$S:$S:rw" "$IMG" -m v2.data.shortcut --rows "$f" \
      --receipt "$S/$name.json" --workers 8 &
  done
  wait
  python3 - "$S" > "$S/shortcut-fail.txt" <<'EOF'
import json, pathlib, sys
for path in sorted(pathlib.Path(sys.argv[1]).glob("*.json")):
    if json.loads(path.read_text()).get("verdict") == "FAIL":
        print(path.stem)
EOF
}

drop_lists() {
  printf -- '%s\n' --drop-groups "$R/quarantine/lists/drop-groups.txt" --drop-groups "$R/g0/scan/drop-groups.txt" \
    --drop-groups "$R/g0u/scan/drop-groups.txt" --drop-dev-groups "$R/quarantine/lists/drop-dev-groups.txt"
}

pass1() {
  fresh quarantine pass1 shortcut
  local O=$R/overlap
  run quarantine -v "$CAND:$CAND:ro" -v "$O:$O:ro" -v "$R/quarantine:$R/quarantine:rw" -v "$PI4:$PI4:ro" \
    "$IMG" -m v2.data.hr2.audit quarantine --candidates "$TRAIN" --candidates "$DEV" \
    --quarantining "$O/piv4q.private.json" --quarantining "$O/piib4.private.json" --full "$O/piv4.private.json" \
    --quarantining-manifest "$PI4/manifest.quarantining.json" --self-scan "$O/self.private.json" \
    --out-dir "$R/quarantine/lists"
  local -a drops
  mapfile -t drops < <(drop_lists)
  run pass1 -v "$CAND:$CAND:ro" -v "$R:$R:ro" -v "$R/pass1:$R/pass1:rw" \
    "$IMG" -m v2.data.ib4.build finalize --cand "$CAND" --out "$R/pass1/out" "${drops[@]}"
  shortcut pass1
}

final() {
  fresh final freeze
  local F=$R/final/out Z=$R/freeze
  local -a fails=() leak=() lists=()
  mapfile -t fails < "$R/shortcut/shortcut-fail.txt"
  mapfile -t lists < <(drop_lists)
  [ ! -e "$R/leak/drop-ids.txt" ] || leak=(--drop-leak-ids "$R/leak/drop-ids.txt")
  run final -v "$CAND:$CAND:ro" -v "$R:$R:ro" -v "$R/final:$R/final:rw" "$IMG" -m v2.data.ib4.build finalize \
    --cand "$CAND" --out "$F" "${lists[@]}" "${leak[@]}" --drop-families "${fails[@]}"
  local TOKM=(-v "$TOKJ:$TOKJ:ro" -v "$Q06:$Q06:ro" -v "$Q08:$Q08:ro" -v "$KAI:$KAI:ro")
  run freeze-train "${TOKM[@]}" -v "$R/final:$R/final:ro" -v "$Z:$Z:rw" "$IMG" -m v2.data.freeze freeze \
    --rows "$F/ib4.train.jsonl" --arm-id IB4 --role train --license-registry "$LIC" --tokenizers "$TOKJ" \
    --out-manifest "$Z/ib4.train.manifest.json" &
  run freeze-dev "${TOKM[@]}" -v "$R/final:$R/final:ro" -v "$Z:$Z:rw" "$IMG" -m v2.data.freeze freeze \
    --rows "$F/ib4.dev.jsonl" --arm-id IB4 --role aho --license-registry "$LIC" --tokenizers "$TOKJ" \
    --out-manifest "$Z/ib4.dev.manifest.json" &
  run isolation -v "$R/final:$R/final:ro" -v "$HS1DEV:$HS1DEV:ro" -v "$HFD:$HFD:ro" "${PRIORS[@]}" \
    -v "$Z:$Z:rw" "$IMG" -m v2.data.freeze isolation --partition "train/IB4=$F/ib4.train.jsonl" \
    --partition "aho/IB4=$F/ib4.dev.jsonl" --partition "aho/IB1=$IB1OUT/ib1.dev.jsonl" \
    --partition "aho/IB2=$IB2OUT/ib2.dev.jsonl" --partition "aho/IB3=$IB3OUT/ib3.dev.jsonl" \
    --partition "aho/HS1=$HS1DEV" \
    --partition "aho/PN1=$PN1DEV/pn1.dev.jsonl" --partition "aho/HR2=$HR2DEV" --report "$Z/isolation.json" &
  local part
  for part in train dev; do
    run "tokens-$part" "${TOKM[@]}" -v "$R/final:$R/final:ro" -v "$Z:$Z:rw" "$IMG" -m v2.data.m2.row_tokens \
      --tokenizers "$TOKJ" --native qwen3.5-0.8b-base@dc7cdfe2 --raw kai-0.6b@7185f514 \
      --rows "$F/ib4.$part.jsonl" --out "$Z/ib4.$part.tokens.jsonl" &
  done
  wait
  run stats -v "$R/final:$R/final:ro" -v "$Z:$Z:rw" "$IMG" -m v2.data.ib4.audit stats \
    --train "$F/ib4.train.jsonl" --dev "$F/ib4.dev.jsonl" --tokens "$Z/ib4.train.tokens.jsonl" \
    --tokens "$Z/ib4.dev.tokens.jsonl" --out "$Z/stats.json"
}

rescan() {
  fresh rescan
  local X=$R/rescan F=$R/final/out
  run rs-g0 -v "$F:$F:ro" -v "$X:$X:rw" -v "$R/g0:$R/g0:ro" "$IMG" -m v2.data.ib4.index_guard scan \
    --reference "$R/g0/ref" --candidates "$F/ib4.train.jsonl" --candidates "$F/ib4.dev.jsonl" \
    --out-dir "$X/g0" --workers "$WORKERS" &
  run rs-g0u -v "$F:$F:ro" -v "$X:$X:rw" -v "$R/g0u:$R/g0u:ro" "$IMG" -m v2.data.ib4.index_guard \
    url-scan --reference "$R/g0u/ref" --candidates "$F/ib4.train.jsonl" --candidates "$F/ib4.dev.jsonl" \
    --out-dir "$X/g0u" &
  run rs-c1 -v "$F:$F:ro" -v "$X:$X:rw" "$IMG" -m v2.eval.sealed.independence names \
    --terms v2/eval/sealed/c1-source-terms.json --root "$F" --output "$X/c1-names.json" &
  wait
  echo "rescan drops: g0=$(grep -c . "$X/g0/drop-groups.txt" || true)" \
    "g0u=$(grep -c . "$X/g0u/drop-groups.txt" || true)" | tee "$X/summary.txt"
}

ix1() {
  fresh ix1
  local F=$R/final/out X=$R/ix1
  run ix1 -e PYTHONHASHSEED=0 -v "$PANEL:$PANEL:ro" -v "$F:$F:ro" -v "$X:$X:rw" "$IMG" \
    -m v2.eval.ix1.contamination --panel "$PANEL" --train "IB4=$F/ib4.train.jsonl,$F/ib4.dev.jsonl" \
    --workers "$WORKERS" --out "$X/out"
  # The upload copy keeps IB4-side totals only: no per-benchmark rows.
  python3 - "$X/out/audit.json" > "$X/ix1.public.json" <<'EOF'
import json, sys
audit = json.load(open(sys.argv[1]))
out = {"schema": "decision2.ib4.ix1-summary.v1", "method": audit.get("method"),
       "planted_control": audit.get("planted_control"), "training_sets": {}}
for name, block in audit["training_sets"].items():
    hits = block.get("benchmarks_with_hits", {})
    out["training_sets"][name] = {k: sum(v.get(k, 0) for v in hits.values()) for k in ("duplicate", "item", "partial")}
print(json.dumps(out, indent=1, sort_keys=True))
EOF
  cat "$X/ix1.public.json"
}

spot() {
  fresh spot
  python3 - "$R/final/out/ib4.train.jsonl" "$SPOT" > "$R/spot/sample.jsonl" <<'EOF'
import collections, hashlib, json, sys
by = collections.defaultdict(list)
for line in open(sys.argv[1], encoding="utf-8"):
    item = json.loads(line)
    by[item["family"]].append(item)
for family, rows in sorted(by.items()):
    rows.sort(key=lambda r: hashlib.sha256(("ib4-spot-v1:" + r["id"]).encode()).hexdigest())
    for r in rows[: int(sys.argv[2])]:
        print(json.dumps({k: r[k] for k in ("id", "family", "state", "instructions", "options", "label")},
                         ensure_ascii=False))
EOF
  wc -l "$R/spot/sample.jsonl"
}

leak() {
  fresh leak
  local F=$R/final/out rc=0
  (cd "$F" && bash "$CODE/v2/common/check_no_private.sh" -- ib4.train.jsonl ib4.dev.jsonl) \
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
  local spec=$R/hf/spec.json name
  # The G0 receipts for upload keep IB4-side counts only; suite statistics stay in the node copy.
  for name in g0/scan/index-guard g0u/scan/index-url-guard; do
    python3 - "$R/$name.public.json" > "$R/hf/$(basename "$name").public.json" <<'EOF'
import json, sys
receipt = json.load(open(sys.argv[1]))
receipt["reference"] = "every Decision Index suite row (selected and added rows); statistics kept on the node"
print(json.dumps(receipt, indent=1, sort_keys=True))
EOF
  done
  python3 - "$R" "$CAND" "$CODE/v2/data/records" "$LIC" "$RUN" > "$spec" <<'EOF'
import json, pathlib, sys
run, cand, rec, lic = map(pathlib.Path, sys.argv[1:5])
phase = sys.argv[5]
items = [
    (rec / "ib4" / phase / "hf-readme.md", "README.md"),
    (rec / "ib4" / phase / "status.json", "status.json"),
    (lic, "license-registry-ib4.json"),
    (run / "final/out/ib4.train.jsonl", "ib4.train.jsonl"),
    (run / "final/out/ib4.dev.jsonl", "ib4.dev.jsonl"),
    (run / "final/out/final.json", "final.json"),
    (cand / "build.json", "build.json"),
    (run / "freeze/ib4.train.tokens.jsonl", "ib4.train.tokens.jsonl"),
    (run / "freeze/ib4.dev.tokens.jsonl", "ib4.dev.tokens.jsonl"),
    (run / "freeze/ib4.train.manifest.json", "train.manifest.json"),
    (run / "freeze/ib4.dev.manifest.json", "dev.manifest.json"),
    (run / "freeze/stats.json", "stats.json"),
    (run / "freeze/isolation.json", "isolation.json"),
    (run / "hf/index-guard.public.json", "audits/index-guard.public.json"),
    (run / "g0/controls.json", "audits/index-guard-controls.json"),
    (run / "hf/index-url-guard.public.json", "audits/index-url-guard.public.json"),
    (run / "g0u/controls.json", "audits/index-url-guard-controls.json"),
    (run / "rescan/summary.txt", "audits/rescan-final-summary.txt"),
    (run / "ix1/ix1.public.json", "audits/ix1-contamination.public.json"),
    (run / "shortcut/g4rows.json", "audits/shortcut-g4rows.json"),
    (run / "overlap/piv4.public.json", "audits/overlap-piv4.public.json"),
    (run / "overlap/piv4q.public.json", "audits/overlap-piv4q.public.json"),
    (run / "overlap/piib4.public.json", "audits/overlap-piib4.public.json"),
    (run / "overlap/self.public.json", "audits/overlap-dev-vs-train.public.json"),
    (run / "quarantine/lists/quarantine.public.json", "audits/quarantine.public.json"),
    (run / "names/g1.json", "audits/names-g1.json"),
    (run / "names/c1-names.json", "audits/c1-names.json"),
    (run / "rescan/c1-names.json", "audits/c1-names-final.json"),
]
items += [(p, f"audits/shortcut/{p.name}") for p in sorted((run / "shortcut").glob("*.json")) if p.name != "g4rows.json"]
print(json.dumps([{"src": str(src), "dst": f"ib4/{phase}/" + dst} for src, dst in items], indent=1))
EOF
  run hf-assemble -v "$R:$R:ro" -v "$CAND:$CAND:ro" -v "$R/hf:$R/hf:rw" "$IMG" -m v2.data.assemble_hf_upload \
    --spec "$spec" --out-dir "$R/hf/upload/m6"
  local rc=0
  (cd "$R/hf/upload/m6/ib4/$RUN" && bash "$CODE/v2/common/check_no_private.sh" -- .) \
    > "$R/logs/hf-leak-guard.out" 2> "$R/logs/hf-leak-guard.err" || rc=$?
  echo "leak-guard exit=$rc $(tail -1 "$R/logs/hf-leak-guard.err")"
  find "$R/hf/upload/m6/ib4/$RUN" -type f -printf '%s\t%P\n' | sort -k2
}

hf_upload() {
  export HF_HUB_CACHE=/data/dev2/hf-cache HF_HUB_DISABLE_TELEMETRY=1
  local repo=llm-semantic-router/decision-2.0-training-data tree=$R/hf/upload/m6/ib4/$RUN dst=m6/ib4/$RUN
  local py=/data/dev2/tools/hf-cli/bin/python log=$R/logs/hf-upload.log
  local counts safe msg
  counts=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); t=d["train"]["rows"]; v=d["dev"]["rows"]; print(f"TRAIN {t:,}, DEV {v:,}")' "$R/final/out/final.json")
  safe=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["release_status"])' "$tree/status.json")
  msg="IB4 $RUN Index-gap data ($safe): $dst ($counts; build ${SHA:0:12})"
  info() {
    hf datasets info "$repo" --expand private,sha | "$py" -c 'import json,sys; d=json.load(sys.stdin); print(d["private"], d["sha"])'
  }
  [ -f "$tree/registry.json" ] || { echo "no assembled tree" >&2; exit 1; }
  [ ! -e "$R/hf/readback" ] || { echo "$R/hf/readback exists" >&2; exit 1; }
  local before parent after head rev
  read -r before parent < <(info)
  [ "$before" = "True" ] || { echo "dataset is not private; refusing to upload" >&2; exit 1; }
  echo "$(date -u +%FT%TZ) parent=$parent private=$before" | tee -a "$log"
  hf upload "$repo" "$tree" "$dst" --repo-type dataset --delete '*' --commit-message "$msg" >> "$log" 2>&1
  read -r after head < <(info)
  [ "$after" = "True" ] || { echo "dataset private flag changed" >&2; exit 1; }
  rev=$("$py" "$CODE/v2/data/hr2/hf_readback.py" pin "$repo" "$parent" "$msg")
  echo "$(date -u +%FT%TZ) head=$head revision=$rev private=$after" | tee -a "$log"
  mkdir -m 700 "$R/hf/readback"
  hf download "$repo" --repo-type dataset --revision "$rev" --include "$dst/*" --include "$dst/**" \
    --local-dir "$R/hf/readback" > /dev/null
  cmp "$R/hf/readback/$dst/registry.json" "$tree/registry.json"
  "$py" "$CODE/v2/data/hr2/hf_readback.py" verify "$repo" "$rev" "$dst" "$tree" \
    "$R/hf/readback/$dst" | tee "$R/hf/readback.json"
}

case "$STAGE" in
  raw | build | scans | pass1 | final | rescan | ix1 | spot | leak) "$STAGE" ;;
  hf-assemble) hf_assemble ;;
  hf-upload) hf_upload ;;
  *)
    echo "unknown stage $STAGE" >&2
    exit 2
    ;;
esac
echo "$(date -u +%FT%TZ) stage $STAGE finished" >> "$R/logs/steps.log"
