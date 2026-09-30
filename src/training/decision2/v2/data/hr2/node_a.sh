#!/bin/bash
# HR2 audits, review sampling and final build on node A (prereg records/hr2-prereg-2026-09-30.md §3-§5).
# CPU only, in the pinned image with --network none; the mirror and every input are mounted
# read-only, only the step's output directory is writable. Every stage refuses to overwrite.
#
#   node_a.sh <commit> scans    PI-hr2 inventory; overlap vs PI-v4 (full and quarantining) and PI-hr2;
#                               DEV-vs-TRAIN self-scan; G1 names; C1 source-term names
#   node_a.sh <commit> pass1    quarantine lists, finalize pass 1, per-family shortcut audits (G4)
#   node_a.sh <commit> review   blind review sample and packets from pass-1 TRAIN
#                               (families listed in shortcut-fail.txt are left out)
#   node_a.sh <commit> splits   R3 packet from review/answers/{r1,r2}.*.jsonl
#   node_a.sh <commit> score    review report and gold-error ids (review/answers/r3.jsonl if any)
#   node_a.sh <commit> final    finalize pass 2, freeze, isolation (G7), tokens and stats (G5, G8)
set -euo pipefail
umask 077
SHA=$1
STAGE=$2
MIRROR=/data/dev2/src/${SHA}-src_training_decision2
CODE=$MIRROR/src/training/decision2
H=/data/dev2/private/data/hr2
R=$H/b2-c54b8d444cac
CAND=$R/cand
T=/data/dev2/tmp/hr2-b2
IMG=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
PI3=/data/dev2/private/data/pi-v3
PI4=/data/dev2/runs/data/m3b/gap/c1/pi
P=/data/dev2/private/panels/goldfree
HFD=/data/dev2/hf-cache/datasets--llm-semantic-router--decision-2.0-training-data
PN1DEV=$HFD/snapshots/27b1d2f130292268b43a618584bebab5d4e4a6b5/m4/pn1/dev
CAL698=$HFD/snapshots/ed87a03ab80ca5b9560780bba51a83a77ff47d14/m2/cal/CAL698/cal.jsonl
HS1DEV=/data/dev2/private/data/hs1/21bdb5e90e2b/build/hs1.dev.jsonl
TOKJ=/data/dev2/private/data/arms-v1/tokenizers.json
Q06=/data/dev2/hf-cache/models--Qwen--Qwen3-0.6B-Base
Q08=/data/dev2/hf-cache/models--Qwen--Qwen3.5-0.8B-Base
KAI=/data/dev2/private/models/Decision-1.0-Kai-0.6B@7185f514f54b8f93c55998b1e8f9c5cc67f0d029/native/tokenizer
LIC=$CODE/v2/data/records/license-registry-hr2.json
TRAIN=$CAND/hr2.train.cand.jsonl
DEV=$CAND/hr2.dev.cand.jsonl
WORKERS=${HR2_WORKERS:-32}

[ -d "$CODE" ] || { echo "no mirror $MIRROR" >&2; exit 1; }
for d in "$T" "$R/logs"; do
  [ -d "$d" ] || mkdir -m 700 "$d"
done
BASE=(--rm --network none --entrypoint python3 -w "$CODE" --cpu-shares 256
  -e PYTHONPATH=. -e TMPDIR="$T" -e CUDA_VISIBLE_DEVICES= -e HIP_VISIBLE_DEVICES=
  --label dev2.track=data-hr2 --label "dev2.commit=$SHA"
  -v "$MIRROR:$MIRROR:ro" -v "$T:$T:rw" -v "$CAND:$CAND:ro")

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
  docker run --name "hr2-b2-$name" "${BASE[@]}" "$@" > "$R/logs/$name.log" 2>&1 || rc=$?
  echo "$(date -u +%FT%TZ) done $name exit=$rc" >> "$R/logs/steps.log"
  return "$rc"
}

scans() {
  fresh overlap names
  local O=$R/overlap
  cat > "$O/pi-hr2.spec.json" <<EOF
[
 {"role": "ht_dev_goldfree", "origin": "$P/ht-dev.prompts.jsonl", "sha256": "30b0bd3569da9dd183f142e606ba6dcb598de7e4d3d9c168f87f91852178fd65", "project": true},
 {"role": "ht_dev2_goldfree", "origin": "$P/ht-dev2.prompts.jsonl", "sha256": "90cd409a1e091a623362c0e5b227d13b7301bf13fe266f7905e09233cf815f74", "project": true},
 {"role": "score5_dev_goldfree", "origin": "$P/score5-dev.prompts.jsonl", "sha256": "a01551c280473c9251ebbf8aed7bf9cfba3e12927def8959e4dd1283c072a86a", "project": true},
 {"role": "score5t_dev_goldfree", "origin": "$P/score5t-dev.prompts.jsonl", "sha256": "8e35bfffc2c3b8e4252d3c39ec1c65054250a3ddf80159d219a16b41bca1d93c", "project": true},
 {"role": "hs1_dev_goldfree", "origin": "$P/hs1-dev.prompts.jsonl", "sha256": "49f192a700242efe46265c4377a3cedb44dd635e5c5d23db1fc2d1e6fac3f072", "project": true},
 {"role": "pn1_dev_goldfree", "origin": "$PN1DEV/pn1.dev.prompts.jsonl", "sha256": "79dbf9996ad09caa622c4a9ea2f28075ddf80a5ae1ba9db8cd633573621bc14a", "project": true},
 {"role": "cal698", "origin": "$CAL698", "sha256": "19cc1a8c4ebe6fd13031079f6b7d131046ce503f435f894e3ea672f91c2ed41f", "project": true}
]
EOF
  run pi-hr2 -v "$P:$P:ro" -v "$HFD:$HFD:ro" -v "$O:$O:rw" "$IMG" \
    -m v2.data.build_protected_inventory --spec "$O/pi-hr2.spec.json" --out-dir "$O/pi-hr2"
  local cand=(--candidates "$TRAIN" --candidates "$DEV")
  run ov-piv4 -v "$PI3:$PI3:ro" -v "$PI4:$PI4:ro" -v "$O:$O:rw" "$IMG" -m v2.data.overlap "${cand[@]}" \
    --protected-inventory "$PI4/manifest.json" --private-receipt "$O/piv4.private.json" \
    --public-receipt "$O/piv4.public.json" --workers "$WORKERS" &
  run ov-piv4q -v "$PI3:$PI3:ro" -v "$PI4:$PI4:ro" -v "$O:$O:rw" "$IMG" -m v2.data.overlap "${cand[@]}" \
    --protected-inventory "$PI4/manifest.quarantining.json" --private-receipt "$O/piv4q.private.json" \
    --public-receipt "$O/piv4q.public.json" --workers "$WORKERS" &
  run ov-pihr2 -v "$O:$O:rw" "$IMG" -m v2.data.overlap "${cand[@]}" \
    --protected-inventory "$O/pi-hr2/manifest.json" --private-receipt "$O/pihr2.private.json" \
    --public-receipt "$O/pihr2.public.json" --workers "$WORKERS" &
  run ov-self -v "$O:$O:rw" "$IMG" -m v2.data.overlap --self-scan --candidates "$DEV" \
    --candidates "$TRAIN" --private-receipt "$O/self.private.json" \
    --public-receipt "$O/self.public.json" --workers "$WORKERS" &
  run names -v "$R/names:$R/names:rw" "$IMG" -m v2.data.hr2.audit names --out "$R/names/g1.json"
  run c1-names -v "$H/raw:$H/raw:ro" -v "$R/names:$R/names:rw" "$IMG" -m v2.eval.sealed.independence \
    names --terms v2/eval/sealed/c1-source-terms.json --root "$H/raw" --root "$CAND" \
    --output "$R/names/c1-names.json"
  wait
}

pass1() {
  fresh quarantine
  local O=$R/overlap Q=$R/quarantine
  run quarantine -v "$O:$O:ro" -v "$PI4:$PI4:ro" -v "$Q:$Q:rw" "$IMG" -m v2.data.hr2.audit quarantine \
    --candidates "$TRAIN" --candidates "$DEV" --quarantining "$O/piv4q.private.json" \
    --quarantining "$O/pihr2.private.json" --full "$O/piv4.private.json" \
    --quarantining-manifest "$PI4/manifest.quarantining.json" --self-scan "$O/self.private.json" \
    --out-dir "$Q/lists"
  fresh pass1 shortcut
  run pass1 -v "$Q:$Q:ro" -v "$R/pass1:$R/pass1:rw" "$IMG" -m v2.data.hr2.build finalize --cand "$CAND" \
    --out "$R/pass1/out" --drop-groups "$Q/lists/drop-groups.txt" \
    --drop-dev-groups "$Q/lists/drop-dev-groups.txt"
  local S=$R/shortcut
  run families -v "$R/pass1:$R/pass1:ro" -v "$S:$S:rw" "$IMG" -m v2.data.hr2.audit families \
    --rows "$R/pass1/out/hr2.train.jsonl" --out-dir "$S/rows"
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
    if json.loads(path.read_text())["verdict"] == "FAIL":
        print(path.stem)
EOF
}

review() {
  fresh review
  local rows=$R/pass1/out/hr2.train.jsonl
  if [ -s "$R/shortcut/shortcut-fail.txt" ]; then
    local -a drops
    mapfile -t drops < "$R/shortcut/shortcut-fail.txt"
    run pass1b -v "$R/quarantine:$R/quarantine:ro" -v "$R/review:$R/review:rw" "$IMG" -m v2.data.hr2.build \
      finalize --cand "$CAND" --out "$R/review/pass1b" --drop-groups "$R/quarantine/lists/drop-groups.txt" \
      --drop-dev-groups "$R/quarantine/lists/drop-dev-groups.txt" --drop-families "${drops[@]}"
    rows=$R/review/pass1b/hr2.train.jsonl
  fi
  run sample -v "$R/pass1:$R/pass1:ro" -v "$R/review:$R/review:rw" "$IMG" -m v2.data.hr2.review sample \
    --train "$rows" --out-dir "$R/review/sample"
}

answers() {
  local who=$1 f
  for f in "$R"/review/answers/"$who".*.jsonl; do
    printf -- '--%s\n%s\n' "$who" "$f"
  done
}

splits() {
  local V=$R/review
  local -a args
  mapfile -t args < <(answers r1; answers r2)
  run splits -v "$V:$V:rw" "$IMG" -m v2.data.hr2.review splits --key "$V/sample/key.jsonl" "${args[@]}" \
    --packets "$V/sample/packet.r1.1.jsonl" --packets "$V/sample/packet.r1.2.jsonl" \
    --out "$V/answers/r3.packet.jsonl"
}

score() {
  local V=$R/review
  local -a args
  mapfile -t args < <(answers r1; answers r2)
  if [ -s "$V/answers/r3.jsonl" ]; then
    args+=(--r3 "$V/answers/r3.jsonl")
  fi
  run score -v "$V:$V:rw" "$IMG" -m v2.data.hr2.review score --sample "$V/sample/sample.json" \
    --key "$V/sample/key.jsonl" "${args[@]}" --out "$V/answers/review.public.json" \
    --private "$V/answers/review.private.json"
  # Gold errors leave TRAIN in any case; P3-failing families (fix rule F1) and G4 failures leave HR2.
  python3 - "$V/answers" "$R/shortcut/shortcut-fail.txt" "$R/final-drop-families.txt" <<'EOF'
import json, os, pathlib, sys
answers, shortcut, out = map(pathlib.Path, sys.argv[1:])
private = json.loads((answers / "review.private.json").read_text())
public = json.loads((answers / "review.public.json").read_text())
ids = sorted(item["id"] for item in private["errors"])
families = set(public["verdict"]["failing_families"])
families |= {line.strip() for line in shortcut.read_text().splitlines() if line.strip()}
for path, lines in ((answers / "drop-ids.txt", ids), (out, sorted(families))):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write("".join(line + "\n" for line in lines))
print(json.dumps({"drop_ids": len(ids), "drop_families": sorted(families)}))
EOF
}

final() {
  fresh final freeze
  local Q=$R/quarantine/lists V=$R/review/answers F=$R/final/out Z=$R/freeze
  local -a drops=()
  [ -e "$R/final-drop-families.txt" ] || { echo "run the score stage first" >&2; exit 1; }
  mapfile -t drops < "$R/final-drop-families.txt"
  run final -v "$R/quarantine:$R/quarantine:ro" -v "$R/review:$R/review:ro" -v "$R/final:$R/final:rw" \
    "$IMG" -m v2.data.hr2.build finalize --cand "$CAND" --out "$F" --drop-groups "$Q/drop-groups.txt" \
    --drop-dev-groups "$Q/drop-dev-groups.txt" --drop-ids "$V/drop-ids.txt" --drop-families "${drops[@]}"
  local TOKM=(-v "$TOKJ:$TOKJ:ro" -v "$Q06:$Q06:ro" -v "$Q08:$Q08:ro" -v "$KAI:$KAI:ro")
  run freeze-train "${TOKM[@]}" -v "$R/final:$R/final:ro" -v "$Z:$Z:rw" "$IMG" -m v2.data.freeze freeze \
    --rows "$F/hr2.train.jsonl" --arm-id HR2 --role train --license-registry "$LIC" --tokenizers "$TOKJ" \
    --out-manifest "$Z/hr2.train.manifest.json" &
  run freeze-dev "${TOKM[@]}" -v "$R/final:$R/final:ro" -v "$Z:$Z:rw" "$IMG" -m v2.data.freeze freeze \
    --rows "$F/hr2.dev.jsonl" --arm-id HR2 --role aho --license-registry "$LIC" --tokenizers "$TOKJ" \
    --out-manifest "$Z/hr2.dev.manifest.json" &
  run isolation -v "$R/final:$R/final:ro" -v "$HS1DEV:$HS1DEV:ro" -v "$PN1DEV:$PN1DEV:ro" -v "$HFD:$HFD:ro" \
    -v "$Z:$Z:rw" "$IMG" -m v2.data.freeze isolation --partition "train/HR2=$F/hr2.train.jsonl" \
    --partition "aho/HR2=$F/hr2.dev.jsonl" --partition "aho/HS1=$HS1DEV" \
    --partition "aho/PN1=$PN1DEV/pn1.dev.jsonl" --report "$Z/isolation.json" &
  local part
  for part in train dev; do
    run "tokens-$part" "${TOKM[@]}" -v "$R/final:$R/final:ro" -v "$Z:$Z:rw" "$IMG" -m v2.data.m2.row_tokens \
      --tokenizers "$TOKJ" --native qwen3.5-0.8b-base@dc7cdfe2 --raw kai-0.6b@7185f514 \
      --rows "$F/hr2.$part.jsonl" --out "$Z/hr2.$part.tokens.jsonl" &
  done
  wait
  run stats -v "$R/final:$R/final:ro" -v "$Z:$Z:rw" "$IMG" -m v2.data.hr2.audit stats \
    --train "$F/hr2.train.jsonl" --dev "$F/hr2.dev.jsonl" --tokens "$Z/hr2.train.tokens.jsonl" \
    --tokens "$Z/hr2.dev.tokens.jsonl" --out "$Z/stats.json"
}

case "$STAGE" in
  scans | pass1 | review | splits | score | final) "$STAGE" ;;
  *)
    echo "unknown stage $STAGE" >&2
    exit 2
    ;;
esac
echo "$(date -u +%FT%TZ) stage $STAGE finished" >> "$R/logs/steps.log"
