#!/bin/bash
# IB1 build, audits, reviews and publication on node A (prereg records/ib1-prereg-2026-10-01.md §2-§5).
# CPU only, in the pinned image with --network none; the mirror and every input are mounted read-only, only the
# step's output directory is writable. Every stage refuses to overwrite.
#
#   node_a.sh <commit> build        candidates from the pinned raw files
#   node_a.sh <commit> scans        G0 Index rows (reference, scan, positive controls); PI-ib1 inventory; overlap
#                                   vs PI-v4 (full, quarantining) and PI-ib1; DEV-vs-TRAIN self-scan; G1; G3 names
#   node_a.sh <commit> pass1        quarantine lists, finalize pass 1, per-family shortcut audits (G4)
#   node_a.sh <commit> screen       stage S sample and packets (G4-failing families left out)
#   node_a.sh <commit> screen-score stage S verdict from screen/answers/s1.*.jsonl
#   node_a.sh <commit> review       stage R sample and packets (screened rows / groups and dropped families left out)
#   node_a.sh <commit> splits       R3 packet from review/answers/{r1,r2}.*.jsonl
#   node_a.sh <commit> score        stage R verdict (review/answers/r3.jsonl if any)
#   node_a.sh <commit> final        finalize pass 2, freeze, isolation (G7), tokens and stats (G5, G8)
#   node_a.sh <commit> leak         ids of final rows with a leak-guard finding; a new `final` drops them
#   node_a.sh <commit> hf-assemble  m6/ib1 upload tree (registry.json inside) and the leak guard
#   node_a.sh <commit> hf-upload    private upload with the HF CLI, pinned revision, read-back check
set -euo pipefail
umask 077
SHA=$1
STAGE=$2
MIRROR=/data/dev2/src/${SHA}-src_training_decision2
CODE=$MIRROR/src/training/decision2
H=/data/dev2/private/data/ib1
RAW=$H/raw
R=$H/${IB1_RUN:-b1}
CAND=$R/cand
T=/data/dev2/tmp/ib1-${IB1_RUN:-b1}
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
TOKJ=/data/dev2/private/data/arms-v1/tokenizers.json
Q06=/data/dev2/hf-cache/models--Qwen--Qwen3-0.6B-Base
Q08=/data/dev2/hf-cache/models--Qwen--Qwen3.5-0.8B-Base
KAI=/data/dev2/private/models/Decision-1.0-Kai-0.6B@7185f514f54b8f93c55998b1e8f9c5cc67f0d029/native/tokenizer
LIC=$CODE/v2/data/records/license-registry-ib1.json
TRAIN=$CAND/ib1.train.cand.jsonl
DEV=$CAND/ib1.dev.cand.jsonl
WORKERS=${IB1_WORKERS:-48}

[ -d "$CODE" ] || { echo "no mirror $MIRROR" >&2; exit 1; }
for d in "$H" "$R" "$T" "$R/logs"; do
  [ -d "$d" ] || mkdir -m 700 "$d"
done
BASE=(--rm --network none --entrypoint python3 -w "$CODE" --cpu-shares 256
  -e PYTHONPATH=. -e TMPDIR="$T" -e CUDA_VISIBLE_DEVICES= -e HIP_VISIBLE_DEVICES=
  --label dev2.track=data-ib1 --label "dev2.commit=$SHA"
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
  docker run --name "ib1-${IB1_RUN:-b1}-$name" "${BASE[@]}" "$@" > "$R/logs/$name.log" 2>&1 || rc=$?
  echo "$(date -u +%FT%TZ) done $name exit=$rc" >> "$R/logs/steps.log"
  return "$rc"
}

build() {
  [ ! -e "$CAND" ] || { echo "$CAND exists" >&2; exit 1; }
  mkdir -m 700 "$CAND"
  run build -v "$RAW:$RAW:ro" -v "$CAND:$CAND:rw" "$IMG" -m v2.data.ib1.build build --raw "$RAW" --out "$CAND"
}

scans() {
  fresh g0 overlap names
  local G=$R/g0 O=$R/overlap
  local CM=(-v "$CAND:$CAND:ro")
  run g0-reference -v "$SUITE:$SUITE:ro" -v "$G:$G:rw" "$IMG" -m v2.data.ib1.index_guard reference \
    --suite "$SUITE/selected-rows.jsonl.gz" --suite "$SUITE/added-rows.jsonl.gz" --out-dir "$G/ref" \
    --workers "$WORKERS"
  run g0-scan "${CM[@]}" -v "$G:$G:rw" "$IMG" -m v2.data.ib1.index_guard scan --reference "$G/ref" \
    --candidates "$TRAIN" --candidates "$DEV" --out-dir "$G/scan" --workers "$WORKERS" &
  run g0-controls -v "$SUITE:$SUITE:ro" -v "$G:$G:rw" "$IMG" -m v2.data.ib1.index_guard controls \
    --suite "$SUITE/selected-rows.jsonl.gz" --suite "$SUITE/added-rows.jsonl.gz" --reference "$G/ref" \
    --out "$G/controls.json" &
  cat > "$O/pi-ib1.spec.json" <<EOF
[
 {"role": "ht_dev_goldfree", "origin": "$P/ht-dev.prompts.jsonl", "sha256": "30b0bd3569da9dd183f142e606ba6dcb598de7e4d3d9c168f87f91852178fd65", "project": true},
 {"role": "ht_dev2_goldfree", "origin": "$P/ht-dev2.prompts.jsonl", "sha256": "90cd409a1e091a623362c0e5b227d13b7301bf13fe266f7905e09233cf815f74", "project": true},
 {"role": "score5_dev_goldfree", "origin": "$P/score5-dev.prompts.jsonl", "sha256": "a01551c280473c9251ebbf8aed7bf9cfba3e12927def8959e4dd1283c072a86a", "project": true},
 {"role": "score5t_dev_goldfree", "origin": "$P/score5t-dev.prompts.jsonl", "sha256": "8e35bfffc2c3b8e4252d3c39ec1c65054250a3ddf80159d219a16b41bca1d93c", "project": true},
 {"role": "hs1_dev_goldfree", "origin": "$P/hs1-dev.prompts.jsonl", "sha256": "49f192a700242efe46265c4377a3cedb44dd635e5c5d23db1fc2d1e6fac3f072", "project": true},
 {"role": "pn1_dev_goldfree", "origin": "$PN1DEV/pn1.dev.prompts.jsonl", "sha256": "79dbf9996ad09caa622c4a9ea2f28075ddf80a5ae1ba9db8cd633573621bc14a", "project": true},
 {"role": "cal698", "origin": "$CAL698", "sha256": "19cc1a8c4ebe6fd13031079f6b7d131046ce503f435f894e3ea672f91c2ed41f", "project": true},
 {"role": "hr2_dev", "origin": "$HR2DEV", "sha256": "697c31428ae821d7c0238a64ced8b997cc5671be47307f88a273d082bc5aaebe", "project": true}
]
EOF
  run pi-ib1 -v "$P:$P:ro" -v "$HFD:$HFD:ro" -v "$O:$O:rw" "$IMG" \
    -m v2.data.build_protected_inventory --spec "$O/pi-ib1.spec.json" --out-dir "$O/pi-ib1"
  local cand=(--candidates "$TRAIN" --candidates "$DEV")
  run ov-piv4 "${CM[@]}" -v "$PI3:$PI3:ro" -v "$PI4:$PI4:ro" -v "$O:$O:rw" "$IMG" -m v2.data.overlap "${cand[@]}" \
    --protected-inventory "$PI4/manifest.json" --private-receipt "$O/piv4.private.json" \
    --public-receipt "$O/piv4.public.json" --workers "$WORKERS" &
  run ov-piv4q "${CM[@]}" -v "$PI3:$PI3:ro" -v "$PI4:$PI4:ro" -v "$O:$O:rw" "$IMG" -m v2.data.overlap "${cand[@]}" \
    --protected-inventory "$PI4/manifest.quarantining.json" --private-receipt "$O/piv4q.private.json" \
    --public-receipt "$O/piv4q.public.json" --workers "$WORKERS" &
  run ov-piib1 "${CM[@]}" -v "$O:$O:rw" "$IMG" -m v2.data.overlap "${cand[@]}" \
    --protected-inventory "$O/pi-ib1/manifest.json" --private-receipt "$O/piib1.private.json" \
    --public-receipt "$O/piib1.public.json" --workers "$WORKERS" &
  run ov-self "${CM[@]}" -v "$O:$O:rw" "$IMG" -m v2.data.overlap --self-scan --candidates "$DEV" \
    --candidates "$TRAIN" --private-receipt "$O/self.private.json" \
    --public-receipt "$O/self.public.json" --workers "$WORKERS" &
  run names -v "$R/names:$R/names:rw" "$IMG" -m v2.data.ib1.audit names --out "$R/names/g1.json"
  run c1-names "${CM[@]}" -v "$RAW:$RAW:ro" -v "$R/names:$R/names:rw" "$IMG" -m v2.eval.sealed.independence \
    names --terms v2/eval/sealed/c1-source-terms.json --root "$RAW" --root "$CAND" \
    --output "$R/names/c1-names.json"
  wait
}

pass1() {
  fresh quarantine
  local O=$R/overlap Q=$R/quarantine
  run quarantine -v "$CAND:$CAND:ro" -v "$O:$O:ro" -v "$PI4:$PI4:ro" -v "$Q:$Q:rw" "$IMG" -m v2.data.hr2.audit quarantine \
    --candidates "$TRAIN" --candidates "$DEV" --quarantining "$O/piv4q.private.json" \
    --quarantining "$O/piib1.private.json" --full "$O/piv4.private.json" \
    --quarantining-manifest "$PI4/manifest.quarantining.json" --self-scan "$O/self.private.json" \
    --out-dir "$Q/lists"
  fresh pass1 shortcut
  run pass1 -v "$CAND:$CAND:ro" -v "$Q:$Q:ro" -v "$R/g0:$R/g0:ro" -v "$R/pass1:$R/pass1:rw" "$IMG" \
    -m v2.data.ib1.build finalize --cand "$CAND" --out "$R/pass1/out" \
    --drop-groups "$Q/lists/drop-groups.txt" --drop-groups "$R/g0/scan/drop-groups.txt" \
    --drop-dev-groups "$Q/lists/drop-dev-groups.txt"
  local S=$R/shortcut
  run families -v "$R/pass1:$R/pass1:ro" -v "$S:$S:rw" "$IMG" -m v2.data.hr2.audit families \
    --rows "$R/pass1/out/ib1.train.jsonl" --out-dir "$S/rows"
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

screen() {
  fresh screen
  run screen-sample -v "$R/pass1:$R/pass1:ro" -v "$R/shortcut:$R/shortcut:ro" -v "$R/screen:$R/screen:rw" "$IMG" \
    -m v2.data.ib1.review screen-sample --train "$R/pass1/out/ib1.train.jsonl" \
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
  run screen-score -v "$S:$S:rw" "$IMG" -m v2.data.ib1.review screen-score --key "$S/sample/key.jsonl" "${args[@]}" \
    --out "$S/screen.public.json" --private "$S/screen.private.json" \
    --drop-families-out "$S/drop-families.txt" --drop-ids-out "$S/drop-ids.txt"
  cat "$R/shortcut/shortcut-fail.txt" "$S/drop-families.txt" | sort -u > "$S/families-out.txt"
}

review() {
  fresh review
  run sample -v "$R/pass1:$R/pass1:ro" -v "$R/screen:$R/screen:ro" -v "$R/review:$R/review:rw" "$IMG" \
    -m v2.data.ib1.review sample --train "$R/pass1/out/ib1.train.jsonl" --screen-key "$R/screen/sample/key.jsonl" \
    --drop-families "$R/screen/families-out.txt" --out-dir "$R/review/sample"
  mkdir -m 700 "$R/review/answers"
}

splits() {
  local V=$R/review
  local -a args
  mapfile -t args < <(answers "$V/answers" r1; answers "$V/answers" r2)
  run splits -v "$V:$V:rw" "$IMG" -m v2.data.ib1.review splits --key "$V/sample/key.jsonl" "${args[@]}" \
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
  run score -v "$V:$V:rw" "$IMG" -m v2.data.ib1.review score --sample "$V/sample/sample.json" \
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
  local Q=$R/quarantine/lists F=$R/final/out Z=$R/freeze
  local -a drops=() leak=()
  mapfile -t drops < "$R/screen/families-out.txt"
  if [ -e "$R/leak/drop-ids.txt" ]; then
    leak=(-v "$R/leak:$R/leak:ro" --drop-leak-ids "$R/leak/drop-ids.txt")
  fi
  run final -v "$CAND:$CAND:ro" -v "$R/quarantine:$R/quarantine:ro" -v "$R/g0:$R/g0:ro" -v "$R:$R:ro" \
    -v "$R/final:$R/final:rw" "${leak[@]:0:2}" "$IMG" -m v2.data.ib1.build finalize --cand "$CAND" --out "$F" \
    --drop-groups "$Q/drop-groups.txt" --drop-groups "$R/g0/scan/drop-groups.txt" \
    --drop-dev-groups "$Q/drop-dev-groups.txt" --drop-ids "$R/drop-ids.txt" "${leak[@]:2}" \
    --drop-families "${drops[@]}"
  local TOKM=(-v "$TOKJ:$TOKJ:ro" -v "$Q06:$Q06:ro" -v "$Q08:$Q08:ro" -v "$KAI:$KAI:ro")
  run freeze-train "${TOKM[@]}" -v "$R/final:$R/final:ro" -v "$Z:$Z:rw" "$IMG" -m v2.data.freeze freeze \
    --rows "$F/ib1.train.jsonl" --arm-id IB1 --role train --license-registry "$LIC" --tokenizers "$TOKJ" \
    --out-manifest "$Z/ib1.train.manifest.json" &
  run freeze-dev "${TOKM[@]}" -v "$R/final:$R/final:ro" -v "$Z:$Z:rw" "$IMG" -m v2.data.freeze freeze \
    --rows "$F/ib1.dev.jsonl" --arm-id IB1 --role aho --license-registry "$LIC" --tokenizers "$TOKJ" \
    --out-manifest "$Z/ib1.dev.manifest.json" &
  run isolation -v "$R/final:$R/final:ro" -v "$HS1DEV:$HS1DEV:ro" -v "$HFD:$HFD:ro" \
    -v "$Z:$Z:rw" "$IMG" -m v2.data.freeze isolation --partition "train/IB1=$F/ib1.train.jsonl" \
    --partition "aho/IB1=$F/ib1.dev.jsonl" --partition "aho/HS1=$HS1DEV" \
    --partition "aho/PN1=$PN1DEV/pn1.dev.jsonl" --partition "aho/HR2=$HR2DEV" --report "$Z/isolation.json" &
  local part
  for part in train dev; do
    run "tokens-$part" "${TOKM[@]}" -v "$R/final:$R/final:ro" -v "$Z:$Z:rw" "$IMG" -m v2.data.m2.row_tokens \
      --tokenizers "$TOKJ" --native qwen3.5-0.8b-base@dc7cdfe2 --raw kai-0.6b@7185f514 \
      --rows "$F/ib1.$part.jsonl" --out "$Z/ib1.$part.tokens.jsonl" &
  done
  wait
  run stats -v "$R/final:$R/final:ro" -v "$Z:$Z:rw" "$IMG" -m v2.data.ib1.audit stats \
    --train "$F/ib1.train.jsonl" --dev "$F/ib1.dev.jsonl" --tokens "$Z/ib1.train.tokens.jsonl" \
    --tokens "$Z/ib1.dev.tokens.jsonl" --out "$Z/stats.json"
}

# Ids of final rows with a leak-guard finding; the next `final` run (old final/ and freeze/ moved aside) drops them.
leak() {
  fresh leak
  local F=$R/final/out rc=0
  (cd "$F" && bash "$CODE/v2/common/check_no_private.sh" -- ib1.train.jsonl ib1.dev.jsonl) \
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
  python3 - "$R" "$CAND" "$CODE/v2/data/records" > "$spec" <<'EOF'
import json, pathlib, sys
run, cand, rec = map(pathlib.Path, sys.argv[1:])
items = [
    (rec / "ib1/hf-readme.md", "README.md"),
    (rec / "ib1/status.json", "status.json"),
    (rec / "license-registry-ib1.json", "license-registry-ib1.json"),
    (run / "final/out/ib1.train.jsonl", "ib1.train.jsonl"),
    (run / "final/out/ib1.dev.jsonl", "ib1.dev.jsonl"),
    (run / "final/out/final.json", "final.json"),
    (cand / "build.json", "build.json"),
    (run / "freeze/ib1.train.tokens.jsonl", "ib1.train.tokens.jsonl"),
    (run / "freeze/ib1.dev.tokens.jsonl", "ib1.dev.tokens.jsonl"),
    (run / "freeze/ib1.train.manifest.json", "train.manifest.json"),
    (run / "freeze/ib1.dev.manifest.json", "dev.manifest.json"),
    (run / "freeze/stats.json", "stats.json"),
    (run / "freeze/isolation.json", "isolation.json"),
    (run / "g0/scan/index-guard.public.json", "audits/index-guard.public.json"),
    (run / "g0/controls.json", "audits/index-guard-controls.json"),
    (run / "overlap/piv4.public.json", "audits/overlap-piv4.public.json"),
    (run / "overlap/piv4q.public.json", "audits/overlap-piv4q.public.json"),
    (run / "overlap/piib1.public.json", "audits/overlap-piib1.public.json"),
    (run / "overlap/self.public.json", "audits/overlap-dev-vs-train.public.json"),
    (run / "quarantine/lists/quarantine.public.json", "audits/quarantine.public.json"),
    (run / "names/g1.json", "audits/names-g1.json"),
    (run / "names/c1-names.json", "audits/c1-names.json"),
    (run / "screen/screen.public.json", "audits/screen.public.json"),
    (run / "screen/sample/sample.json", "audits/screen-sample.json"),
    (run / "review/answers/review.public.json", "audits/review.public.json"),
    (run / "review/sample/sample.json", "audits/review-sample.json"),
]
items += [(p, f"audits/shortcut/{p.name}") for p in sorted((run / "shortcut").glob("*.json"))]
print(json.dumps([{"src": str(src), "dst": "ib1/" + dst} for src, dst in items], indent=1))
EOF
  run hf-assemble -v "$R:$R:ro" -v "$CAND:$CAND:ro" -v "$R/hf:$R/hf:rw" "$IMG" -m v2.data.assemble_hf_upload \
    --spec "$spec" --out-dir "$R/hf/upload/m6"
  local rc=0
  (cd "$R/hf/upload/m6/ib1" && bash "$CODE/v2/common/check_no_private.sh" -- .) \
    > "$R/logs/hf-leak-guard.out" 2> "$R/logs/hf-leak-guard.err" || rc=$?
  echo "leak-guard exit=$rc $(tail -1 "$R/logs/hf-leak-guard.err")"
  find "$R/hf/upload/m6/ib1" -type f -printf '%s\t%P\n' | sort -k2
}

hf_upload() {
  export HF_HUB_CACHE=/data/dev2/hf-cache HF_HUB_DISABLE_TELEMETRY=1
  local repo=llm-semantic-router/decision-2.0-training-data tree=$R/hf/upload/m6/ib1
  local py=/data/dev2/tools/hf-cli/bin/python log=$R/logs/hf-upload.log
  local counts safe msg
  counts=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); t=d["train"]["rows"]; v=d["dev"]["rows"]; print(f"TRAIN {t:,}, DEV {v:,}")' "$R/final/out/final.json")
  safe=$(python3 -c 'import json,sys; print("release-safe" if json.load(open(sys.argv[1]))["release_safe"] else "NOT release-safe")' "$tree/status.json")
  msg="IB1 Index-family breadth block ($safe): m6/ib1 ($counts; build ${SHA:0:12})"
  info() {
    hf datasets info "$repo" --expand private,sha | "$py" -c 'import json,sys; d=json.load(sys.stdin); print(d["private"], d["sha"])'
  }
  [ -f "$tree/registry.json" ] || { echo "no assembled tree" >&2; exit 1; }
  [ ! -e "$R/hf/readback" ] || { echo "$R/hf/readback exists" >&2; exit 1; }
  local before parent after head rev
  read -r before parent < <(info)
  [ "$before" = "True" ] || { echo "dataset is not private; refusing to upload" >&2; exit 1; }
  echo "$(date -u +%FT%TZ) parent=$parent private=$before" | tee -a "$log"
  hf upload "$repo" "$tree" m6/ib1 --repo-type dataset --commit-message "$msg" >> "$log" 2>&1
  read -r after head < <(info)
  [ "$after" = "True" ] || { echo "dataset private flag changed" >&2; exit 1; }
  rev=$("$py" "$CODE/v2/data/hr2/hf_readback.py" pin "$repo" "$parent" "$msg")
  echo "$(date -u +%FT%TZ) head=$head revision=$rev private=$after" | tee -a "$log"
  mkdir -m 700 "$R/hf/readback"
  hf download "$repo" --repo-type dataset --revision "$rev" --include 'm6/ib1/*' --include 'm6/ib1/**' \
    --local-dir "$R/hf/readback" > /dev/null
  cmp "$R/hf/readback/m6/ib1/registry.json" "$tree/registry.json"
  "$py" "$CODE/v2/data/hr2/hf_readback.py" verify "$repo" "$rev" m6/ib1 "$tree" \
    "$R/hf/readback/m6/ib1" | tee "$R/hf/readback.json"
}

case "$STAGE" in
  build | scans | pass1 | screen | review | splits | score | final | leak) "$STAGE" ;;
  screen-score) screen_score ;;
  hf-assemble) hf_assemble ;;
  hf-upload) hf_upload ;;
  *)
    echo "unknown stage $STAGE" >&2
    exit 2
    ;;
esac
echo "$(date -u +%FT%TZ) stage $STAGE finished" >> "$R/logs/steps.log"
