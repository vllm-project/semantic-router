#!/bin/bash
# HR2-r2 on node A (prereg amendment 4). CPU only, in the pinned image with --network none; the
# mirror, the HR2 run directory and the raw files are mounted read-only, only the step's output
# directory is writable. Every stage refuses to overwrite.
#
#   node_a_r2.sh <commit> analyze   round-1 error analysis and filter counts (reads round-1 files only)
#   node_a_r2.sh <commit> pass      PRM800K boundary list (C2), then finalize r2 from the HR2 candidates
#   node_a_r2.sh <commit> audits    G1 names, C1 source terms, overlap re-scan of the r2 files (PI-v4
#                                   full and quarantining, PI-hr2, DEV vs TRAIN) and the quarantine
#                                   recheck, per-family shortcuts (G4), leak guard on the r2 files
#   node_a_r2.sh <commit> review    round-2 sample and packets (round-1 rows and groups left out)
#   node_a_r2.sh <commit> splits    R3 packet from review/answers/{r1,r2}.*.jsonl
#   node_a_r2.sh <commit> score     review report and round-2 gold-error ids (no fix rule)
#   node_a_r2.sh <commit> final     finalize with the round-2 gold errors, freeze, isolation (G7),
#                                   tokens and stats (G5, G8), leak guard on the final files
#   node_a_r2.sh <commit> hf-assemble   m5/hr2 upload tree for r2 (registry.json inside), leak guard
#   node_a_r2.sh <commit> hf-upload     private upload replacing m5/hr2, pinned revision, read-back check
set -euo pipefail
umask 077
SHA=$1
STAGE=$2
MIRROR=/data/dev2/src/${SHA}-src_training_decision2
CODE=$MIRROR/src/training/decision2
H=/data/dev2/private/data/hr2
RAW=$H/raw
R1=$H/b2-c54b8d444cac
CAND=$R1/cand
R=$H/b2-c54b8d444cac-r2
T=/data/dev2/tmp/hr2-r2
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
WORKERS=${HR2_WORKERS:-32}
# amendment 4: the round-1 drops (G4 families, quarantine, gold errors, leak guard) plus L1, L2, C1, C2
DROPS=(--drop-groups "$R1/quarantine/lists/drop-groups.txt"
  --drop-dev-groups "$R1/quarantine/lists/drop-dev-groups.txt"
  --drop-ids "$R1/review/answers/drop-ids.txt" --drop-leak-ids "$R1/leak/drop-ids.txt"
  --drop-construction-ids "$R/lists/prm/construction-ids.txt"
  --drop-families indonli kob_boolq --drop-licence-families vitc allegro
  --drop-construction-families hs3_help)

[ -d "$CODE" ] || { echo "no mirror $MIRROR" >&2; exit 1; }
for d in "$T" "$R" "$R/logs"; do
  [ -d "$d" ] || mkdir -m 700 "$d"
done
BASE=(--rm --network none --entrypoint python3 -w "$CODE" --cpu-shares 256
  -e PYTHONPATH=. -e TMPDIR="$T" -e CUDA_VISIBLE_DEVICES= -e HIP_VISIBLE_DEVICES=
  --label dev2.track=data-hr2 --label "dev2.commit=$SHA"
  -v "$MIRROR:$MIRROR:ro" -v "$T:$T:rw" -v "$R1:$R1:ro")

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
  docker run --name "hr2-r2-$name" "${BASE[@]}" "$@" > "$R/logs/$name.log" 2>&1 || rc=$?
  echo "$(date -u +%FT%TZ) done $name exit=$rc" >> "$R/logs/steps.log"
  return "$rc"
}

# leak_ids <out dir> <rows dir> <files ...>: ids of rows with a shared leak-guard finding
# (amendment 3 rule).
leak_ids() {
  local out=$1 dir=$2 rc=0
  shift 2
  (cd "$dir" && bash "$CODE/v2/common/check_no_private.sh" -- "$@") > "$out/findings.txt" \
    2> "$out/guard.err" || rc=$?
  python3 - "$dir" "$out/findings.txt" > "$out/drop-ids.txt" <<'EOF'
import collections, json, pathlib, sys
root, findings = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
wanted = collections.defaultdict(set)
for line in findings.read_text().split("\n"):
    if line.strip():
        name, number, _ = line.split(":", 2)
        wanted[name.removeprefix("./")].add(int(number))
ids = set()
for name, numbers in wanted.items():
    lines = (root / name).read_text(encoding="utf-8").split("\n")
    ids |= {json.loads(lines[n - 1])["id"] for n in numbers}
print("\n".join(sorted(ids)))
EOF
  echo "leak-guard exit=$rc findings=$(grep -c . "$out/findings.txt" || true) rows=$(grep -c . "$out/drop-ids.txt" || true)"
}

analyze() {
  fresh analysis
  local A=$R/analysis
  run analyze -v "$RAW:$RAW:ro" -v "$A:$A:rw" "$IMG" -m v2.data.hr2.r2 analyze \
    --review "$R1/review/answers/review.private.json" \
    --reviewed "$R1/review/pass1b/hr2.train.jsonl" --train "$R1/final/out/hr2.train.jsonl" \
    --dev "$R1/final/out/hr2.dev.jsonl" --tokens "$R1/freeze/hr2.train.tokens.jsonl" \
    --tokens "$R1/freeze/hr2.dev.tokens.jsonl" --raw "$RAW" \
    --out "$A/analysis.public.json" --private "$A/analysis.private.json"
  tail -1 "$R/logs/analyze.log"
}

pass() {
  fresh lists pass
  run boundary -v "$RAW:$RAW:ro" -v "$R/lists:$R/lists:rw" "$IMG" -m v2.data.hr2.r2 boundary \
    --cand "$CAND" --raw "$RAW" --out-dir "$R/lists/prm"
  run pass -v "$R/lists:$R/lists:ro" -v "$R/pass:$R/pass:rw" "$IMG" -m v2.data.hr2.build \
    finalize --cand "$CAND" --out "$R/pass/out" "${DROPS[@]}"
  tail -1 "$R/logs/boundary.log"
  tail -1 "$R/logs/pass.log"
}

audits() {
  fresh overlap names quarantine shortcut leak-pass
  local O=$R/overlap F=$R/pass/out
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
  local PASS=(-v "$R/pass:$R/pass:ro")
  local cand=(--candidates "$F/hr2.train.jsonl" --candidates "$F/hr2.dev.jsonl")
  run ov-piv4 "${PASS[@]}" -v "$PI3:$PI3:ro" -v "$PI4:$PI4:ro" -v "$O:$O:rw" "$IMG" -m v2.data.overlap \
    "${cand[@]}" --protected-inventory "$PI4/manifest.json" --private-receipt "$O/piv4.private.json" \
    --public-receipt "$O/piv4.public.json" --workers "$WORKERS" &
  run ov-piv4q "${PASS[@]}" -v "$PI3:$PI3:ro" -v "$PI4:$PI4:ro" -v "$O:$O:rw" "$IMG" -m v2.data.overlap \
    "${cand[@]}" --protected-inventory "$PI4/manifest.quarantining.json" \
    --private-receipt "$O/piv4q.private.json" --public-receipt "$O/piv4q.public.json" \
    --workers "$WORKERS" &
  run ov-pihr2 "${PASS[@]}" -v "$O:$O:rw" "$IMG" -m v2.data.overlap "${cand[@]}" \
    --protected-inventory "$O/pi-hr2/manifest.json" --private-receipt "$O/pihr2.private.json" \
    --public-receipt "$O/pihr2.public.json" --workers "$WORKERS" &
  run ov-self "${PASS[@]}" -v "$O:$O:rw" "$IMG" -m v2.data.overlap --self-scan \
    --candidates "$F/hr2.dev.jsonl" --candidates "$F/hr2.train.jsonl" \
    --private-receipt "$O/self.private.json" --public-receipt "$O/self.public.json" \
    --workers "$WORKERS" &
  run names -v "$R/names:$R/names:rw" "$IMG" -m v2.data.hr2.audit names --out "$R/names/g1.json"
  run c1-names "${PASS[@]}" -v "$RAW:$RAW:ro" -v "$R/names:$R/names:rw" "$IMG" \
    -m v2.eval.sealed.independence names --terms v2/eval/sealed/c1-source-terms.json \
    --root "$RAW" --root "$F" --output "$R/names/c1-names.json"
  local S=$R/shortcut
  run families "${PASS[@]}" -v "$S:$S:rw" "$IMG" -m v2.data.hr2.audit families \
    --rows "$F/hr2.train.jsonl" --out-dir "$S/rows"
  local f name
  for f in "$S"/rows/*.jsonl; do
    name=$(basename "$f" .jsonl)
    run "sc-$name" -v "$S:$S:rw" "$IMG" -m v2.data.shortcut --rows "$f" \
      --receipt "$S/$name.json" --workers 4 &
  done
  wait
  run quarantine "${PASS[@]}" -v "$O:$O:ro" -v "$PI4:$PI4:ro" -v "$R/quarantine:$R/quarantine:rw" \
    "$IMG" -m v2.data.hr2.audit quarantine --candidates "$F/hr2.train.jsonl" \
    --candidates "$F/hr2.dev.jsonl" --quarantining "$O/piv4q.private.json" \
    --quarantining "$O/pihr2.private.json" --full "$O/piv4.private.json" \
    --quarantining-manifest "$PI4/manifest.quarantining.json" --self-scan "$O/self.private.json" \
    --out-dir "$R/quarantine/lists"
  python3 - "$S" > "$S/shortcut-fail.txt" <<'EOF'
import json, pathlib, sys
for path in sorted(pathlib.Path(sys.argv[1]).glob("*.json")):
    if json.loads(path.read_text())["verdict"] == "FAIL":
        print(path.stem)
EOF
  leak_ids "$R/leak-pass" "$F" hr2.train.jsonl hr2.dev.jsonl
  echo "quarantine recheck: $(tail -1 "$R/logs/quarantine.log")"
  echo "shortcut FAIL: $(tr '\n' ' ' < "$S/shortcut-fail.txt")"
  echo "names: $(tail -1 "$R/logs/names.log")"
  echo "c1-names: $(tail -1 "$R/logs/c1-names.log")"
}

review() {
  fresh review
  run sample -v "$R/pass:$R/pass:ro" -v "$R/review:$R/review:rw" "$IMG" -m v2.data.hr2.review \
    sample --round hr2-r2 --train "$R/pass/out/hr2.train.jsonl" \
    --exclude-key "$R1/review/sample/key.jsonl" --out-dir "$R/review/sample"
  tail -1 "$R/logs/sample.log"
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
  run splits -v "$V:$V:rw" "$IMG" -m v2.data.hr2.review splits --round hr2-r2 \
    --key "$V/sample/key.jsonl" "${args[@]}" --packets "$V/sample/packet.r1.1.jsonl" \
    --packets "$V/sample/packet.r1.2.jsonl" --out "$V/answers/r3.packet.jsonl"
  tail -1 "$R/logs/splits.log"
}

score() {
  local V=$R/review
  local -a args
  mapfile -t args < <(answers r1; answers r2)
  if [ -s "$V/answers/r3.jsonl" ]; then
    args+=(--r3 "$V/answers/r3.jsonl")
  fi
  run score -v "$V:$V:rw" "$IMG" -m v2.data.hr2.review score --round hr2-r2 \
    --sample "$V/sample/sample.json" --key "$V/sample/key.jsonl" "${args[@]}" \
    --out "$V/answers/review.public.json" --private "$V/answers/review.private.json"
  # Round-2 gold errors leave TRAIN in any case; amendment 4 has no fix rule.
  python3 - "$V/answers" <<'EOF'
import json, os, pathlib, sys
answers = pathlib.Path(sys.argv[1])
private = json.loads((answers / "review.private.json").read_text())
ids = sorted(item["id"] for item in private["errors"])
fd = os.open(answers / "drop-ids.txt", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w") as stream:
    stream.write("".join(line + "\n" for line in ids))
print(json.dumps({"drop_ids": len(ids)}))
EOF
  tail -1 "$R/logs/score.log"
}

final() {
  fresh final freeze leak-final
  local F=$R/final/out Z=$R/freeze
  [ -s "$R/review/answers/review.public.json" ] || { echo "run the score stage first" >&2; exit 1; }
  run final -v "$R/lists:$R/lists:ro" -v "$R/review:$R/review:ro" -v "$R/final:$R/final:rw" \
    "$IMG" -m v2.data.hr2.build finalize --cand "$CAND" --out "$F" "${DROPS[@]}" \
    --drop-ids "$R/review/answers/drop-ids.txt"
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
  leak_ids "$R/leak-final" "$F" hr2.train.jsonl hr2.dev.jsonl
  tail -1 "$R/logs/final.log"
  tail -1 "$R/logs/stats.log"
  grep -h "exit=" "$R/logs/steps.log" | tail -9
}

hf_assemble() {
  fresh hf
  local spec=$R/hf/spec.json
  python3 - "$R" "$R1" "$CODE/v2/data/records" > "$spec" <<'EOF'
import json, pathlib, sys
run, r1, rec = map(pathlib.Path, sys.argv[1:])
items = [
    (rec / "hr2/r2/hf-readme.md", "README.md"),
    (rec / "hr2/r2/status.json", "status.json"),
    (rec / "license-registry-hr2.json", "license-registry-hr2.json"),
    (run / "final/out/hr2.train.jsonl", "hr2.train.jsonl"),
    (run / "final/out/hr2.dev.jsonl", "hr2.dev.jsonl"),
    (run / "final/out/final.json", "final.json"),
    (r1 / "cand/build.json", "build.json"),
    (run / "freeze/hr2.train.tokens.jsonl", "hr2.train.tokens.jsonl"),
    (run / "freeze/hr2.dev.tokens.jsonl", "hr2.dev.tokens.jsonl"),
    (run / "freeze/hr2.train.manifest.json", "train.manifest.json"),
    (run / "freeze/hr2.dev.manifest.json", "dev.manifest.json"),
    (run / "freeze/stats.json", "stats.json"),
    (run / "freeze/isolation.json", "isolation.json"),
    (run / "analysis/analysis.public.json", "audits/r1-error-analysis.public.json"),
    (run / "lists/prm/boundary.public.json", "audits/construction-prm-boundary.public.json"),
    (r1 / "quarantine/lists/quarantine.public.json", "audits/quarantine.public.json"),
    (run / "quarantine/lists/quarantine.public.json", "audits/quarantine-recheck.public.json"),
    (run / "overlap/piv4.public.json", "audits/overlap-piv4.public.json"),
    (run / "overlap/piv4q.public.json", "audits/overlap-piv4q.public.json"),
    (run / "overlap/pihr2.public.json", "audits/overlap-pihr2.public.json"),
    (run / "overlap/self.public.json", "audits/overlap-dev-vs-train.public.json"),
    (run / "names/g1.json", "audits/names-g1.json"),
    (run / "names/c1-names.json", "audits/c1-names.json"),
    (run / "review/answers/review.public.json", "audits/review.public.json"),
    (run / "review/sample/sample.json", "audits/review-sample.json"),
    (r1 / "review/answers/review.public.json", "audits/review-r1.public.json"),
    (r1 / "review/sample/sample.json", "audits/review-r1-sample.json"),
]
items += [(p, f"audits/shortcut/{p.name}") for p in sorted((run / "shortcut").glob("*.json"))]
print(json.dumps([{"src": str(src), "dst": "hr2/" + dst} for src, dst in items], indent=1))
EOF
  run hf-assemble -v "$R:$R:ro" -v "$R/hf:$R/hf:rw" "$IMG" -m v2.data.assemble_hf_upload \
    --spec "$spec" --out-dir "$R/hf/upload/m5"
  local rc=0
  (cd "$R/hf/upload/m5/hr2" && bash "$CODE/v2/common/check_no_private.sh" -- .) \
    > "$R/logs/hf-leak-guard.out" 2> "$R/logs/hf-leak-guard.err" || rc=$?
  echo "leak-guard exit=$rc $(tail -1 "$R/logs/hf-leak-guard.err")"
  find "$R/hf/upload/m5/hr2" -type f -printf '%s\t%P\n' | sort -k2
}

hf_upload() {
  export HF_HUB_CACHE=/data/dev2/hf-cache HF_HUB_DISABLE_TELEMETRY=1
  local repo=llm-semantic-router/decision-2.0-training-data tree=$R/hf/upload/m5/hr2
  local py=/data/dev2/tools/hf-cli/bin/python log=$R/logs/hf-upload.log
  local counts safe msg
  counts=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); t=d["train"]["rows"]; v=d["dev"]["rows"]; print(f"TRAIN {t:,}, DEV {v:,}")' "$R/final/out/final.json")
  safe=$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["release_safe"]).lower())' "$tree/status.json")
  msg="HR2-r2 human-rated data (release_safe: $safe): m5/hr2 ($counts; amendment 4; build ${SHA:0:12})"
  info() {
    hf datasets info "$repo" --expand private,sha | "$py" -c 'import json,sys; d=json.load(sys.stdin); print(d["private"], d["sha"])'
  }
  [ -f "$tree/registry.json" ] || { echo "no assembled tree" >&2; exit 1; }
  [ ! -e "$R/hf/readback" ] || { echo "$R/hf/readback exists" >&2; exit 1; }
  local before parent after head rev
  read -r before parent < <(info)
  [ "$before" = "True" ] || { echo "dataset is not private; refusing to upload" >&2; exit 1; }
  echo "$(date -u +%FT%TZ) parent=$parent private=$before" | tee -a "$log"
  # --delete '*' is relative to m5/hr2: files of the round-1 folder that r2 does not carry go.
  hf upload "$repo" "$tree" m5/hr2 --repo-type dataset --delete '*' --commit-message "$msg" >> "$log" 2>&1
  read -r after head < <(info)
  [ "$after" = "True" ] || { echo "dataset private flag changed" >&2; exit 1; }
  rev=$("$py" "$CODE/v2/data/hr2/hf_readback.py" pin "$repo" "$parent" "$msg")
  echo "$(date -u +%FT%TZ) head=$head revision=$rev private=$after" | tee -a "$log"
  mkdir -m 700 "$R/hf/readback"
  hf download "$repo" --repo-type dataset --revision "$rev" --include 'm5/hr2/*' \
    --local-dir "$R/hf/readback" > /dev/null
  cmp "$R/hf/readback/m5/hr2/registry.json" "$tree/registry.json"
  "$py" "$CODE/v2/data/hr2/hf_readback.py" verify "$repo" "$rev" m5/hr2 "$tree" \
    "$R/hf/readback/m5/hr2" | tee "$R/hf/readback.json"
}

case "$STAGE" in
  analyze | pass | audits | review | splits | score | final) "$STAGE" ;;
  hf-assemble) hf_assemble ;;
  hf-upload) hf_upload ;;
  *)
    echo "unknown stage $STAGE" >&2
    exit 2
    ;;
esac
echo "$(date -u +%FT%TZ) stage $STAGE finished" >> "$R/logs/steps.log"
