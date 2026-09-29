#!/usr/bin/env bash
# ~27B M3 F2 completion, phase 2 (node B host). Every 27B stage runs from mirror 35fa052d2,
# as F1's did; the gate and overlap tools run from the eval track's mirrors, as for F1.
#   f2-phase2.sh contrast      development contrast (v2.27b.m3_contrast through score_m3.sh; host CPU, 10,000
#                              draws, per seed and pooled, report-only floors), 17:15 soup rule and proxy screen
#                              -> m3-contrast/contrast.json; needs all six READOUT.json files
#   f2-phase2.sh pick          M3-S's candidate artifact from contrast.json -> "NAME LABEL MEMBER..."
#   f2-phase2.sh prepkg GPU    run_finalist.sh STAGES=cal698,adopt on the default frozen cache 583241fb (as F1)
#   f2-phase2.sh snapshot      verified cp -a of F1's post-run scored cache (03b172f1) -> m3-f2/f1-scored-cache
#   f2-phase2.sh formal GPU    run_finalist.sh STAGES=package,formal from the snapshot with F1 as EXTRA_COMPARATOR
#                              (formal/PAIRED-vs-M3-A-soup.json is F2 - F1), then the descriptive M2-S s1 compare
#                              (run_finalist.sh takes one extra comparator)
#   f2-phase2.sh gates         v2.eval.gates paired (three peers, and F1 - F2 outside both run dirs), types, the
#                              F1 record's item_cis.py driver (public 231 by tier, CSS15 per task), typed-FINAL
#                              families and Score levels (host CPU, eval mirror c8a5504b5, as for F1)
#   f2-phase2.sh overlap SPEC  v2.eval.overlap_effects exposure + run (host CPU, eval mirror 1bbebc2fe; 35fa052d2 has
#                              no overlap_effects.py); SPEC = the committed F2 spec (overlap-spec-27b-f2.json)
# A seed artifact gets its own NAME (<seed>-finalist): run_finalist.sh would otherwise write into the arm-seed
# directory and count its training receipts in GPU-HOURS.json.
set -euo pipefail
R=/data/dev2/runs/27b
SRC=35fa052d2b7c3ad0f9b2ee9bd1529e6b28076029-src_training_decision2
S=/data/dev2/src/$SRC/src/training/decision2
F1=$R/M3-A-soup/formal
F1_POST=03b172f1a6adeef6c6a6c491d04389b355c9d8579480008023f408c8659b502b
SNAP=$R/m3-f2/f1-scored-cache
CONTRAST=$R/m3-contrast/contrast.json
export TMPDIR=/data/dev2/tmp PYTHONDONTWRITEBYTECODE=1

artifact() {  # -> NAME LABEL MEMBER...; fails if the proxy screen drops the artifact
  python3 - "$CONTRAST" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
art = c["soup_rule"]["M3-S"]["artifact"]
if not c["proxy_screen"]["candidates"][art]["kept"]:
    raise SystemExit(f"{art} is dropped by the proxy screen")
print("M3-S-soup M3-S-soup M3-S-s1 M3-S-s2" if art == "M3-S-soup" else f"{art}-finalist {art} {art}")
EOF
}
finalist() {  # STAGES GPU [KEY=VALUE]...: run_finalist.sh on the artifact, appended to its driver log
  local stages=$1 gpu=$2 out
  shift 2
  out=$(artifact)
  read -r -a pick <<< "$out"
  echo "=== $(date -u +%FT%TZ) $stages ${pick[*]} on GPU$gpu" >> "$R/${pick[0]}.driver.log"
  (cd "$S" && env STAGES="$stages" LABEL="${pick[1]}" "$@" \
    bash v2/27b/run_finalist.sh "${pick[0]}" "$gpu" "$SRC" "${pick[@]:2}") >> "$R/${pick[0]}.driver.log" 2>&1
}

case "${1:-}" in
  contrast)
    for d in M3-A-s1 M3-A-s2 M3-S-s1 M3-S-s2 M3-A-soup M3-S-soup; do
      [ -f "$R/$d/readout-kernel-32768/READOUT.json" ] || { echo "$d: readout not scored yet" >&2; exit 1; }
    done
    mkdir -p "$(dirname "$CONTRAST")"
    SOUP_A=$R/M3-A-soup/readout-kernel-32768 SOUP_S=$R/M3-S-soup/readout-kernel-32768 \
      bash "$S/v2/27b/score_m3.sh" "$SRC" "$CONTRAST" 32768 > "${CONTRAST%.json}.log" 2>&1
    python3 - "$CONTRAST" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
print(json.dumps({"soup_rule": c["soup_rule"], "proxy_screen": c["proxy_screen"]["candidates"]}, indent=1))
EOF
    ;;
  pick)
    artifact
    ;;
  prepkg)
    finalist cal698,adopt "${2:?GPU}"
    ;;
  snapshot)
    [ ! -e "$SNAP" ] || { echo "$SNAP exists" >&2; exit 66; }
    cd "$S"
    export PYTHONPATH=$S
    [ "$(python3 -m v2.27b.triton_cache digest "$F1/triton-cache")" = "$F1_POST" ] \
      || { echo "F1's post-run cache no longer hashes to $F1_POST" >&2; exit 1; }
    python3 -m v2.27b.triton_cache copy --frozen "$F1/triton-cache" --expect "$F1_POST" --dest "$SNAP"
    python3 -m v2.27b.triton_cache digest "$SNAP"
    ;;
  formal)
    [ -d "$SNAP" ] || { echo "no snapshot $SNAP" >&2; exit 2; }
    finalist package,formal "${2:?GPU}" FROZEN="$SNAP" CACHE_SHA="$F1_POST" EXTRA_COMPARATOR="M3-A-soup=$F1"
    out=$(artifact)
    read -r -a pick <<< "$out"
    (cd "$S" && PYTHONPATH=$S python3 -m v2.eval.same_panel compare --run-dir "$R/${pick[0]}/formal" \
      --comparator-run-dir "$R/m2-S1-formal" --left-name "${pick[1]}" --right-name M2-S-s1-ref8192) \
      >> "$R/${pick[0]}.driver.log" 2>&1
    ;;
  gates)
    out=$(artifact)
    read -r -a pick <<< "$out"
    E=/data/dev2/src/c8a5504b5b39049390fccfbe3bcac5ef65e1e710-src_training_decision2/src/training/decision2
    G=$R/m3-f2/gates CAND=$R/${pick[0]}/formal
    [ ! -e "$G" ] || { echo "$G exists" >&2; exit 66; }
    mkdir -p "$G"
    cd "$E"
    export PYTHONPATH=$E
    date -u +%FT%TZ > "$G/run.start"
    python3 -m v2.eval.panels verify --panel typed-final --panel css15 --panel public231 > "$G/panels-verify.json"
    while read -r key name run; do
      python3 -m v2.eval.gates paired --left "$CAND" --right "$run" --left-name "DEV2.0-27B (F2)" \
        --right-name "$name" --output "$G/paired-vs-$key.json" > "$G/paired-vs-$key.log"
    done <<EOF
autojev27 AutoJev-27B $R/m2-peer-autojev27-nodeB-kernel
eikos27b Eikos-27B $R/m3-peer-eikos27-nodeB-kernel
jebadiah27b Jebadiah-27B $R/m3-peer-jebadiah-nodeB-kernel
EOF
    python3 -m v2.eval.gates paired --left "$F1" --right "$CAND" --left-name "DEV2.0-27B (F1)" \
      --right-name "DEV2.0-27B (F2)" --output "$G/paired-F1-minus-F2.json" > "$G/paired-F1-minus-F2.log"
    python3 -m v2.eval.gates types --run "$CAND" --label "DEV2.0-27B (F2)" --output "$G/types-cand.json" \
      > "$G/types-cand.log"
    cfg=$(python3 -c 'import json, sys; r, c, f1, g = sys.argv[1:]; peers = {"AutoJev-27B": f"{r}/m2-peer-autojev27-nodeB-kernel", "Eikos-27B": f"{r}/m3-peer-eikos27-nodeB-kernel", "Jebadiah-27B": f"{r}/m3-peer-jebadiah-nodeB-kernel"}; print(json.dumps({"models": {"DEV2.0-27B (F2)": c, "DEV2.0-27B (F1)": f1, **peers}, "pairs": [["DEV2.0-27B (F2)", n] for n in [*peers, "DEV2.0-27B (F1)"]], "output": f"{g}/item-cis.json", "jobs": 8}))' "$R" "$CAND" "$F1" "$G")
    python3 - "$cfg" > "$G/item-cis.log" \
      < /data/dev2/src/5553298a1a7e1c8b192eab1bb7c2699689b7a3dd-src_training_decision2/src/training/decision2/v2/eval/records/m4-dev2-27b-f1-gates/item_cis.py
    python3 - "$F1/REPORT.json" "$CAND/REPORT.json" "$G/types-cand.json" > "$G/families-score-levels.log" <<'EOF'
import json, sys
f1, f2, types = (json.load(open(p)) for p in sys.argv[1:])
fam1, fam2 = (r["panels"]["typed-final"]["by_family"] for r in (f1, f2))
for k in sorted(fam1):  # point values only: the tooling gives intervals for T, not per family
    print(f"{k}: F1 {fam1[k]:.4f}  F2 {fam2[k]:.4f}  F1-F2 {fam1[k] - fam2[k]:+.4f}")
score = types["types"]["score"]
print("Score predicted", score["predicted_distribution"], "gold", score["gold_distribution"])
print("Score recall by level", score["recall_by_level"])
EOF
    date -u +%FT%TZ > "$G/run.end"
    (cd "$G" && sha256sum -- * > SHA256SUMS.txt)
    ;;
  overlap)
    SPEC=${2:?SPEC: the committed F2 overlap spec}
    out=$(artifact)
    read -r -a pick <<< "$out"
    E=/data/dev2/src/1bbebc2fe3136858bcda63a12c2cde13c45d9238-src_training_decision2/src/training/decision2
    O=/data/dev2/runs/eval/m5/overlap-effects V=$R/m3-f2/overlap
    [ ! -e "$V" ] || { echo "$V exists" >&2; exit 66; }
    mkdir -p "$V"
    cd "$E"
    export PYTHONPATH=$E
    python3 -m v2.eval.overlap_effects exposure --groups "$O/final/excluded-groups.json" \
      --train /data/dev2/private/27b/m3-data/mixtures-m3-1/score_rep.train.jsonl \
      --expect-sha256 8402fe1c6dbc2d973dd13489b7a4098cb7c4a7cec486e93cf064be3dacb892e9 \
      --label "DEV2.0-27B F2 (${pick[1]}; M3-S mixture score_rep.train.jsonl)" \
      --output "$V/exposure-dev2-27b-f2.json" > "$V/exposure.log"
    python3 -m v2.eval.overlap_effects run --spec "$SPEC" --flagged "$O/final-27b/flagged.json" \
      --output "$V/overlap-effects.json" --jobs 8 > "$V/run.log" 2>&1
    ;;
  *)
    sed -n '2,18p' "$0" >&2
    exit 2
    ;;
esac
