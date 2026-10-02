#!/usr/bin/env bash
# 0.8B / 2B Index-first releases (user decision 2026-10-02 09:55; the winners M16 08b-RA-a75 / 2b-RA-a75): release
# inputs on node A, CPU only.
#   bf16   the v2.release.bf16_copy that the Index ran on (dec-indexpath/<point>-bf16-ckpt, copied from node C by
#          ix.sh push) -> inputs/dev2-<key>-ixf/bf16/{checkpoint,bf16-copy.json} (SHA-256 lists equal)
#   adopt  the sealed M16 formal run (node B collection, T = 1: CAL698 was rejected, no calibration) adopted unchanged,
#          sealed, reported and compared (the current revision's T = 1 run, own 1.0, the 16K control and the card
#          peers); its mlx-diag run scored; the reference evidence on that run (types, public 231 and card-eligible
#          mlx-diag vs the current revision); every check prints hashes only
#   paired v2.eval.gates paired files with named runs, each checked equal to the same_panel compare
# Usage (node A, from the exact mirror holding this file): bash <mirror>/.../ops/prep.sh 0p8b|2b bf16|adopt|paired
set -euo pipefail
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
KEY=${1:?0p8b|2b} STAGE=${2:?bf16|adopt|paired}
G=/data/dev2/private/panels/goldfree
MLXP=/data/dev2/private/panels/mlx-diag-v1
F=/data/dev2/runs/dec/formal/m16
REL=/data/dev2/runs/release
E=/data/dev2/runs/eval
DF=/data/dev2/runs/dec/formal
case "$KEY" in
  0p8b)
    POINT=08b-RA-a75 TIER=0.8B NAME=Decision-2.0-Eos-0.8B CUR=$REL/dev2-0p8b-t1-derived
    CURM_PRED=$REL/inputs/dev2-0p8b-t1/derived/mlx-diag.predictions.jsonl CURM_SCORE=$REL/dev2-0p8b-t1-derived-mlx/mlx-diag.score.json
    COMPARATORS="$CUR dev2-0p8b
$E/m1/r4-eos1 adopted-1.0
$DF/m2/eos1-16k same-limit-16k
$E/m2/q2b-intern08b intern
$E/m2/q1-kev08b kev
$E/m1/p2-jpt08b jpt08b"
    PAIRED_NAMES="dev2-0p8b adopted-1.0 intern kev" ;;
  2b)
    POINT=2b-RA-a75 TIER=2B NAME=Decision-2.0-Sol-2B CUR=$REL/dev2-2b-t1-derived
    CURM_PRED=$REL/inputs/dev2-2b-t1/derived/mlx-diag.predictions.jsonl CURM_SCORE=$REL/dev2-2b-t1-derived-mlx/mlx-diag.score.json
    COMPARATORS="$CUR dev2-2b
$DF/m3/sol1-16k same-limit-16k
$E/m1-adopt/sol1 adopted-1.0
$E/m1-adopt/decider2b decider2b
$E/m2/q4-thisthat12 thisthat12
$E/m1/p3-bosun17b bosun17b"
    PAIRED_NAMES="dev2-2b same-limit-16k decider2b thisthat12" ;;
  *) echo "tier key 0p8b or 2b" >&2; exit 2 ;;
esac
RUN=$F/m16-$POINT
MLX=$F/m16-$POINT-mlx
IN=$REL/inputs/dev2-$KEY-ixf
D=$REL/dev2-$KEY-ixf-t1
DM=$D-mlx
CURM=$IN/current-mlx
GATES=$IN/gates
BCK=/data/dev2/models/ix1/dec-indexpath/$POINT-bf16-ckpt
LEFT="$NAME M16 $POINT (T = 1)"
LABEL="$NAME successor M16 $POINT at T = 1 (the sealed M16 formal run, collected without calibration; post-key same-panel)"
export PYTHONPATH=$S:$S/v2/9b
cd "$S"
sums() { (cd "$1" && find . -type f | LC_ALL=C sort | xargs -r sha256sum); }

case "$STAGE" in
bf16)
  test ! -e "$IN/bf16" || { echo "$IN/bf16 exists" >&2; exit 3; }
  python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["model_sha256"] and r["source_model_sha256"]; print("bf16 copy", r["source_model_sha256"][:12], "->", r["model_sha256"][:12])' \
    "$BCK.receipt/bf16-copy.json"
  mkdir -p "$IN/bf16"
  cp -a "$BCK" "$IN/bf16/checkpoint"
  cp -p "$BCK.receipt/bf16-copy.json" "$IN/bf16/bf16-copy.json"
  [ "$(sums "$BCK")" = "$(sums "$IN/bf16/checkpoint")" ] || { echo "copy differs" >&2; exit 3; }
  echo "checkpoint $(sums "$IN/bf16/checkpoint" | wc -l) files; receipt $(sha256sum < "$IN/bf16/bf16-copy.json" | cut -c1-64)" ;;
adopt)
  test -d "$IN/bf16/checkpoint" || { echo "run bf16 first" >&2; exit 3; }
  test ! -e "$D" || { echo "$D exists" >&2; exit 3; }
  python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["calibration_used"] == "none", r["calibration_used"]; print("formal run", r["name"], "image", r["image_id"][:19], "calibration", r["calibration_used"])' \
    "$RUN/M6-RECEIPT.json"
  python3 - "$RUN/SEAL.json" "$RUN/output" <<'PY'
import hashlib, json, sys
seal = json.load(open(sys.argv[1]))
for panel, v in seal["panels"].items():
    got = hashlib.sha256(open(f"{sys.argv[2]}/{panel}.predictions.jsonl", "rb").read()).hexdigest()
    assert got == v["predictions_sha256"], (panel, got)
    print("sealed predictions match:", panel, got[:12])
PY
  mkdir -p "$IN"
  python3 -B -m v2.eval.same_panel adopt --run-dir "$D" \
    --typed-final "$RUN/output/typed-final.predictions.jsonl" --css15 "$RUN/output/css15.predictions.jsonl" \
    --public231 "$RUN/output/public231.predictions.jsonl" \
    --prior-receipt "$RUN/COLLECT.json" --prior-receipt "$RUN/SEAL.json" --prior-receipt "$RUN/M6-RECEIPT.json" \
    --reason "T = 1 (uncalibrated) predictions of the $TIER Index-first successor M16 $POINT: the sealed M16 formal run m16-$POINT (node B, image dbe5f32b), collected without calibration (CAL698 rejected), adopted unchanged; every Decision 2.0 model keeps T = 1"
  python3 -B -m v2.eval.same_panel seal --run-dir "$D"
  python3 -B -m v2.eval.same_panel report --run-dir "$D" --label "$LABEL" --tier "$TIER" --family decision2 \
    --count-safetensors "$IN/bf16/checkpoint"
  while read -r dir cname; do
    python3 -B -m v2.eval.same_panel compare --run-dir "$D" --comparator-run-dir "$dir" \
      --left-name "$LEFT" --right-name "$cname" > "$D/compare-$cname.log"
  done <<< "$COMPARATORS"
  python3 - "$D/REPORT.json" "$RUN/REPORT.json" <<'PY'
import json, sys
a, b = (json.load(open(p))["v3"] for p in sys.argv[1:3])
assert abs(a["score"] - b["score"]) < 1e-9 and a["T"] == b["T"] and a["H"] == b["H"], (a, b)
print("adopted run reproduces the formal report:", round(a["score"], 3))
PY
  mkdir -p "$DM/output" "$CURM/output"
  cp "$MLX/output/mlx-diag.predictions.jsonl" "$DM/output/"
  python3 -B -m v2.eval.multilingual_panel score --panel "$MLXP" \
    --predictions "$DM/output/mlx-diag.predictions.jsonl" --output "$DM/mlx-diag.score.json"
  cp "$CURM_PRED" "$CURM/output/"
  cp "$CURM_SCORE" "$CURM/"
  mkdir "$GATES"
  python3 -B -m v2.eval.gates types --run "$D" --label "$LEFT" --output "$GATES/types.json"
  python3 -B -m v2.eval.gates public231 --left "$D" --right "$CUR" --left-name "$LEFT" \
    --right-name "$NAME (current, T = 1)" --output "$GATES/public231-vs-current.json"
  python3 -B -m lux9b.mlx_paired --left "$DM" --right "$CURM" --panel "$MLXP" --left-name "$LEFT" \
    --right-name "$NAME (current, T = 1)" --output "$GATES/mlx-paired-vs-current.json"
  cp -p "$F/exposure-$POINT.json" "$GATES/exposure.json"
  sha256sum "$D"/*.json "$DM/mlx-diag.score.json" "$DM/output/"* "$CURM/output/"* "$GATES"/* | cut -c1-80 ;;
paired)
  while read -r dir cname; do
    [[ " $PAIRED_NAMES " == *" $cname "* ]] || continue
    python3 -B -m v2.eval.gates paired --left "$D" --right "$dir" --left-name "$LEFT" \
      --right-name "$cname" --output "$GATES/paired-vs-$cname.json"
    python3 - "$GATES/paired-vs-$cname.json" "$D/PAIRED-vs-$cname.json" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
assert a["point"] == b["point"] and a["ci95"] == b["ci95"] and a["axis_ci95"] == b["axis_ci95"], "paired differs"
print("paired equal to same_panel compare:", sys.argv[1].rsplit("/", 1)[-1])
PY
  done <<< "$COMPARATORS"
  sha256sum "$GATES"/paired-vs-*.json | cut -c1-80 ;;
*) echo "stage bf16|adopt|paired" >&2; exit 2 ;;
esac
