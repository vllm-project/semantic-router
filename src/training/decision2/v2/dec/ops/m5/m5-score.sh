#!/usr/bin/env bash
# Decoder M5 formal scoring on node A (prereg dec-m5-prereg-2026-09-29.md, "Formal runs"), from an exact mirror, on
# node-B run directories relayed gold-free by m5-relay.sh pull into /data/dev2/runs/dec/formal/m5 (M5_FORMAL_ROOT).
#
#   m5-score.sh <run>            seal check, report (tier 4B, family decision2, parameters from the node-B header
#                                stubs), compare with the N4XF node-B reference (finalists; the successor bar), the
#                                adopted Nox 1.0 and Decider 4B runs; the reference is compared and repeat-checked
#                                against the node-A formal N4XF run instead of itself
#   m5-score.sh mlx <run>        mlx-diag score + aggregates (only after <run>/REPORT.json exists)
#   m5-score.sh check <existing run dir> <scratch dir> <label> <comparator run dir> <comparator name>
#                                read-only reproduction: copy an existing sealed run's gold-free files into a scratch
#                                dir, re-run report + compare there and diff against the existing PAIRED file
set -u
S=$(cd "$(dirname "$0")/../../../.." && pwd)
FA=${M5_FORMAL_ROOT:-/data/dev2/runs/dec/formal/m5}
REF_RUN=m5-ref-N4XF-soup
NODEA_N4XF=/data/dev2/runs/dec/formal/m4/m4-N4XF-soup-nodeA
NOX1=/data/dev2/runs/eval/m1-adopt/nox1
DECIDER=/data/dev2/runs/eval/m1-adopt/decider4b
MLX_PANEL=/data/dev2/private/panels/mlx-diag-v1
mkdir -p "$FA"
flog() { echo "$(date -u +%FT%TZ) score $*" >> "$FA/OPERATIONS.log"; }
die() { flog "$*"; echo "$*" >&2; exit 1; }
py() { (cd "$S" && PYTHONPATH=$S python3 -B "$@"); }
jget() {
  python3 - "$@" <<'EOF'
import json, sys
v = json.load(open(sys.argv[1]))
for k in sys.argv[2:]:
    v = v[k]
print(json.dumps(v) if isinstance(v, (list, dict)) else v)
EOF
}
compare() { # compare <run dir> <label> <comparator dir> <comparator name>
  py -m v2.eval.same_panel compare --run-dir "$1" --comparator-run-dir "$3" --left-name "$2" --right-name "$4" \
    > "$1.compare-$4.log" 2>&1 || die "$(basename "$1") compare $4 FAILED"
}

case ${1:-} in
  mlx)
    [ $# -eq 2 ] || { sed -n '2,15p' "$0"; exit 2; }
    R=$FA/$2 MR=$FA/$2-mlx
    [ -f "$R/REPORT.json" ] || die "$2: v3 report not sealed; mlx-diag is scored only after it"
    [ -f "$MR/output/mlx-diag.predictions.jsonl" ] || die "$2-mlx not relayed"
    [ -f "$MR/mlx-diag.score.json" ] || py -m v2.eval.multilingual_panel score --panel "$MLX_PANEL" \
      --predictions "$MR/output/mlx-diag.predictions.jsonl" --output "$MR/mlx-diag.score.json" > "$MR.score.log" 2>&1 \
      || die "$2 mlx score FAILED"
    python3 -B "$S/v2/dec/ops/m5/m5-mlx-agg.py" --panel "$MLX_PANEL" --predictions "$MR/output/mlx-diag.predictions.jsonl" \
      --score "$MR/mlx-diag.score.json" --output "$MR/mlx-agg.json" > "$MR.agg.log" 2>&1 || die "$2 mlx aggregates FAILED"
    flog "$2 mlx scored: $(tail -1 "$MR.agg.log")"
    ;;
  check)
    [ $# -eq 6 ] || { sed -n '2,15p' "$0"; exit 2; }
    SRC_RUN=$2 X=$3 label=$4 cmp=$5 name=$6
    [ ! -e "$X" ] || die "$X exists"
    mkdir -p "$X/output"
    cp "$SRC_RUN"/SEAL.json "$X/"
    for f in COLLECT.json ADOPT.json; do [ -f "$SRC_RUN/$f" ] && cp "$SRC_RUN/$f" "$X/"; done
    cp "$SRC_RUN"/output/*.predictions.jsonl "$X/output/"
    py -m v2.eval.same_panel report --run-dir "$X" --label "$label" --tier 4B --family decision2 > "$X.report.log" 2>&1 \
      || die "check report FAILED"
    compare "$X" "$label" "$cmp" "$name"
    python3 - "$SRC_RUN" "$X" "$name" <<'EOF'
import json, re, sys
a, b, name = sys.argv[1:]
safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
old, new = (json.load(open(f"{d}/PAIRED-vs-{safe}.json")) for d in (a, b))
ra, rb = (json.load(open(f"{d}/REPORT.json")) for d in (a, b))
print(json.dumps({
    "v3": [ra["v3"]["score"], rb["v3"]["score"]],
    "delta": [old["point"]["delta"]["score"], new["point"]["delta"]["score"]],
    "ci95": [old["ci95"], new["ci95"]],
    "identical_point_and_ci": old["point"] == new["point"] and old["ci95"] == new["ci95"]
    and old["axis_ci95"] == new["axis_ci95"],
}))
EOF
    ;;
  ""|-*) sed -n '2,15p' "$0"; exit 2 ;;
  *)
    RUN=$1 R=$FA/$1
    [ -f "$R/M5-RECEIPT.json" ] || die "$RUN not relayed (m5-relay.sh pull $RUN)"
    kind=$(jget "$R/M5-RECEIPT.json" kind) label=$(jget "$R/M5-RECEIPT.json" label)
    [ "$kind" = ref ] || [ "$kind" = finalist ] || die "$RUN is a $kind run"
    [ -f "$R/SEAL.json" ] || py -m v2.eval.same_panel seal --run-dir "$R" > "$R.seal.log" 2>&1 || die "$RUN seal FAILED"
    [ "$(sha256sum "$R/SEAL.json" | cut -d' ' -f1)" = "$(jget "$R/M5-RECEIPT.json" seal_sha256)" ] \
      || die "$RUN SEAL.json differs from the node-B receipt"
    if [ ! -f "$R/REPORT.json" ]; then
      py -m v2.eval.same_panel report --run-dir "$R" --label "$label" --tier 4B --family decision2 \
        --count-safetensors "$R/pkg-headers" > "$R.report.log" 2>&1 || die "$RUN report FAILED"
    fi
    [ "$(jget "$R/REPORT.json" parameters safetensors_count)" = "$(jget "$R/PARAMS.json" parameters)" ] \
      || die "$RUN parameter count differs from the node-B count"
    if [ "$kind" = ref ]; then
      compare "$R" "$label" "$NODEA_N4XF" n4xf-nodeA
      py -m v2.eval.same_panel repeat --run-dir "$R" --comparator-run-dir "$NODEA_N4XF" --name n4xf-nodeA \
        > "$R.repeat-n4xf-nodeA.log" 2>&1 || die "$RUN repeat vs node-A N4XF FAILED"
    else
      [ -f "$FA/$REF_RUN/REPORT.json" ] || die "reference $REF_RUN not scored yet"
      compare "$R" "$label" "$FA/$REF_RUN" n4xf-ref
    fi
    compare "$R" "$label" "$NOX1" adopted-1.0
    compare "$R" "$label" "$DECIDER" decider4b
    python3 - "$R" > "$R.summary.log" <<'EOF'
import glob, json, os, sys
r = sys.argv[1]
rep = json.load(open(f"{r}/REPORT.json"))
out = {"v3": rep["v3"]["score"], "public231": rep["panels"]["public231"]["correct"]}
for f in sorted(glob.glob(f"{r}/PAIRED-vs-*.json")):
    p = json.load(open(f))
    out[os.path.basename(f)[10:-5]] = {
        "delta": round(p["point"]["delta"]["score"], 3), "ci95": [round(p["ci95"]["low"], 3), round(p["ci95"]["high"], 3)],
        "H_delta": round(p["point"]["delta"]["H"], 4),
        "H_ci95": [round(p["axis_ci95"]["H"]["delta"]["low"], 4), round(p["axis_ci95"]["H"]["delta"]["high"], 4)]}
if os.path.exists(f"{r}/REPEAT-vs-n4xf-nodeA.json"):
    rp = json.load(open(f"{r}/REPEAT-vs-n4xf-nodeA.json"))["panels"]
    out["answer_changes_vs_nodeA"] = {k: v["category_changes"] for k, v in rp.items()}
print(json.dumps(out))
EOF
    flog "$RUN scored: $(cat "$R.summary.log")"
    cat "$R.summary.log"
    ;;
esac
