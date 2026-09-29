#!/usr/bin/env bash
# Decoder M6 formal scoring on node A (prereg dec-m6-prereg-2026-09-29.md, "Formal runs" and "Successor rule"), from an
# exact mirror, on run directories under /data/dev2/runs/dec/formal/m6 (M6_FORMAL_ROOT): relayed gold-free from node B
# by m6-relay.sh pull (4b, 2b) or collected on node A (08b, 2b fallback).
#
#   m6-score.sh 2b ref                  node-B S2T reference: seal check, report, repeat vs the node-A S2T run and the
#                                       T = 1 derived run; REF-EXACT.json (exact = 0 differing answers on typed FINAL,
#                                       CSS15 and public 231 against both); relay it back with m6-relay.sh mark-ref
#   m6-score.sh <tier> <run>            finalist: seal check vs the receipt, report (tier, family decision2, parameters
#                                       from the header stubs), compares with the successor bar (T = 1 run), the tier
#                                       gates (own 1.0) and the peers, types gate (TYPES.json)
#   m6-score.sh <tier> mlx <run>        mlx-diag score + aggregates of <run>-mlx, then v2.dec.mlx_paired against the
#                                       tier's reference mlx-diag run (MLX-PAIRED.json)
#   m6-score.sh <tier> overlap <run>    v2.eval.overlap_effects run (flagged 0ed60dd7...) for successor item 6(b); the
#                                       candidate's exposure receipts are the incumbent's (inherited) plus the new
#                                       training files' (M6_EXPOSURE=comma list, from the data lock)
#   m6-score.sh <tier> successor <run> [<run> ...]
#                                       successor rule items 1-6 per run (m6_successor.py evaluate) and the tier choice
#                                       (m6_successor.py choose) -> successor/<tier>-*.json / .md
set -u
S=$(cd "$(dirname "$0")/../../../.." && pwd)
MIRROR=${S%/src/training/decision2}
[ -f "$MIRROR/.dev2-mirror.json" ] || { echo "run from an exact mirror under /data/dev2/src" >&2; exit 2; }
FA=${M6_FORMAL_ROOT:-/data/dev2/runs/dec/formal/m6}
D=/data/dev2/runs/dec/formal E=/data/dev2/runs/eval RL=/data/dev2/runs/release
MLX_PANEL=/data/dev2/private/panels/mlx-diag-v1
FLAGGED=$E/m5/overlap-effects/final/flagged.json
FLAGGED_SHA=0ed60dd751f2f33326b1b58859e27065166ea18e529812caa6ccec9fa894f41e
TIER=${1:-}
case $TIER in
  4b)
    TLABEL=4B
    CMP=("$RL/dev2-4b-t1-derived" bar-t1 "$D/m5/m5-ref-N4XF-soup" n4xf-ref "$E/m1-adopt/nox1" adopted-1.0
      "$E/m1-adopt/decider4b" decider4b "$E/m2/q5b-jet62" jet62)
    OWN=(adopted-1.0) PEERS=(decider4b jet62)
    MLX_REF=$D/m5/m5-ref-N4XF-soup-mlx MLX_REF_NAME=n4xf-ref
    INHERITED=$E/m5/overlap-effects/final-4b/exposure-dev2-4b-n4xf.json ;;
  2b)
    TLABEL=2B
    CMP=("$RL/dev2-2b-t1-derived" bar-t1 "$D/m3/sol1-16k" same-limit-16k "$E/m1-adopt/sol1" adopted-1.0
      "$E/m1-adopt/decider2b" decider2b "$E/m2/q4-thisthat12" thisthat12)
    OWN=(same-limit-16k adopted-1.0) PEERS=(decider2b thisthat12)
    if [ "${M6_2B_NODE:-B}" = A ]; then MLX_REF=$D/m3/m3-S2T-soup-mlx MLX_REF_NAME=s2t-nodeA
    else MLX_REF=$FA/m6-ref-S2T-soup-mlx MLX_REF_NAME=s2t-ref; fi
    INHERITED=$E/m5/overlap-effects/final/exposure-dev2-2b-s2t.json ;;
  08b)
    TLABEL=0.8B
    CMP=("$RL/dev2-0p8b-t1-derived" bar-t1 "$E/m1/r4-eos1" adopted-1.0 "$D/m2/eos1-16k" same-limit-16k
      "$E/m2/q2b-intern08b" intern "$E/m2/q1-kev08b" kev)
    OWN=(adopted-1.0 same-limit-16k) PEERS=(intern kev)
    MLX_REF=$D/m2/m2-E8F-soup-nodeA-mlx MLX_REF_NAME=e8f-nodeA
    INHERITED=$E/m5/overlap-effects/final/exposure-dev2-0p8b.json ;;
  *) sed -n '2,24p' "$0"; exit 2 ;;
esac
shift
mkdir -p "$FA"
flog() { echo "$(date -u +%FT%TZ) score $TIER $*" >> "$FA/OPERATIONS.log"; }
die() { flog "$*"; echo "$*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
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
  [ -f "$1/PAIRED-vs-$4.json" ] && return 0
  py -m v2.eval.same_panel compare --run-dir "$1" --comparator-run-dir "$3" --left-name "$2" --right-name "$4" \
    > "$1.compare-$4.log" 2>&1 || die "$(basename "$1") compare $4 FAILED"
}
# sealed_report <run dir> <label>: seal check against the receipt, report, parameter check.
sealed_report() {
  local R=$1 label=$2
  [ -f "$R/M6-RECEIPT.json" ] || die "$(basename "$R") has no M6-RECEIPT.json (m6-relay.sh pull, or collected elsewhere)"
  [ -f "$R/SEAL.json" ] || py -m v2.eval.same_panel seal --run-dir "$R" > "$R.seal.log" 2>&1 || die "$(basename "$R") seal FAILED"
  [ "$(sha "$R/SEAL.json")" = "$(jget "$R/M6-RECEIPT.json" seal_sha256)" ] || die "$(basename "$R") SEAL.json differs from the receipt"
  if [ ! -f "$R/REPORT.json" ]; then
    py -m v2.eval.same_panel report --run-dir "$R" --label "$label" --tier "$TLABEL" --family decision2 \
      --count-safetensors "$R/pkg-headers" > "$R.report.log" 2>&1 || die "$(basename "$R") report FAILED"
  fi
  [ "$(jget "$R/REPORT.json" parameters safetensors_count)" = "$(jget "$R/PARAMS.json" parameters)" ] \
    || die "$(basename "$R") parameter count differs from the staging count"
}
summary() {
  python3 - "$1" <<'EOF'
import glob, json, os, sys
r = sys.argv[1]
rep = json.load(open(f"{r}/REPORT.json"))
out = {"v3": rep["v3"]["score"], "T": rep["v3"]["T"], "H": rep["v3"]["H"], "public231": rep["panels"]["public231"]["correct"]}
for f in sorted(glob.glob(f"{r}/PAIRED-vs-*.json")):
    p = json.load(open(f))
    out[os.path.basename(f)[10:-5]] = {
        "delta": round(p["point"]["delta"]["score"], 3), "ci95": [round(p["ci95"]["low"], 3), round(p["ci95"]["high"], 3)],
        "H_ci95": [round(p["axis_ci95"]["H"]["delta"]["low"], 4), round(p["axis_ci95"]["H"]["delta"]["high"], 4)]}
if os.path.exists(f"{r}/TYPES.json"):
    out["types"] = {k: v["verdict"] for k, v in json.load(open(f"{r}/TYPES.json"))["types"].items()}
print(json.dumps(out))
EOF
}

CMD=${1:-}
case $CMD in
  ref)
    [ "$TIER" = 2b ] || die "ref is the 2B step"
    R=$FA/m6-ref-S2T-soup
    sealed_report "$R" "$(jget "$R/M6-RECEIPT.json" label)"
    for name in s2t-nodeA s2t-t1-derived; do
      if [ "$name" = s2t-nodeA ]; then dir=$D/m3/m3-S2T-soup-nodeA; else dir=$RL/dev2-2b-t1-derived; fi
      [ -f "$R/REPEAT-vs-$name.json" ] || py -m v2.eval.same_panel repeat --run-dir "$R" --comparator-run-dir "$dir" \
        --name "$name" > "$R.repeat-$name.log" 2>&1 || die "reference repeat vs $name FAILED"
      compare "$R" "$(jget "$R/M6-RECEIPT.json" label)" "$dir" "$name"
    done
    python3 - "$R" > "$R.exact.log" <<'EOF' || die "REF-EXACT FAILED"
import datetime as dt, hashlib, json, sys
from pathlib import Path
r = Path(sys.argv[1])
panels = ("typed-final", "css15", "public231")
out = {"schema": "dec-m6-ref-exact/1", "run": str(r), "rule": "0 differing answers on typed FINAL, CSS15 and public 231",
       "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "against": {}}
exact = True
for name in ("s2t-nodeA", "s2t-t1-derived"):
    f = r / f"REPEAT-vs-{name}.json"
    rep = json.loads(f.read_text())["panels"]
    changes = {p: rep[p]["category_changes"] if p in rep else None for p in panels}
    ok = all(c == 0 for c in changes.values()) and all(rep[p]["missing_left"] == rep[p]["missing_right"] == 0 for p in rep)
    out["against"][name] = {"repeat": str(f), "repeat_sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                            "category_changes": changes, "exact": ok}
    exact &= ok
out["exact"] = exact
out["seal_sha256"] = hashlib.sha256((r / "SEAL.json").read_bytes()).hexdigest()
out["report_sha256"] = hashlib.sha256((r / "REPORT.json").read_bytes()).hexdigest()
(r / "REF-EXACT.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
print(json.dumps({"exact": exact, **{k: v["category_changes"] for k, v in out["against"].items()}}))
EOF
    flog "2B reference: $(cat "$R.exact.log")"
    cat "$R.exact.log"
    ;;
  mlx)
    [ $# -eq 2 ] || { sed -n '2,24p' "$0"; exit 2; }
    R=$FA/$2 MR=$FA/$2-mlx
    [ -f "$R/REPORT.json" ] || die "$2: v3 report not written; mlx-diag is scored only after it"
    [ -f "$MR/output/mlx-diag.predictions.jsonl" ] || die "$2-mlx not collected / relayed"
    [ -f "$MR/mlx-diag.score.json" ] || py -m v2.eval.multilingual_panel score --panel "$MLX_PANEL" \
      --predictions "$MR/output/mlx-diag.predictions.jsonl" --output "$MR/mlx-diag.score.json" > "$MR.score.log" 2>&1 \
      || die "$2 mlx score FAILED"
    [ -f "$MR/mlx-agg.json" ] || python3 -B "$S/v2/dec/ops/m5/m5-mlx-agg.py" --panel "$MLX_PANEL" \
      --predictions "$MR/output/mlx-diag.predictions.jsonl" --score "$MR/mlx-diag.score.json" --output "$MR/mlx-agg.json" \
      > "$MR.agg.log" 2>&1 || die "$2 mlx aggregates FAILED"
    if [ "$2" != m6-ref-S2T-soup ]; then
      REFP=$MLX_REF/output/mlx-diag.predictions.jsonl
      [ -f "$REFP" ] || die "mlx-diag reference $MLX_REF has no predictions (2B: score m6-ref-S2T-soup-mlx first)"
      [ -f "$MR/MLX-PAIRED.json" ] || py -m v2.dec.mlx_paired --panel "$MLX_PANEL" --candidate "$MR/output/mlx-diag.predictions.jsonl" \
        --reference "$REFP" --candidate-name "$2" --reference-name "$MLX_REF_NAME" --output "$MR/MLX-PAIRED.json" \
        > "$MR.paired.log" 2>&1 || die "$2 mlx paired FAILED (see $MR.paired.log)"
      flog "$2 mlx paired vs $MLX_REF_NAME: $(tail -1 "$MR.paired.log")"
    fi
    flog "$2 mlx scored: $(tail -1 "$MR.agg.log")"
    ;;
  overlap)
    [ $# -eq 2 ] || { sed -n '2,24p' "$0"; exit 2; }
    R=$FA/$2 O=$FA/$2.overlap
    [ -f "$R/REPORT.json" ] || die "$2 not scored yet"
    [ "$(sha "$FLAGGED")" = "$FLAGGED_SHA" ] || die "flagged ids $FLAGGED are not $FLAGGED_SHA"
    [ ! -e "$O/overlap-effects.json" ] || die "$O/overlap-effects.json exists"
    mkdir -p "$O"
    python3 - "$O/spec.json" "$2" "$R" "$INHERITED" "${M6_EXPOSURE:-}" "${OWN[*]}" "${PEERS[*]}" "${CMP[@]}" <<'EOF' || die "overlap spec FAILED"
import json, sys
from pathlib import Path
out, cand, run, inherited, new, own, peers, *cmp = sys.argv[1:]
comparators = dict(zip(cmp[1::2], cmp[0::2]))
exposure = [inherited] + [p for p in new.split(",") if p]
models = {cand: {"run": run, "exposure": exposure,
                 "note": "M6 finalist; exposure = the incumbent's receipt (inherited through the incumbent weights) + the new training files'"}}
own, peers = own.split(), peers.split()
names = ["bar-t1"] + own + peers
for n in names:
    models[n] = {"run": comparators[n]}
spec = {
    "label": "post-key same-panel",
    "panel_root": "/data/dev2/private/panels",
    "focus_task": "media_ideology",
    "models": models,
    "tiers": {"m6": {"candidate": cand, "own_1_0": own, "peers": peers, "internal_peers": ["bar-t1"],
                     "threshold_pool": peers}},
    "reproduce": [{"left": cand, "right": n, "stored": f"{run}/PAIRED-vs-{n}.json"} for n in names
                  if Path(f"{run}/PAIRED-vs-{n}.json").is_file()],
}
Path(out).write_text(json.dumps(spec, indent=1) + "\n")
EOF
    py -m v2.eval.overlap_effects run --spec "$O/spec.json" --flagged "$FLAGGED" --output "$O/overlap-effects.json" \
      > "$O.log" 2>&1 || die "$2 overlap_effects FAILED or reported problems (see $O.log)"
    flog "$2 overlap: $(tail -1 "$O.log")"
    ;;
  successor)
    [ $# -ge 2 ] || { sed -n '2,24p' "$0"; exit 2; }
    shift
    OUTD=$FA/successor
    mkdir -p "$OUTD"
    results=()
    for run in "$@"; do
      R=$FA/$run
      exp=()
      IFS=, read -r -a new <<< "${M6_EXPOSURE:-}"
      for e in "${new[@]}"; do exp+=(--exposure "$e"); done
      python3 -B "$S/v2/dec/ops/m6/m6_successor.py" evaluate --tier "$TIER" --run "$R" --types "$R/TYPES.json" \
        --mlx-paired "$FA/$run-mlx/MLX-PAIRED.json" --overlap "$FA/$run.overlap/overlap-effects.json" "${exp[@]}" \
        --output "$OUTD/$TIER-$run" > "$OUTD/$TIER-$run.log" 2>&1 || die "$run successor evaluation FAILED"
      results+=(--result "$OUTD/$TIER-$run.json")
      flog "$run successor: $(tail -1 "$OUTD/$TIER-$run.log")"
    done
    python3 -B "$S/v2/dec/ops/m6/m6_successor.py" choose --tier "$TIER" "${results[@]}" --output "$OUTD/$TIER-choice" \
      > "$OUTD/$TIER-choice.log" 2>&1 || die "successor choice FAILED"
    flog "$TIER choice: $(tail -1 "$OUTD/$TIER-choice.log")"
    cat "$OUTD/$TIER-choice.log"
    ;;
  ""|-*) sed -n '2,24p' "$0"; exit 2 ;;
  *)
    RUN=$CMD R=$FA/$CMD
    [ "$(jget "$R/M6-RECEIPT.json" kind)" = finalist ] || die "$RUN is not a finalist run"
    [ "$(jget "$R/M6-RECEIPT.json" tier)" = "$TIER" ] || die "$RUN is not a tier-$TIER run"
    label=$(jget "$R/M6-RECEIPT.json" label)
    sealed_report "$R" "$label"
    set -- "${CMP[@]}"
    while [ $# -ge 2 ]; do compare "$R" "$label" "$1" "$2"; shift 2; done
    [ -f "$R/TYPES.json" ] || py -m v2.eval.gates types --run "$R" --label "$label" --output "$R/TYPES.json" \
      > "$R.types.log" 2>&1 || die "$RUN types gate FAILED"
    summary "$R" > "$R.summary.log"
    flog "$RUN scored: $(cat "$R.summary.log")"
    cat "$R.summary.log"
    ;;
esac
