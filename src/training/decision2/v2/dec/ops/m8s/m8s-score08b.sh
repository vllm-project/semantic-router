#!/usr/bin/env bash
# Decoder M8-small 0.8B formal scoring on node A (prereg dec-m8s-prereg-2026-09-30.md, "Formal", 0.8B): the node-B
# reference of the released weights (m8s-ref-08b-I) was not answer-identical to the node-A T = 1 binding, so it is
# the paired bar ("bar-t1") for items 1, 6(b) and 7, and its mlx-diag run pairs item 4. Everything else is
# ops/m6/m6-score.sh (reports, types gate, the tier-gate and peer comparators, mlx scoring, successor rule), run with
# M6_FORMAL_ROOT=/data/dev2/runs/dec/formal/m8s on runs relayed by m8s-relay.sh pull.
#   m8s-score08b.sh ref                  report of the node-B reference (parameters from its stubs)
#   m8s-score08b.sh run <run>            m6-score finalist step, then PAIRED-vs-bar-t1 against the node-B reference
#                                        (m6-score's node-A pairing is kept as PAIRED-vs-bar-t1-nodeA.json)
#   m8s-score08b.sh mlx <run>            MLX-PAIRED.json against the node-B reference's mlx-diag, then m6-score mlx
#   m8s-score08b.sh overlap <run>        overlap_effects with the node-B bar (exposure: inherited E8F + the top-up)
#   m8s-score08b.sh successor <run> ...  public-231 guard vs the node-B bar, then m6-score successor (items 1-8)
set -euo pipefail
S=$(cd "$(dirname "$0")/../../../.." && pwd)
export M6_FORMAL_ROOT=/data/dev2/runs/dec/formal/m8s
FA=$M6_FORMAL_ROOT E=/data/dev2/runs/eval D=/data/dev2/runs/dec/formal RL=/data/dev2/runs/release
REF=$FA/m8s-ref-08b-I REF_MLX=$FA/m8s-ref-08b-I-mlx
MLX_PANEL=/data/dev2/private/panels/mlx-diag-v1
FLAGGED=$E/m5/overlap-effects/final/flagged.json
INHERITED=$E/m5/overlap-effects/final/exposure-dev2-0p8b.json
TOPUP_EXPOSURE=$FA/exposure-08b.json
SCORE=$S/v2/dec/ops/m6/m6-score.sh
py() { (cd "$S" && PYTHONPATH=$S python3 -B "$@"); }
log() { echo "$(date -u +%FT%TZ) score08b $*" | tee -a "$FA/OPERATIONS.log"; }
CMD=${1:-}
case $CMD in
  ref)
    [ -f "$REF/SEAL.json" ] || { echo "no sealed reference at $REF" >&2; exit 1; }
    [ -f "$REF/REPORT.json" ] || py -m v2.eval.same_panel report --run-dir "$REF" --label "DEV2.0-0.8B bede7938 node-B reference" \
      --tier 0.8B --family decision2 --count-safetensors "$REF/pkg-headers" > "$REF.report.log" 2>&1
    log "reference report: $(python3 -c 'import json,sys; r=json.load(open(sys.argv[1]))["v3"]; print(r["score"], r["T"], r["H"])' "$REF/REPORT.json")"
    ;;
  run)
    R=$FA/$2
    bash "$SCORE" 08b "$2" > /dev/null
    if [ -f "$R/PAIRED-vs-bar-t1.json" ] && [ ! -f "$R/PAIRED-vs-bar-t1-nodeA.json" ]; then
      mv "$R/PAIRED-vs-bar-t1.json" "$R/PAIRED-vs-bar-t1-nodeA.json"
      py -m v2.eval.same_panel compare --run-dir "$R" --comparator-run-dir "$REF" --left-name "$2" --right-name bar-t1 \
        > "$R.compare-bar-t1-nodeB.log" 2>&1
    fi
    log "$2 vs node-B bar: $(python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print(round(p["point"]["delta"]["score"],3), [round(p["ci95"]["low"],3), round(p["ci95"]["high"],3)])' "$R/PAIRED-vs-bar-t1.json")"
    ;;
  mlx)
    MR=$FA/$2-mlx
    [ -f "$MR/MLX-PAIRED.json" ] || py -m v2.dec.mlx_paired --panel "$MLX_PANEL" --candidate "$MR/output/mlx-diag.predictions.jsonl" \
      --reference "$REF_MLX/output/mlx-diag.predictions.jsonl" --candidate-name "$2" --reference-name e8f-ref-nodeB \
      --output "$MR/MLX-PAIRED.json" > "$MR.paired-nodeB.log" 2>&1
    bash "$SCORE" 08b mlx "$2"
    log "$2 mlx vs node-B reference: $(tail -1 "$MR.paired-nodeB.log")"
    ;;
  overlap)
    R=$FA/$2 O=$FA/$2.overlap
    [ -f "$R/REPORT.json" ] || { echo "$2 not scored" >&2; exit 1; }
    [ ! -e "$O/overlap-effects.json" ] || { echo "$O exists" >&2; exit 1; }
    mkdir -p "$O"
    python3 - "$O/spec.json" "$2" "$R" "$INHERITED" "$TOPUP_EXPOSURE" "$REF" "$E" "$D" <<'EOF'
import json, sys
from pathlib import Path
out, cand, run, inherited, topup, ref, e, d = sys.argv[1:]
comparators = {"bar-t1": ref, "adopted-1.0": f"{e}/m1/r4-eos1", "same-limit-16k": f"{d}/m2/eos1-16k",
               "intern": f"{e}/m2/q2b-intern08b", "kev": f"{e}/m2/q1-kev08b"}
models = {cand: {"run": run, "exposure": [inherited, topup],
                 "note": "M8-small 0.8B finalist; exposure = the released E8F soup's receipt (inherited) + the top-up file's"}}
for n, path in comparators.items():
    models[n] = {"run": path}
spec = {"label": "post-key same-panel", "panel_root": "/data/dev2/private/panels", "focus_task": "media_ideology",
        "models": models,
        "tiers": {"m8s": {"candidate": cand, "own_1_0": ["adopted-1.0", "same-limit-16k"], "peers": ["intern", "kev"],
                          "internal_peers": ["bar-t1"], "threshold_pool": ["intern", "kev"]}},
        "reproduce": [{"left": cand, "right": n, "stored": f"{run}/PAIRED-vs-{n}.json"} for n in comparators
                      if Path(f"{run}/PAIRED-vs-{n}.json").is_file()]}
Path(out).write_text(json.dumps(spec, indent=1) + "\n")
EOF
    py -m v2.eval.overlap_effects run --spec "$O/spec.json" --flagged "$FLAGGED" --output "$O/overlap-effects.json" > "$O.log" 2>&1
    log "$2 overlap: $(tail -1 "$O.log" | cut -c1-200)"
    ;;
  successor)
    shift
    mkdir -p "$FA/successor"
    for run in "$@"; do
      PUB=$FA/successor/08b-$run.public231.json
      [ -f "$PUB" ] || py -m v2.eval.gates public231 --left "$FA/$run" --right "$REF" --left-name "$run" --right-name bar-t1 \
        --output "$PUB" > "$PUB.log" 2>&1
    done
    M6_EXPOSURE=$TOPUP_EXPOSURE bash "$SCORE" 08b successor "$@"
    ;;
  *) sed -n '2,13p' "$0"; exit 2 ;;
esac
