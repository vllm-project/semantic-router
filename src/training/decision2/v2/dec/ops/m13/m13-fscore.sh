#!/usr/bin/env bash
# Decoder M13 formal scoring on node A (amendment 2, dec-m13-amendment-2-2026-10-01.md), CPU only, from an exact mirror,
# on run directories under /data/dev2/runs/dec/formal/m13 pulled gold-free from node E / F over the transfer key.
# Successor items 1-7 against two bars per tier (as the 0.8B fast track, ops/f08):
#   4b   bar-lh  the released LH's stored formal run (formal/m10/m10-4b-LH-t1-derived, v3 67.345, T = 1)
#        bar-f   m13-4b-LH, node F's M13 formal-path collection of the same weights (the parity run)
#   08b  bar-t1  DEV2.0-0.8B's stored formal run (runs/release/dev2-0p8b-t1-derived, v3 50.236)
#        bar-e   m13-08b-C0, node E's M13 formal-path collection of the same weights
# Items 1, 2, 4, 6(b) and 7 must pass against both bars (m13_successor.py). m6-score.sh's own compares (its bar-t1,
# the adopted 1.0 run, peers) are kept; for 4B its bar-t1 (DEV2.0-4B) is reported only.
#
#   m13-fscore.sh <tier> formal-pull <user@node> <run>   formal/m13/<run> (gold-free outputs, receipts, seal, header
#                                                        stubs; no weights or caches), content-manifest checked
#   m13-fscore.sh <tier> formal-mark <user@node> <run>   REPORT.json / SEAL.json hashes -> <node>:<run>/V3-SEALED.json
#   m13-fscore.sh <tier> parity                          m6-score.sh on the tier's parity run, then same_panel repeat
#                                                        vs the stored bar -> <parity run>/PARITY-EXACT.json
#   m13-fscore.sh <tier> run <run>                       m6-score.sh <tier> <run>, then PAIRED-vs-<bar>.json for both bars
#   m13-fscore.sh <tier> mlx <run>                       mlx-diag score (m6-score.sh mlx), then MLX-PAIRED-<bar>.json
#                                                        against each bar's mlx-diag run
#   m13-fscore.sh <tier> exposure <user@node> <point>    overlap_effects exposure of <point>'s M13 TRAIN (streamed from
#                                                        its node, hash-checked) against the r2 payload
#   m13-fscore.sh <tier> overlap <run> <point>           overlap_effects run with both bars (item 6(b))
#   m13-fscore.sh <tier> successor <run> <point>         public-231 guards vs both bars, then m13_successor.py
set -u
S=$(cd "$(dirname "$0")/../../../.." && pwd)
MIRROR=${S%/src/training/decision2}
[ -f "$MIRROR/.dev2-mirror.json" ] || { echo "run from an exact mirror under /data/dev2/src" >&2; exit 2; }
export M6_FORMAL_ROOT=/data/dev2/runs/dec/formal/m13
FA=$M6_FORMAL_ROOT D=/data/dev2/runs/dec/formal E=/data/dev2/runs/eval RL=/data/dev2/runs/release
MLX_PANEL=/data/dev2/private/panels/mlx-diag-v1
FLAGGED=$E/m5/overlap-effects/final/flagged.json
FLAGGED_SHA=0ed60dd751f2f33326b1b58859e27065166ea18e529812caa6ccec9fa894f41e
PAYLOAD=$E/m5/overlap-effects/final/excluded-groups.json
PAYLOAD_SHA=2194716a179b2e6c3ba529dee3952c5dd62c0f3281638b7236029d5d542f4914
SCORE=$S/v2/dec/ops/m6/m6-score.sh
KEY=/root/.ssh/d2_temp_cd
TIER=${1:-}
case $TIER in
  4b)
    BAR1=bar-lh BAR1_RUN=$D/m10/m10-4b-LH-t1-derived BAR1_MLX=$D/m10/m10-4b-LH-mlx
    BAR2=bar-f PARITY=m13-4b-LH
    OTHERS=("adopted-1.0=$E/m1-adopt/nox1" "n4xf-ref=$D/m5/m5-ref-N4XF-soup" "decider4b=$E/m1-adopt/decider4b"
      "jet62=$E/m2/q5b-jet62")
    OWN='["adopted-1.0"]' PEERS='["decider4b", "jet62"]' ;;
  08b)
    BAR1=bar-t1 BAR1_RUN=$RL/dev2-0p8b-t1-derived BAR1_MLX=$D/m2/m2-E8F-soup-nodeA-mlx
    BAR2=bar-e PARITY=m13-08b-C0
    OTHERS=("adopted-1.0=$E/m1/r4-eos1" "same-limit-16k=$D/m2/eos1-16k" "intern=$E/m2/q2b-intern08b"
      "kev=$E/m2/q1-kev08b")
    OWN='["adopted-1.0", "same-limit-16k"]' PEERS='["intern", "kev"]' ;;
  *) sed -n '2,25p' "$0"; exit 2 ;;
esac
shift
BAR2_RUN=$FA/$PARITY
train_of() { # train_of <point> -> "<path> <sha256>" (M13 data lock)
  case $1 in
    4b-LHA10SD) echo "/data/dev2/runs/dec/m13/data/4b/4b-LHA10SD/train.jsonl d41cdd1aa1e63b47394ed8fa61a87bf461cb4d2cea08cc7ff12f2e58b36dc9d5" ;;
    08b-RASD) echo "/data/dev2/runs/dec/m13/data/08b/08b-RASD/train.jsonl 12bd63d8b215877b6f471c12652e534e0f0be794c15bff74ad5ac50e98018bf3" ;;
    08b-RAAG) echo "/data/dev2/runs/dec/m13/data/08b/08b-RAAG/train.jsonl 69b8d00d6cdd2d48ee0c0d840ba07954eba404aa48ffbaf34a6cd3d8aefbb6cd" ;;
    *) return 1 ;;
  esac
}
mkdir -p "$FA/successor"
log() { echo "$(date -u +%FT%TZ) m13-fscore $TIER $*" | tee -a "$FA/OPERATIONS-nodeA.log"; }
die() { log "$*"; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
py() { (cd "$S" && PYTHONPATH=$S python3 -B "$@"); }
jget() { python3 -c 'import json,sys; v=json.load(open(sys.argv[1]))
for k in sys.argv[2:]: v=v[k]
print(json.dumps(v) if isinstance(v,(list,dict)) else v)' "$@"; }
exec 9> "$FA/.m13-fscore.lock"
flock -n 9 || die "another m13-fscore.sh is running"
paired_line() { python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print(round(p["point"]["delta"]["score"],3), [round(p["ci95"]["low"],3), round(p["ci95"]["high"],3)])' "$1"; }

case ${1:-} in
  formal-pull)
    [ $# -eq 3 ] || { sed -n '2,25p' "$0"; exit 2; }
    from=$2 run=$3
    on() { ssh -i "$KEY" -o BatchMode=yes "$from" "$@"; }
    manifest='find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -d" " -f1'
    [ ! -e "$FA/$run" ] || die "$FA/$run exists on node A"
    on "test -f '$FA/$run/M6-RECEIPT.json'" || die "$run has no M6 receipt on its node"
    on "! find '$FA/$run' -iname '*gold*' | grep -q ." || die "$run contains gold-named files"
    on "! find '$FA/$run' -name '*.safetensors' -size +1M | grep -q ." || die "$run contains weights"
    mb=$(on "cd '$FA/$run' && $manifest")
    rsync -a -e "ssh -i $KEY -o BatchMode=yes" "$from:$FA/$run/" "$FA/$run/" || die "pull of $run failed"
    ma=$(cd "$FA/$run" && eval "$manifest")
    [ "$ma" = "$mb" ] || die "$run manifest differs after the pull ($mb vs $ma)"
    log "formal run $run pulled ($ma)"
    ;;
  formal-mark)
    [ $# -eq 3 ] || { sed -n '2,25p' "$0"; exit 2; }
    from=$2 run=$3
    on() { ssh -i "$KEY" -o BatchMode=yes "$from" "$@"; }
    seal_a=$(sha "$FA/$run/SEAL.json")
    seal_b=$(on "sha256sum '$FA/$run/SEAL.json' | cut -d' ' -f1")
    [ "$seal_a" = "$seal_b" ] || die "$run: node-A seal differs from its node"
    [ -f "$FA/$run/REPORT.json" ] || die "$run has no report yet"
    python3 -c 'import json,sys,datetime as d; print(json.dumps({"run": sys.argv[1], "report_sha256": sys.argv[2], "seal_sha256": sys.argv[3], "marked_utc": d.datetime.now(d.timezone.utc).isoformat()}))' \
      "$run" "$(sha "$FA/$run/REPORT.json")" "$seal_a" \
      | on "set -o noclobber; cat > '$FA/$run/V3-SEALED.json'" || die "mark of $run failed"
    log "formal run $run marked on its node"
    ;;
  parity)
    R=$BAR2_RUN
    bash "$SCORE" "$TIER" "$PARITY" > /dev/null || die "parity run $PARITY report / compares FAILED"
    [ -f "$R/REPEAT-vs-$BAR1.json" ] || py -m v2.eval.same_panel repeat --run-dir "$R" --comparator-run-dir "$BAR1_RUN" \
      --name "$BAR1" > "$R.repeat-$BAR1.log" 2>&1 || die "parity repeat vs $BAR1 FAILED"
    python3 - "$R" "$BAR1" > "$R.exact.log" << 'EOF' || die "PARITY-EXACT FAILED"
import datetime as dt, hashlib, json, sys
from pathlib import Path
r, bar = Path(sys.argv[1]), sys.argv[2]
panels = ("typed-final", "css15", "public231")
f = r / f"REPEAT-vs-{bar}.json"
rep = json.loads(f.read_text())["panels"]
changes = {p: rep[p]["category_changes"] if p in rep else None for p in panels}
ok = all(c == 0 for c in changes.values()) and all(rep[p]["missing_left"] == rep[p]["missing_right"] == 0 for p in rep)
out = {"schema": "dec-m13-parity-exact/1", "run": str(r), "stored_bar": bar,
       "rule": "0 differing answers on typed FINAL, CSS15 and public 231", "category_changes": changes, "exact": ok,
       "repeat": str(f), "repeat_sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
       "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
       "seal_sha256": hashlib.sha256((r / "SEAL.json").read_bytes()).hexdigest(),
       "report_sha256": hashlib.sha256((r / "REPORT.json").read_bytes()).hexdigest()}
(r / "PARITY-EXACT.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
print(json.dumps({"exact_vs_stored_bar": ok, "category_changes": changes}))
EOF
    log "parity $PARITY: $(cat "$R.exact.log"); $(cat "$R.summary.log")"
    cat "$R.exact.log"
    ;;
  run)
    [ $# -eq 2 ] || { sed -n '2,25p' "$0"; exit 2; }
    R=$FA/$2
    [ -f "$BAR2_RUN/PARITY-EXACT.json" ] || die "score the parity run first (m13-fscore.sh $TIER parity)"
    bash "$SCORE" "$TIER" "$2" > /dev/null || die "$2 m6-score FAILED"
    label=$(jget "$R/M6-RECEIPT.json" label)
    for pair in "$BAR1=$BAR1_RUN" "$BAR2=$BAR2_RUN"; do
      name=${pair%%=*} dir=${pair#*=}
      [ -f "$R/PAIRED-vs-$name.json" ] || py -m v2.eval.same_panel compare --run-dir "$R" --comparator-run-dir "$dir" \
        --left-name "$label" --right-name "$name" > "$R.compare-$name.log" 2>&1 || die "$2 compare vs $name FAILED"
    done
    log "$2 scored: $(cat "$R.summary.log"); vs $BAR1 $(paired_line "$R/PAIRED-vs-$BAR1.json"); vs $BAR2 $(paired_line "$R/PAIRED-vs-$BAR2.json")"
    ;;
  mlx)
    [ $# -eq 2 ] || { sed -n '2,25p' "$0"; exit 2; }
    MR=$FA/$2-mlx
    bash "$SCORE" "$TIER" mlx "$2" || die "$2 m6-score mlx FAILED"
    if [ "$2" != "$PARITY" ]; then
      for pair in "$BAR1=$BAR1_MLX" "$BAR2=$BAR2_RUN-mlx"; do
        name=${pair%%=*} REFP=${pair#*=}/output/mlx-diag.predictions.jsonl
        [ -f "$REFP" ] || die "$name has no mlx-diag predictions yet"
        [ -f "$MR/MLX-PAIRED-$name.json" ] || py -m v2.dec.mlx_paired --panel "$MLX_PANEL" \
          --candidate "$MR/output/mlx-diag.predictions.jsonl" --reference "$REFP" --candidate-name "$2" \
          --reference-name "$name" --output "$MR/MLX-PAIRED-$name.json" > "$MR.paired-$name.log" 2>&1 \
          || die "$2 mlx paired vs $name FAILED (see $MR.paired-$name.log)"
        log "$2 mlx vs $name: $(tail -1 "$MR.paired-$name.log" | cut -c1-300)"
      done
    fi
    ;;
  exposure)
    [ $# -eq 3 ] || { sed -n '2,25p' "$0"; exit 2; }
    from=$2 point=$3
    read -r train want < <(train_of "$point") || die "no M13 TRAIN for $point"
    [ ${#want} -eq 64 ] || die "$point has no full TRAIN hash in the data lock"
    X=$FA/exposure-$point.json
    [ ! -e "$X" ] || die "$X exists"
    [ "$(sha "$PAYLOAD")" = "$PAYLOAD_SHA" ] || die "r2 payload is not $PAYLOAD_SHA"
    ssh -i "$KEY" -o BatchMode=yes "$from" "cat '$train'" | py -m v2.eval.overlap_effects exposure --groups "$PAYLOAD" \
      --train - --expect-sha256 "$want" --label "decoder M13 $point TRAIN (data lock debd36e1a)" \
      --output "$X" > "$FA/exposure-$point.log" 2>&1 || die "exposure receipt FAILED (see $FA/exposure-$point.log)"
    log "exposure $point: $(tail -1 "$FA/exposure-$point.log" | cut -c1-300)"
    ;;
  overlap)
    [ $# -eq 3 ] || { sed -n '2,25p' "$0"; exit 2; }
    R=$FA/$2 O=$FA/$2.overlap X=$FA/exposure-$3.json
    [ -f "$R/PAIRED-vs-$BAR2.json" ] || die "$2 not scored yet"
    [ -f "$X" ] || die "no exposure receipt for $3 yet"
    [ "$(sha "$FLAGGED")" = "$FLAGGED_SHA" ] || die "flagged ids $FLAGGED are not $FLAGGED_SHA"
    [ ! -e "$O/overlap-effects.json" ] || die "$O/overlap-effects.json exists"
    mkdir -p "$O"
    python3 - "$O/spec.json" "$2" "$R" "$X" "$OWN" "$PEERS" "$BAR1=$BAR1_RUN" "$BAR2=$BAR2_RUN" "${OTHERS[@]}" << 'EOF' || die "overlap spec FAILED"
import json, sys
from pathlib import Path
out, cand, run, exposure, own, peers, *pairs = sys.argv[1:]
comparators = dict(p.split("=", 1) for p in pairs)
bars = list(comparators)[:2]
models = {cand: {"run": run, "exposure": [exposure],
                 "note": "M13 finalist: no incumbent weight (retrained from its Qwen base / released recipe); exposure = its TRAIN file's receipt"}}
for n, path in comparators.items():
    models[n] = {"run": path}
own, peers = json.loads(own), json.loads(peers)
spec = {"label": "post-key same-panel", "panel_root": "/data/dev2/private/panels", "focus_task": "media_ideology",
        "models": models,
        "tiers": {"m13": {"candidate": cand, "own_1_0": own, "peers": peers, "internal_peers": bars,
                          "threshold_pool": peers}},
        "reproduce": [{"left": cand, "right": n, "stored": f"{run}/PAIRED-vs-{n}.json"} for n in [*bars, *own, *peers]
                      if Path(f"{run}/PAIRED-vs-{n}.json").is_file()]}
Path(out).write_text(json.dumps(spec, indent=1) + "\n")
EOF
    py -m v2.eval.overlap_effects run --spec "$O/spec.json" --flagged "$FLAGGED" --output "$O/overlap-effects.json" \
      > "$O.log" 2>&1 || die "$2 overlap_effects FAILED or reported problems (see $O.log)"
    log "$2 overlap: $(tail -1 "$O.log" | cut -c1-300)"
    ;;
  successor)
    [ $# -eq 3 ] || { sed -n '2,25p' "$0"; exit 2; }
    run=$2 R=$FA/$2 OUTD=$FA/successor
    bars=()
    for pair in "$BAR1=$BAR1_RUN" "$BAR2=$BAR2_RUN"; do
      name=${pair%%=*} dir=${pair#*=}
      PUB=$OUTD/$TIER-$run.public231-$name.json
      [ -f "$PUB" ] || py -m v2.eval.gates public231 --left "$R" --right "$dir" --left-name "$run" --right-name "$name" \
        --output "$PUB" > "$PUB.log" 2>&1 || die "$run public231 guard vs $name FAILED to run"
      bars+=(--bar "$name=$FA/$run-mlx/MLX-PAIRED-$name.json,$PUB")
    done
    c1=()
    [ -f "$FA/$run.c1/SUMMARY.json" ] && c1=(--c1 "$FA/$run.c1/SUMMARY.json")
    python3 -B "$S/v2/dec/ops/m13/m13_successor.py" --tier "$TIER" --run "$R" --types "$R/TYPES.json" "${bars[@]}" \
      --overlap "$FA/$run.overlap/overlap-effects.json" --exposure "$FA/exposure-$3.json" \
      "${c1[@]}" --output "$OUTD/$TIER-$run" > "$OUTD/$TIER-$run.log" 2>&1 || die "$run successor evaluation FAILED"
    log "$run successor: $(tail -1 "$OUTD/$TIER-$run.log")"
    cat "$OUTD/$TIER-$run.log"
    ;;
  *) sed -n '2,25p' "$0"; exit 2 ;;
esac
