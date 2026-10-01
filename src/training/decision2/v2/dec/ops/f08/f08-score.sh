#!/usr/bin/env bash
# Decoder 0.8B fast track: node-A scoring of the 08b-RA formal attempt (record dec-08bfast-formal-lock-2026-10-01.md),
# CPU only, from an exact mirror, on run directories under /data/dev2/runs/dec/formal/f08 pulled gold-free from node E
# over the temporary transfer key. Two bars:
#   bar-t1  the stored formal run of DEV2.0-0.8B (runs/release/dev2-0p8b-t1-derived, v3 50.236): m6-score.sh's bar;
#   bar-e   f08-08b-C0, node E's formal-path collection of the same weights (the M12 prereg's parity run).
# Items 1, 2, 4, 6(b) and 7 must pass against both (f08_successor.py); node B's m8s-ref-08b-I is a reported repeat.
#
#   f08-score.sh formal-pull <user@node> <run>   formal/f08/<run> from node E (gold-free outputs, receipts, seal,
#                                                header stubs; no weights or caches), content-manifest checked
#   f08-score.sh formal-mark <user@node> <run>   REPORT.json / SEAL.json hashes -> <node>:<run>/V3-SEALED.json
#                                                (gates m6-formal.sh mlx there; no score leaves node A)
#   f08-score.sh c0                              m6-score.sh 08b f08-08b-C0, then same_panel repeat vs the stored bar
#                                                and node B's m8s-ref-08b-I -> f08-08b-C0/C0-EXACT.json
#   f08-score.sh run <run>                       m6-score.sh 08b <run>, then PAIRED-vs-bar-e.json against f08-08b-C0
#   f08-score.sh mlx <run>                       m6-score.sh 08b mlx <run> (MLX-PAIRED.json vs node A's E8F mlx-diag),
#                                                then MLX-PAIRED-bar-e.json against f08-08b-C0-mlx
#   f08-score.sh exposure <user@node>            v2.eval.overlap_effects exposure of the 08b-RA TRAIN (streamed from
#                                                node E, hash-checked) against the r2 payload -> exposure-08b-RA.json
#   f08-score.sh overlap <run>                   overlap_effects run with both bars (item 6(b))
#   f08-score.sh successor <run>                 public-231 guards vs both bars, then f08_successor.py (items 1-8)
set -u
S=$(cd "$(dirname "$0")/../../../.." && pwd)
MIRROR=${S%/src/training/decision2}
[ -f "$MIRROR/.dev2-mirror.json" ] || { echo "run from an exact mirror under /data/dev2/src" >&2; exit 2; }
export M6_FORMAL_ROOT=/data/dev2/runs/dec/formal/f08
FA=$M6_FORMAL_ROOT D=/data/dev2/runs/dec/formal E=/data/dev2/runs/eval RL=/data/dev2/runs/release
BAR_T1=$RL/dev2-0p8b-t1-derived
BAR_B=$D/m8s/m8s-ref-08b-I
C0=$FA/f08-08b-C0
MLX_PANEL=/data/dev2/private/panels/mlx-diag-v1
FLAGGED=$E/m5/overlap-effects/final/flagged.json
FLAGGED_SHA=0ed60dd751f2f33326b1b58859e27065166ea18e529812caa6ccec9fa894f41e
PAYLOAD=$E/m5/overlap-effects/final/excluded-groups.json
PAYLOAD_SHA=2194716a179b2e6c3ba529dee3952c5dd62c0f3281638b7236029d5d542f4914
RA_TRAIN=/data/dev2/runs/dec/m12/data/08b/08b-RA/train.jsonl
RA_TRAIN_SHA=12bd63d8b215877b6f471c12652e534e0f0be794c15bff74ad5ac50e98018bf3
EXPOSURE=$FA/exposure-08b-RA.json
SCORE=$S/v2/dec/ops/m6/m6-score.sh
KEY=/root/.ssh/d2_temp_cd
mkdir -p "$FA/successor"
log() { echo "$(date -u +%FT%TZ) f08-score $*" | tee -a "$FA/OPERATIONS-nodeA.log"; }
die() { log "$*"; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
py() { (cd "$S" && PYTHONPATH=$S python3 -B "$@"); }
jget() { python3 -c 'import json,sys; v=json.load(open(sys.argv[1]))
for k in sys.argv[2:]: v=v[k]
print(json.dumps(v) if isinstance(v,(list,dict)) else v)' "$@"; }
exec 9> "$FA/.f08-score.lock"
flock -n 9 || die "another f08-score.sh is running"

case ${1:-} in
  formal-pull)
    [ $# -eq 3 ] || { sed -n '2,25p' "$0"; exit 2; }
    from=$2 run=$3
    on() { ssh -i "$KEY" -o BatchMode=yes "$from" "$@"; }
    manifest='find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -d" " -f1'
    [ ! -e "$FA/$run" ] || die "$FA/$run exists on node A"
    on "test -f '$FA/$run/M6-RECEIPT.json'" || die "$run has no M6 receipt on $from"
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
    [ "$seal_a" = "$seal_b" ] || die "$run: node-A seal differs from $from"
    [ -f "$FA/$run/REPORT.json" ] || die "$run has no report yet"
    python3 -c 'import json,sys,datetime as d; print(json.dumps({"run": sys.argv[1], "report_sha256": sys.argv[2], "seal_sha256": sys.argv[3], "marked_utc": d.datetime.now(d.timezone.utc).isoformat()}))' \
      "$run" "$(sha "$FA/$run/REPORT.json")" "$seal_a" \
      | on "set -o noclobber; cat > '$FA/$run/V3-SEALED.json'" || die "mark of $run failed"
    log "formal run $run marked on its node"
    ;;
  c0)
    bash "$SCORE" 08b f08-08b-C0 > /dev/null || die "C0 report / compares FAILED"
    for pair in "bar-t1=$BAR_T1" "m8s-ref-08b-I=$BAR_B"; do
      name=${pair%%=*} dir=${pair#*=}
      [ -f "$C0/REPEAT-vs-$name.json" ] || py -m v2.eval.same_panel repeat --run-dir "$C0" --comparator-run-dir "$dir" \
        --name "$name" > "$C0.repeat-$name.log" 2>&1 || die "C0 repeat vs $name FAILED"
    done
    python3 - "$C0" > "$C0.exact.log" << 'EOF' || die "C0-EXACT FAILED"
import datetime as dt, hashlib, json, sys
from pathlib import Path
r = Path(sys.argv[1])
panels = ("typed-final", "css15", "public231")
out = {"schema": "dec-f08-c0-exact/1", "run": str(r), "rule": "0 differing answers on typed FINAL, CSS15 and public 231",
       "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "against": {}}
for name in ("bar-t1", "m8s-ref-08b-I"):
    f = r / f"REPEAT-vs-{name}.json"
    rep = json.loads(f.read_text())["panels"]
    changes = {p: rep[p]["category_changes"] if p in rep else None for p in panels}
    ok = all(c == 0 for c in changes.values()) and all(rep[p]["missing_left"] == rep[p]["missing_right"] == 0 for p in rep)
    out["against"][name] = {"repeat": str(f), "repeat_sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                            "category_changes": changes, "exact": ok}
out["exact_vs_stored_bar"] = out["against"]["bar-t1"]["exact"]
out["seal_sha256"] = hashlib.sha256((r / "SEAL.json").read_bytes()).hexdigest()
out["report_sha256"] = hashlib.sha256((r / "REPORT.json").read_bytes()).hexdigest()
(r / "C0-EXACT.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
print(json.dumps({"exact_vs_stored_bar": out["exact_vs_stored_bar"], **{k: v["category_changes"] for k, v in out["against"].items()}}))
EOF
    log "C0 parity: $(cat "$C0.exact.log"); C0 $(cat "$C0.summary.log")"
    cat "$C0.exact.log"
    ;;
  run)
    [ $# -eq 2 ] || { sed -n '2,25p' "$0"; exit 2; }
    R=$FA/$2
    [ -f "$C0/C0-EXACT.json" ] || die "score the C0 parity run first (f08-score.sh c0)"
    bash "$SCORE" 08b "$2" > /dev/null || die "$2 m6-score FAILED"
    [ -f "$R/PAIRED-vs-bar-e.json" ] || py -m v2.eval.same_panel compare --run-dir "$R" --comparator-run-dir "$C0" \
      --left-name "$(jget "$R/M6-RECEIPT.json" label)" --right-name bar-e > "$R.compare-bar-e.log" 2>&1 \
      || die "$2 compare vs bar-e FAILED"
    log "$2 scored: $(cat "$R.summary.log"); vs bar-e $(python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print(round(p["point"]["delta"]["score"],3), [round(p["ci95"]["low"],3), round(p["ci95"]["high"],3)])' "$R/PAIRED-vs-bar-e.json")"
    ;;
  mlx)
    [ $# -eq 2 ] || { sed -n '2,25p' "$0"; exit 2; }
    MR=$FA/$2-mlx
    bash "$SCORE" 08b mlx "$2" || die "$2 m6-score mlx FAILED"
    if [ "$2" != f08-08b-C0 ]; then
      REFP=$C0-mlx/output/mlx-diag.predictions.jsonl
      [ -f "$REFP" ] || die "C0's mlx-diag run is not here yet"
      [ -f "$MR/MLX-PAIRED-bar-e.json" ] || py -m v2.dec.mlx_paired --panel "$MLX_PANEL" \
        --candidate "$MR/output/mlx-diag.predictions.jsonl" --reference "$REFP" --candidate-name "$2" \
        --reference-name bar-e --output "$MR/MLX-PAIRED-bar-e.json" > "$MR.paired-bar-e.log" 2>&1 \
        || die "$2 mlx paired vs bar-e FAILED (see $MR.paired-bar-e.log)"
      log "$2 mlx vs bar-e: $(tail -1 "$MR.paired-bar-e.log" | cut -c1-300)"
    fi
    ;;
  exposure)
    [ $# -eq 2 ] || { sed -n '2,25p' "$0"; exit 2; }
    [ ! -e "$EXPOSURE" ] || die "$EXPOSURE exists"
    [ "$(sha "$PAYLOAD")" = "$PAYLOAD_SHA" ] || die "r2 payload is not $PAYLOAD_SHA"
    ssh -i "$KEY" -o BatchMode=yes "$2" "cat '$RA_TRAIN'" | py -m v2.eval.overlap_effects exposure --groups "$PAYLOAD" \
      --train - --expect-sha256 "$RA_TRAIN_SHA" --label "decoder 0.8B fast track: M12 08b-RA TRAIN (data lock d506595c2)" \
      --output "$EXPOSURE" > "$FA/exposure-08b-RA.log" 2>&1 || die "exposure receipt FAILED (see $FA/exposure-08b-RA.log)"
    log "exposure: $(tail -1 "$FA/exposure-08b-RA.log" | cut -c1-300)"
    ;;
  overlap)
    [ $# -eq 2 ] || { sed -n '2,25p' "$0"; exit 2; }
    R=$FA/$2 O=$FA/$2.overlap
    [ -f "$R/PAIRED-vs-bar-e.json" ] || die "$2 not scored yet"
    [ -f "$EXPOSURE" ] || die "no exposure receipt yet (f08-score.sh exposure)"
    [ "$(sha "$FLAGGED")" = "$FLAGGED_SHA" ] || die "flagged ids $FLAGGED are not $FLAGGED_SHA"
    [ ! -e "$O/overlap-effects.json" ] || die "$O/overlap-effects.json exists"
    mkdir -p "$O"
    python3 - "$O/spec.json" "$2" "$R" "$EXPOSURE" "$BAR_T1" "$C0" "$E" "$D" << 'EOF' || die "overlap spec FAILED"
import json, sys
from pathlib import Path
out, cand, run, exposure, bar_t1, bar_e, e, d = sys.argv[1:]
comparators = {"bar-t1": bar_t1, "bar-e": bar_e, "adopted-1.0": f"{e}/m1/r4-eos1", "same-limit-16k": f"{d}/m2/eos1-16k",
               "intern": f"{e}/m2/q2b-intern08b", "kev": f"{e}/m2/q1-kev08b"}
models = {cand: {"run": run, "exposure": [exposure],
                 "note": "0.8B fast track (M12 08b-RA): no incumbent weight; exposure = its TRAIN file's receipt"}}
for n, path in comparators.items():
    models[n] = {"run": path}
spec = {"label": "post-key same-panel", "panel_root": "/data/dev2/private/panels", "focus_task": "media_ideology",
        "models": models,
        "tiers": {"f08": {"candidate": cand, "own_1_0": ["adopted-1.0", "same-limit-16k"], "peers": ["intern", "kev"],
                          "internal_peers": ["bar-t1", "bar-e"], "threshold_pool": ["intern", "kev"]}},
        "reproduce": [{"left": cand, "right": n, "stored": f"{run}/PAIRED-vs-{n}.json"} for n in comparators
                      if Path(f"{run}/PAIRED-vs-{n}.json").is_file()]}
Path(out).write_text(json.dumps(spec, indent=1) + "\n")
EOF
    py -m v2.eval.overlap_effects run --spec "$O/spec.json" --flagged "$FLAGGED" --output "$O/overlap-effects.json" \
      > "$O.log" 2>&1 || die "$2 overlap_effects FAILED or reported problems (see $O.log)"
    log "$2 overlap: $(tail -1 "$O.log" | cut -c1-300)"
    ;;
  successor)
    [ $# -eq 2 ] || { sed -n '2,25p' "$0"; exit 2; }
    run=$2 R=$FA/$2 OUTD=$FA/successor
    for pair in "bar-t1=$BAR_T1" "bar-e=$C0"; do
      name=${pair%%=*} dir=${pair#*=}
      PUB=$OUTD/08b-$run.public231-$name.json
      [ -f "$PUB" ] || py -m v2.eval.gates public231 --left "$R" --right "$dir" --left-name "$run" --right-name "$name" \
        --output "$PUB" > "$PUB.log" 2>&1 || die "$run public231 guard vs $name FAILED to run"
    done
    c1=()
    [ -f "$FA/$run.c1/SUMMARY.json" ] && c1=(--c1 "$FA/$run.c1/SUMMARY.json")
    python3 -B "$S/v2/dec/ops/f08/f08_successor.py" --run "$R" --types "$R/TYPES.json" \
      --mlx-paired "$FA/$run-mlx/MLX-PAIRED.json" --mlx-paired-e "$FA/$run-mlx/MLX-PAIRED-bar-e.json" \
      --overlap "$FA/$run.overlap/overlap-effects.json" --exposure "$EXPOSURE" \
      --public231 "$OUTD/08b-$run.public231-bar-t1.json" --public231-e "$OUTD/08b-$run.public231-bar-e.json" \
      "${c1[@]}" --output "$OUTD/08b-$run" > "$OUTD/08b-$run.log" 2>&1 || die "$run successor evaluation FAILED"
    log "$run successor: $(tail -1 "$OUTD/08b-$run.log")"
    cat "$OUTD/08b-$run.log"
    ;;
  *) sed -n '2,25p' "$0"; exit 2 ;;
esac
