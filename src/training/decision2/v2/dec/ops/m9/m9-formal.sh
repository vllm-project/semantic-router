#!/usr/bin/env bash
# Decoder M9 formal step on node A (prereg dec-m9-prereg-2026-10-01.md, "Formal"): the M6 / M8 formal procedure with
# the 4B node-A option (ops/m6: M6_4B_NODE=A, image dbe5f32b, HIP_FORCE_DEV_KERNARG=1, copies of node B's frozen
# masters formal/m5/cache-frozen f6d0f920... and cache-frozen-mlx 65d7d38f...). Outputs under
# /data/dev2/runs/dec/m9/formal; GPU M9_GPU (6|7), every GPU job a co-tenant. The run is a PILOT, not a release
# candidate: nothing is handed off, uploaded or sent to the C1 custodian.
#
#   m9-formal.sh setup             .sha256 manifests of the node-A master copies, checked against the pinned hashes
#   m9-formal.sh parity            DEV2.0-4B's weights (the N4XF soup staged on node A, list df602ca9; T = 1) collected
#                                  on the node-A formal path (typed FINAL, CSS15, public 231), sealed, reported, and
#                                  repeated against the bar run dev2-4b-t1-derived -> PARITY.json (exact = 0 category
#                                  changes on all three panels); then its mlx-diag
#   m9-formal.sh select            m9/select/4b-pick.json -> m9/select-formal/4b-finalists.json (the single H9 pick)
#   m9-formal.sh smoke|finalist    m6-formal.sh 4b smoke|finalist <pick> (CAL698 16K fit + 23:15 rule, 8-item smoke,
#                                  collection on a copy of the master)
#   m9-formal.sh score <run>       m6-score.sh 4b <run> (bar, n4xf-ref, Nox 1.0, Decider 4B, Jet v6.2, types) plus the
#                                  paired comparison with the node-A incumbent run (n4xf-nodeA-m9)
#   m9-formal.sh mlx <run>         m6-formal.sh 4b mlx <run>, m6-score.sh 4b mlx <run>, plus mlx paired vs the node-A
#                                  incumbent's mlx-diag
#   m9-formal.sh readout <run>     overlap (item 6b) and the successor items 1-7 as a read-out (m6-score.sh successor;
#                                  exposure = the H9 TRAIN receipt), labelled pilot / not a candidate
set -u
S=$(cd "$(dirname "$0")/../../../.." && pwd)
MIRROR=${S%/src/training/decision2}
# shellcheck disable=SC2034  # SRC and TIER are read by the sourced M6 formal library
SRC=${MIRROR##*/}
[ -f "$MIRROR/.dev2-mirror.json" ] || { echo "run from an exact mirror under /data/dev2/src" >&2; exit 2; }
CMD=${1:-}
R=/data/dev2/runs/dec
M=$R/m9
export M6_4B_NODE=A M6_PREFIX=m9 M6_FORMAL_ROOT=$M/formal M6_SELECT=$M/select-formal M6_4B_MASTER_DIR=$M/caches/formal
export M6_GPU=${M9_GPU:-} M6_SHARED_LEASE=dec-m9-formal
F=$M6_FORMAL_ROOT
OPS=$S/v2/dec/ops
RL=/data/dev2/runs/release
BAR=$RL/dev2-4b-t1-derived
I=$R/m4/formal-candidates/staging-e8656221/m4/N4XF-soup/checkpoint
I_LIST=df602ca9f2574bcf127322c9b232ac94b2470dd00cfbefd8320b3f5fca5351f4
REF=m9-ref-N4XF
EXPOSURE_H9=$M/exposure/m9-4b-H9/exposure-m9-4b-H9.json
mkdir -p "$F" "$M/select-formal"
log() { echo "$(date -u +%FT%TZ) m9-formal $*" | tee -a "$F/OPERATIONS.log"; }
die() { log "$*"; echo "$*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
tree_manifest() { (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum); }
py() { (cd "$S" && PYTHONPATH=$S python3 -B "$@"); }
case $CMD in
  parity | smoke | finalist | mlx)
    case ${M9_GPU:-} in 6 | 7) ;; *) die "set M9_GPU to 6 or 7 (the GPUs lent to M9)" ;; esac ;;
esac

case $CMD in
  setup)
    for pair in "cache-frozen f6d0f9207e436aa0090f51d1d4163e9f58edeb067ac28b894fb5e58612ea11d8" \
      "cache-frozen-mlx 65d7d38f267bb168b18b71dd0c1b80b8b3e1aa0bbaca141d8d03b3fe7223c569"; do
      read -r name pin <<< "$pair"
      d=$M6_4B_MASTER_DIR/$name
      [ -d "$d" ] || die "missing $d (copy it from node B over the direct link)"
      [ -f "$d.sha256" ] || tree_manifest "$d" > "$d.sha256"
      [ "$(sha "$d.sha256")" = "$pin" ] || die "$d manifest $(sha "$d.sha256") is not $pin"
      chmod -R a-w "$d"
      log "master $d ready (manifest $pin)"
    done
    ;;
  parity)
    # shellcheck disable=SC2034
    TIER=4b
    # shellcheck source=src/training/decision2/v2/dec/ops/m6/m6-formal-lib.sh
    . "$OPS/m6/m6-formal-lib.sh"
    tier_setup
    [ "$NODE" = A ] || die "parity runs on node A"
    [ -f "$MASTER.sha256" ] || die "run m9-formal.sh setup first"
    PKG=$F/pkg/$REF LIST=$F/pkg/$REF.sha256
    if [ ! -f "$LIST" ]; then
      [ "$(tree_manifest "$I" | sha256sum | cut -d' ' -f1)" = "$I_LIST" ] || die "incumbent $I differs from $I_LIST"
      mkdir -p "$PKG/m6/$REF"
      cp -al "$I" "$PKG/m6/$REF/checkpoint" || die "incumbent package link FAILED"
      tree_manifest "$PKG" > "$LIST"
      log "$REF package: $(wc -l < "$LIST") files, revision $(sha "$LIST")"
    fi
    REV=$(sha "$LIST") MODEL=$PKG/m6/$REF/checkpoint
    if [ ! -f "$F/$REF/M6-RECEIPT.json" ]; then
      [ ! -e "$F/$REF" ] || die "$F/$REF exists without a receipt (failed earlier; not rerun)"
      cache_copy "$MASTER" "$MASTER_SHA" "$F/$REF-cache"
      args=(--extra "model_id=decision2-dec-$REF")
      log "$REF collection start (GPU$GPU, copy of $MASTER)"
      run_collect "$REF" "$MODEL" "$PKG" "$REV" "$SPEC_T1" "$F/$REF-cache" \
        "M9 formal-path parity: DEV2.0-4B weights, T = 1 (node A)" "${args[@]}" || die "$REF collection FAILED"
      finish "$REF" ref N4XF-soup "decision2-dec-$REF (DEV2.0-4B weights, M9 node-A formal path)" "$PKG" "$LIST" \
        "$REV" "$MODEL" "$SPEC_T1" "$F/$REF-cache" "$COPY_BEFORE" none none - - "${args[@]}"
    fi
    RD=$F/$REF
    [ -f "$RD/REPORT.json" ] || py -m v2.eval.same_panel report --run-dir "$RD" --label "DEV2.0-4B weights (M9 node-A path)" \
      --tier 4B --family decision2 --count-safetensors "$RD/pkg-headers" > "$RD.report.log" 2>&1 || die "$REF report FAILED"
    [ -f "$RD/REPEAT-vs-bar-t1.json" ] || py -m v2.eval.same_panel repeat --run-dir "$RD" --comparator-run-dir "$BAR" \
      --name bar-t1 > "$RD.repeat-bar-t1.log" 2>&1 || die "$REF repeat vs the bar FAILED"
    [ -f "$RD/PAIRED-vs-bar-t1.json" ] || py -m v2.eval.same_panel compare --run-dir "$RD" --comparator-run-dir "$BAR" \
      --left-name "$REF" --right-name bar-t1 > "$RD.compare-bar-t1.log" 2>&1 || die "$REF compare vs the bar FAILED"
    python3 - "$RD" > "$RD.parity.log" <<'EOF' || die "PARITY FAILED"
import datetime as dt, hashlib, json, sys
from pathlib import Path
r = Path(sys.argv[1])
rep = json.loads((r / "REPEAT-vs-bar-t1.json").read_text())["panels"]
panels = ("typed-final", "css15", "public231")
changes = {p: rep[p]["category_changes"] if p in rep else None for p in panels}
exact = all(c == 0 for c in changes.values()) and all(
    rep[p]["missing_left"] == rep[p]["missing_right"] == 0 for p in rep)
out = {"schema": "dec-m9-formal-parity/1",
       "rule": "0 category changes on typed FINAL, CSS15 and public 231 vs dev2-4b-t1-derived -> the stored run is the bar; "
               "otherwise the node-A incumbent run is the paired reference and both comparisons are reported",
       "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "category_changes": changes, "exact": exact,
       "max_abs_numeric_drift": {p: rep[p].get("max_abs_numeric_drift") for p in rep},
       "repeat_sha256": hashlib.sha256((r / "REPEAT-vs-bar-t1.json").read_bytes()).hexdigest(),
       "report_sha256": hashlib.sha256((r / "REPORT.json").read_bytes()).hexdigest()}
(r / "PARITY.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
print(json.dumps({"exact": exact, "category_changes": changes}))
EOF
    log "parity: $(cat "$RD.parity.log")"
    if [ ! -f "$F/$REF-mlx/M6-RECEIPT.json" ]; then
      bash "$OPS/m6/m6-formal.sh" 4b mlx "$REF" || die "$REF mlx-diag collection FAILED"
    fi
    if [ ! -f "$F/$REF-mlx/mlx-agg.json" ]; then
      MR=$F/$REF-mlx
      py -m v2.eval.multilingual_panel score --panel /data/dev2/private/panels/mlx-diag-v1 \
        --predictions "$MR/output/mlx-diag.predictions.jsonl" --output "$MR/mlx-diag.score.json" > "$MR.score.log" 2>&1 \
        || die "$REF mlx score FAILED"
      python3 -B "$OPS/m5/m5-mlx-agg.py" --panel /data/dev2/private/panels/mlx-diag-v1 \
        --predictions "$MR/output/mlx-diag.predictions.jsonl" --score "$MR/mlx-diag.score.json" --output "$MR/mlx-agg.json" \
        > "$MR.agg.log" 2>&1 || die "$REF mlx aggregates FAILED"
      log "$REF mlx scored: $(tail -1 "$MR.agg.log")"
    fi
    ;;
  select)
    out=$M6_SELECT/4b-finalists.json
    [ -f "$out" ] && { echo "exists: $out"; exit 0; }
    [ -f "$M/select/4b-pick.json" ] || die "no m9/select/4b-pick.json (m9-lines.sh rules)"
    python3 - "$M/select/4b-pick.json" "$out" <<'EOF' || die "select FAILED"
import hashlib, json, sys
pick_path, out = sys.argv[1:]
doc = json.load(open(pick_path))
pick = doc["pick"]
if pick is None:
    raise SystemExit("no development passer: no formal run (prereg)")
entry = {"slot": 1, "line": "L-H9", **pick}
json.dump({"schema": "dec-m9-formal-select/1", "role": "the single M9 formal run (pilot, not a release candidate)",
           "source": pick_path, "source_sha256": hashlib.sha256(open(pick_path, "rb").read()).hexdigest(),
           "finalists": [entry]}, open(out, "x"), indent=1, sort_keys=True)
print(entry["point"])
EOF
    log "formal select: $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["finalists"][0]["point"])' "$out")"
    ;;
  smoke | finalist)
    [ $# -ge 2 ] || { sed -n '2,24p' "$0"; exit 2; }
    bash "$OPS/m6/m6-formal.sh" 4b "$CMD" "$2" "${@:3}"
    ;;
  score)
    [ $# -eq 2 ] || { sed -n '2,24p' "$0"; exit 2; }
    bash "$OPS/m6/m6-score.sh" 4b "$2" || die "m6-score.sh 4b $2 FAILED"
    RD=$F/$2
    [ -f "$RD/PAIRED-vs-n4xf-nodeA-m9.json" ] || py -m v2.eval.same_panel compare --run-dir "$RD" \
      --comparator-run-dir "$F/$REF" --left-name "$2" --right-name n4xf-nodeA-m9 > "$RD.compare-n4xf-nodeA-m9.log" 2>&1 \
      || die "$2 compare vs the node-A incumbent FAILED"
    log "$2 scored (pilot, not a release candidate)"
    ;;
  mlx)
    [ $# -eq 2 ] || { sed -n '2,24p' "$0"; exit 2; }
    [ -f "$F/$2-mlx/M6-RECEIPT.json" ] || bash "$OPS/m6/m6-formal.sh" 4b mlx "$2" || die "$2 mlx-diag collection FAILED"
    bash "$OPS/m6/m6-score.sh" 4b mlx "$2" || die "m6-score.sh 4b mlx $2 FAILED"
    MR=$F/$2-mlx
    [ -f "$MR/MLX-PAIRED-vs-n4xf-nodeA-m9.json" ] || py -m v2.dec.mlx_paired --panel /data/dev2/private/panels/mlx-diag-v1 \
      --candidate "$MR/output/mlx-diag.predictions.jsonl" --reference "$F/$REF-mlx/output/mlx-diag.predictions.jsonl" \
      --candidate-name "$2" --reference-name n4xf-nodeA-m9 --output "$MR/MLX-PAIRED-vs-n4xf-nodeA-m9.json" \
      > "$MR.paired-nodeA.log" 2>&1 || die "$2 mlx paired vs the node-A incumbent FAILED"
    log "$2 mlx scored"
    ;;
  readout)
    [ $# -eq 2 ] || { sed -n '2,24p' "$0"; exit 2; }
    export M6_EXPOSURE=$EXPOSURE_H9
    [ -f "$F/$2.overlap/overlap-effects.json" ] || bash "$OPS/m6/m6-score.sh" 4b overlap "$2" || die "$2 overlap FAILED"
    bash "$OPS/m6/m6-score.sh" 4b successor "$2" || die "$2 successor read-out FAILED"
    log "$2 items 1-7 read out (pilot, not a release candidate; no hand-off, no C1)"
    ;;
  *) sed -n '2,24p' "$0"; exit 2 ;;
esac
