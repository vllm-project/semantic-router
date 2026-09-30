#!/usr/bin/env bash
# Decoder M8-small formal collections on node B (prereg dec-m8s-prereg-2026-09-30.md, "Formal"), from an exact mirror;
# outputs under /data/dev2/runs/dec/m8s/formal. The eval track's frozen runner (v2/eval/run_same_panel.sh, gold never
# mounted, --shared-lease dec-m8s-formal) at 16,384 tokens, each collection on its own cp -a copy of a master autotune
# cache, gold-free seal, then the M5 receipt and an M6-style receipt, so node A's ops/m6/m6-score.sh scores the
# relayed run (M6_FORMAL_ROOT=/data/dev2/runs/dec/formal/m8s).
#
#   m8s-formal.sh ref08b                    the verified 0.8B start on a FRESH cache (typed FINAL, CSS15, public 231;
#                                           T = 1 spec), sealed, cache frozen (formal/cache-frozen-08b); then
#                                           mlx-diag on a fresh cache (frozen as formal/cache-frozen-08b-mlx)
#   m8s-formal.sh finalist <tier> <point>   stage (checkpoint link, per-file list, parameters, CAL698 16K fit and the
#                                           23:15 rule on the point's 16K typed DEV / CSS pilot readouts), an 8-item
#                                           smoke on a throwaway master copy, the collection on a fresh copy, seal
#   m8s-formal.sh mlx <tier> <point>        mlx-diag on a copy of the tier's mlx master cache
#
# Masters: 2b formal/m6/cache-frozen-2b(-mlx) (the M6 node-B S2T reference, exact against node A); 08b the ref08b
# caches (usable once node A finds the reference exact against the T = 1 binding). GPU: M8S_GPU (node B 2, 6, 7).
set -u
. "$(dirname "$0")/m8s-lib.sh"
CMD=${1:-}
F=$M/formal H=/data/dev2/hf-cache
M5OPS=$S/v2/dec/ops/m5
SPEC_T1=$S/v2/dec/ops/m5/m5-adapter-infer-dec-t1.json SPEC_CAL=$S/v2/dec/adapter-spec-infer-dec.json
PANELS=/data/dev2/private/panels
CAL698_SHA=19cc1a8c4ebe6fd13031079f6b7d131046ce503f435f894e3ea672f91c2ed41f
declare -A SOURCE=([2b]=$H/models--llm-semantic-router--Decision-1.0-Sol-2B/snapshots/ce0c018a28de16d6639b1cd203b761bf643b89e6
  [08b]=$H/models--llm-semantic-router--Decision-1.0-Eos-0.8B/snapshots/363c4a5e56afc115b1c78c837633956d0bbb63ab)
declare -A TLABEL=([2b]=2B [08b]=0.8B)
declare -A MASTER=([2b]=$R/formal/m6/cache-frozen-2b [08b]=$F/cache-frozen-08b)
declare -A MASTER_MLX=([2b]=$R/formal/m6/cache-frozen-2b-mlx [08b]=$F/cache-frozen-08b-mlx)
GPU=${M8S_GPU:-}
mkdir -p "$F/pkg" "$F/stage" "$F/params"
die() { log "formal: $*"; exit 1; }
py() { (cd "$S" && PYTHONPATH=$S python3 -B "$@"); }
tree_manifest() { (cd "$1" && find . -type f -not -path './.cache/*' -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum); }
jget() { python3 -c 'import json,sys; v=json.load(open(sys.argv[1]));
for k in sys.argv[2:]: v=v[k]
print(v)' "$@"; }
gpu_ok() {
  [ -n "$GPU" ] || die "set M8S_GPU (node B 2, 6 or 7)"
  dec_env "$GPU" || exit 2
  [ "$(vram_free_gb "$GPU")" -ge 60 ] || die "GPU$GPU has < 60 GB free VRAM"
}
cache_copy() {  # <master> <copy>: the master must match its frozen manifest
  [ -f "$1.sha256" ] || die "no frozen manifest $1.sha256"
  [ ! -e "$2" ] || die "cache copy $2 exists"
  cp -a "$1" "$2" || die "cache copy FAILED"
  diff -q <(tree_manifest "$2") "$1.sha256" > /dev/null || die "cache copy of $1 differs from its frozen manifest"
  tree_manifest "$2" > "$2.before.sha256"
}
cache_freeze() {  # <cache> <master>
  [ ! -e "$2" ] || die "master $2 exists"
  cp -a "$1" "$2" || die "cache freeze FAILED"
  tree_manifest "$2" > "$2.sha256" || die "cache manifest FAILED"
  log "formal: cache frozen $2 ($(wc -l < "$2.sha256") files, manifest $(sha "$2.sha256" | cut -c1-12))"
}
collect() {  # <run> <tier> <model> <pkg> <rev> <spec> <cache> <purpose> [collect args]
  local run=$1 t=$2 model=$3 pkg=$4 rev=$5 spec=$6 cache=$7 purpose=$8 rc
  shift 8
  gpu_ok
  [ ! -e "$F/$run" ] || die "$run exists"
  bash "$S/v2/eval/run_same_panel.sh" --gpu "$GPU" --track dec-small --src "$SRC" --image "$IMAGE" --run-dir "$F/$run" \
    --model-dir "$model" --mount "$H" --mount "$pkg" --env TRITON_CACHE_AUTOTUNING=1 --env "TRITON_CACHE_DIR=$cache" \
    --mount-rw "$cache" --shared-lease dec-m8s-formal --purpose "$purpose" --expected-end "$(date -u -d '+30 min' +%FT%TZ)" \
    -- --adapter-spec "$spec" --model-path "$model" --revision "$rev" --extra "source=${SOURCE[$t]}" \
    --extra max_length=16384 "$@" > "$F/$run.collect.log" 2>&1
  rc=$?
  tree_manifest "$cache" > "$F/$run.cache-after.sha256"
  [ -f "$F/$run/GPU-TIME.json" ] && python3 - "$F/$run/GPU-TIME.json" "$run" "$purpose" >> "$M/GPU-SECONDS.jsonl" <<'EOF'
import json, sys
g = json.load(open(sys.argv[1]))
print(json.dumps({"item": "formal", "purpose": sys.argv[3], "job": sys.argv[2], "gpu": g.get("gpu"),
                  "start_utc": g.get("start_utc"), "end_utc": g.get("end_utc"), "exit_status": g.get("exit_code"),
                  "gpu_seconds": g.get("wall_seconds"), "receipt": sys.argv[1]}))
EOF
  [ $rc = 0 ] || die "$run collection FAILED (see $F/$run.collect.log)"
}
finish() {  # <run> <kind> <tier> <label> <pkg> <list> <rev> <model> <spec> <cache> <cal used|none> <decision|none> <params dir> <point|->
  local run=$1 kind=$2 t=$3 label=$4 pkg=$5 list=$6 rev=$7 model=$8 spec=$9 cache=${10} cal=${11} dec=${12} params=${13} point=${14}
  local RD=$F/$run
  if [ "$kind" != mlx ] && [ "$kind" != smoke ]; then
    py -m v2.eval.same_panel seal --run-dir "$RD" > "$RD.seal.log" 2>&1 || die "$run seal FAILED"
  fi
  cp -a "$params/pkg-headers" "$RD/pkg-headers" || die "$run parameter stubs copy FAILED"
  cp "$params/PARAMS.json" "$RD/PARAMS.json" || die "$run parameter count copy FAILED"
  cp "$list" "$RD/PACKAGE.sha256"
  py "$M5OPS/m5-receipt.py" --run-dir "$RD" --kind "$kind" --name "$run" --label "$label" --package-dir "$pkg" \
    --package-list "$list" --revision "$rev" --model "$model" --spec "$spec" --calibration-used "$cal" \
    --calibration-decision "$dec" --cache-dir "$cache" --cache-before "$cache.before.sha256" \
    --cache-after-manifest "$RD.cache-after.sha256" --mirror "$SRC" --image "$IMAGE" > "$RD.receipt.log" 2>&1 \
    || die "$run receipt FAILED"
  python3 - "$RD" "$t" "${TLABEL[$t]}" "$point" "$M/select/$t-finalists.json" "$pkg" <<'EOF' || die "$run M6-style receipt FAILED"
import hashlib, json, sys
from pathlib import Path
rd, tier, tlabel, point, finalists, pkg = sys.argv[1:]
rd = Path(rd)
m5 = rd / "M5-RECEIPT.json"
base = json.loads(m5.read_text())
slot = None
if point != "-" and Path(finalists).is_file():
    for f in json.loads(Path(finalists).read_text())["finalists"]:
        if f["point"] == point:
            slot = {k: f.get(k) for k in ("slot", "line", "step", "T", "G", "H3", "proxy", "htdev2", "htdev2_delta")}
weights = next(Path(pkg).glob("m8s/*/weights.json"), None)
out = {
    "schema": "dec-m8s-formal-receipt/1 (M6 receipt fields)",
    "tier": tier, "tier_label": tlabel, "node": "B", "point": None if point == "-" else point, "selection": slot,
    "weights_json": str(weights) if weights else None,
    "weights_json_sha256": hashlib.sha256(weights.read_bytes()).hexdigest() if weights else None,
    "m5_receipt_sha256": hashlib.sha256(m5.read_bytes()).hexdigest(),
    **{k: base.get(k) for k in ("kind", "name", "label", "revision", "revision_matches_list", "package_dir", "model", "spec",
                                "calibration_used", "calibration_rule", "collect_args", "image_id", "gpu", "wall_seconds",
                                "gpu_hours", "seal_sha256", "parameters", "cache")},
}
with open(rd / "M6-RECEIPT.json", "x") as f:
    json.dump(out, f, indent=1, sort_keys=True)
    f.write("\n")
EOF
  log "formal: $run done ($(tail -1 "$RD.receipt.log" | cut -c1-160))"
}
params_of() {  # <model> <params dir>
  [ -f "$2/PARAMS.json" ] && return 0
  mkdir -p "$2"
  py "$M5OPS/m5-params.py" --package "$1" --stubs "$2/pkg-headers" --output "$2/PARAMS.json" > "$2.log" 2>&1 \
    || die "parameter count FAILED ($1)"
}

case $CMD in
  ref08b)
    t=08b I=${START[08b]} run=m8s-ref-08b-I
    [ -f "$I/START-VERIFIED.json" ] || die "0.8B start not verified"
    list=$F/pkg/$run.sha256
    [ -f "$list" ] || tree_manifest "$I" > "$list"
    params_of "$I" "$F/params/$run"
    if [ ! -f "$F/$run/M6-RECEIPT.json" ]; then
      mkdir -p "$F/cache-ref-08b" && tree_manifest "$F/cache-ref-08b" > "$F/cache-ref-08b.before.sha256"
      collect "$run" 08b "$I" "$I" "${START_REV[08b]}" "$SPEC_T1" "$F/cache-ref-08b" "M8s 0.8B node-B reference (fresh cache)"
      finish "$run" ref 08b "DEV2.0-0.8B bede7938 node-B reference" "$I" "$list" "${START_REV[08b]}" "$I" "$SPEC_T1" \
        "$F/cache-ref-08b" none none "$F/params/$run" -
      cache_freeze "$F/cache-ref-08b" "${MASTER[08b]}"
    fi
    if [ ! -f "$F/$run-mlx/M6-RECEIPT.json" ]; then
      mkdir -p "$F/cache-ref-08b-mlx" && tree_manifest "$F/cache-ref-08b-mlx" > "$F/cache-ref-08b-mlx.before.sha256"
      collect "$run-mlx" 08b "$I" "$I" "${START_REV[08b]}" "$SPEC_T1" "$F/cache-ref-08b-mlx" "M8s 0.8B node-B mlx-diag reference" \
        --panels mlx-diag
      finish "$run-mlx" mlx 08b "DEV2.0-0.8B bede7938 node-B mlx reference" "$I" "$list" "${START_REV[08b]}" "$I" "$SPEC_T1" \
        "$F/cache-ref-08b-mlx" none none "$F/params/$run" -
      cache_freeze "$F/cache-ref-08b-mlx" "${MASTER_MLX[08b]}"
    fi
    ;;
  finalist | mlx)
    [ $# -eq 3 ] || { sed -n '2,20p' "$0"; exit 2; }
    t=$2 point=$3 run=m8s-$3
    case $t in 2b | 08b) ;; *) die "tier must be 2b or 08b" ;; esac
    fj=$M/select/$t-finalists.json
    python3 -c 'import json,sys; sys.exit(0 if any(f["point"]==sys.argv[2] for f in json.load(open(sys.argv[1]))["finalists"]) else 1)' \
      "$fj" "$point" || die "$point is not a finalist in $fj"
    P=$M/lines/$t/$point
    art=$(jget "$P/weights.json" checkpoint)
    [ "$(tree_manifest "$art" | sha256sum | cut -d' ' -f1)" = "$(jget "$P/weights.json" files_sha256_list_sha256)" ] \
      || die "$point checkpoint differs from its line list"
    pkg=$F/pkg/$run list=$F/pkg/$run.sha256 C16=$F/stage/$run-cal698-16k DC=$F/stage/$run-devcal.json
    if [ ! -f "$list" ]; then
      if [ ! -f "$C16/calibration.json" ]; then
        [ "$(sha "$SELCAL/cal.jsonl")" = "$CAL698_SHA" ] || die "CAL698 differs"
        [ ! -e "$C16.launch.json" ] || die "$run: earlier 16K calibration failed; not rerun"
        gpu_ok
        bash "$LAUNCH" "m8s-$run-cal698-16k" "$SRC" "$C16" -- -m v2.dec.calibrate_ckpt --checkpoint "$(incontainer "$art")" \
          --cal /data/cal.jsonl --max-length 16384 --output /out/calibration.json
        rc=$?
        gpu_seconds "$C16.launch.json" formal "$run CAL698 16K fit"
        [ $rc = 0 ] || die "$run 16K calibration FAILED"
      fi
      [ -f "$DC" ] || py -m v2.release.dev_calibration --label "$run" --panel-root "$PANELS" \
        --typed-dev "$P/dev/dev.predictions.jsonl" --css-pilot "$P/css-pilot/css-pilot.predictions.jsonl" \
        --candidate "$C16/calibration.json" --work "${DC%.json}-work" --output "$DC" > "${DC%.json}.log" 2>&1 \
        || die "$run 23:15 rule check FAILED"
      D=$pkg/m8s/$run
      mkdir -p "$D/cal698-16k"
      cp -al "$art" "$D/checkpoint" || die "$run checkpoint link FAILED"
      cp "$C16/calibration.json" "$D/cal698-16k/calibration.json"
      cp "$DC" "$D/calibration-decision.json"
      cp "$P/weights.json" "$D/weights.json"
      tree_manifest "$pkg" > "$list"
      log "formal: $run staged ($(wc -l < "$list") files, revision $(sha "$list" | cut -c1-12), CAL698 adopt=$(jget "$DC" adopt))"
    fi
    (cd "$pkg" && sha256sum -c --quiet "$list") > /dev/null 2>&1 || die "$run package changed since staging"
    model=$pkg/m8s/$run/checkpoint rev=$(sha "$list") dec=$pkg/m8s/$run/calibration-decision.json
    params_of "$model" "$F/params/$run"
    if [ "$(jget "$dec" adopt)" = True ]; then
      spec=$SPEC_CAL cal=$pkg/m8s/$run/cal698-16k/calibration.json calx=(--extra "calibration=$cal")
    else
      spec=$SPEC_T1 cal=none calx=()
    fi
    if [ "$CMD" = finalist ]; then
      [ -f "${MASTER[$t]}.sha256" ] || die "no frozen master for $t (08b: ref08b first)"
      if [ ! -f "$F/$run-smoke/GPU-TIME.json" ]; then
        cache_copy "${MASTER[$t]}" "$F/cache-$run-smoke"
        collect "$run-smoke" "$t" "$model" "$pkg" "$rev" "$spec" "$F/cache-$run-smoke" "M8s $run smoke" "${calx[@]}" --max-items 8
      fi
      if [ ! -f "$F/$run/M6-RECEIPT.json" ]; then
        cache_copy "${MASTER[$t]}" "$F/cache-$run"
        collect "$run" "$t" "$model" "$pkg" "$rev" "$spec" "$F/cache-$run" "M8s $run formal collection" "${calx[@]}"
        finish "$run" finalist "$t" "$run" "$pkg" "$list" "$rev" "$model" "$spec" "$F/cache-$run" "$cal" "$dec" \
          "$F/params/$run" "$point"
      fi
    else
      [ -f "${MASTER_MLX[$t]}.sha256" ] || die "no frozen mlx master for $t"
      if [ ! -f "$F/$run-mlx/M6-RECEIPT.json" ]; then
        cache_copy "${MASTER_MLX[$t]}" "$F/cache-$run-mlx"
        collect "$run-mlx" "$t" "$model" "$pkg" "$rev" "$spec" "$F/cache-$run-mlx" "M8s $run mlx-diag" "${calx[@]}" --panels mlx-diag
        finish "$run-mlx" mlx "$t" "$run" "$pkg" "$list" "$rev" "$model" "$spec" "$F/cache-$run-mlx" "$cal" "$dec" \
          "$F/params/$run" "$point"
      fi
    fi
    ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
