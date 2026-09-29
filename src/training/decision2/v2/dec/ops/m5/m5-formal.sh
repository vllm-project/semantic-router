#!/usr/bin/env bash
# Decoder M5 formal runs on node B (prereg dec-m5-prereg-2026-09-29.md, "Formal runs (finalists only)"), run from
# an exact mirror; outputs under /data/dev2/runs/dec/formal/m5 (M5_FORMAL_ROOT), GPU M5_GPU (default 3).
#
#   m5-formal.sh ref                           N4XF soup reference: verify the node-B package against its staged
#                                              list, collect v3 + public 231 on a FRESH cache, seal, freeze the cache
#   m5-formal.sh stage <ARM> <artifact-dir>    finalist package: CAL698 16K fit, 23:15 rule on typed DEV + CSS pilot,
#                                              checkpoint + calibrations + per-file SHA-256 list (revision = its hash)
#   m5-formal.sh finalist <ARM> <artifact-dir> stage (if needed), then collect + seal on a copy of the frozen cache
#   m5-formal.sh mlx <run>                     mlx-diag collection (needs <run>/V3-SEALED.json from m5-relay.sh mark)
#   m5-formal.sh smoke [max-items] [t1]        reference package, 8 items, throwaway cache, shared lease entry
#                                              owner.m5-formal-smoke (removed afterwards); needs >= 80 GB free VRAM;
#                                              t1 = the temperature-1 adapter spec (CAL698 rejected) instead
#
# <artifact-dir> is the arm's soup build /data/dev2/runs/dec/m5/soup/<ARM>/build/<ARM>-soup or the median seed's
# BEST checkpoint /data/dev2/runs/dec/m5/arms/full/m5-<ARM>-sN/<ckpt>. No upload anywhere (04:55 policy).
set -u
S=$(cd "$(dirname "$0")/../../../.." && pwd)
MIRROR=${S%/src/training/decision2}
SRC=${MIRROR##*/}
[ -f "$MIRROR/.dev2-mirror.json" ] || { echo "run from an exact mirror under /data/dev2/src" >&2; exit 2; }
# shellcheck source=src/training/decision2/v2/dec/ops/m5/m5-formal-lib.sh
. "$S/v2/dec/ops/m5/m5-formal-lib.sh"

devcal() { # devcal <label> <typed-dev preds> <css-pilot preds> <source (8K) calibration> <candidate (16K)> <receipt>
  [ -f "$6" ] && return 0
  py -m v2.release.dev_calibration --label "$1" --panel-root /data/dev2/private/panels --typed-dev "$2" \
    --css-pilot "$3" --source-calibration "$4" --candidate "$5" --work "${6%.json}-work" --output "$6" \
    > "${6%.json}.log" 2>&1
}

# stage <ARM> <artifact-dir>: sets NAME PKG LIST REV MODEL SPEC CAL DECISION CALX
stage() {
  local arm=$1 art=${2%/} tag D best
  case $art in
    "$M/soup/$arm/build/$arm-soup")
      tag=soup D=$M/soup/$arm
      SCAL=$D/cal698/calibration.json DEV=$D/dev/dev.predictions.jsonl CSS=$D/css-pilot/css-pilot.predictions.jsonl ;;
    "$M/arms/full/m5-$arm-"s[123]/*)
      D=$(dirname "$art") tag=${D##*-}
      best=$(jget "$D/BEST.json" checkpoint)
      [ "$(basename "$art")" = "$best" ] || die "$art is not the BEST checkpoint ($best) of $D"
      D=$D-post
      SCAL=$D/cal/calibration.json DEV=$D/dev/dev.predictions.jsonl CSS=$D/css-pilot/css-pilot.predictions.jsonl ;;
    *) die "artifact must be $M/soup/$arm/build/$arm-soup or a BEST checkpoint under $M/arms/full/m5-$arm-sN" ;;
  esac
  for f in "$art/decision_config.json" "$SCAL" "$DEV" "$CSS"; do [ -f "$f" ] || die "missing $f"; done
  NAME=m5-$arm-$tag PKG=$F/pkg/m5-$arm-$tag LIST=$F/pkg/m5-$arm-$tag.sha256
  local C16=$F/stage-cal/$NAME-cal698-16k DC=$F/stage-cal/$NAME-devcal.json
  mkdir -p "$F/stage-cal"
  if [ -f "$LIST" ]; then
    (cd "$PKG" && sha256sum -c --quiet "$LIST") > "$LIST.verify.log" 2>&1 || die "$NAME package changed since staging"
  else
    if [ ! -f "$C16/calibration.json" ]; then
      DEC_IMAGE=$IMAGE DEC_DATA=/data/dev2/runs/dec/m3/data-sel700-cal698 DEC_RENDER=${RENDER[$GPU]} \
        DEC_GPU_LABEL="node B GPU$GPU" bash "$S/v2/dec/launch.sh" "$NAME-cal698-16k" "$SRC" "$C16" -- \
        -m v2.dec.calibrate_ckpt --checkpoint "/runs/${art#/data/dev2/runs/dec/}" --cal /data/cal.jsonl \
        --max-length 16384 --output /out/calibration.json || die "$NAME 16K calibration FAILED"
    fi
    devcal "$NAME" "$DEV" "$CSS" "$SCAL" "$C16/calibration.json" "$DC" || die "$NAME 23:15 rule check FAILED"
    mkdir -p "$PKG/m5/$NAME/cal698-16k" "$PKG/m5/$NAME/cal698-8k-dev"
    cp -al "$art" "$PKG/m5/$NAME/checkpoint" || die "$NAME checkpoint copy FAILED"
    cp "$C16/calibration.json" "$PKG/m5/$NAME/cal698-16k/calibration.json"
    cp "$SCAL" "$PKG/m5/$NAME/cal698-8k-dev/calibration.json"
    cp "$DC" "$PKG/m5/$NAME/calibration-decision.json"
    tree_manifest "$PKG" > "$LIST"
    flog "$NAME staged: $(wc -l < "$LIST") files, revision $(sha "$LIST"), CAL698 16K adopt=$(jget "$DC" adopt)"
  fi
  REV=$(sha "$LIST") MODEL=$PKG/m5/$NAME/checkpoint DECISION=$PKG/m5/$NAME/calibration-decision.json
  if [ "$(jget "$DECISION" adopt)" = True ]; then
    SPEC=$SPEC_CAL CAL=$PKG/m5/$NAME/cal698-16k/calibration.json CALX=(--extra "calibration=$CAL")
  else
    SPEC=$SPEC_T1 CAL=none CALX=()
  fi
}

cmd=${1:-}
case $cmd in
  ref)
    verify_ref_package
    D4=/data/dev2/runs/dec/m4/soup/N4XF
    devcal N4XF-soup "$D4/dev/dev.predictions.jsonl" "$D4/css-pilot/css-pilot.predictions.jsonl" \
      "$D4/cal698/calibration.json" "$REF_CAL" "$F/stage-cal/N4XF-soup-ref-devcal.json" \
      || flog "N4XF 23:15 rule check FAILED (record only; the reference keeps the node-A calibration)"
    [ ! -e "$F/$REF_RUN" ] || die "$F/$REF_RUN exists"
    [ ! -e "$REF_CACHE" ] || [ -z "$(ls -A "$REF_CACHE")" ] || die "$REF_CACHE is not fresh"
    mkdir -p "$REF_CACHE"
    args=(--extra model_id=decision2-dec-m4-N4XF-soup --extra "calibration=$REF_CAL")
    flog "$REF_RUN collection start (GPU$GPU, fresh cache $REF_CACHE)"
    run_collect "$REF_RUN" "$REF_MODEL" "$REF_PKG" "$REF_LIST_SHA" "$SPEC_CAL" "$REF_CACHE" \
      "M5 formal reference N4XF soup (node B)" "${args[@]}" || die "$REF_RUN collection FAILED"
    finish "$REF_RUN" ref N4XF-soup decision2-dec-m4-N4XF-soup-nodeB-ref "$REF_PKG" "$F/pkg/N4XF-soup-ref.sha256" \
      "$REF_LIST_SHA" "$REF_MODEL" "$SPEC_CAL" "$REF_CACHE" none "$REF_CAL" "$F/stage-cal/N4XF-soup-ref-devcal.json" "${args[@]}"
    cache_freeze "$REF_CACHE" "$MASTER"
    ;;
  stage)
    [ $# -eq 3 ] || { sed -n '2,20p' "$0"; exit 2; }
    stage "$2" "$3"
    echo "$NAME revision $REV spec $SPEC calibration $CAL"
    ;;
  finalist)
    [ $# -eq 3 ] || { sed -n '2,20p' "$0"; exit 2; }
    [ -f "$MASTER.sha256" ] || die "reference cache not frozen yet (run: m5-formal.sh ref)"
    stage "$2" "$3"
    [ ! -e "$F/$NAME" ] || die "$F/$NAME exists"
    cache_copy "$MASTER" "$MASTER.sha256" "$F/$NAME-cache"
    args=(--extra "model_id=decision2-dec-$NAME" "${CALX[@]}")
    flog "$NAME collection start (GPU$GPU, copy of $MASTER)"
    run_collect "$NAME" "$MODEL" "$PKG" "$REV" "$SPEC" "$F/$NAME-cache" "M5 formal finalist $NAME (node B)" "${args[@]}" \
      || die "$NAME collection FAILED"
    finish "$NAME" finalist "$NAME" "decision2-dec-$NAME" "$PKG" "$LIST" "$REV" "$MODEL" "$SPEC" "$F/$NAME-cache" \
      "$MASTER.sha256" "$CAL" "$DECISION" "${args[@]}"
    ;;
  mlx)
    [ $# -eq 2 ] || { sed -n '2,20p' "$0"; exit 2; }
    RUN=$2 R=$F/$2
    [ -f "$R/V3-SEALED.json" ] || die "$RUN: v3 report not sealed on node A yet (m5-relay.sh mark $RUN)"
    [ ! -e "$F/$RUN-mlx" ] || die "$F/$RUN-mlx exists"
    if [ "$RUN" = "$REF_RUN" ]; then base=$MASTER; else base=$MASTER_MLX; fi
    [ -f "$base.sha256" ] || die "no frozen cache $base (the reference mlx-diag run comes first)"
    mapfile -t args < <(jget "$R/M5-RECEIPT.json" collect_args | python3 -c 'import json,sys; print("\n".join(json.load(sys.stdin)))')
    model=$(jget "$R/M5-RECEIPT.json" model) pkg=$(jget "$R/M5-RECEIPT.json" package_dir)
    rev=$(jget "$R/M5-RECEIPT.json" revision) spec=$(jget "$R/M5-RECEIPT.json" spec)
    cache_copy "$base" "$base.sha256" "$F/$RUN-mlx-cache"
    flog "$RUN-mlx collection start (GPU$GPU, copy of $base)"
    run_collect "$RUN-mlx" "$model" "$pkg" "$rev" "$spec" "$F/$RUN-mlx-cache" "M5 mlx-diag $RUN (node B)" \
      "${args[@]}" --panels mlx-diag || die "$RUN-mlx collection FAILED"
    finish "$RUN-mlx" mlx "$(jget "$R/M5-RECEIPT.json" name)" "$(jget "$R/M5-RECEIPT.json" label)" "$pkg" "$R/PACKAGE.sha256" \
      "$rev" "$model" "$spec" "$F/$RUN-mlx-cache" "$base.sha256" "$(jget "$R/M5-RECEIPT.json" calibration_used)" \
      "$(jget "$R/M5-RECEIPT.json" calibration_decision)" "${args[@]}"
    if [ "$RUN" = "$REF_RUN" ]; then cache_freeze "$F/$RUN-mlx-cache" "$MASTER_MLX"; fi
    ;;
  smoke)
    N=${2:-8}
    spec=$SPEC_CAL args=(--extra model_id=decision2-dec-m4-N4XF-soup --extra "calibration=$REF_CAL") cal=$REF_CAL
    [ "${3:-}" = t1 ] && spec=$SPEC_T1 args=(--extra model_id=decision2-dec-m4-N4XF-soup) cal=none
    verify_ref_package
    free=$(vram_free_gb "$GPU")
    [ "$free" -ge 80 ] || die "smoke: GPU$GPU has $free GB free VRAM (< 80)"
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    RUN=dryrun/smoke-$ts C=$TMPDIR/m5-formal-smoke-cache-$ts
    export M5_SHARED_LEASE=m5-formal-smoke
    trap 'rm -rf "$C"; rm -f "/data/dev2/leases/gpu$GPU.lock/owner.$M5_SHARED_LEASE"' EXIT
    mkdir -p "$C" "$F/dryrun"
    flog "smoke $RUN start (GPU$GPU shared, $free GB free, throwaway cache $C, spec $spec)"
    run_collect "$RUN" "$REF_MODEL" "$REF_PKG" "$REF_LIST_SHA" "$spec" "$C" "M5 formal tooling smoke ($N items)" \
      "${args[@]}" --max-items "$N" || die "smoke collection FAILED (log $F/$RUN.collect.log)"
    finish "$RUN" smoke N4XF-soup decision2-dec-m4-N4XF-soup-nodeB-smoke "$REF_PKG" "$F/pkg/N4XF-soup-ref.sha256" \
      "$REF_LIST_SHA" "$REF_MODEL" "$spec" "$C" none "$cal" none "${args[@]}" --max-items "$N"
    ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
