#!/usr/bin/env bash
# Decoder M6 formal runs (prereg dec-m6-prereg-2026-09-29.md, "Formal runs (finalists only)"), run from an exact
# mirror on the collecting node; outputs under /data/dev2/runs/dec/formal/m6 (M6_FORMAL_ROOT). GPU M6_GPU (node B 3|4,
# node A 5); every GPU job is a co-tenant (>= 60 GB free VRAM, lease entry owner.dec-formal / owner.m6-formal-smoke).
#
#   m6-formal.sh 2b ref                      node-B S2T reference: verify the released staging folder against its
#                                            list (22e7fe86...), collect v3 + public 231 on a FRESH cache with the
#                                            node-A run's spec and calibration, seal, freeze the cache (cache-frozen-2b)
#   m6-formal.sh <tier> stage <point>        finalist package: 16K CAL698 fit, 23:15 rule (v2.release.dev_calibration)
#                                            on the point's 16K typed DEV + CSS pilot readouts, checkpoint (verified
#                                            against its files.sha256) + calibrations + weights.json, per-file SHA-256
#                                            list (revision = its hash), parameter count + header stubs
#   m6-formal.sh <tier> smoke <point> [N]    stage, then N (default 8) items on a throwaway copy of the master cache
#   m6-formal.sh <tier> finalist <point>     stage (if needed), collect + seal on a copy of the tier's master cache
#   m6-formal.sh <tier> mlx <run>            mlx-diag collection (node B: needs <run>/V3-SEALED.json from
#                                            m6-relay.sh mark; node A: needs <run>/REPORT.json from m6-score.sh)
#   m6-formal.sh <tier> gpuh                 GPU-hours of the formal jobs so far (GPU-SECONDS.jsonl)
#
# <point> is a finalist named in /data/dev2/runs/dec/m6/select/<tier>-finalists.json (m6-rules.sh; M6_SELECT names
# another select directory, e.g. a later milestone's preregistered finalists). Tiers: 4b node B
# (master formal/m5/cache-frozen f6d0f920..., mlx cache-frozen-mlx 65d7d38f...); 2b node B after `2b ref` and a
# node-A exactness check (REF-EXACT.json relayed by m6-relay.sh mark-ref), or node A with M6_2B_NODE=A (image
# f83b1d10, HIP_FORCE_DEV_KERNARG=1, copy of formal/m3/m3-S2T-soup-nodeA-triton, package relayed by m6-relay.sh pkg,
# or staged on node A with M6_2B_STAGE_A=1 when the points live there); M6_PREFIX (default m6) names runs/packages;
# 08b node A (copy of formal/m2/m2-E8F-soup-nodeA-triton). CAL698 is data-sel700-cal698/cal.jsonl (19cc1a8c...;
# relay it to node A with m6-relay.sh cal698). No upload anywhere.
set -u
S=$(cd "$(dirname "$0")/../../../.." && pwd)
MIRROR=${S%/src/training/decision2}
SRC=${MIRROR##*/}
[ -f "$MIRROR/.dev2-mirror.json" ] || { echo "run from an exact mirror under /data/dev2/src" >&2; exit 2; }
TIER=${1:-} CMD=${2:-}
case $TIER in 4b|2b|08b) ;; *) sed -n '2,28p' "$0"; exit 2 ;; esac
# shellcheck source=src/training/decision2/v2/dec/ops/m6/m6-formal-lib.sh
. "$S/v2/dec/ops/m6/m6-formal-lib.sh"
tier_setup
if [ "$NODE" = A ] && [ "$TIER" = 4b ]; then
  [ -d "$MASTER" ] && [ -d "$MASTER_MLX" ] || die "4B on node A needs the master caches $MASTER and $MASTER_MLX"
elif [ "$NODE" != A ]; then
  [ -d "$R/formal/m5/cache-frozen" ] || die "tier $TIER collects on node $NODE (no formal/m5/cache-frozen here)"
else
  [ -d "$R/formal/m2/m2-E8F-soup-nodeA-triton" ] || die "tier $TIER collects on node A (no formal/m2/m2-E8F-soup-nodeA-triton here)"
fi

devcal() { # devcal <label> <typed-dev preds> <css-pilot preds> <candidate (16K)> <receipt>
  [ -f "$5" ] && return 0
  py -m v2.release.dev_calibration --label "$1" --panel-root "$PANELS" --typed-dev "$2" --css-pilot "$3" \
    --candidate "$4" --work "${5%.json}-work" --output "$5" > "${5%.json}.log" 2>&1
}

# stage <point>: sets NAME PKG LIST REV MODEL SPEC CAL DECISION CALX PARAMS
stage() {
  local point=$1
  NAME=$PFX-$point PKG=$F/pkg/$PFX-$point LIST=$F/pkg/$PFX-$point.sha256 PARAMS=$F/stage-params/$PFX-$point
  if [ -f "$LIST" ]; then
    (cd "$PKG" && sha256sum -c --quiet "$LIST") > "$LIST.verify.log" 2>&1 || die "$NAME package changed since staging"
    [ "$(tree_manifest "$PKG" | sha256sum | cut -d' ' -f1)" = "$(sha "$LIST")" ] || die "$NAME package has extra or missing files"
  else
    [ "$NODE" = A ] && [ "$TIER" = 2b ] && [ -z "${M6_2B_STAGE_A:-}" ] \
      && die "$NAME: stage on node B and relay the package (m6-relay.sh pkg $point)"
    local fj=$SEL/$TIER-finalists.json
    [ -f "$fj" ] || die "no $fj (m6-rules.sh $TIER)"
    local info
    info=$(python3 - "$fj" "$point" <<'EOF'
import json, sys
for f in json.load(open(sys.argv[1]))["finalists"]:
    if f["point"] == sys.argv[2]:
        print(f["checkpoint"], f["weights_json"], f["files_sha256_list"], f["typed_dev_predictions"], f["css_pilot_predictions"])
        break
else:
    raise SystemExit(f"{sys.argv[2]} is not a finalist in {sys.argv[1]}")
EOF
    ) || die "$point is not a finalist ($fj)"
    local art wj fl dev css
    read -r art wj fl dev css <<< "$info"
    for f in "$art/decision_config.json" "$wj" "$fl" "$dev" "$css"; do [ -f "$f" ] || die "missing $f"; done
    (cd "$art" && sha256sum -c --quiet "$fl") > "$F/stage-cal/$NAME.checkpoint-verify.log" 2>&1 \
      || die "$NAME checkpoint differs from its line hash list $fl"
    [ "$(tree_manifest "$art" | sha256sum | cut -d' ' -f1)" = "$(sha "$fl")" ] || die "$NAME checkpoint has extra or missing files"
    local C16=$F/stage-cal/$NAME-cal698-16k DC=$F/stage-cal/$NAME-devcal.json
    if [ ! -f "$C16/calibration.json" ]; then
      [ "$(sha "$CAL698_DIR/cal.jsonl")" = "$CAL698_SHA" ] || die "CAL698 $CAL698_DIR/cal.jsonl is not $CAL698_SHA"
      [ ! -e "$C16.launch.json" ] || die "$NAME: earlier 16K calibration failed (receipt $C16.launch.json); not rerun"
      gpu_check "$NAME CAL698 16K fit"
      lease_entry dec-formal "$NAME CAL698 16K fit"
      DEC_IMAGE=$IMAGE DEC_DATA=$CAL698_DIR DEC_RENDER=${RENDER_OF[$GPU]} DEC_GPU_LABEL="node $NODE GPU$GPU" \
        bash "$S/v2/dec/launch.sh" "$NAME-cal698-16k" "$SRC" "$C16" -- \
        -m v2.dec.calibrate_ckpt --checkpoint "/runs/${art#"$R"/}" --cal /data/cal.jsonl \
        --max-length 16384 --output /out/calibration.json
      local rc=$?
      lease_clear dec-formal
      gpu_seconds launch "$C16.launch.json" "$NAME" "CAL698 16K fit"
      [ $rc = 0 ] || die "$NAME 16K calibration FAILED"
    fi
    devcal "$NAME" "$dev" "$css" "$C16/calibration.json" "$DC" || die "$NAME 23:15 rule check FAILED"
    local D=$PKG/m6/$NAME
    mkdir -p "$D/cal698-16k"
    cp -al "$art" "$D/checkpoint" || die "$NAME checkpoint copy FAILED"
    cp "$C16/calibration.json" "$D/cal698-16k/calibration.json"
    cp "$DC" "$D/calibration-decision.json"
    cp "$wj" "$D/weights.json"
    cp "$fl" "$D/checkpoint.files.sha256"
    tree_manifest "$PKG" > "$LIST"
    if [ ! -f "$PARAMS/PARAMS.json" ]; then
      mkdir -p "$PARAMS"
      py "$M5OPS/m5-params.py" --package "$D/checkpoint" --stubs "$PARAMS/pkg-headers" --output "$PARAMS/PARAMS.json" \
        > "$PARAMS.log" 2>&1 || die "$NAME parameter count FAILED"
    fi
    flog "$NAME staged: $(wc -l < "$LIST") files, revision $(sha "$LIST"), CAL698 16K adopt=$(jget "$DC" adopt), parameters $(jget "$PARAMS/PARAMS.json" parameters)"
  fi
  [ -f "$PARAMS/PARAMS.json" ] || die "$NAME: no parameter count $PARAMS/PARAMS.json"
  REV=$(sha "$LIST") MODEL=$PKG/m6/$NAME/checkpoint DECISION=$PKG/m6/$NAME/calibration-decision.json
  if [ "$(jget "$DECISION" adopt)" = True ]; then
    SPEC=$SPEC_CAL CAL=$PKG/m6/$NAME/cal698-16k/calibration.json CALX=(--extra "calibration=$CAL")
  else
    SPEC=$SPEC_T1 CAL=none CALX=()
  fi
}

# master_ready: the finalist master cache of this tier / node exists (2B node B: reference exact on node A).
master_ready() {
  if [ "$TIER" = 2b ] && [ "$NODE" = B ]; then
    [ -f "$MASTER.sha256" ] || die "2B reference cache not frozen yet (m6-formal.sh 2b ref)"
    [ -f "$F/$REF2_RUN/REF-EXACT.json" ] || die "2B reference not checked on node A yet (m6-score.sh 2b ref; m6-relay.sh mark-ref)"
    [ "$(jget "$F/$REF2_RUN/REF-EXACT.json" exact)" = True ] \
      || die "2B node-B reference is not exact: collect 2B finalists on node A (M6_2B_NODE=A)"
  fi
}

case $CMD in
  ref)
    [ "$TIER" = 2b ] && [ "$NODE" = B ] || die "ref is the 2B node-B step"
    [ "$(sha "$REF2_LIST")" = "$REF2_LIST_SHA" ] || die "S2T staged list hash differs from $REF2_LIST_SHA"
    if [ ! -d "$REF2_PKG" ]; then
      mkdir -p "$F/pkg"
      cp -al "$REF2_STAGE" "$REF2_PKG" || die "reference package copy FAILED"
    fi
    (cd "$REF2_PKG" && sha256sum -c --quiet "$REF2_LIST") > "$F/pkg/S2T-soup-ref.verify.log" 2>&1 \
      || die "reference package hash verification FAILED"
    tree_manifest "$REF2_PKG" > "$F/pkg/S2T-soup-ref.sha256"
    [ "$(sha "$F/pkg/S2T-soup-ref.sha256")" = "$REF2_LIST_SHA" ] || die "reference package has extra or missing files"
    [ ! -e "$F/$REF2_RUN" ] || die "$F/$REF2_RUN exists"
    [ ! -e "$REF2_CACHE" ] || [ -z "$(ls -A "$REF2_CACHE")" ] || die "$REF2_CACHE is not fresh"
    mkdir -p "$REF2_CACHE"
    args=(--extra model_id=decision2-dec-m3-S2T-soup --extra "calibration=$REF2_CAL")
    flog "$REF2_RUN collection start (GPU$GPU, fresh cache $REF2_CACHE)"
    run_collect "$REF2_RUN" "$REF2_MODEL" "$REF2_PKG" "$REF2_LIST_SHA" "$SPEC_CAL" "$REF2_CACHE" \
      "M6 2B reference S2T soup (node B)" "${args[@]}" || die "$REF2_RUN collection FAILED"
    finish "$REF2_RUN" ref S2T-soup decision2-dec-m3-S2T-soup-nodeB-ref "$REF2_PKG" "$F/pkg/S2T-soup-ref.sha256" \
      "$REF2_LIST_SHA" "$REF2_MODEL" "$SPEC_CAL" "$REF2_CACHE" none "$REF2_CAL" none - - "${args[@]}"
    cache_freeze "$REF2_CACHE" "$MASTER"
    ;;
  stage)
    [ $# -eq 3 ] || { sed -n '2,28p' "$0"; exit 2; }
    stage "$3"
    echo "$NAME revision $REV spec $SPEC calibration $CAL"
    ;;
  smoke)
    [ $# -ge 3 ] || { sed -n '2,28p' "$0"; exit 2; }
    master_ready
    stage "$3"
    N=${4:-8} ts=$(date -u +%Y%m%dT%H%M%SZ)
    RUN=dryrun/smoke-$3-$ts C=$TMPDIR/m6-formal-smoke-cache-$3-$ts
    export M6_SHARED_LEASE=m6-formal-smoke
    trap 'rm -rf "$C" "$C.master.sha256"' EXIT
    mkdir -p "$F/dryrun"
    cache_copy "$MASTER" "$MASTER_SHA" "$C"
    args=(--extra "model_id=decision2-dec-$NAME" "${CALX[@]}")
    run_collect "$RUN" "$MODEL" "$PKG" "$REV" "$SPEC" "$C" "M6 formal smoke $NAME ($N items)" "${args[@]}" --max-items "$N" \
      || die "smoke collection FAILED (log $F/$RUN.collect.log)"
    finish "$RUN" smoke "$NAME" "decision2-dec-$NAME-smoke" "$PKG" "$LIST" "$REV" "$MODEL" "$SPEC" "$C" "$COPY_BEFORE" \
      "$CAL" "$DECISION" "$PARAMS" "$3" "${args[@]}" --max-items "$N"
    ;;
  finalist)
    [ $# -eq 3 ] || { sed -n '2,28p' "$0"; exit 2; }
    master_ready
    stage "$3"
    [ ! -e "$F/$NAME" ] || die "$F/$NAME exists"
    cache_copy "$MASTER" "$MASTER_SHA" "$F/$NAME-cache"
    args=(--extra "model_id=decision2-dec-$NAME" "${CALX[@]}")
    flog "$NAME collection start (GPU$GPU, copy of $MASTER)"
    run_collect "$NAME" "$MODEL" "$PKG" "$REV" "$SPEC" "$F/$NAME-cache" "M6 formal finalist $NAME (node $NODE)" "${args[@]}" \
      || die "$NAME collection FAILED"
    finish "$NAME" finalist "$NAME" "decision2-dec-$NAME" "$PKG" "$LIST" "$REV" "$MODEL" "$SPEC" "$F/$NAME-cache" \
      "$COPY_BEFORE" "$CAL" "$DECISION" "$PARAMS" "$3" "${args[@]}"
    ;;
  mlx)
    [ $# -eq 3 ] || { sed -n '2,28p' "$0"; exit 2; }
    RUN=$3 RD=$F/$3
    [ -f "$RD/M6-RECEIPT.json" ] || die "$RUN has no M6 receipt"
    if [ "$NODE" != A ]; then
      [ -f "$RD/V3-SEALED.json" ] || die "$RUN: v3 report not sealed on node A yet (m6-relay.sh mark $RUN)"
    else
      [ -f "$RD/REPORT.json" ] || die "$RUN: v3 report not written yet (m6-score.sh $TIER $RUN)"
    fi
    [ ! -e "$F/$RUN-mlx" ] || die "$F/$RUN-mlx exists"
    if [ "$RUN" = "$REF2_RUN" ]; then base=$MASTER pin=$MASTER_SHA; else master_ready; base=$MASTER_MLX pin=$MASTER_MLX_SHA; fi
    mapfile -t args < <(jget "$RD/M5-RECEIPT.json" collect_args | python3 -c 'import json,sys; print("\n".join(json.load(sys.stdin)))')
    model=$(jget "$RD/M5-RECEIPT.json" model) pkg=$(jget "$RD/M5-RECEIPT.json" package_dir)
    rev=$(jget "$RD/M5-RECEIPT.json" revision) spec=$(jget "$RD/M5-RECEIPT.json" spec)
    point=$(jget "$RD/M6-RECEIPT.json" point)
    [ "$point" = None ] && point=-
    params=-
    [ "$point" != - ] && params=$F/stage-params/$PFX-$point
    cache_copy "$base" "$pin" "$F/$RUN-mlx-cache"
    flog "$RUN-mlx collection start (GPU$GPU, copy of $base)"
    run_collect "$RUN-mlx" "$model" "$pkg" "$rev" "$spec" "$F/$RUN-mlx-cache" "M6 mlx-diag $RUN (node $NODE)" \
      "${args[@]}" --panels mlx-diag || die "$RUN-mlx collection FAILED"
    finish "$RUN-mlx" mlx "$(jget "$RD/M5-RECEIPT.json" name)" "$(jget "$RD/M5-RECEIPT.json" label)" "$pkg" "$RD/PACKAGE.sha256" \
      "$rev" "$model" "$spec" "$F/$RUN-mlx-cache" "$COPY_BEFORE" "$(jget "$RD/M5-RECEIPT.json" calibration_used)" \
      "$(jget "$RD/M5-RECEIPT.json" calibration_decision)" "$params" "$point" "${args[@]}"
    if [ "$RUN" = "$REF2_RUN" ]; then cache_freeze "$F/$RUN-mlx-cache" "$MASTER_MLX"; fi
    ;;
  gpuh)
    [ -f "$F/GPU-SECONDS.jsonl" ] || { echo "no GPU jobs yet"; exit 0; }
    python3 - "$F/GPU-SECONDS.jsonl" <<'EOF'
import json, sys
from collections import defaultdict
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by = defaultdict(float)
for r in rows:
    by[r["run"].split("/")[0]] += r["gpu_seconds"] or 0
for k, v in sorted(by.items()):
    print(f"{k:40s} {v / 3600:.3f} GPU-h")
print(f"{'total':40s} {sum(by.values()) / 3600:.3f} GPU-h over {len(rows)} jobs")
EOF
    ;;
  *) sed -n '2,28p' "$0"; exit 2 ;;
esac
