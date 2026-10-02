#!/usr/bin/env bash
# Decoder M16 development readouts (prereg dec-m16-prereg-2026-10-01.md, "Development readouts"), node A / B, from an
# exact mirror. Every point is read as M14: v2.dec.infer_dec through m16-launch.sh, one container per panel, 16,384
# tokens, T = 1, image dbe5f32b, the node's <tier>-read Triton cache (tier = the point's prefix: 2b, 08b or 4b).
# Outputs: /data/dev2/runs/dec/m16/lines/<point>/<panel>/<panel>.predictions.jsonl (+ manifest and launch receipt). A
# panel already read is skipped; a failed read is recorded and not rerun. The caller holds the GPU's chain flock
# (m16-chain.sh).
#
#   M16_NODE=a|b m16-lines.sh read <mirror-dir> <gpu> <point> <checkpoint> <source> [panel ...]
#       <checkpoint> / <source> are host paths under /data/dev2/runs/dec or /data/dev2/hf-cache.
#       Default panels: dev css-pilot ht-dev2 score5t-dev hs1-dev pn1-dev m10-probes ib-dev.
#   M16_NODE=a|b m16-lines.sh mlx <mirror-dir> <gpu> <point> <checkpoint> <source>
#       MLX-DEV readout: v2.dec.eval_rows (raw probabilities, max length 8,192, as SELECT700) on the tier's panel
#       (m16/mlxdev/<tier>/panel.jsonl) -> lines/<point>/mlxdev/mlxdev-predictions.jsonl; read once, never rerun.
#   M16_NODE=a|b m16-lines.sh mlxcmp <mirror-dir> <point> <reference>
#       v2.dec.mlx_dev score of <point> and compare (A = <reference>, B = <point>; 2,000 draws, the tool's seed), CPU
#       -> lines/<point>/mlxcmp/<point>.mlxdev.json (+ <point>.mlxscore.json)
set -u
MODE=${1:-} SRC=${2:-}
NODE=${M16_NODE:?set M16_NODE=a or b}
M=/data/dev2/runs/dec/m16
L=$M/lines
CODE=/data/dev2/src/$SRC/src/training/decision2
OPS=$CODE/v2/dec/ops/m16
mkdir -p "$L"
log() { echo "$(date -u +%FT%TZ) lines-$NODE $*" | tee -a "$L/OPERATIONS.log"; }
incontainer() {
  case $1 in
    /data/dev2/runs/dec/*) echo "/runs/${1#/data/dev2/runs/dec/}" ;;
    /data/dev2/hf-cache/*) echo "/hf/${1#/data/dev2/hf-cache/}" ;;
    *) echo "path outside the mounted roots: $1" >&2; return 1 ;;
  esac
}
lease_busy() {  # <gpu> <purpose>
  printf 'track=dec-m16\nstatus=busy\npurpose=decoder M16 %s\nstart_utc=%s\nexpected_end_utc=%s\n' \
    "$2" "$(date -u +%FT%TZ)" "$(date -u -d '+60 min' +%FT%TZ)" > "/data/dev2/leases/gpu$1.lock/owner"
}
lease_idle() {  # <gpu> <purpose>
  printf 'track=dec-m16\nstatus=idle\npurpose=decoder M16 %s\nstart_utc=%s\nexpected_end_utc=%s\n' \
    "$2" "$(date -u +%FT%TZ)" "$(date -u -d '+30 min' +%FT%TZ)" > "/data/dev2/leases/gpu$1.lock/owner"
}

case $MODE in
  read)
    GPU=$3 POINT=$4 CK=$5 SOURCE=$6
    shift 6
    PANELS=${*:-dev css-pilot ht-dev2 score5t-dev hs1-dev pn1-dev m10-probes ib-dev}
    case ${POINT%%-*} in 2b | 08b | 4b) ;; *) log "$POINT: point names start with 2b-, 08b- or 4b-"; exit 2 ;; esac
    ck=$(incontainer "$CK") || exit 2
    source=$(incontainer "$SOURCE") || exit 2
    lease_busy "$GPU" "readouts ($POINT)"
    failed=0
    for panel in $PANELS; do
      out=$L/$POINT/$panel
      if [ -f "$out.launch.json" ]; then
        grep -q '"exit_status": 0' "$out.launch.json" || { log "$POINT $panel failed earlier; not rerun"; failed=1; }
        continue
      fi
      [ -f "/data/dev2/runs/dec/panels/$panel.prompts.jsonl" ] || { log "$POINT $panel: no prompts"; failed=1; continue; }
      if M16_NODE=$NODE M16_CACHE=${POINT%%-*}-read bash "$OPS/m16-launch.sh" "lines-$POINT-$panel" "$SRC" "$out" --gpu "$GPU" -- \
        -m v2.dec.infer_dec --checkpoint "$ck" --source-path "$source" --input "/panels/$panel.prompts.jsonl" \
        --output "/out/$panel.predictions.jsonl" --model-id "decision2-dec-m16-$POINT" --model-revision "$POINT" \
        --max-length 16384; then
        log "$POINT $panel read: $(wc -l < "$out/$panel.predictions.jsonl") items"
      else
        log "$POINT $panel FAILED: $(tail -c 300 "$out.stderr.log" | tr '\n' ' ')"
        failed=1
      fi
    done
    lease_idle "$GPU" "(readouts of $POINT finished)"
    exit $failed
    ;;
  mlx)
    GPU=$3 POINT=$4 CK=$5 SOURCE=$6 t=${4%%-*}
    out=$L/$POINT/mlxdev
    if [ -f "$out.launch.json" ]; then
      grep -q '"exit_status": 0' "$out.launch.json" || { log "$POINT MLX-DEV failed earlier; not rerun"; exit 1; }
      exit 0
    fi
    [ -f "$M/mlxdev/$t/panel.jsonl" ] || { log "$POINT: no MLX-DEV panel for tier $t"; exit 1; }
    ck=$(incontainer "$CK") || exit 2
    source=$(incontainer "$SOURCE") || exit 2
    lease_busy "$GPU" "MLX-DEV readout ($POINT)"
    mkdir -p "$L/$POINT"
    if M16_NODE=$NODE M16_CACHE=$t-read bash "$OPS/m16-launch.sh" "lines-$POINT-mlxdev" "$SRC" "$out" --gpu "$GPU" -- \
      -m v2.dec.eval_rows --checkpoint "$ck" --source-path "$source" --rows "/runs/m16/mlxdev/$t/panel.jsonl" \
      --tag mlxdev --output /out; then
      log "$POINT MLX-DEV read: $(tail -1 "$out.stdout.log" | cut -c1-200)"
    else
      log "$POINT MLX-DEV FAILED: $(tail -c 300 "$out.stderr.log" | tr '\n' ' ')"
      lease_idle "$GPU" "(MLX-DEV of $POINT failed)"
      exit 1
    fi
    lease_idle "$GPU" "(MLX-DEV of $POINT finished)"
    ;;
  mlxcmp)
    POINT=$3 REF=$4 t=${3%%-*}
    out=$L/$POINT/mlxcmp
    [ -f "$out/$POINT.mlxdev.json" ] && exit 0
    [ ! -e "$out.launch.json" ] || { log "$POINT MLX-DEV compare failed earlier; not rerun"; exit 1; }
    for p in "$POINT" "$REF"; do
      [ -f "$L/$p/mlxdev/mlxdev-predictions.jsonl" ] || { log "$POINT MLX-DEV compare: no readout of $p"; exit 1; }
    done
    idx=/runs/m16/mlxdev/$t/panel.jsonl.index.jsonl
    M16_NODE=$NODE bash "$OPS/m16-launch.sh" "mlxcmp-$POINT" "$SRC" "$out" --cpu -- -c "
import subprocess, sys
for args in (['score', '--predictions', '/runs/m16/lines/$POINT/mlxdev/mlxdev-predictions.jsonl',
              '--output', '/out/$POINT.mlxscore.json'],
             ['compare', '--a', '/runs/m16/lines/$REF/mlxdev/mlxdev-predictions.jsonl',
              '--b', '/runs/m16/lines/$POINT/mlxdev/mlxdev-predictions.jsonl', '--output', '/out/$POINT.mlxdev.json']):
    subprocess.run([sys.executable, '-m', 'v2.dec.mlx_dev', args[0], '--index', '$idx', *args[1:]], check=True)
" || { log "$POINT MLX-DEV compare FAILED (see $out.stderr.log)"; exit 1; }
    log "$POINT MLX-DEV vs $REF: $(tail -1 "$out.stdout.log" | cut -c1-300)"
    ;;
  *) sed -n '2,17p' "$0"; exit 2 ;;
esac
