#!/usr/bin/env bash
# Decoder M15 development readouts (prereg dec-m15-prereg-2026-10-01.md, "Development readouts"), node E / F, from an
# exact mirror. Every point is read the same way: v2.dec.infer_dec through m15-launch.sh, one container per panel,
# 16,384 tokens, T = 1, image dbe5f32b, the node's <tier>-read Triton cache (tier = the point's prefix: 2b, 08b or
# 4b, stage 2).
# Outputs: /data/dev2/runs/dec/m15/lines/<point>/<panel>/<panel>.predictions.jsonl (+ manifest and launch receipt). A
# panel already read is skipped; a failed read is recorded and not rerun. The caller holds the GPU's chain flock
# (m15-post.sh, m15-refs.sh), so reads never share a GPU with training.
#
#   M15_NODE=e|f m15-lines.sh read <mirror-dir> <gpu> <point> <checkpoint> <source> [panel ...]
#       <checkpoint> / <source> are host paths under /data/dev2/runs/dec or /data/dev2/models.
#       Default panels: dev css-pilot ht-dev2 score5t-dev hs1-dev pn1-dev m10-probes.
#   M15_NODE=e|f m15-lines.sh mlx <mirror-dir> <gpu> <point> <checkpoint> <source>
#       MLX-DEV-M15 readout: v2.dec.eval_rows (raw probabilities, max length 8,192, as SELECT700) on the tier's panel
#       (m15/mlxdev/<tier>/panel.jsonl) -> lines/<point>/mlxdev/mlxdev-predictions.jsonl; read once, never rerun.
#   M15_NODE=e|f m15-lines.sh mlxcmp <mirror-dir> <point> <reference>
#       v2.dec.mlx_dev score of <point> and compare (B = <point>, A = <reference>; 2,000 draws, seed 20260929), CPU
#       -> lines/<point>/mlxcmp/<point>.mlxdev.json (+ <point>.mlxscore.json)
#   M15_NODE=e|f m15-lines.sh parity <mirror-dir> <point> <reference-dir> [panel ...]
#       answer agreement and maximum probability drift of <point>'s readouts against stored predictions
#       (<reference-dir>/<panel>/<panel>.predictions.jsonl) -> lines/<point>/parity-<name>.json
set -u
MODE=${1:-} SRC=${2:-}
NODE=${M15_NODE:?set M15_NODE=e or f}
M=/data/dev2/runs/dec/m15
L=$M/lines
CODE=/data/dev2/src/$SRC/src/training/decision2
OPS=$CODE/v2/dec/ops/m15
mkdir -p "$L"
log() { echo "$(date -u +%FT%TZ) lines-$NODE $*" | tee -a "$L/OPERATIONS.log"; }
incontainer() {
  case $1 in
    /data/dev2/runs/dec/*) echo "/runs/${1#/data/dev2/runs/dec/}" ;;
    /data/dev2/models/*) echo "/models/${1#/data/dev2/models/}" ;;
    *) echo "path outside the mounted roots: $1" >&2; return 1 ;;
  esac
}

case $MODE in
  read)
    GPU=$3 POINT=$4 CK=$5 SOURCE=$6
    shift 6
    PANELS=${*:-dev css-pilot ht-dev2 score5t-dev hs1-dev pn1-dev m10-probes}
    lease=/data/dev2/leases/gpu$GPU.lock/owner
    if grep -qs "^status=busy" "$lease" && grep -qs "training" "$lease"; then
      log "$POINT: GPU$GPU is training; not read"
      exit 1
    fi
    mkdir -p "$(dirname "$lease")"
    printf 'track=dec-m15\nstatus=busy\npurpose=decoder M15 readouts (%s)\nstart_utc=%s\nexpected_end_utc=%s\n' \
      "$POINT" "$(date -u +%FT%TZ)" "$(date -u -d '+60 min' +%FT%TZ)" > "$lease"
    case ${POINT%%-*} in 2b | 08b | 4b) ;; *) log "$POINT: point names start with 2b-, 08b- or 4b-"; exit 2 ;; esac
    ck=$(incontainer "$CK") || exit 2
    source=$(incontainer "$SOURCE") || exit 2
    for panel in $PANELS; do
      out=$L/$POINT/$panel
      if [ -f "$out.launch.json" ]; then
        grep -q '"exit_status": 0' "$out.launch.json" || log "$POINT $panel failed earlier; not rerun"
        continue
      fi
      [ -f "/data/dev2/runs/dec/panels/$panel.prompts.jsonl" ] || { log "$POINT $panel: no prompts"; continue; }
      if M15_NODE=$NODE M15_CACHE=${POINT%%-*}-read bash "$OPS/m15-launch.sh" "lines-$POINT-$panel" "$SRC" "$out" --gpu "$GPU" -- \
        -m v2.dec.infer_dec --checkpoint "$ck" --source-path "$source" --input "/panels/$panel.prompts.jsonl" \
        --output "/out/$panel.predictions.jsonl" --model-id "decision2-dec-m15-$POINT" --model-revision "$POINT" \
        --max-length 16384; then
        log "$POINT $panel read: $(wc -l < "$out/$panel.predictions.jsonl") items"
      else
        log "$POINT $panel FAILED: $(tail -c 300 "$out.stderr.log" | tr '\n' ' ')"
      fi
    done
    printf 'track=dec-m15\nstatus=idle\npurpose=decoder M15 (readouts of %s finished)\nstart_utc=%s\nexpected_end_utc=%s\n' \
      "$POINT" "$(date -u +%FT%TZ)" "$(date -u -d '+30 min' +%FT%TZ)" > "$lease"
    ;;
  mlx)
    GPU=$3 POINT=$4 CK=$5 SOURCE=$6 t=${4%%-*}
    out=$L/$POINT/mlxdev
    if [ -f "$out.launch.json" ]; then
      grep -q '"exit_status": 0' "$out.launch.json" || { log "$POINT MLX-DEV failed earlier; not rerun"; exit 1; }
      exit 0
    fi
    [ -f "$M/mlxdev/$t/panel.jsonl" ] || { log "$POINT: no MLX-DEV-M15 panel for tier $t"; exit 1; }
    lease=/data/dev2/leases/gpu$GPU.lock/owner
    printf 'track=dec-m15\nstatus=busy\npurpose=decoder M15 MLX-DEV readout (%s)\nstart_utc=%s\nexpected_end_utc=%s\n' \
      "$POINT" "$(date -u +%FT%TZ)" "$(date -u -d '+60 min' +%FT%TZ)" > "$lease"
    ck=$(incontainer "$CK") || exit 2
    source=$(incontainer "$SOURCE") || exit 2
    mkdir -p "$L/$POINT"
    if M15_NODE=$NODE M15_CACHE=$t-read bash "$OPS/m15-launch.sh" "lines-$POINT-mlxdev" "$SRC" "$out" --gpu "$GPU" -- \
      -m v2.dec.eval_rows --checkpoint "$ck" --source-path "$source" --rows "/runs/m15/mlxdev/$t/panel.jsonl" \
      --tag mlxdev --output /out; then
      log "$POINT MLX-DEV read: $(tail -1 "$out.stdout.log" | cut -c1-200)"
    else
      log "$POINT MLX-DEV FAILED: $(tail -c 300 "$out.stderr.log" | tr '\n' ' ')"
      exit 1
    fi
    printf 'track=dec-m15\nstatus=idle\npurpose=decoder M15 (MLX-DEV of %s finished)\nstart_utc=%s\nexpected_end_utc=%s\n' \
      "$POINT" "$(date -u +%FT%TZ)" "$(date -u -d '+30 min' +%FT%TZ)" > "$lease"
    ;;
  mlxcmp)
    POINT=$3 REF=$4 t=${3%%-*}
    out=$L/$POINT/mlxcmp
    [ -f "$out/$POINT.mlxdev.json" ] && exit 0
    [ ! -e "$out.launch.json" ] || { log "$POINT MLX-DEV compare failed earlier; not rerun"; exit 1; }
    for p in "$POINT" "$REF"; do
      [ -f "$L/$p/mlxdev/mlxdev-predictions.jsonl" ] || { log "$POINT MLX-DEV compare: no readout of $p"; exit 1; }
    done
    idx=/runs/m15/mlxdev/$t/panel.jsonl.index.jsonl
    M15_NODE=$NODE bash "$OPS/m15-launch.sh" "mlxcmp-$POINT" "$SRC" "$out" --cpu -- -c "
import subprocess, sys
for args in (['score', '--predictions', '/runs/m15/lines/$POINT/mlxdev/mlxdev-predictions.jsonl',
              '--output', '/out/$POINT.mlxscore.json'],
             ['compare', '--a', '/runs/m15/lines/$REF/mlxdev/mlxdev-predictions.jsonl',
              '--b', '/runs/m15/lines/$POINT/mlxdev/mlxdev-predictions.jsonl', '--output', '/out/$POINT.mlxdev.json']):
    subprocess.run([sys.executable, '-m', 'v2.dec.mlx_dev', args[0], '--index', '$idx', *args[1:]], check=True)
" || { log "$POINT MLX-DEV compare FAILED (see $out.stderr.log)"; exit 1; }
    log "$POINT MLX-DEV vs $REF: $(tail -1 "$out.stdout.log" | cut -c1-300)"
    ;;
  parity)
    POINT=$3 REF=$4
    shift 4
    PANELS=${*:-dev css-pilot ht-dev2 score5t-dev hs1-dev}
    args=()
    for panel in $PANELS; do
      args+=(--pair "$panel=$L/$POINT/$panel/$panel.predictions.jsonl,$REF/$panel/$panel.predictions.jsonl")
    done
    python3 "$CODE/v2/dec/ops/m10/m10_compare.py" "${args[@]}" --output "$L/$POINT/parity-$(basename "$REF").json"
    ;;
  *) sed -n '2,16p' "$0"; exit 2 ;;
esac
