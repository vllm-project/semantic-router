#!/usr/bin/env bash
# 9B M9 development readouts (prereg "Development readouts") on node A GPU6-7, from an exact mirror. Every point is
# read the same way: v2.dec.infer_dec through m9/launch.sh, one container per panel, 16,384 tokens, T = 1 (no
# calibration file), image f83b1d10, M9's node-A readout cache; MLX-DEV-9B through v2.dec.eval_rows (raw
# probabilities, as M7). Outputs: /data/dev2/runs/9b/m9/lines/<point>/<panel>/<panel>.predictions.jsonl (+ manifest
# and launch receipt; MLX-DEV-9B: lines/<point>/mlxdev/mlxdev-predictions.jsonl). A panel already read is skipped; a
# failed read is recorded and not rerun.
#
#   M9_NODE=a lines.sh read <mirror-dir> <gpu> <point> <checkpoint> <source> [panel ...]
#       <checkpoint> / <source>: host paths under /data/dev2/runs/9b or /data/dev2/models, or "lux" (the Lux 1.0
#       package the K recipe used). Default panels: dev css-pilot ht-dev2 score5t-dev hs1-dev pn1-dev m9-probes mlxdev.
#   M9_NODE=a lines.sh parity <mirror-dir> <point> <reference-prefix> [panel ...]
#       answer agreement and maximum probability drift of <point>'s readouts against stored predictions
#       <reference-prefix>-<panel>/<panel>.predictions.jsonl (M7 / M8 layout) -> lines/<point>/parity-<name>.json
set -u
MODE=${1:-} SRC=${2:-}
NODE=${M9_NODE:?set M9_NODE=a}
M=/data/dev2/runs/9b/m9
L=$M/lines
CODE=/data/dev2/src/$SRC/src/training/decision2
OPS=$CODE/v2/9b/lux9b/m9
mkdir -p "$L"
log() { echo "$(date -u +%FT%TZ) lines-$NODE $*" | tee -a "$L/OPERATIONS.log"; }
incontainer() {
  case $1 in
    lux) echo /lux ;;
    /data/dev2/runs/9b/*) echo "/runs/${1#/data/dev2/runs/9b/}" ;;
    /data/dev2/models/*) echo "/models/${1#/data/dev2/models/}" ;;
    *) echo "path outside the mounted roots: $1" >&2; return 1 ;;
  esac
}

case $MODE in
  read)
    GPU=$3 POINT=$4 CK=$5 SOURCE=$6
    shift 6
    PANELS=${*:-dev css-pilot ht-dev2 score5t-dev hs1-dev pn1-dev m9-probes mlxdev}
    lease=/data/dev2/leases/gpu$GPU.lock/owner
    if [ "$GPU" = 7 ] && [ -f /data/dev2/leases/gpu7.lock/owner.eval ] \
      && ! grep -qsE '^status=(idle|released)' /data/dev2/leases/gpu7.lock/owner.eval; then
      log "$POINT: GPU7's eval entry is active; not read"
      exit 1
    fi
    if [ -f "$lease" ] && ! grep -qs '^track=9b-m9' "$lease" && ! grep -qsE '^status=idle' "$lease"; then
      log "$POINT: GPU$GPU lease is held by another track; not read"
      exit 1
    fi
    mkdir -p "$(dirname "$lease")"
    [ -f "$lease" ] && ! grep -qs '^track=9b-m9' "$lease" && cp "$lease" "$(dirname "$lease")/owner.before-9b-m9"
    printf 'track=9b-m9\nstatus=busy\npurpose=9B M9 development readouts (%s)\nstart_utc=%s\nexpected_end_utc=%s\n' \
      "$POINT" "$(date -u +%FT%TZ)" "$(date -u -d '+60 min' +%FT%TZ)" > "$lease"
    ck=$(incontainer "$CK") || exit 2
    source=$(incontainer "$SOURCE") || exit 2
    for panel in $PANELS; do
      out=$L/$POINT/$panel
      if [ -f "$out.launch.json" ]; then
        grep -q '"exit_status": 0' "$out.launch.json" || log "$POINT $panel failed earlier; not rerun"
        continue
      fi
      if [ "$panel" = mlxdev ]; then
        if M9_NODE=$NODE bash "$OPS/launch.sh" "lines-$POINT-$panel" "$SRC" "$out" --gpu "$GPU" -- \
          -m v2.dec.eval_rows --checkpoint "$ck" --source-path "$source" \
          --rows /runs/m7/data/mlxdev/build/panel.jsonl --tag mlxdev --output /out --max-length 16384; then
          log "$POINT $panel read: $(wc -l < "$out/mlxdev-predictions.jsonl") rows"
        else
          log "$POINT $panel FAILED: $(tail -c 300 "$out.stderr.log" | tr '\n' ' ')"
        fi
        continue
      fi
      input=/panels/$panel.prompts.jsonl host=/data/dev2/runs/dec/panels/$panel.prompts.jsonl
      if [ "$panel" = m9-probes ]; then
        input=/runs/m9/panels/m9-probes.prompts.jsonl host=$M/panels/m9-probes.prompts.jsonl
      fi
      [ -f "$host" ] || { log "$POINT $panel: no prompts"; continue; }
      if M9_NODE=$NODE bash "$OPS/launch.sh" "lines-$POINT-$panel" "$SRC" "$out" --gpu "$GPU" -- \
        -m v2.dec.infer_dec --checkpoint "$ck" --source-path "$source" --input "$input" \
        --output "/out/$panel.predictions.jsonl" --model-id "decision2-9b-m9-$POINT" --model-revision "$POINT" \
        --max-length 16384; then
        log "$POINT $panel read: $(wc -l < "$out/$panel.predictions.jsonl") items"
      else
        log "$POINT $panel FAILED: $(tail -c 300 "$out.stderr.log" | tr '\n' ' ')"
      fi
    done
    printf 'track=9b-m9\nstatus=idle\npurpose=9B M9 (readouts of %s finished)\nstart_utc=%s\nexpected_end_utc=%s\n' \
      "$POINT" "$(date -u +%FT%TZ)" "$(date -u -d '+30 min' +%FT%TZ)" > "$lease"
    ;;
  parity)
    POINT=$3 REF=$4
    shift 4
    PANELS=${*:-dev css-pilot ht-dev2 pn1-dev}
    args=()
    for panel in $PANELS; do
      args+=(--pair "$panel=$L/$POINT/$panel/$panel.predictions.jsonl,$REF-$panel/$panel.predictions.jsonl")
    done
    python3 "$CODE/v2/dec/ops/m10/m10_compare.py" "${args[@]}" --output "$L/$POINT/parity-$(basename "$REF").json"
    ;;
  *) sed -n '2,16p' "$0"; exit 2 ;;
esac
