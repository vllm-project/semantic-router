#!/usr/bin/env bash
# Decoder M14 development readouts (prereg dec-m14-prereg-2026-10-01.md, "Development readouts"), node A / B, from an
# exact mirror. Every point is read the same way as M12 / M13: v2.dec.infer_dec through m14-launch.sh, one container
# per panel, 16,384 tokens, T = 1, image dbe5f32b, the node's <tier>-read Triton cache (tier = the point's prefix:
# 2b, 08b or 4b).
# Outputs: /data/dev2/runs/dec/m14/lines/<point>/<panel>/<panel>.predictions.jsonl (+ manifest and launch receipt). A
# panel already read is skipped; a failed read is recorded and not rerun. The caller holds the GPU's chain flock
# (m14-post.sh, m14-refs.sh), so reads never share a GPU with training.
#
#   M14_NODE=a|b m14-lines.sh read <mirror-dir> <gpu> <point> <checkpoint> <source> [panel ...]
#       <checkpoint> / <source> are host paths under /data/dev2/runs/dec or /data/dev2/hf-cache.
#       Default panels: dev css-pilot ht-dev2 score5t-dev hs1-dev pn1-dev m10-probes ib-dev.
#   M14_NODE=a|b m14-lines.sh parity <mirror-dir> <point> <reference-dir> [panel ...]
#       answer agreement and maximum probability drift of <point>'s readouts against stored predictions
#       (<reference-dir>/<panel>/<panel>.predictions.jsonl) -> lines/<point>/parity-<name>.json
set -u
MODE=${1:-} SRC=${2:-}
NODE=${M14_NODE:?set M14_NODE=a or b}
M=/data/dev2/runs/dec/m14
L=$M/lines
CODE=/data/dev2/src/$SRC/src/training/decision2
OPS=$CODE/v2/dec/ops/m14
mkdir -p "$L"
log() { echo "$(date -u +%FT%TZ) lines-$NODE $*" | tee -a "$L/OPERATIONS.log"; }
incontainer() {
  case $1 in
    /data/dev2/runs/dec/*) echo "/runs/${1#/data/dev2/runs/dec/}" ;;
    /data/dev2/hf-cache/*) echo "/hf/${1#/data/dev2/hf-cache/}" ;;
    *) echo "path outside the mounted roots: $1" >&2; return 1 ;;
  esac
}

case $MODE in
  read)
    GPU=$3 POINT=$4 CK=$5 SOURCE=$6
    shift 6
    PANELS=${*:-dev css-pilot ht-dev2 score5t-dev hs1-dev pn1-dev m10-probes ib-dev}
    lease=/data/dev2/leases/gpu$GPU.lock/owner
    if grep -qs "^status=busy" "$lease" && grep -qs "training" "$lease"; then
      log "$POINT: GPU$GPU is training; not read"
      exit 1
    fi
    mkdir -p "$(dirname "$lease")"
    printf 'track=dec-m14\nstatus=busy\npurpose=decoder M14 readouts (%s)\nstart_utc=%s\nexpected_end_utc=%s\n' \
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
      if M14_NODE=$NODE M14_CACHE=${POINT%%-*}-read bash "$OPS/m14-launch.sh" "lines-$POINT-$panel" "$SRC" "$out" --gpu "$GPU" -- \
        -m v2.dec.infer_dec --checkpoint "$ck" --source-path "$source" --input "/panels/$panel.prompts.jsonl" \
        --output "/out/$panel.predictions.jsonl" --model-id "decision2-dec-m14-$POINT" --model-revision "$POINT" \
        --max-length 16384; then
        log "$POINT $panel read: $(wc -l < "$out/$panel.predictions.jsonl") items"
      else
        log "$POINT $panel FAILED: $(tail -c 300 "$out.stderr.log" | tr '\n' ' ')"
      fi
    done
    printf 'track=dec-m14\nstatus=idle\npurpose=decoder M14 (readouts of %s finished)\nstart_utc=%s\nexpected_end_utc=%s\n' \
      "$POINT" "$(date -u +%FT%TZ)" "$(date -u -d '+30 min' +%FT%TZ)" > "$lease"
    ;;
  parity)
    POINT=$3 REF=$4
    shift 4
    PANELS=${*:-dev css-pilot ht-dev2 score5t-dev hs1-dev pn1-dev m10-probes ib-dev}
    args=()
    for panel in $PANELS; do
      args+=(--pair "$panel=$L/$POINT/$panel/$panel.predictions.jsonl,$REF/$panel/$panel.predictions.jsonl")
    done
    python3 "$CODE/v2/dec/ops/m10/m10_compare.py" "${args[@]}" --output "$L/$POINT/parity-$(basename "$REF").json"
    ;;
  *) sed -n '2,16p' "$0"; exit 2 ;;
esac
