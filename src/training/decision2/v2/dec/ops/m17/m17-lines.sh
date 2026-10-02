#!/usr/bin/env bash
# Decoder M17 development readouts (prereg dec-m17-prereg-2026-10-02.md, "Development readouts"), node F, from an
# exact mirror. Every point is read the same way: v2.dec.infer_dec through m17-launch.sh, one container per panel,
# 16,384 tokens, T = 1, image dbe5f32b, the node's 4b-read Triton cache.
# Outputs: /data/dev2/runs/dec/m17/lines/<point>/<panel>/<panel>.predictions.jsonl (+ manifest and launch receipt). A
# panel already read is skipped; a failed read is recorded and not rerun. The caller holds the GPU's chain flock
# (m17-post.sh, m17-anchor.sh), so reads never share a GPU with training.
#
#   M17_NODE=f m17-lines.sh read <mirror-dir> <gpu> <point> <checkpoint> <source> [panel ...]
#       <checkpoint> / <source> are host paths under /data/dev2/runs/dec or /data/dev2/models.
#       Default panels: dev css-pilot ht-dev2 score5t-dev hs1-dev pn1-dev m10-probes ib-dev.
#   M17_NODE=f m17-lines.sh mlx <mirror-dir> <gpu> <point> <checkpoint> <source>
#       old MLX-DEV readout (report only): v2.dec.eval_rows (raw probabilities, max length 8,192, as SELECT700) on
#       M15's 4B panel (m17/mlxdev/4b/panel.jsonl) -> lines/<point>/mlxdev/mlxdev-predictions.jsonl; read once.
#   M17_NODE=f m17-lines.sh mlxcmp <mirror-dir> <point> <reference>
#       v2.dec.mlx_dev score of <point> and compare (B = <point>, A = <reference>; 2,000 draws, seed 20260929), CPU
#       -> lines/<point>/mlxcmp/<point>.mlxdev.json (+ <point>.mlxscore.json)
set -u
MODE=${1:-} SRC=${2:-}
NODE=${M17_NODE:?set M17_NODE=f}
M=/data/dev2/runs/dec/m17
L=$M/lines
CODE=/data/dev2/src/$SRC/src/training/decision2
OPS=$CODE/v2/dec/ops/m17
mkdir -p "$L"
log() { echo "$(date -u +%FT%TZ) lines-$NODE $*" | tee -a "$L/OPERATIONS.log"; }
incontainer() {
  case $1 in
    /data/dev2/runs/dec/*) echo "/runs/${1#/data/dev2/runs/dec/}" ;;
    /data/dev2/models/*) echo "/models/${1#/data/dev2/models/}" ;;
    *) echo "path outside the mounted roots: $1" >&2; return 1 ;;
  esac
}
lease() {  # <gpu> <status> <purpose> <minutes>
  mkdir -p "/data/dev2/leases/gpu$1.lock"
  printf 'track=dec-m17\nstatus=%s\npurpose=decoder M17 %s\nstart_utc=%s\nexpected_end_utc=%s\n' \
    "$2" "$3" "$(date -u +%FT%TZ)" "$(date -u -d "+$4 min" +%FT%TZ)" > "/data/dev2/leases/gpu$1.lock/owner"
}

case $MODE in
  read)
    GPU=$3 POINT=$4 CK=$5 SOURCE=$6
    shift 6
    PANELS=${*:-dev css-pilot ht-dev2 score5t-dev hs1-dev pn1-dev m10-probes ib-dev}
    if grep -qs "^status=busy" "/data/dev2/leases/gpu$GPU.lock/owner" && grep -qs "training" "/data/dev2/leases/gpu$GPU.lock/owner"; then
      log "$POINT: GPU$GPU is training; not read"
      exit 1
    fi
    lease "$GPU" busy "readouts ($POINT)" 60
    case $POINT in 4b-*) ;; *) log "$POINT: M17 point names start with 4b-"; exit 2 ;; esac
    ck=$(incontainer "$CK") || exit 2
    source=$(incontainer "$SOURCE") || exit 2
    for panel in $PANELS; do
      out=$L/$POINT/$panel
      if [ -f "$out.launch.json" ]; then
        grep -q '"exit_status": 0' "$out.launch.json" || log "$POINT $panel failed earlier; not rerun"
        continue
      fi
      [ -f "/data/dev2/runs/dec/panels/$panel.prompts.jsonl" ] || { log "$POINT $panel: no prompts"; continue; }
      if M17_NODE=$NODE M17_CACHE=4b-read bash "$OPS/m17-launch.sh" "lines-$POINT-$panel" "$SRC" "$out" --gpu "$GPU" -- \
        -m v2.dec.infer_dec --checkpoint "$ck" --source-path "$source" --input "/panels/$panel.prompts.jsonl" \
        --output "/out/$panel.predictions.jsonl" --model-id "decision2-dec-m17-$POINT" --model-revision "$POINT" \
        --max-length 16384; then
        log "$POINT $panel read: $(wc -l < "$out/$panel.predictions.jsonl") items"
      else
        log "$POINT $panel FAILED: $(tail -c 300 "$out.stderr.log" | tr '\n' ' ')"
      fi
    done
    lease "$GPU" idle "(readouts of $POINT finished)" 30
    ;;
  mlx)
    GPU=$3 POINT=$4 CK=$5 SOURCE=$6
    out=$L/$POINT/mlxdev
    if [ -f "$out.launch.json" ]; then
      grep -q '"exit_status": 0' "$out.launch.json" || { log "$POINT MLX-DEV failed earlier; not rerun"; exit 1; }
      exit 0
    fi
    [ -f "$M/mlxdev/4b/panel.jsonl" ] || { log "$POINT: no old MLX-DEV 4B panel"; exit 1; }
    lease "$GPU" busy "old MLX-DEV readout ($POINT)" 60
    ck=$(incontainer "$CK") || exit 2
    source=$(incontainer "$SOURCE") || exit 2
    mkdir -p "$L/$POINT"
    if M17_NODE=$NODE M17_CACHE=4b-read bash "$OPS/m17-launch.sh" "lines-$POINT-mlxdev" "$SRC" "$out" --gpu "$GPU" -- \
      -m v2.dec.eval_rows --checkpoint "$ck" --source-path "$source" --rows "/runs/m17/mlxdev/4b/panel.jsonl" \
      --tag mlxdev --output /out; then
      log "$POINT MLX-DEV read: $(tail -1 "$out.stdout.log" | cut -c1-200)"
    else
      log "$POINT MLX-DEV FAILED: $(tail -c 300 "$out.stderr.log" | tr '\n' ' ')"
      exit 1
    fi
    lease "$GPU" idle "(old MLX-DEV of $POINT finished)" 30
    ;;
  mlxcmp)
    POINT=$3 REF=$4
    out=$L/$POINT/mlxcmp
    [ -f "$out/$POINT.mlxdev.json" ] && exit 0
    [ ! -e "$out.launch.json" ] || { log "$POINT MLX-DEV compare failed earlier; not rerun"; exit 1; }
    for p in "$POINT" "$REF"; do
      [ -f "$L/$p/mlxdev/mlxdev-predictions.jsonl" ] || { log "$POINT MLX-DEV compare: no readout of $p"; exit 1; }
    done
    idx=/runs/m17/mlxdev/4b/panel.jsonl.index.jsonl
    M17_NODE=$NODE bash "$OPS/m17-launch.sh" "mlxcmp-$POINT" "$SRC" "$out" --cpu -- -c "
import subprocess, sys
for args in (['score', '--predictions', '/runs/m17/lines/$POINT/mlxdev/mlxdev-predictions.jsonl',
              '--output', '/out/$POINT.mlxscore.json'],
             ['compare', '--a', '/runs/m17/lines/$REF/mlxdev/mlxdev-predictions.jsonl',
              '--b', '/runs/m17/lines/$POINT/mlxdev/mlxdev-predictions.jsonl', '--output', '/out/$POINT.mlxdev.json']):
    subprocess.run([sys.executable, '-m', 'v2.dec.mlx_dev', args[0], '--index', '$idx', *args[1:]], check=True)
" || { log "$POINT MLX-DEV compare FAILED (see $out.stderr.log)"; exit 1; }
    log "$POINT MLX-DEV vs $REF: $(tail -1 "$out.stdout.log" | cut -c1-300)"
    ;;
  *) sed -n '2,19p' "$0"; exit 2 ;;
esac
