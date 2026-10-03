#!/usr/bin/env bash
# Decoder M17 MLX-DEV2 reads on node F (prereg "Development readouts and gates", MLX-DEV2), from an exact mirror.
#
#   M17_NODE=f m17-mlx2.sh ref <mirror-dir> <gpu>
#       LH's own IX1 package (DEV2.0-4B 13d42143, copied from node C; tree digest checked) read with a fresh Triton
#       cache, which is then frozen (private mlx2/cache-frozen).
#   M17_NODE=f m17-mlx2.sh point <mirror-dir> <gpu> <point> <checkpoint-dir> <model_sha256>
#       <checkpoint-dir> (merged FP32 weights) restaged onto LH's package (v2.eval.ix1.restage) under
#       /data/dev2/models/ix1/dec-m17/, then read with a copy of the frozen cache.
#
# Reads go through v2/eval/mlx_dev2_run.sh (frozen image decision20-train-fast:host2, the GPU's render node only,
# --network none, the gold-free prompts 35747a26...). Its probe.sh accepts only an eval-ix1 lease owner and removes the
# owner file on exit, so the M17 owner file is handed to that form for the read (purpose naming dec-m17) and restored
# after. Outputs (private, gold-free): /data/dev2/private/dec/m17/mlx2/<name>/mlx-dev2.predictions.jsonl; scored on
# node A (m17-score.sh mlx2). Receipts m17/mlx2/<name>.launch.json count GPU-h; markers m17/mlx2/<name>.{DONE,FAILED}.
# A failed restage or read stops that point and is never rerun.
set -u
MODE=${1:-} SRC=${2:-} GPU=${3:-}
NODE=${M17_NODE:?set M17_NODE=f}
[ "$NODE" = f ] || { echo "MLX-DEV2 reads run on node F" >&2; exit 2; }
[[ " 2 3 6 7 " == *" $GPU "* ]] || { echo "GPU $GPU on node F is not an M17 GPU" >&2; exit 2; }
M=/data/dev2/runs/dec/m17
S=/data/dev2/src/$SRC/src/training/decision2
P=/data/dev2/private/dec/m17/mlx2
PKG=/data/dev2/models/ix1/DEV2.0-4B-13d42143 PKG_TREE=29bc37d40e4044d886e2b229541c93098f0d53de8c471af0d825f3e35958e37e
PANEL=/data/dev2/private/panels/mlx-dev2-v1-goldfree
PROMPTS_SHA=35747a2679c86cac6577ed2e254f6fe9fafb673ef1af6fb5ff29589d2f9d8b73
FROZEN=$P/cache-frozen
mkdir -p "$M/mlx2" "$P"
chmod 700 /data/dev2/private/dec/m17 "$P" 2> /dev/null || true
log() { echo "$(date -u +%FT%TZ) mlx2 $*" | tee -a "$M/OPERATIONS.log"; }
tree() { (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1); }
[ "$(sha256sum "$PANEL/prompts.jsonl" | cut -d' ' -f1)" = "$PROMPTS_SHA" ] || { log "prompts are not $PROMPTS_SHA"; exit 1; }

read_pkg() {  # <name> <package> [<cache>]: one read with the lease handed to the harness form
  local name=$1 pkg=$2 cache=${3:-} owner=/data/dev2/leases/gpu$GPU.lock/owner saved start end rc extra=()
  [ -n "$cache" ] && extra=(--cache "$cache")
  [ -f "$M/mlx2/$name.DONE" ] && { log "$name already read"; return 0; }
  [ -f "$M/mlx2/$name.FAILED" ] && { log "$name failed earlier; not rerun"; return 1; }
  mkdir -p "$(dirname "$owner")"
  saved=$(cat "$owner" 2> /dev/null || true)
  printf 'track=eval-ix1\npurpose=dec-m17 MLX-DEV2 read %s (decoder M17 lease in the harness form)\nstart_utc=%s\n' \
    "$name" "$(date -u +%FT%TZ)" > "$owner"
  start=$(date -u +%FT%TZ)
  bash "$S/v2/eval/mlx_dev2_run.sh" --gpu "$GPU" --package "$pkg" --work "$P/$name" --panel "$PANEL" \
    --name "m17-mlx2-$name" "${extra[@]}" > "$M/mlx2/$name.log" 2>&1
  rc=$?
  end=$(date -u +%FT%TZ)
  if [ -n "$saved" ]; then printf '%s\n' "$saved" > "$owner"; else
    printf 'track=dec-m17\nstatus=idle\npurpose=decoder M17 (MLX-DEV2 read %s finished)\nstart_utc=%s\n' "$name" "$end" > "$owner"
  fi
  python3 - "$M/mlx2/$name.launch.json" "m17-mlx2-$name" "$start" "$end" "$rc" "node F GPU$GPU" "$pkg" "$P/$name" "$SRC" << 'EOF'
import json, sys
path, job, start, end, rc, gpu, pkg, work, src = sys.argv[1:]
json.dump({"job": job, "start_utc": start, "end_utc": end, "exit_status": int(rc), "gpu": gpu,
           "image": "decision20-train-fast:host2", "package": pkg, "work": work, "mirror": src}, open(path, "x"), indent=2)
EOF
  if [ $rc = 0 ] && [ -f "$P/$name/mlx-dev2.predictions.jsonl" ]; then
    echo "$(date -u +%FT%TZ) $(wc -l < "$P/$name/mlx-dev2.predictions.jsonl")" > "$M/mlx2/$name.DONE"
    log "$name read: $(wc -l < "$P/$name/mlx-dev2.predictions.jsonl") predictions"
    return 0
  fi
  echo "exit $rc" > "$M/mlx2/$name.FAILED"
  log "$name FAILED (exit $rc; see $M/mlx2/$name.log)"
  return 1
}

case $MODE in
  ref)
    if [ ! -f "$M/mlx2/4b-LH-f.DONE" ]; then
      [ "$(tree "$PKG")" = "$PKG_TREE" ] || { log "LH package tree is not $PKG_TREE"; exit 1; }
      read_pkg 4b-LH-f "$PKG" || exit 1
    fi
    if [ ! -d "$FROZEN" ]; then
      cp -a "$P/4b-LH-f/triton" "$FROZEN.tmp" && mv -T "$FROZEN.tmp" "$FROZEN"
      tree "$FROZEN" > "$M/mlx2/cache-frozen.tree"
      log "Triton cache frozen from the LH read ($(find "$FROZEN" -type f | wc -l) files, $(cut -c1-16 "$M/mlx2/cache-frozen.tree"))"
    fi
    ;;
  point)
    POINT=$4 CK=$5 SHA=$6
    [ -d "$FROZEN" ] || { log "$POINT: no frozen cache (the LH reference read comes first)"; exit 1; }
    [ "$(tree "$FROZEN")" = "$(cat "$M/mlx2/cache-frozen.tree")" ] || { log "$POINT: frozen cache changed"; exit 1; }
    DST=/data/dev2/models/ix1/dec-m17/DEV2.0-4B-$POINT-${SHA:0:8}-r13d42143
    if [ ! -f "$DST/MODEL_MANIFEST.json" ]; then
      [ -f "$M/mlx2/$POINT.restage.FAILED" ] && { log "$POINT restage failed earlier; not rerun"; exit 1; }
      mkdir -p "$(dirname "$DST")"
      if (cd "$S" && PYTHONPATH=$S python3 -B -m v2.eval.ix1.restage --package "$PKG" --out "$DST" --checkpoint "$CK" \
        --model-sha256 "$SHA") > "$M/mlx2/$POINT.restage.log" 2>&1; then
        log "$POINT restaged onto LH's package: $DST"
      else
        echo failed > "$M/mlx2/$POINT.restage.FAILED"
        log "$POINT restage FAILED (see $M/mlx2/$POINT.restage.log)"
        exit 1
      fi
    fi
    read_pkg "$POINT" "$DST" "$FROZEN" || exit 1
    ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
