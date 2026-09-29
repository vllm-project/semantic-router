#!/usr/bin/env bash
# shellcheck disable=SC2034  # constants used by m5-formal.sh, which sources this file
# Decoder M5 formal helpers for node B (sourced by m5-formal.sh; prereg dec-m5-prereg-2026-09-29.md, "Formal
# runs"). Collection uses the frozen runner v2/eval/run_same_panel.sh (gold never mounted) with the node-B image,
# 16,384 tokens and the frozen-cache procedure: the reference fills a fresh autotune cache, which is then frozen
# (master copy + SHA-256 manifest); every later run gets its own cp -a copy of the master and its receipt records
# whether new cache entries appeared.
F=${M5_FORMAL_ROOT:-/data/dev2/runs/dec/formal/m5}
M=/data/dev2/runs/dec/m5
H=/data/dev2/hf-cache
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
NOX=$H/models--llm-semantic-router--Decision-1.0-Nox-4B/snapshots/cde2a68dbaa557ea65dc458104d410a0802ee259
REF_STAGE=/data/dev2/runs/dec/m4/hf-staging/N4XF-soup
REF_LIST=/data/dev2/runs/dec/m4/hf-staging/N4XF-soup.sha256
REF_LIST_SHA=cea4cc9a2a5997e02ebb55ed3cb653b348f2067855a51b13bbbba7f95da984f3
REF_RUN=m5-ref-N4XF-soup
REF_PKG=$F/pkg/N4XF-soup-ref
REF_MODEL=$REF_PKG/m4/N4XF-soup/checkpoint
REF_CAL=$REF_PKG/m4/N4XF-soup/cal698-16k/calibration.json
REF_CACHE=$F/cache-ref
MASTER=$F/cache-frozen
MASTER_MLX=$F/cache-frozen-mlx
SPEC_CAL=$S/v2/dec/adapter-spec-infer-dec.json
SPEC_T1=$S/v2/dec/ops/m5/m5-adapter-infer-dec-t1.json
GPU=${M5_GPU:-3}
declare -A RENDER=([3]=/dev/dri/renderD153 [4]=/dev/dri/renderD161)
export TMPDIR=/data/dev2/tmp
mkdir -p "$F" "$TMPDIR"

flog() { echo "$(date -u +%FT%TZ) $*" >> "$F/OPERATIONS.log"; }
die() { flog "$*"; echo "$*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
tree_manifest() { (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum); }
# jget <json file> <key> [<key> ...]: print one value (lists / objects as JSON).
jget() {
  python3 - "$@" <<'EOF'
import json, sys
v = json.load(open(sys.argv[1]))
for k in sys.argv[2:]:
    v = v[k]
print(json.dumps(v) if isinstance(v, (list, dict)) else v)
EOF
}
py() { (cd "$S" && PYTHONPATH=$S python3 -B "$@"); }

# Reference package: hard-link copy of the node-B N4XF staging folder, verified file by file against its staged
# list, and its own list (same find | sort | sha256sum as m4-stage.sh) must hash to the staged list's SHA-256.
verify_ref_package() {
  [ "$(sha "$REF_LIST")" = "$REF_LIST_SHA" ] || die "N4XF staged list hash differs from $REF_LIST_SHA"
  if [ ! -d "$REF_PKG" ]; then
    mkdir -p "$F/pkg"
    cp -al "$REF_STAGE" "$REF_PKG" || die "reference package copy FAILED"
  fi
  (cd "$REF_PKG" && sha256sum -c --quiet "$REF_LIST") > "$F/pkg/N4XF-soup-ref.verify.log" 2>&1 \
    || die "reference package hash verification FAILED"
  tree_manifest "$REF_PKG" > "$F/pkg/N4XF-soup-ref.sha256"
  [ "$(sha "$F/pkg/N4XF-soup-ref.sha256")" = "$REF_LIST_SHA" ] || die "reference package has extra or missing files"
  flog "reference package verified: $(wc -l < "$REF_LIST") files, list $REF_LIST_SHA"
}

# vram_free_gb <gpu>
vram_free_gb() {
  rocm-smi -d "$1" --showmeminfo vram | awk -F': ' '/Total Memory/ {t=$NF} /Total Used Memory/ {u=$NF} END {printf "%d\n", (t-u)/1e9}'
}

# cache_copy <master> <master manifest> <copy>: verify the master against its manifest, then cp -a it.
cache_copy() {
  local master=$1 manifest=$2 copy=$3
  [ -f "$manifest" ] || die "no frozen cache manifest $manifest"
  [ ! -e "$copy" ] || die "cache copy $copy exists"
  [ "$(tree_manifest "$master" | sha256sum | cut -d' ' -f1)" = "$(sha "$manifest")" ] || die "frozen master $master changed"
  cp -a "$master" "$copy" || die "cache copy FAILED"
  chmod -R u+w "$copy"
}

# cache_freeze <cache> <master>: read-only master copy of a finished run's cache plus <master>.sha256.
cache_freeze() {
  local cache=$1 master=$2
  [ ! -e "$master" ] || die "frozen master $master exists"
  cp -a "$cache" "$master" || die "cache freeze FAILED"
  tree_manifest "$master" > "$master.sha256"
  chmod -R a-w "$master"
  flog "cache frozen: $master ($(wc -l < "$master.sha256") files, manifest $(sha "$master.sha256"))"
}

# run_collect <run> <model> <package root> <revision> <spec> <cache> <purpose> [same_panel collect args ...]
run_collect() {
  local run=$1 model=$2 pkg=$3 rev=$4 spec=$5 cache=$6 purpose=$7
  shift 7
  local lease=()
  [ -n "${M5_SHARED_LEASE:-}" ] && lease=(--shared-lease "$M5_SHARED_LEASE")
  bash "$S/v2/eval/run_same_panel.sh" --gpu "$GPU" --track dec --src "$SRC" --image "$IMAGE" --model-dir "$model" \
    --mount "$H" --mount "$pkg" --env TRITON_CACHE_AUTOTUNING=1 --env "TRITON_CACHE_DIR=$cache" --mount-rw "$cache" \
    --purpose "$purpose" --expected-end "$(date -u -d '+20 min' +%FT%TZ)" "${lease[@]}" --run-dir "$F/$run" \
    -- --adapter-spec "$spec" --model-path "$model" --revision "$rev" --extra "source=$NOX" --extra max_length=16384 "$@" \
    > "$F/$run.collect.log" 2>&1
}

# finish <run> <kind> <name> <label> <package root> <package list> <revision> <model> <spec> <cache> <cache-before|none>
#        <calibration used|none> <calibration decision|none> [collect args ...]
# Seal (gold-free, formal panels only), parameter count + header stubs for node-A report, receipt.
finish() {
  local run=$1 kind=$2 name=$3 label=$4 pkg=$5 list=$6 rev=$7 model=$8 spec=$9 cache=${10} before=${11} cal=${12} dec=${13}
  shift 13
  local R=$F/$run cargs=()
  for a in "$@"; do cargs+=(--collect-arg "$a"); done
  if [ "$kind" = ref ] || [ "$kind" = finalist ]; then
    py -m v2.eval.same_panel seal --run-dir "$R" > "$R.seal.log" 2>&1 || die "$run seal FAILED"
  fi
  py "$S/v2/dec/ops/m5/m5-params.py" --package "$model" --stubs "$R/pkg-headers" --output "$R/PARAMS.json" \
    > "$R.params.log" 2>&1 || die "$run parameter count FAILED"
  cp "$list" "$R/PACKAGE.sha256"
  py "$S/v2/dec/ops/m5/m5-receipt.py" --run-dir "$R" --kind "$kind" --name "$name" --label "$label" \
    --package-dir "$pkg" --package-list "$list" --revision "$rev" --model "$model" --spec "$spec" \
    --calibration-used "$cal" --calibration-decision "$dec" --cache-dir "$cache" --cache-before "$before" \
    --cache-after-manifest "$R.cache-after.sha256" --mirror "$SRC" --image "$IMAGE" "${cargs[@]}" \
    > "$R.receipt.log" 2>&1 || die "$run receipt FAILED"
  flog "$run done: $(tail -1 "$R.receipt.log")"
}
