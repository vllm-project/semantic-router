#!/usr/bin/env bash
# shellcheck disable=SC2034  # constants used by m6-formal.sh / m6-score.sh, which source this file
# Decoder M6 formal helpers (sourced; prereg dec-m6-prereg-2026-09-29.md, "Formal runs"). The M5 procedure
# (ops/m5/m5-formal-lib.sh) parameterized by tier and node: collection with the frozen runner
# v2/eval/run_same_panel.sh (gold never mounted) at 16,384 tokens, every run on its own cp -a copy of a master
# autotune cache whose receipt records new cache entries, gold-free seal, header stubs for the node-A parameter
# count, receipts (ops/m5/m5-receipt.py + M6-RECEIPT.json with tier / node / point) and GPU-seconds.
#
# The caller sets S (decision2 dir of the exact mirror), SRC (mirror dir name) and TIER (4b|2b|08b), then calls
# tier_setup. Node: 4b / 2b on node B, 08b on node A; M6_2B_NODE=A selects the 2B node-A fallback (prereg: used
# only if the node-B S2T reference does not reproduce the node-A S2T answers exactly).
F=${M6_FORMAL_ROOT:-/data/dev2/runs/dec/formal/m6}
R=/data/dev2/runs/dec
M=$R/m6
H=/data/dev2/hf-cache
SEL=${M6_SELECT:-$M/select}
PFX=${M6_PREFIX:-m6}
LEASES=/data/dev2/leases
PANELS=/data/dev2/private/panels
CAL698_DIR=$R/m3/data-sel700-cal698
CAL698_SHA=19cc1a8c4ebe6fd13031079f6b7d131046ce503f435f894e3ea672f91c2ed41f
IMAGE_B=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
IMAGE_A=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
NOX=$H/models--llm-semantic-router--Decision-1.0-Nox-4B/snapshots/cde2a68dbaa557ea65dc458104d410a0802ee259
SOL=$H/models--llm-semantic-router--Decision-1.0-Sol-2B/snapshots/ce0c018a28de16d6639b1cd203b761bf643b89e6
EOS=$H/models--llm-semantic-router--Decision-1.0-Eos-0.8B/snapshots/363c4a5e56afc115b1c78c837633956d0bbb63ab
SPEC_CAL=$S/v2/dec/adapter-spec-infer-dec.json
SPEC_T1=$S/v2/dec/ops/m5/m5-adapter-infer-dec-t1.json
M5OPS=$S/v2/dec/ops/m5
MIN_FREE_GB=${M6_MIN_FREE_GB:-60}
# 2B node-B S2T reference (prereg "Formal runs", 2B): the released staging folder, collected on a fresh cache.
REF2_STAGE=$R/m3/hf-staging/S2T-soup
REF2_LIST=$R/m3/hf-staging/S2T-soup.sha256
REF2_LIST_SHA=22e7fe86fba8b925d4930e2691c046a10235dc30ae45b9933560d8e81c9a47dd
REF2_RUN=m6-ref-S2T-soup
REF2_PKG=$F/pkg/S2T-soup-ref
REF2_MODEL=$REF2_PKG/m3/S2T-soup/checkpoint
REF2_CAL=$REF2_PKG/m3/S2T-soup/cal698-16k/calibration.json
REF2_CACHE=$F/cache-ref-2b
export TMPDIR=/data/dev2/tmp
mkdir -p "$F/stage-cal" "$F/stage-params" "$TMPDIR"

flog() { echo "$(date -u +%FT%TZ) $TIER $*" >> "$F/OPERATIONS.log"; }
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
vram_free_gb() {
  rocm-smi -d "$1" --showmeminfo vram | awk -F': ' '/Total Memory/ {t=$NF} /Total Used Memory/ {u=$NF} END {printf "%d\n", (t-u)/1e9}'
}

# tier_setup: node, image, source, report tier, master caches, GPU and runner environment for $TIER.
tier_setup() {
  ENVX=() ISOLATE=()
  case $TIER in
    4b)
      NODE=B IMAGE=$IMAGE_B SOURCE=$NOX TLABEL=4B INCUMBENT=N4XF-soup
      MASTER=$R/formal/m5/cache-frozen MASTER_SHA=f6d0f9207e436aa0090f51d1d4163e9f58edeb067ac28b894fb5e58612ea11d8
      MASTER_MLX=$R/formal/m5/cache-frozen-mlx MASTER_MLX_SHA=65d7d38f267bb168b18b71dd0c1b80b8b3e1aa0bbaca141d8d03b3fe7223c569
      # M6_4B_NODE=E|F (decoder M10): the same image and copies of node B's frozen masters on node E / F; every
      # container gets only its GPU's render node (nodes C-F rule); Nox from the node's plain model directory.
      case ${M6_4B_NODE:-B} in
        E | F)
          NODE=$M6_4B_NODE H=/data/dev2/models ISOLATE=(--isolate)
          SOURCE=$H/Decision-1.0-Nox-4B/cde2a68dbaa557ea65dc458104d410a0802ee259 ;;
        B) ;;
        *) echo "M6_4B_NODE must be B, E or F" >&2; exit 2 ;;
      esac ;;
    2b)
      TLABEL=2B SOURCE=$SOL INCUMBENT=S2T-soup
      if [ "${M6_2B_NODE:-B}" = A ]; then
        NODE=A IMAGE=$IMAGE_A ENVX=(--env HIP_FORCE_DEV_KERNARG=1)
        MASTER=$R/formal/m3/m3-S2T-soup-nodeA-triton MASTER_SHA=live
        MASTER_MLX=$R/formal/m3/m3-S2T-soup-triton MASTER_MLX_SHA=live
      else
        NODE=B IMAGE=$IMAGE_B
        MASTER=$F/cache-frozen-2b MASTER_SHA=frozen
        MASTER_MLX=$F/cache-frozen-2b-mlx MASTER_MLX_SHA=frozen
      fi ;;
    08b)
      NODE=A IMAGE=$IMAGE_A SOURCE=$EOS TLABEL=0.8B INCUMBENT=E8F-soup ENVX=(--env HIP_FORCE_DEV_KERNARG=1)
      MASTER=$R/formal/m2/m2-E8F-soup-nodeA-triton MASTER_SHA=live
      MASTER_MLX=$R/formal/m2/m2-E8F-soup-nodeA-triton MASTER_MLX_SHA=live ;;
    *) echo "tier must be 4b, 2b or 08b" >&2; exit 2 ;;
  esac
  if [ "$NODE" = B ]; then
    GPU=${M6_GPU:-} RENDER_OF=([3]=/dev/dri/renderD153 [4]=/dev/dri/renderD161)
  elif [ "$NODE" = E ] || [ "$NODE" = F ]; then
    local g bdf allowed
    GPU=${M6_GPU:-}
    [ "$NODE" = E ] && allowed="0 1 2 3" || allowed="2 3 4 5 6 7"
    for g in $allowed; do
      bdf=$(amd-smi list 2>/dev/null | awk -v n="GPU: $g" '$0 ~ "^"n"$" {getline; print tolower($2)}')
      [ -n "$bdf" ] && RENDER_OF[$g]=$(readlink -f "/dev/dri/by-path/pci-$bdf-render")
    done
  else
    GPU=${M6_GPU:-5} RENDER_OF=([5]=/dev/dri/renderD169)
  fi
}
declare -A RENDER_OF

# gpu_check <purpose>: the co-tenant rule (>= 60 GB free VRAM on a decoder GPU of this node).
gpu_check() {
  [ -n "$GPU" ] || die "set M6_GPU (node B 3|4, node A 5)"
  [ -n "${RENDER_OF[$GPU]:-}" ] || die "GPU$GPU is not a node-$NODE decoder GPU"
  local free
  free=$(vram_free_gb "$GPU") || die "rocm-smi failed on GPU$GPU"
  [ "$free" -ge "$MIN_FREE_GB" ] || die "$1: GPU$GPU has $free GB free VRAM (< $MIN_FREE_GB); not started"
  flog "$1 on GPU$GPU ($free GB free)"
}

# gpu_seconds <kind runner|launch> <file> <run> <purpose>: one JSON line per GPU job in GPU-SECONDS.jsonl.
gpu_seconds() {
  python3 - "$@" >> "$F/GPU-SECONDS.jsonl" <<'EOF'
import datetime as dt, json, sys
kind, path, run, purpose = sys.argv[1:]
d = json.load(open(path))
if kind == "runner":
    out = {"start_utc": d.get("start_utc"), "end_utc": d.get("end_utc"), "gpu_seconds": d.get("wall_seconds"),
           "gpu": d.get("gpu"), "shared": d.get("shared", False), "exit_status": d.get("exit_code")}
else:
    t = lambda s: dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
    out = {"start_utc": d["start_utc"], "end_utc": d["end_utc"],
           "gpu_seconds": (t(d["end_utc"]) - t(d["start_utc"])).total_seconds(), "gpu": d["gpu"],
           "shared": True, "exit_status": d["exit_status"]}
print(json.dumps({"run": run, "purpose": purpose, "source": path, **out}))
EOF
}

# lease_entry <name> <purpose> / lease_clear <name>: co-tenant entry for launch.sh jobs (the runner writes its own).
lease_entry() {
  local e=$LEASES/gpu$GPU.lock/owner.$1
  [ ! -e "$e" ] || die "$e exists"
  mkdir -p "$(dirname "$e")"
  printf 'track=dec\nstatus=busy (co-tenant)\npurpose=decoder M6 %s\nstart_utc=%s\nexpected_end_utc=%s\n' "$2" \
    "$(date -u +%FT%TZ)" "$(date -u -d '+30 min' +%FT%TZ)" > "$e"
}
lease_clear() { rm -f "$LEASES/gpu$GPU.lock/owner.$1"; }

# cache_copy <master> <frozen|live|sha256> <copy>: cp -a copy of a master autotune cache. A frozen master
# (<master>.sha256 manifest) must still match its manifest (and the pinned hash if given); a live persisted cache
# (node A) gets its manifest taken before the copy and the copy must match it. Sets COPY_BEFORE (manifest file).
cache_copy() {
  local master=$1 pin=$2 copy=$3
  [ -d "$master" ] || die "no master cache $master"
  [ ! -e "$copy" ] || die "cache copy $copy exists"
  if [ "$pin" = live ]; then
    tree_manifest "$master" > "$copy.master.sha256"
    cp -a "$master" "$copy" || die "cache copy FAILED"
    [ "$(tree_manifest "$copy" | sha256sum | cut -d' ' -f1)" = "$(sha "$copy.master.sha256")" ] \
      || die "cache copy of $master differs from the master (changed during the copy?)"
    COPY_BEFORE=$copy.master.sha256
  else
    [ -f "$master.sha256" ] || die "no frozen cache manifest $master.sha256"
    [ "$pin" = frozen ] || [ "$(sha "$master.sha256")" = "$pin" ] || die "frozen manifest of $master is not $pin"
    [ "$(tree_manifest "$master" | sha256sum | cut -d' ' -f1)" = "$(sha "$master.sha256")" ] || die "frozen master $master changed"
    cp -a "$master" "$copy" || die "cache copy FAILED"
    COPY_BEFORE=$master.sha256
  fi
  chmod -R u+w "$copy"
  flog "cache copy $copy of $master (manifest $(sha "$COPY_BEFORE"))"
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
# Always a co-tenant (--shared-lease dec-formal, or M6_SHARED_LEASE) after the VRAM check.
run_collect() {
  local run=$1 model=$2 pkg=$3 rev=$4 spec=$5 cache=$6 purpose=$7
  shift 7
  gpu_check "collect $run"
  bash "$S/v2/eval/run_same_panel.sh" --gpu "$GPU" --track dec --src "$SRC" --image "$IMAGE" --model-dir "$model" \
    "${ISOLATE[@]}" --mount "$H" --mount "$pkg" "${ENVX[@]}" --env TRITON_CACHE_AUTOTUNING=1 --env "TRITON_CACHE_DIR=$cache" \
    --mount-rw "$cache" --purpose "$purpose" --expected-end "$(date -u -d '+25 min' +%FT%TZ)" \
    --shared-lease "${M6_SHARED_LEASE:-dec-formal}" --run-dir "$F/$run" \
    -- --adapter-spec "$spec" --model-path "$model" --revision "$rev" --extra "source=$SOURCE" --extra max_length=16384 "$@" \
    > "$F/$run.collect.log" 2>&1
  local rc=$?
  [ -f "$F/$run/GPU-TIME.json" ] && gpu_seconds runner "$F/$run/GPU-TIME.json" "$run" "$purpose"
  return $rc
}

# finish <run> <kind> <name> <label> <package root> <package list> <revision> <model> <spec> <cache>
#        <cache-before manifest|none> <calibration used|none> <calibration decision|none> <params dir|-> <point|->
#        [collect args ...]
# Seal (gold-free; ref / finalist), parameter count + header stubs (from staging, or computed now), M5 receipt,
# then M6-RECEIPT.json (tier, node, point, slot, line, weights) beside it.
finish() {
  local run=$1 kind=$2 name=$3 label=$4 pkg=$5 list=$6 rev=$7 model=$8 spec=$9 cache=${10} before=${11} cal=${12}
  local dec=${13} params=${14} point=${15}
  shift 15
  local RD=$F/$run cargs=() a
  for a in "$@"; do cargs+=("--collect-arg=$a"); done
  if [ "$kind" = ref ] || [ "$kind" = finalist ]; then
    py -m v2.eval.same_panel seal --run-dir "$RD" > "$RD.seal.log" 2>&1 || die "$run seal FAILED"
  fi
  if [ "$params" != - ]; then
    cp -a "$params/pkg-headers" "$RD/pkg-headers" || die "$run parameter stubs copy FAILED"
    cp "$params/PARAMS.json" "$RD/PARAMS.json" || die "$run parameter count copy FAILED"
  else
    py "$M5OPS/m5-params.py" --package "$model" --stubs "$RD/pkg-headers" --output "$RD/PARAMS.json" \
      > "$RD.params.log" 2>&1 || die "$run parameter count FAILED"
  fi
  cp "$list" "$RD/PACKAGE.sha256"
  py "$M5OPS/m5-receipt.py" --run-dir "$RD" --kind "$kind" --name "$name" --label "$label" \
    --package-dir "$pkg" --package-list "$list" --revision "$rev" --model "$model" --spec "$spec" \
    --calibration-used "$cal" --calibration-decision "$dec" --cache-dir "$cache" --cache-before "$before" \
    --cache-after-manifest "$RD.cache-after.sha256" --mirror "$SRC" --image "$IMAGE" "${cargs[@]}" \
    > "$RD.receipt.log" 2>&1 || die "$run receipt FAILED"
  python3 - "$RD" "$TIER" "$TLABEL" "$NODE" "$point" "$SEL/$TIER-finalists.json" "$pkg" <<'EOF' || die "$run M6 receipt FAILED"
import hashlib, json, sys
from pathlib import Path
rd, tier, tlabel, node, point, finalists, pkg = sys.argv[1:]
rd = Path(rd)
m5 = rd / "M5-RECEIPT.json"
base = json.loads(m5.read_text())
slot = None
if point != "-" and Path(finalists).is_file():
    for f in json.loads(Path(finalists).read_text())["finalists"]:
        if f["point"] == point:
            slot = {k: f.get(k) for k in ("slot", "line", "step", "T", "G", "H3", "proxy", "effective_weights")}
weights = next(Path(pkg).glob("m6/*/weights.json"), None)
out = {
    "schema": "dec-m6-formal-receipt/1",
    "tier": tier, "tier_label": tlabel, "node": node, "point": None if point == "-" else point,
    "selection": slot,
    "weights_json": str(weights) if weights else None,
    "weights_json_sha256": hashlib.sha256(weights.read_bytes()).hexdigest() if weights else None,
    "m5_receipt_sha256": hashlib.sha256(m5.read_bytes()).hexdigest(),
    "m5_receipt_note": "M5-RECEIPT.json comes from ops/m5/m5-receipt.py, whose node field is fixed to B; node here is authoritative",
    **{k: base[k] for k in ("kind", "name", "label", "revision", "revision_matches_list", "package_dir", "model", "spec",
                            "calibration_used", "calibration_rule", "collect_args", "image_id", "gpu", "wall_seconds",
                            "gpu_hours", "seal_sha256", "parameters", "cache")},
}
with open(rd / "M6-RECEIPT.json", "x") as f:
    json.dump(out, f, indent=1, sort_keys=True)
    f.write("\n")
EOF
  flog "$run done: $(tail -1 "$RD.receipt.log")"
}
