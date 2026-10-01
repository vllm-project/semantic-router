#!/usr/bin/env bash
# 27B MoE milestone: private Decision Index 0.2.1 run of a frozen MoE package with IX1's harness (kit 87d4650b, the
# IX1 panel builder, parity gate, merge and dual scorer), through the package's native path (v2.27b.moe.index_engine).
# Index values stay in the node private directories and never reach commits, gists, cards, COORDINATION or STATUS.
# Label: independent provisional 0.2.1 reproduction.
# Usage: moe-index.sh MODE MIRROR ARGS...   (MIRROR: the mirror directory name under /data/dev2/src)
#   offer  NAME               node B host: R/NAME's frozen package (PACKAGE.json, calibration.json, params.json and the
#                             checkpoint) -> X/index/NAME with SHA256SUMS, then X/index/NAME.OFFERED
#   stage  NAME               node A host: pull X/index/NAME over the temporary link -> R/index-stage/NAME; verify the
#                             SHA-256 list and the frozen calibration -> STAGED.json
#   panel  SHARDS             node A host (CPU): v2.eval.ix1.panel over the verified suite -> P/panel-SHARDS plus the
#                             gold-free compatibility sample; the IX1 row count and run-ID digest must match
#   parity NAME GPU [ROWS]    node A GPU (foreground): (1) the package's formal collector (v2.27b.moe.collect) over
#                             ROWS (default: the compatibility sample) with a fresh Triton cache, frozen to
#                             P/parity/NAME/cache-frozen; (2) the kit runner with the MoE engine over the same rows with
#                             a copy of it; (3) v2.eval.ix1.parity -> P/parity/NAME/parity.json
#   run    NAME SHARDS GPU... node A: one detached kit-runner job per listed GPU (shard k of SHARDS on the k-th GPU),
#                             each with a copy of the frozen cache; a shard starts once the previous one is ready
#   resume NAME SHARDS K GPU  restart ended shard K in place on GPU (the kit skips final rows and retries errors)
#   extra  NAME GPU ROWS TAG  the kit runner over ROWS alone -> P/runs/NAME/extra-TAG (foreground)
#   scan   NAME SHARDS        node A (CPU container): request sizes of P/panel-SHARDS under the package's own encoder
#                             (v2.27b.moe.index_scan; inputs only, nothing answered) -> P/scan/NAME-panel-SHARDS.json
#   presplit NAME SHARDS T    node A host: requests of >= T padded tokens are skipped in their shard and rerun alone
#                             after it (v2.27b.moe.index_skip presplit), so a request over one GPU's memory cannot
#                             halt a shard
#   auto-offer NAME           node B host (start detached): wait for the Stage B chain's frozen package, then offer;
#                             ends without an offer if the chain skips the package (X/mlx/NAME.SKIP) or after 8 h
#   auto-run NAME SHARDS GPU...  node A host (start detached): wait for X/index/NAME.OFFERED, then stage, scan (CPU,
#                             beside the parity gate), parity on the first GPU over the compatibility sample and, only
#                             on a pass, presplit at MOE_INDEX_MIN_PADDED (if set) and run on the listed GPUs
# R = MOE_ROOT (default /data/dev2/runs/27b-moe; a path check sets R/pathcheck), P = MOE_INDEX_ROOT (default
# /data/dev2/private/eval/index021/ix1), X = /data/dev2/xfer/27b-moe (node B). Every GPU job goes through
# v2.27b.moe.launch (the track's lease, render node, wall-clock cap and GPU-hour receipt): no network, the mirror, kit,
# package, base and rows mounted read-only, only the job's directory writable. Score with v2/eval/ix1/score.sh.
set -euo pipefail
MODE=${1:?MODE} MIR=${2:?MIRROR}
shift 2
S=/data/dev2/src/$MIR/src/training/decision2
[ -f "$S/v2/27b/moe/moe-index.sh" ] || { echo "missing mirror $MIR" >&2; exit 2; }
R=${MOE_ROOT:-/data/dev2/runs/27b-moe}
P=${MOE_INDEX_ROOT:-/data/dev2/private/eval/index021/ix1}
X=/data/dev2/xfer/27b-moe
I=/data/dev2/private/eval/index021
KIT=$I/kit-87d4650b
KIT_REVISION=87d4650b42b377c0291a89c1f1a879f9b31082bf
KEY=/data/dev2/tmp/27b-moe-xfer
BASE=/data/dev2/models/moe/gemma-4-26B-A4B-it
MODEL_ID=llm-semantic-router/DEV2.0-27B-MoE-candidate
ENGINE=v2.27b.moe.index_engine:MoEPackageIndexEngine
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
PANEL_ROWS=120226
PANEL_DIGEST=6455d7be7ce3e902a73f4d854884f817e20a0fa8ba74252a299261c4f1d6ad50
COMPAT_PREFIX=1356ceaf
CAP_HOURS=${MOE_INDEX_CAP_HOURS:-9}
cd "$S"
export PYTHONPATH=$S PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/dev2/tmp
umask 077
name_ok() { [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad NAME $1" >&2; exit 2; }; }
gpu_ok() { [[ "$1" =~ ^[345]$ ]] || { echo "node A GPU$1 is outside the MoE allocation (3, 4, 5)" >&2; exit 2; }; }
json() { python3 -c "import functools,json,sys; print(functools.reduce(lambda v, k: v[k], sys.argv[2:], json.load(open(sys.argv[1]))))" "$@"; }
rs() { rsync -a --mkpath -e "ssh -i $KEY/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=$KEY/known_hosts -o StrictHostKeyChecking=yes -o BatchMode=yes" "$@"; }
digest_dir() { (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -c1-64); }
kit_ok() { [ "$(git -C "$KIT" rev-parse HEAD)" = "$KIT_REVISION" ] || { echo "kit is not at $KIT_REVISION" >&2; exit 1; }; }
staged() {  # NAME -> the verified staged package directory
  local d=$R/index-stage/$1
  [ -f "$d/STAGED.json" ] || { echo "$1 is not staged on this node: $d" >&2; exit 2; }
  echo "$d"
}
job() {  # GPU NAME PURPOSE RECEIPT WORK PKG ROWS_DIR SCRIPT: one container through the track launcher (foreground)
  local gpu=$1 name=$2 purpose=$3 receipt=$4 work=$5 pkg=$6 rows_dir=$7 script=$8
  mkdir -p "$work/home" "$work/triton"
  DEV2_NODE=a python3 -m v2.27b.moe.launch --name "$name" --gpu "$gpu" --cap-hours "$CAP_HOURS" --purpose "$purpose" \
    --receipt "$receipt" --mount "$S:/code" --mount "$KIT:$KIT" --mount "$pkg:$pkg" --mount "$BASE:$BASE" \
    --mount "$rows_dir:$rows_dir" --mount "$work:$work:rw" --env "PYTHONPATH=/code:$KIT" --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 --env "HOME=$work/home" --env "TRITON_CACHE_DIR=$work/triton" \
    --env TRITON_CACHE_AUTOTUNING=1 --env HIP_FORCE_DEV_KERNARG=1 --env "DECISION2_MOE_PACKAGE_DIR=$pkg" \
    --env "DECISION2_BASE_DIR=$BASE" -- bash -c "$script"
}
kit_script() {  # ROWS WORK PKG_SHA -> the kit runner command, with start / end / exit markers for the merge
  # shellcheck disable=SC2016  # $? and $code expand inside the container
  printf 'date +%%s > %q/start_epoch; python3 -m decision_index run --engine %s --option model_id=%s --option package_sha256=%s --option device=cuda:0 --rows %q --out %q --compact > %q/runner.log 2>&1; code=$?; echo $code > %q/exit_code; date +%%s > %q/end_epoch; exit $code' \
    "$2" "$ENGINE" "$MODEL_ID" "$3" "$1" "$2" "$2" "$2" "$2"
}
pkg_sha() { sha256sum "$1/PACKAGE.json" | cut -c1-64; }

case "$MODE" in
  offer)
    NAME=${1:?NAME}; name_ok "$NAME"
    PK=$R/$NAME/package D=$X/index/$NAME
    [ -f "$PK/PACKAGE.json" ] || { echo "$NAME has no frozen package" >&2; exit 2; }
    [ ! -e "$X/index/$NAME.OFFERED" ] || { echo "$NAME is already offered" >&2; exit 66; }
    python3 - "$PK/PACKAGE.json" <<'EOF'
import hashlib, json, sys
package = json.load(open(sys.argv[1]))
path = package["calibration"]["path"]
if hashlib.sha256(open(path, "rb").read()).hexdigest() != package["calibration"]["sha256"]:
    raise SystemExit("package calibration changed after the freeze")
EOF
    CKPT=$(json "$PK/PACKAGE.json" checkpoint)
    rm -rf "$D" && mkdir -p "$D/checkpoint"
    cp -p "$PK/PACKAGE.json" "$PK/calibration.json" "$PK/params.json" "$D/"
    rsync -a --exclude trainer_state.pt "$CKPT/" "$D/checkpoint/"
    (cd "$D" && find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > "$D.SHA256SUMS"
    mv "$D.SHA256SUMS" "$D/SHA256SUMS"
    chmod -R go+rX "$D"
    date -u +%FT%TZ > "$X/index/$NAME.OFFERED"
    echo "offered $NAME: $(wc -l < "$D/SHA256SUMS") files, PACKAGE.json $(pkg_sha "$D")" ;;
  stage)
    NAME=${1:?NAME}; name_ok "$NAME"
    PEER=$(cat "$KEY/peer") D=$R/index-stage/$NAME
    [ ! -e "$D/STAGED.json" ] || { echo "$NAME is already staged: $D" >&2; exit 66; }
    rm -rf "$D" && mkdir -p "$D"
    rs "root@$PEER:index/$NAME.OFFERED" "$D.OFFERED"
    rs "root@$PEER:index/$NAME/" "$D/"
    (cd "$D" && sha256sum -c --quiet SHA256SUMS)
    python3 - "$D" "$(cat "$D.OFFERED")" <<'EOF'
import hashlib, json, pathlib, sys
from datetime import datetime, timezone
d, offered = pathlib.Path(sys.argv[1]), sys.argv[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
package = json.loads((d / "PACKAGE.json").read_text())
if sha(d / "calibration.json") != package["calibration"]["sha256"]:
    raise SystemExit("staged calibration differs from the frozen package")
config = json.loads((d / "checkpoint/decision_config.json").read_text())
if config["lora"]["base_revision"] != package["base"]["revision"]:
    raise SystemExit("staged checkpoint pins another base revision")
record = {"schema": "decision2-27b-moe-index-stage/1", "name": d.name, "offered_utc": offered,
          "staged_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
          "package_sha256": sha(d / "PACKAGE.json"), "model_sha256": package["model_sha256"],
          "files": len((d / "SHA256SUMS").read_text().splitlines()), "sha256sums_sha256": sha(d / "SHA256SUMS")}
(d / "STAGED.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
print(json.dumps(record, sort_keys=True))
EOF
    rm -f "$D.OFFERED"
    chmod -R go+rX "$D" ;;
  panel)
    SHARDS=${1:?SHARDS}
    [[ "$SHARDS" =~ ^[1-8]$ ]] || { echo "SHARDS must be 1..8" >&2; exit 2; }
    OUT=$P/panel-$SHARDS
    [ ! -e "$OUT/panel.json" ] || { echo "$OUT exists" >&2; exit 66; }
    [ "$(sha256sum < "$I/compat-86.jsonl.gz" | cut -c1-8)" = "$COMPAT_PREFIX" ] || { echo "compatibility sample changed" >&2; exit 2; }
    mkdir -p "$OUT"
    (cd "$I" && PYTHONPATH="$S:$I/kit-19ad28ec" venv/bin/python -m v2.eval.ix1.panel --suite-dir suite-0.2 \
      --shards "$SHARDS" --out "$OUT" --compat compat-86.jsonl.gz)
    python3 - "$OUT/panel.json" "$PANEL_ROWS" "$PANEL_DIGEST" <<'EOF'
import json, sys
panel, rows, digest = json.load(open(sys.argv[1])), int(sys.argv[2]), sys.argv[3]
if panel["rows"] != rows or panel["run_ids_sha256"] != digest or panel["compat"]["rows"] != 86:
    raise SystemExit(f"panel differs from IX1's: {panel['rows']} rows, digest {panel['run_ids_sha256']}")
print(json.dumps({"rows": panel["rows"], "run_ids_sha256": panel["run_ids_sha256"], "shards": len(panel["shards"])}))
EOF
    ;;
  parity)
    NAME=${1:?NAME} GPU=${2:?GPU} ROWS=${3:-}
    name_ok "$NAME"; gpu_ok "$GPU"; kit_ok
    PKG=$(staged "$NAME")
    SHA=$(pkg_sha "$PKG")
    if [ -z "$ROWS" ]; then
      ROWS=$(ls "$P"/panel-*/compat-86.gold-free.jsonl.gz | head -1)
    fi
    [ -f "$ROWS" ] || { echo "no parity rows" >&2; exit 2; }
    W=$P/parity/$NAME REF=$P/parity/$NAME/ref KW=$P/parity/$NAME/kit
    [ ! -e "$W/parity.json" ] && [ ! -e "$REF/predictions.jsonl" ] || { echo "$W already has a parity run" >&2; exit 66; }
    mkdir -p "$REF" "$KW" "$W/receipts"
    python3 -m v2.27b.moe.index_ref prompts --rows "$ROWS" --out "$REF/prompts.jsonl"
    REVISION="checkpoint-sha256:$(python3 - "$PKG/checkpoint" <<'EOF'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
files = sorted(p for p in root.rglob("*") if p.is_file() and p.name not in ("trainer_state.pt",))
rows = [[str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest()] for p in files]
print(hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest())
EOF
)"
    LIMIT=$(json "$PKG/PACKAGE.json" max_input_tokens)
    job "$GPU" "d2-27b-moe-ix-$NAME-ref-g$GPU" "27b-moe Index parity reference $NAME (formal collector)" \
      "$W/receipts/ref.json" "$REF" "$PKG" "$(dirname "$ROWS")" \
      "python3 -m v2.27b.moe.collect --checkpoint $PKG/checkpoint --source-path $BASE --model-id $MODEL_ID --model-revision $REVISION --max-length $LIMIT --calibration $PKG/calibration.json --input $REF/prompts.jsonl --output $REF/predictions.jsonl > $REF/collect.log 2>&1"
    python3 -m v2.27b.moe.index_ref convert --rows "$ROWS" --predictions "$REF/predictions.jsonl" --out "$REF/ref.jsonl"
    cp -a "$REF/triton" "$W/cache-frozen"
    digest_dir "$W/cache-frozen" > "$W/cache-frozen.sha256"
    mkdir -p "$KW/triton" && cp -a "$W/cache-frozen/." "$KW/triton/"
    job "$GPU" "d2-27b-moe-ix-$NAME-kit-g$GPU" "27b-moe Index parity kit pass $NAME" "$W/receipts/kit.json" "$KW" \
      "$PKG" "$(dirname "$ROWS")" "$(kit_script "$ROWS" "$KW" "$SHA")"
    python3 -m v2.eval.ix1.parity --kit "$KW/results.jsonl" --ref "$REF/ref.jsonl" --out "$W/parity.json" ;;
  run | resume)
    NAME=${1:?NAME} SHARDS=${2:?SHARDS}
    shift 2
    name_ok "$NAME"; kit_ok
    staged "$NAME" > /dev/null
    PANEL=$P/panel-$SHARDS CACHE=$P/parity/$NAME/cache-frozen
    [ -f "$PANEL/panel.json" ] || { echo "no $SHARDS-way panel" >&2; exit 2; }
    python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['pass'] else 1)" "$P/parity/$NAME/parity.json" \
      || { echo "$NAME has not passed the parity gate" >&2; exit 2; }
    [ "$(digest_dir "$CACHE")" = "$(cat "$CACHE.sha256")" ] || { echo "frozen cache $CACHE changed" >&2; exit 1; }
    gpus=()
    if [ "$MODE" = run ]; then
      [ $# -eq "$SHARDS" ] || { echo "run needs one GPU per shard" >&2; exit 2; }
      gpus=("$@")
      mapfile -t shards < <(seq 0 $((SHARDS - 1)))
    else
      shards=("${1:?K}")
      gpus[$1]=${2:?GPU}
    fi
    for k in "${shards[@]}"; do
      g=${gpus[$k]}
      gpu_ok "$g"
      W=$P/runs/$NAME/shard-$k suffix=""
      if [ "$MODE" = run ]; then
        [ ! -e "$W/results.jsonl" ] || { echo "$W already has results; use resume" >&2; exit 66; }
        mkdir -p "$W/triton" && cp -a "$CACHE/." "$W/triton/"
      else
        [ -f "$W/end_epoch" ] || { echo "$W has not ended" >&2; exit 2; }
        i=1
        while [ -e "$W/end_epoch.$i" ]; do i=$((i + 1)); done
        for f in start_epoch end_epoch exit_code status.json; do [ ! -e "$W/$f" ] || mv "$W/$f" "$W/$f.$i"; done
        suffix=.$i
      fi
      nohup bash "$S/v2/27b/moe/moe-index.sh" _shard "$MIR" "$NAME" "$SHARDS" "$k" "$g" "$suffix" \
        > "$W/launch$suffix.log" 2>&1 < /dev/null &
      echo "started $NAME shard $k of $SHARDS on GPU$g (pid $!)"
      for _ in $(seq 120); do
        grep -q '"event": *"\(ready\|progress\|complete\|failed\)"' "$W/status.json" 2> /dev/null && break
        [ -f "$W/end_epoch" ] && break
        sleep 15
      done
    done ;;
  _shard)  # NAME SHARDS K GPU SUFFIX: one shard in the foreground (started detached by run / resume)
    # A request that aborts the device is skipped (v2.27b.moe.index_skip: rows.override.jsonl.gz) and the shard
    # resumes, at most 4 attempts; then every skipped request is rerun alone (extra-sK-n), and one that fails again
    # is recorded as a final error. Any other failure stops the shard for a person.
    NAME=$1 SHARDS=$2 K=$3 GPU=$4 SUFFIX=${5:-}
    PKG=$(staged "$NAME")
    SHA=$(pkg_sha "$PKG")
    PANEL=$P/panel-$SHARDS W=$P/runs/$NAME/shard-$K
    SOURCE=$PANEL/shard-$K-of-$SHARDS.jsonl.gz
    for _ in 1 2 3 4; do
      ROWS=$SOURCE
      [ ! -f "$W/rows.override.jsonl.gz" ] || ROWS=$W/rows.override.jsonl.gz
      status=0
      job "$GPU" "d2-27b-moe-ix-$NAME-s$K-g$GPU${SUFFIX:+-r${SUFFIX#.}}" \
        "27b-moe Index 0.2.1 private run $NAME shard $K of $SHARDS" "$W/receipt$SUFFIX.json" "$W" "$PKG" "$PANEL" \
        "$(kit_script "$ROWS" "$W" "$SHA")" || status=$?
      [ "$status" != 0 ] || break
      python3 -m v2.27b.moe.index_skip skip --shard-dir "$W" --rows "$SOURCE" || exit "$status"
      i=1
      while [ -e "$W/end_epoch.$i" ]; do i=$((i + 1)); done
      for f in start_epoch end_epoch exit_code status.json; do [ ! -e "$W/$f" ] || mv "$W/$f" "$W/$f.$i"; done
      SUFFIX=.$i
    done
    [ "$status" = 0 ] || { echo "shard $K still failing after 4 attempts" >&2; exit "$status"; }
    for f in "$W"/extra-rows/*.jsonl.gz; do
      [ -e "$f" ] || continue
      tag=s$K-$(basename "$f" .jsonl.gz)
      [ ! -e "$P/runs/$NAME/extra-$tag/end_epoch" ] || continue
      bash "$0" extra "$MIR" "$NAME" "$GPU" "$f" "$tag" \
        || python3 -m v2.27b.moe.index_skip abort --extra-dir "$P/runs/$NAME/extra-$tag" --rows "$f"
    done ;;
  extra)
    NAME=${1:?NAME} GPU=${2:?GPU} ROWS=${3:?ROWS} TAG=${4:?TAG}
    name_ok "$NAME"; gpu_ok "$GPU"; kit_ok
    [[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "bad TAG" >&2; exit 2; }
    PKG=$(staged "$NAME")
    SHA=$(pkg_sha "$PKG") CACHE=$P/parity/$NAME/cache-frozen
    W=$P/runs/$NAME/extra-$TAG
    [ ! -e "$W/results.jsonl" ] || { echo "$W already has results" >&2; exit 66; }
    mkdir -p "$W/triton" && cp -a "$CACHE/." "$W/triton/"
    job "$GPU" "d2-27b-moe-ix-$NAME-extra-$TAG-g$GPU" "27b-moe Index extra $NAME $TAG" "$W/receipt.json" "$W" "$PKG" \
      "$(dirname "$ROWS")" "$(kit_script "$ROWS" "$W" "$SHA")" ;;
  scan)
    NAME=${1:?NAME} SHARDS=${2:?SHARDS}
    name_ok "$NAME"
    PKG=$(staged "$NAME")
    PANEL=$P/panel-$SHARDS OUT=$P/scan/$NAME-panel-$SHARDS.json
    [ -f "$PANEL/panel.json" ] || { echo "no $SHARDS-way panel" >&2; exit 2; }
    [ ! -e "$OUT" ] || { echo "$OUT exists" >&2; exit 66; }
    mkdir -p "$P/scan"
    docker run --rm --name "d2-27b-moe-ix-$NAME-scan" --network none --cpus 32 -e HIP_VISIBLE_DEVICES= \
      -e ROCR_VISIBLE_DEVICES= -e PYTHONPATH=/code -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 \
      -e TRANSFORMERS_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false \
      --mount "type=bind,src=$S,dst=/code,readonly" --mount "type=bind,src=$PKG,dst=$PKG,readonly" \
      --mount "type=bind,src=$PANEL,dst=$PANEL,readonly" --mount "type=bind,src=$P/scan,dst=$P/scan" \
      -w /code --entrypoint python3 "$IMAGE" -m v2.27b.moe.index_scan --checkpoint "$PKG/checkpoint" \
      --panel "$PANEL/panel.json" --out "$OUT" --max-length "$(json "$PKG/PACKAGE.json" max_input_tokens)" --workers 32 ;;
  presplit)
    NAME=${1:?NAME} SHARDS=${2:?SHARDS} MIN=${3:?MIN_PADDED_TOKENS}
    name_ok "$NAME"
    [[ "$MIN" =~ ^[0-9]+$ ]] || { echo "MIN_PADDED_TOKENS must be an integer" >&2; exit 2; }
    python3 -m v2.27b.moe.index_skip presplit --scan "$P/scan/$NAME-panel-$SHARDS.json" \
      --panel "$P/panel-$SHARDS/panel.json" --run "$P/runs/$NAME" --min-padded-tokens "$MIN" ;;
  auto-offer)
    NAME=${1:?NAME}; name_ok "$NAME"
    F=$R/$NAME/package/PACKAGE.json
    for _ in $(seq 960); do
      if python3 -c "import json,os,sys,time; d=json.load(open(sys.argv[1])); sys.exit(0 if 'active_parameters' in d and time.time() - os.path.getmtime(sys.argv[1]) > 30 else 1)" "$F" 2> /dev/null; then
        echo "$(date -u +%FT%TZ) $NAME package frozen; offering"
        exec bash "$0" offer "$MIR" "$NAME"
      fi
      [ ! -f "$X/mlx/$NAME.SKIP" ] || { echo "$(date -u +%FT%TZ) no package: $(cat "$X/mlx/$NAME.SKIP")"; exit 0; }
      sleep 30
    done
    echo "$(date -u +%FT%TZ) no package after 8 h"; exit 1 ;;
  auto-run)
    NAME=${1:?NAME} SHARDS=${2:?SHARDS}
    shift 2
    name_ok "$NAME"
    [ $# -eq "$SHARDS" ] || { echo "auto-run needs one GPU per shard" >&2; exit 2; }
    PEER=$(cat "$KEY/peer") T=/data/dev2/tmp/27b-moe-index
    mkdir -p "$T"
    for _ in $(seq 960); do
      if rs "root@$PEER:index/$NAME.OFFERED" "$T/" 2> /dev/null; then
        echo "$(date -u +%FT%TZ) $NAME offered; staging"
        bash "$0" stage "$MIR" "$NAME"
        mkdir -p "$P/logs"
        bash "$0" scan "$MIR" "$NAME" "$SHARDS" > "$P/logs/scan-$NAME.log" 2>&1 &
        scan=$!
        echo "$(date -u +%FT%TZ) parity gate on GPU$1 (scan pid $scan)"
        bash "$0" parity "$MIR" "$NAME" "$1"
        wait "$scan" || { echo "$(date -u +%FT%TZ) scan failed (logs/scan-$NAME.log)"; exit 1; }
        if [ -n "${MOE_INDEX_MIN_PADDED:-}" ]; then
          bash "$0" presplit "$MIR" "$NAME" "$SHARDS" "$MOE_INDEX_MIN_PADDED"
        fi
        echo "$(date -u +%FT%TZ) parity passed; starting $SHARDS shards on GPUs $*"
        exec bash "$0" run "$MIR" "$NAME" "$SHARDS" "$@"
      fi
      if rs "root@$PEER:mlx/$NAME.SKIP" "$T/" 2> /dev/null; then
        echo "$(date -u +%FT%TZ) no package: $(cat "$T/$NAME.SKIP")"; exit 0
      fi
      sleep 60
    done
    echo "$(date -u +%FT%TZ) no offer after 16 h"; exit 1 ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
