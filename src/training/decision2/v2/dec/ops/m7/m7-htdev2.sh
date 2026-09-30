#!/usr/bin/env bash
# Decoder M7 HT-DEV v2 diagnostic (COORDINATION 2026-09-30 04:10: a running milestone that did not adopt HT-DEV v2
# before its first development readout reports it as a diagnostic; M7's prereg predates it). Never selected on.
# Each point and the tier reference <tier>-I is collected the way the M7 development readouts are (m7-lines.sh
# infer: v2.dec.infer_dec through launch.sh on the tier's node, image and kernel path, 16,384 tokens, T = 1), so the
# comparison is paired within M7. Gold stays on node A: node-B predictions are relayed there (m7-relay.sh htdev2).
#
#   m7-htdev2.sh <tier> collect <point> [<point> ...]  tier's node: <tier>-I first if missing, then each recorded
#                                                       point -> lines/<tier>/<point>/ht-dev2/ht-dev2.predictions.jsonl
#   m7-htdev2.sh <tier> score <point> [<point> ...]    node A: m7_htdev2.py vs <tier>-I -> lines/<tier>/diag/
#                                                       <point>.htdev2.json (verdict FLAG / TIE / GAIN at ±0.02); also
#                                                       <tier>-I vs the eval track's reference collection of the same
#                                                       weights (<tier>-I.htdev2-vs-eval.json, collection-path check)
#   m7-htdev2.sh <tier> status
#
# Prompts: /data/dev2/runs/dec/panels/ht-dev2.prompts.jsonl (gold-free, 90cd409a..., m7-relay.sh htdev2-panel).
# Collections are co-tenants (lease entry owner.dec-htdev2, refused below 60 GB free VRAM); GPU seconds go to
# lines/<tier>/GPU-SECONDS.jsonl with the line readouts.
set -u
S=$(cd "$(dirname "$0")/../../../.." && pwd)
MIRROR=${S%/src/training/decision2}
SRC=${MIRROR##*/}
[ -f "$MIRROR/.dev2-mirror.json" ] || { echo "run from an exact mirror under /data/dev2/src" >&2; exit 2; }
TIER=${1:-} CMD=${2:-}
R=/data/dev2/runs/dec
M=$R/m7
LAUNCH=$S/v2/dec/launch.sh
OPS=$S/v2/dec/ops/m7
PROMPTS=$R/panels/ht-dev2.prompts.jsonl PROMPTS_SHA=90cd409a1e091a623362c0e5b227d13b7301bf13fe266f7905e09233cf815f74
GOLD=/data/dev2/private/panels/gold/ht-dev2.gold.jsonl GOLD_SHA=659c92b45d2a5b37e250ccb728cdbf5580ba361b3fc4b2f2ab505a13e0a556cc
EVAL_REF=/data/dev2/runs/eval/htdev2/collect
LEASES=/data/dev2/leases
MIN_FREE_GB=${M7_MIN_FREE_GB:-60}
case $TIER in
  4b)
    NODE=B IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
    SOURCE=/hf/models--llm-semantic-router--Decision-1.0-Nox-4B/snapshots/cde2a68dbaa557ea65dc458104d410a0802ee259
    REF_KEY=dec-m4-N4XF-soup GPU=${M7_GPU:-} ;;
  2b)
    NODE=A IMAGE=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
    SOURCE=/hf/models--llm-semantic-router--Decision-1.0-Sol-2B/snapshots/ce0c018a28de16d6639b1cd203b761bf643b89e6
    REF_KEY=dec-m3-S2T-soup GPU=${M7_GPU:-5} ;;
  *) sed -n '2,18p' "$0"; exit 2 ;;
esac
declare -A RENDER=([B3]=/dev/dri/renderD153 [B4]=/dev/dri/renderD161 [A5]=/dev/dri/renderD169)
L=$M/lines/$TIER
mkdir -p "$L/diag"
log() { echo "$(date -u +%FT%TZ) $TIER htdev2 $*" >> "$L/OPERATIONS.log"; }
die() { log "$*"; echo "$*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
incontainer() { echo "/runs/${1#"$R"/}"; }
model_of() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$1/weights.json"; }
vram_free_gb() {
  rocm-smi -d "$1" --showmeminfo vram | awk -F': ' '/Total Memory/ {t=$NF} /Total Used Memory/ {u=$NF} END {printf "%d\n", (t-u)/1e9}'
}
export DEC_IMAGE=$IMAGE
LEASE=''
cleanup() { [ -n "$LEASE" ] && rm -f "$LEASE"; return 0; }
trap cleanup EXIT

gpu_begin() {  # <purpose>
  [ -n "$GPU" ] || die "set M7_GPU (node B 3|4, node A 5)"
  [ -n "${RENDER[$NODE$GPU]:-}" ] || die "GPU$GPU is not a node-$NODE decoder GPU for tier $TIER"
  local free
  free=$(vram_free_gb "$GPU") || die "rocm-smi failed on GPU$GPU"
  [ "$free" -ge "$MIN_FREE_GB" ] || die "GPU$GPU has $free GB free VRAM (< $MIN_FREE_GB); collection not started"
  if [ -z "$LEASE" ]; then
    LEASE=$LEASES/gpu$GPU.lock/owner.dec-htdev2
    [ ! -e "$LEASE" ] || die "$LEASE exists (another HT-DEV v2 collection on GPU$GPU?)"
    printf 'track=dec\nstatus=busy (co-tenant)\npurpose=decoder M7 %s\nstart_utc=%s\nexpected_end_utc=%s\n' "$1" \
      "$(date -u +%FT%TZ)" "$(date -u -d '+60 min' +%FT%TZ)" > "$LEASE"
  fi
  export DEC_RENDER=${RENDER[$NODE$GPU]} DEC_GPU_LABEL="node $NODE GPU$GPU"
  log "GPU$GPU $1 ($free GB free)"
}

gpu_seconds() {  # <launch receipt> <point>
  python3 - "$1" "$2" >> "$L/GPU-SECONDS.jsonl" <<'EOF'
import datetime as dt, json, sys
r = json.load(open(sys.argv[1]))
t = lambda s: dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
print(json.dumps({"point": sys.argv[2], "purpose": "readout ht-dev2 (diagnostic)", "job": r["job"], "gpu": r["gpu"],
                  "start_utc": r["start_utc"], "end_utc": r["end_utc"], "exit_status": r["exit_status"],
                  "gpu_seconds": (t(r["end_utc"]) - t(r["start_utc"])).total_seconds(), "shared": True,
                  "receipt": sys.argv[1]}))
EOF
}

collect() {  # <point>
  local point=$1 P=$L/$1 ck rc
  [ -f "$P/weights.json" ] || die "$point is not a recorded point (m7-lines.sh)"
  [ -f "$P/ht-dev2/ht-dev2.predictions.jsonl" ] && return 0
  [ ! -e "$P/ht-dev2.launch.json" ] || die "$point ht-dev2: earlier collection failed (receipt $P/ht-dev2.launch.json); not rerun"
  ck=$(model_of "$P")
  [ -f "$ck/decision_config.json" ] || die "$point: $ck is not a checkpoint"
  gpu_begin "ht-dev2 $point"
  bash "$LAUNCH" "m7-$point-ht-dev2" "$SRC" "$P/ht-dev2" -- -m v2.dec.infer_dec --checkpoint "$(incontainer "$ck")" \
    --source-path "$SOURCE" --input /panels/ht-dev2.prompts.jsonl --output /out/ht-dev2.predictions.jsonl \
    --model-id "decision2-dec-m7-$point" --model-revision "$(sha "$P/files.sha256")" --max-length 16384
  rc=$?
  gpu_seconds "$P/ht-dev2.launch.json" "$point"
  [ $rc = 0 ] || die "$point ht-dev2 collection FAILED (see $P/ht-dev2.stderr.log)"
  log "$point ht-dev2 collected: $(wc -l < "$P/ht-dev2/ht-dev2.predictions.jsonl") rows"
}

score() {  # <point>
  local point=$1 out=$L/diag/$1.htdev2.json
  local left=$L/$1/ht-dev2/ht-dev2.predictions.jsonl right=$L/$TIER-I/ht-dev2/ht-dev2.predictions.jsonl
  [ -f "$out" ] && return 0
  for f in "$left" "$right"; do [ -f "$f" ] || die "missing $f (collect it; node-B files: m7-relay.sh htdev2)"; done
  (cd "$S" && PYTHONPATH=$S python3 -B "$OPS/m7_htdev2.py" --gold "$GOLD" --left "$left" --left-name "$point" \
    --right "$right" --right-name "$TIER-I" --output "$out") > "$L/diag/$point.htdev2.log" 2>&1 \
    || die "$point ht-dev2 score FAILED (see $L/diag/$point.htdev2.log)"
  log "$point ht-dev2 vs $TIER-I: $(tail -1 "$L/diag/$point.htdev2.log")"
}

case $CMD in
  collect)
    [ $# -ge 3 ] || { sed -n '2,18p' "$0"; exit 2; }
    [ "$(sha "$PROMPTS")" = "$PROMPTS_SHA" ] || die "$PROMPTS is not the HT-DEV v2 prompt file $PROMPTS_SHA"
    collect "$TIER-I"
    shift 2
    for p in "$@"; do [ "$p" = "$TIER-I" ] || collect "$p"; done
    ;;
  score)
    [ $# -ge 3 ] || { sed -n '2,18p' "$0"; exit 2; }
    [ -f "$GOLD" ] && [ "$(sha "$GOLD")" = "$GOLD_SHA" ] || die "HT-DEV v2 gold $GOLD missing or not $GOLD_SHA (score on node A)"
    shift 2
    for p in "$@"; do [ "$p" = "$TIER-I" ] || score "$p"; done
    ref=$EVAL_REF/$REF_KEY/output/ht-dev2.predictions.jsonl out=$L/diag/$TIER-I.htdev2-vs-eval.json
    if [ -f "$ref" ] && [ ! -f "$out" ]; then
      if (cd "$S" && PYTHONPATH=$S python3 -B "$OPS/m7_htdev2.py" --gold "$GOLD" \
        --left "$L/$TIER-I/ht-dev2/ht-dev2.predictions.jsonl" --left-name "$TIER-I" --right "$ref" \
        --right-name "eval:$REF_KEY" --output "$out") > "${out%.json}.log" 2>&1; then
        log "$TIER-I (M7 path) vs eval:$REF_KEY: $(tail -1 "${out%.json}.log")"
      else
        log "$TIER-I vs eval:$REF_KEY comparison FAILED (report only)"
      fi
    fi
    ;;
  status)
    for P in "$L"/"$TIER"-*/; do
      p=$(basename "$P")
      printf '%-18s ht-dev2=%s diag=%s\n' "$p" "$([ -f "$P/ht-dev2/ht-dev2.predictions.jsonl" ] && echo y || echo n)" \
        "$([ -f "$L/diag/$p.htdev2.json" ] && echo y || echo n)"
    done
    ;;
  *) sed -n '2,18p' "$0"; exit 2 ;;
esac
