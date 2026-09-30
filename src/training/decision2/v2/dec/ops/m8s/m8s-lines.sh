#!/usr/bin/env bash
# Decoder M8-small development lines and diagnostics on node B (prereg dec-m8s-prereg-2026-09-30.md, "Candidates and
# development readouts", "Development gates", "Diagnostics"). Outputs under /data/dev2/runs/dec/m8s/lines/<tier>.
#
#   m8s-lines.sh <tier> refs          <tier>-I (the verified released start) read at 16K: typed DEV, CSS pilot,
#                                     HT-DEV v2; then its HT-DEV v2 vs the eval track's reference collection of the
#                                     same weights (diag/<tier>-I.htdev2-vs-eval.json, collection-path check)
#   m8s-lines.sh <tier> line <ARM>    alpha 1/3, 1/2 (members [A, I, I], [A, I]; v2.dec.soup, CPU) and alpha 1 (the arm
#                                     soup m8s/soup/<tier>-<ARM>), 16K readouts, readout/L-<ARM>.json
#                                     (v2.dec.dev_readout vs <tier>-I) and diag/<point>.htdev2.json (vs <tier>-I)
#   m8s-lines.sh <tier> diag <point>  hs1-dev at 16K for <tier>-I and <point>, scored against <tier>-I (report only)
#   m8s-lines.sh <tier> status
#
# Every point and the reference are read the same way: v2.dec.infer_dec through launch.sh, image dbe5f32b, the M8s
# Triton cache, 16,384 tokens, T = 1. GPU: M8S_GPU (node B 2, 6 or 7), a co-tenant only of this track's own jobs
# (lease entry owner.dec-m8s-readout, refused below 60 GB free VRAM); GPU seconds go to m8s/GPU-SECONDS.jsonl.
set -u
. "$(dirname "$0")/m8s-lib.sh"
TIER=${1:-} CMD=${2:-}
case $TIER in
  2b) SOURCE10=/hf/models--llm-semantic-router--Decision-1.0-Sol-2B/snapshots/ce0c018a28de16d6639b1cd203b761bf643b89e6
    REF_KEY=dec-m3-S2T-soup ;;
  08b) SOURCE10=/hf/models--llm-semantic-router--Decision-1.0-Eos-0.8B/snapshots/363c4a5e56afc115b1c78c837633956d0bbb63ab
    REF_KEY=dec-m2-E8F-soup ;;
  *) sed -n '2,17p' "$0"; exit 2 ;;
esac
GPU=${M8S_GPU:-}
I=${START[$TIER]}
L=$M/lines/$TIER
DIAG_GOLD=/data/dev2/private/dec/m7/hs1-dev.gold.jsonl
mkdir -p "$L/readout" "$L/diag"
LEASE=''
cleanup() { [ -n "$LEASE" ] && rm -f "$LEASE"; return 0; }
trap cleanup EXIT
die() { log "lines $TIER: $*"; exit 1; }
tree_manifest() { (cd "$1" && find . -type f -not -path './.cache/*' -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum); }
model_of() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$1/weights.json"; }

gpu_begin() {  # <purpose>
  [ -n "$GPU" ] || die "set M8S_GPU (node B 2, 6 or 7)"
  dec_env "$GPU" || exit 2
  local free
  free=$(vram_free_gb "$GPU")
  [ "$free" -ge 60 ] || die "GPU$GPU has $free GB free VRAM (< 60); readout not started"
  if [ -z "$LEASE" ]; then
    LEASE=/data/dev2/leases/gpu$GPU.lock/owner.dec-m8s-readout
    [ ! -e "$LEASE" ] || die "$LEASE exists (another M8s readout on GPU$GPU?)"
    printf 'track=dec-small\nstatus=busy (co-tenant of this track only)\npurpose=decoder M8-small %s\nstart_utc=%s\n' \
      "$1" "$(date -u +%FT%TZ)" > "$LEASE"
  fi
}

record() {  # <point> <line> <alpha> <checkpoint> <member name=path ...>
  local point=$1 line=$2 alpha=$3 ck=$4 P=$L/$1
  shift 4
  tree_manifest "$ck" > "$P/files.sha256"
  python3 - "$P" "$TIER" "$point" "$line" "$alpha" "$ck" "$@" <<'EOF'
import hashlib, json, sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
P, tier, point, line, alpha, ck, *members = sys.argv[1:]
names = [m.split("=", 1)[0] for m in members]
paths = dict(m.split("=", 1) for m in members)
counts = Counter(names)
build = Path(P) / "build.stdout.log"
files = Path(P) / "files.sha256"
out = {
    "schema": "dec-m8s-point/1", "tier": tier, "point": point, "line": line, "alpha": alpha, "checkpoint": ck,
    "members": [{"name": n, "path": paths[n]} for n in names],
    "effective_weights": {n: str(Fraction(c, len(names))) for n, c in sorted(counts.items())},
    "soup_output": json.loads(build.read_text().strip().splitlines()[-1]) if build.is_file() else None,
    "files_sha256_list_sha256": hashlib.sha256(files.read_bytes()).hexdigest(),
    "files": sum(1 for x in files.read_text().splitlines() if x.strip()),
}
with open(Path(P) / "weights.json", "x") as f:
    json.dump(out, f, indent=1, sort_keys=True)
    f.write("\n")
print(json.dumps({"point": point, "weights": out["effective_weights"]}))
EOF
}

build() {  # <point> <line> <alpha> <member name=path ...> (A first: the soup keeps the first member's root files)
  local point=$1 line=$2 alpha=$3 P=$L/$1 args=() m
  shift 3
  [ -f "$P/weights.json" ] && return 0
  mkdir -p "$P"
  for m in "$@"; do args+=(--member "$(incontainer "${m#*=}")"); done
  if [ ! -d "$P/build/$point" ]; then
    [ ! -e "$P/build.launch.json" ] || die "$point: earlier build failed (receipt $P/build.launch.json); not rerun"
    bash "$LAUNCH" "m8s-$point-build" "$SRC" "$P/build" --cpu -- -m v2.dec.soup "${args[@]}" --output "/out/$point" \
      || die "$point build FAILED (see $P/build.stderr.log)"
  fi
  record "$point" "$line" "$alpha" "$P/build/$point" "$@" >> "$L/OPERATIONS.log" || die "$point record FAILED"
  log "lines $TIER: $point built"
}

alias_point() {  # <point> <line> <alpha> <name> <path>
  local P=$L/$1
  [ -f "$P/weights.json" ] && return 0
  [ -f "$5/decision_config.json" ] || die "$1: $5 is not a checkpoint"
  mkdir -p "$P"
  record "$1" "$2" "$3" "$5" "$4=$5" >> "$L/OPERATIONS.log" || die "$1 record FAILED"
}

infer() {  # <point> <panel> <prompts file under /panels>
  local point=$1 panel=$2 prompts=$3 P=$L/$1 ck rc
  ck=$(model_of "$P")
  [ -f "$P/$panel/$panel.predictions.jsonl" ] && return 0
  [ ! -e "$P/$panel.launch.json" ] || die "$point $panel: earlier readout failed (receipt $P/$panel.launch.json); not rerun"
  gpu_begin "readout $point $panel"
  bash "$LAUNCH" "m8s-$point-$panel" "$SRC" "$P/$panel" -- -m v2.dec.infer_dec --checkpoint "$(incontainer "$ck")" \
    --source-path "$SOURCE10" --input "/panels/$prompts" --output "/out/$panel.predictions.jsonl" \
    --model-id "decision2-dec-m8s-$point" --model-revision "$(sha "$P/files.sha256")" --max-length 16384
  rc=$?
  gpu_seconds "$P/$panel.launch.json" "lines:$TIER" "readout $point $panel"
  [ $rc = 0 ] || die "$point $panel readout FAILED (see $P/$panel.stderr.log)"
}
readout() {
  infer "$1" dev dev.prompts.jsonl
  infer "$1" css-pilot css-pilot.prompts.jsonl
  infer "$1" ht-dev2 ht-dev2.prompts.jsonl
}
htdev2() {  # <out> <left point file> <left name> <right file> <right name>
  [ -f "$1" ] && return 0
  (cd "$S" && PYTHONPATH=$S python3 -B "$OPS/m8s_rules.py" htdev2 --gold "$HT_GOLD" --left "$2" --left-name "$3" \
    --right "$4" --right-name "$5" --output "$1") >> "$L/diag/htdev2.log" 2>&1 || die "HT-DEV v2 score FAILED ($1)"
  log "lines $TIER: $(tail -1 "$L/diag/htdev2.log")"
}
arm_spec() { echo "$1=$L/$1/dev/dev.predictions.jsonl,$L/$1/css-pilot/css-pilot.predictions.jsonl"; }

[ "$(sha "$HT_GOLD")" = "$HT_GOLD_SHA" ] || die "HT-DEV v2 gold differs from $HT_GOLD_SHA"
case $CMD in
  refs)
    [ -f "$I/START-VERIFIED.json" ] || die "start $I is not verified (m8s-prep.sh)"
    alias_point "$TIER-I" ref 0 I "$I"
    readout "$TIER-I"
    htdev2 "$L/diag/$TIER-I.htdev2-vs-eval.json" "$L/$TIER-I/ht-dev2/ht-dev2.predictions.jsonl" "$TIER-I" \
      "$M/eval-ref/$REF_KEY/ht-dev2.predictions.jsonl" "eval:$REF_KEY"
    ;;
  line)
    [ $# -eq 3 ] || { sed -n '2,17p' "$0"; exit 2; }
    ARM=$3 A=$M/soup/$TIER-$3/build/$TIER-$3-soup
    [ -f "$M/soup/$TIER-$ARM/DONE" ] || die "L-$ARM: arm soup not DONE"
    [ -f "$L/$TIER-I/ht-dev2/ht-dev2.predictions.jsonl" ] || die "reference $TIER-I not read yet (refs)"
    build "$TIER-$ARM-a1_3" "L-$ARM" 1/3 "A=$A" "I=$I" "I=$I"
    build "$TIER-$ARM-a1_2" "L-$ARM" 1/2 "A=$A" "I=$I"
    alias_point "$TIER-$ARM-a1" "L-$ARM" 1 A "$A"
    args=(--arm "$(arm_spec "$TIER-I")")
    for p in "$TIER-$ARM-a1" "$TIER-$ARM-a1_2" "$TIER-$ARM-a1_3"; do
      readout "$p"
      args+=(--arm "$(arm_spec "$p")" --compare "$TIER-I:$p")
      htdev2 "$L/diag/$p.htdev2.json" "$L/$p/ht-dev2/ht-dev2.predictions.jsonl" "$p" \
        "$L/$TIER-I/ht-dev2/ht-dev2.predictions.jsonl" "$TIER-I"
    done
    out=$L/readout/L-$ARM.json
    [ -f "$out" ] && mv "$out" "$L/readout/L-$ARM.$(date -u +%Y%m%dT%H%M%SZ).json"
    (cd "$S" && PYTHONPATH=$S python3 -B -m v2.dec.dev_readout --typed-gold "$GOLD/typed-dev.gold.jsonl" \
      --css-gold "$GOLD/css-pilot.gold.jsonl" "${args[@]}" --output "$out" > "$L/readout/L-$ARM.log" 2>&1) \
      || die "L-$ARM dev_readout FAILED (see $L/readout/L-$ARM.log)"
    printf 'L-%s:1=%s,1/2=%s,1/3=%s\n' "$ARM" "$TIER-$ARM-a1" "$TIER-$ARM-a1_2" "$TIER-$ARM-a1_3" > "$L/readout/L-$ARM.line"
    log "lines $TIER: L-$ARM done ($(sha "$out" | cut -c1-12))"
    ;;
  diag)
    [ $# -eq 3 ] || { sed -n '2,17p' "$0"; exit 2; }
    [ -f "$L/$3/weights.json" ] || die "$3 is not a recorded point"
    for p in "$TIER-I" "$3"; do infer "$p" hs1-dev hs1-dev.prompts.jsonl; done
    [ "$3" = "$TIER-I" ] && exit 0
    [ -f "$L/diag/$3.hs1.json" ] || (cd "$S" && PYTHONPATH=$S python3 -B -m v2.data.hs1.validity --gold "$DIAG_GOLD" \
      --left "$L/$3/hs1-dev/hs1-dev.predictions.jsonl" --left-name "$3" \
      --right "$L/$TIER-I/hs1-dev/hs1-dev.predictions.jsonl" --right-name "$TIER-I" --output "$L/diag/$3.hs1.json") \
      > "$L/diag/$3.hs1.log" 2>&1 || die "$3 hs1-dev score FAILED"
    log "lines $TIER: $3 hs1-dev scored (report only)"
    ;;
  status)
    for P in "$L"/"$TIER"-*/; do
      p=$(basename "$P")
      printf '%-20s weights=%s dev=%s css=%s ht=%s hs1=%s gate=%s\n' "$p" "$([ -f "$P/weights.json" ] && echo y || echo n)" \
        "$([ -f "$P/dev/dev.predictions.jsonl" ] && echo y || echo n)" "$([ -f "$P/css-pilot/css-pilot.predictions.jsonl" ] && echo y || echo n)" \
        "$([ -f "$P/ht-dev2/ht-dev2.predictions.jsonl" ] && echo y || echo n)" "$([ -f "$P/hs1-dev/hs1-dev.predictions.jsonl" ] && echo y || echo n)" \
        "$([ -f "$L/diag/$p.htdev2.json" ] && python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["verdict"], round(d["delta"],4))' "$L/diag/$p.htdev2.json" || echo -)"
    done
    ;;
  *) sed -n '2,17p' "$0"; exit 2 ;;
esac
