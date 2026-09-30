#!/usr/bin/env bash
# Decoder M8 development lines and diagnostics (prereg dec-m8-prereg-2026-09-30.md, "Lines and the development rule",
# "Diagnostics"), node B, from an exact mirror under /data/dev2/src. Outputs under /data/dev2/runs/dec/m8/lines/4b.
# The M7 line tooling (ops/m7/m7-lines.sh) with the M8 arms, alpha grid and panels.
#
#   m8-lines.sh refs            4b-I (the released N4XF soup, list df602ca9): typed DEV, CSS pilot, HT-DEV v2 and hs1-dev
#                               reused from M7's node-B readouts of the same weights (hash-checked); Score5-typed-DEV read
#   m8-lines.sh line <ARM>      alpha 1/3, 2/3 (members [I,I,A], [I,A,A]; v2.dec.soup, CPU) and alpha 1 (the arm soup
#                               m8/soup/<ARM>/build/<ARM>-soup); 16K typed DEV, CSS pilot, HT-DEV v2, Score5-typed-DEV;
#                               then readout/L-<ARM>.json (v2.dec.dev_readout, --compare 4b-I:<point>)
#   m8-lines.sh diag <point>    hs1-dev at 16K (report only), scored against 4b-I: diag/<point>.hs1.json
#   m8-lines.sh status
#
# Every point and 4b-I are read the same way: v2.dec.infer_dec through launch.sh, node B GPU M8_GPU (3|4), image
# dbe5f32b, 16,384 tokens, T = 1. Readouts are co-tenants (lease entry owner.dec-readout, refused below 60 GB free VRAM);
# wall seconds go to GPU-SECONDS.jsonl. HT-DEV v2 and Score5-typed-DEV are scored on node A (m8-relay.sh lines, then
# m8-score.sh), where their gold is.
set -u
S=$(cd "$(dirname "$0")/../../../.." && pwd)
MIRROR=${S%/src/training/decision2}
SRC=${MIRROR##*/}
[ -f "$MIRROR/.dev2-mirror.json" ] || { echo "run from an exact mirror under /data/dev2/src" >&2; exit 2; }
CMD=${1:-}
R=/data/dev2/runs/dec
M=$R/m8
M7I=$R/m7/lines/4b/4b-I
LAUNCH=$S/v2/dec/launch.sh
GOLD=/data/dev2/private/panels/gold
DIAG_GOLD=/data/dev2/private/dec/m7
LEASES=/data/dev2/leases
MIN_FREE_GB=${M8_MIN_FREE_GB:-60}
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
DATA=$R/m3/data-sel700-cal698
I=$R/m4/soup/N4XF/build/N4XF-soup I_LIST=df602ca9f2574bcf127322c9b232ac94b2470dd00cfbefd8320b3f5fca5351f4
SOURCE=/hf/models--llm-semantic-router--Decision-1.0-Nox-4B/snapshots/cde2a68dbaa557ea65dc458104d410a0802ee259
ARMS="D1 D2 C" GPU=${M8_GPU:-}
TIER=4b
declare -A RENDER=([3]=/dev/dri/renderD153 [4]=/dev/dri/renderD161)
declare -A PROMPTS=([dev]=dev.prompts.jsonl [css-pilot]=css-pilot.prompts.jsonl [ht-dev2]=ht-dev2.prompts.jsonl
  [score5t-dev]=score5t-dev.prompts.jsonl [hs1-dev]=hs1-dev.prompts.jsonl)
L=$M/lines/$TIER
mkdir -p "$L/readout" "$L/diag"

log() { echo "$(date -u +%FT%TZ) $TIER $*" >> "$L/OPERATIONS.log"; }
die() { log "$*"; echo "$*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
tree_manifest() { (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum); }
incontainer() { echo "/runs/${1#"$R"/}"; }
vram_free_gb() {
  rocm-smi -d "$1" --showmeminfo vram | awk -F': ' '/Total Memory/ {t=$NF} /Total Used Memory/ {u=$NF} END {printf "%d\n", (t-u)/1e9}'
}
export DEC_IMAGE=$IMAGE DEC_DATA=$DATA
LEASE='' LOCK=''
cleanup() { [ -n "$LEASE" ] && rm -f "$LEASE"; [ -n "$LOCK" ] && rmdir "$LOCK" 2>/dev/null; return 0; }
trap cleanup EXIT

gpu_seconds() {  # <launch receipt> <point> <purpose>
  python3 - "$1" "$2" "$3" >> "$L/GPU-SECONDS.jsonl" <<'EOF'
import datetime as dt, json, sys
r = json.load(open(sys.argv[1]))
t = lambda s: dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
print(json.dumps({"point": sys.argv[2], "purpose": sys.argv[3], "job": r["job"], "gpu": r["gpu"],
                  "start_utc": r["start_utc"], "end_utc": r["end_utc"], "exit_status": r["exit_status"],
                  "gpu_seconds": (t(r["end_utc"]) - t(r["start_utc"])).total_seconds(), "shared": True,
                  "receipt": sys.argv[1]}))
EOF
}

gpu_begin() {  # <purpose>: co-tenant guard and lease entry (removed on exit)
  [ -n "$GPU" ] || die "set M8_GPU (node B 3|4)"
  [ -n "${RENDER[$GPU]:-}" ] || die "GPU$GPU is not a node-B decoder GPU"
  local free
  free=$(vram_free_gb "$GPU") || die "rocm-smi failed on GPU$GPU"
  [ "$free" -ge "$MIN_FREE_GB" ] || die "GPU$GPU has $free GB free VRAM (< $MIN_FREE_GB); readout not started"
  if [ -z "$LEASE" ]; then
    LEASE=$LEASES/gpu$GPU.lock/owner.dec-readout
    [ ! -e "$LEASE" ] || die "$LEASE exists (another readout on GPU$GPU?)"
    printf 'track=dec\nstatus=busy (co-tenant)\npurpose=decoder M8 %s\nstart_utc=%s\nexpected_end_utc=%s\n' "$1" \
      "$(date -u +%FT%TZ)" "$(date -u -d '+90 min' +%FT%TZ)" > "$LEASE"
  fi
  export DEC_RENDER=${RENDER[$GPU]} DEC_GPU_LABEL="node B GPU$GPU"
  log "GPU$GPU $1 ($free GB free)"
}

model_of() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$1/weights.json"; }

# record <point> <line> <kind alpha|ref> <step> <checkpoint> <anchor name=path ...> -- <member name ...>
record() {
  local point=$1 line=$2 kind=$3 step=$4 ck=$5
  shift 5
  local P=$L/$point
  tree_manifest "$ck" > "$P/files.sha256"
  python3 - "$P" "$TIER" "$point" "$line" "$kind" "$step" "$ck" "$@" <<'EOF'
import hashlib, json, sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
P, tier, point, line, kind, step, ck, *rest = sys.argv[1:]
cut = rest.index("--")
anchors = dict(a.split("=", 1) for a in rest[:cut])
members = rest[cut + 1:]
counts = Counter(members)
build = Path(P) / "build.stdout.log"
soup = json.loads(build.read_text().strip().splitlines()[-1]) if build.is_file() else None
files = Path(P) / "files.sha256"
out = {
    "schema": "dec-m6-point/1",
    "milestone": "M8",
    "tier": tier, "point": point, "line": line, "kind": kind, "step": step,
    "checkpoint": ck,
    "anchors": anchors,
    "members": [{"name": m, "path": anchors[m]} for m in members],
    "effective_weights": {m: str(Fraction(c, len(members))) for m, c in sorted(counts.items())},
    "weights_note": "v2.dec.soup records a uniform mean over the member list; repeated members give these rational weights",
    "soup_output": soup,
    "files_sha256_list": str(files),
    "files_sha256_list_sha256": hashlib.sha256(files.read_bytes()).hexdigest(),
    "files": sum(1 for line in files.read_text().splitlines() if line.strip()),
}
if soup is not None and len(soup["members"]) != len(members):
    raise SystemExit("soup member count differs from the member list")
with open(Path(P) / "weights.json", "x") as f:
    json.dump(out, f, indent=1, sort_keys=True)
    f.write("\n")
print(json.dumps({"point": point, "weights": out["effective_weights"], "list": out["files_sha256_list_sha256"]}))
EOF
}

build() {  # <point> <line> <step> <anchor name=path ...> -- <member name ...>
  local point=$1 line=$2 step=$3
  shift 3
  local P=$L/$point args=() cut=0 anchors=() names=() a m
  for a in "$@"; do
    if [ "$a" = -- ]; then cut=1; continue; fi
    if [ $cut = 0 ]; then anchors+=("$a"); else names+=("$a"); fi
  done
  [ -f "$P/weights.json" ] && return 0
  mkdir -p "$P"
  declare -A path
  for a in "${anchors[@]}"; do path[${a%%=*}]=${a#*=}; done
  for m in "${names[@]}"; do
    [ -f "${path[$m]}/decision_config.json" ] || die "$point: member $m ${path[$m]} is not a checkpoint"
    args+=(--member "$(incontainer "${path[$m]}")")
  done
  if [ ! -d "$P/build/$point" ]; then
    [ ! -e "$P/build.launch.json" ] || die "$point: earlier build failed (receipt $P/build.launch.json); not rerun"
    log "$point build start (${names[*]})"
    bash "$LAUNCH" "m8-$point-build" "$SRC" "$P/build" --cpu -- -m v2.dec.soup "${args[@]}" --output "/out/$point" \
      || die "$point build FAILED (see $P/build.stderr.log)"
  fi
  record "$point" "$line" alpha "$step" "$P/build/$point" "${anchors[@]}" -- "${names[@]}" >> "$L/OPERATIONS.log" \
    || die "$point record FAILED"
  log "$point built"
}

alias_point() {  # <point> <line> <kind> <step> <name> <path>
  local point=$1 line=$2 kind=$3 step=$4 name=$5 ck=$6 P=$L/$1
  [ -f "$P/weights.json" ] && return 0
  [ -f "$ck/decision_config.json" ] || die "$point: $ck is not a checkpoint"
  mkdir -p "$P"
  record "$point" "$line" "$kind" "$step" "$ck" "$name=$ck" -- "$name" >> "$L/OPERATIONS.log" || die "$point record FAILED"
  log "$point recorded (no build): $ck"
}

infer() {  # <point> <panel>
  local point=$1 panel=$2 P=$L/$1 ck
  ck=$(model_of "$P")
  [ -f "$P/$panel/$panel.predictions.jsonl" ] && return 0
  [ ! -e "$P/$panel.launch.json" ] || die "$point $panel: earlier readout failed (receipt $P/$panel.launch.json); not rerun"
  [ -f "$R/panels/${PROMPTS[$panel]}" ] || die "missing gold-free prompts $R/panels/${PROMPTS[$panel]} (m8-relay.sh panels)"
  gpu_begin "readout $point $panel"
  bash "$LAUNCH" "m8-$point-$panel" "$SRC" "$P/$panel" -- -m v2.dec.infer_dec --checkpoint "$(incontainer "$ck")" \
    --source-path "$SOURCE" --input "/panels/${PROMPTS[$panel]}" --output "/out/$panel.predictions.jsonl" \
    --model-id "decision2-dec-m8-$point" --model-revision "$(sha "$P/files.sha256")" --max-length 16384
  local rc=$?
  gpu_seconds "$P/$panel.launch.json" "$point" "readout $panel"
  [ $rc = 0 ] || die "$point $panel readout FAILED (see $P/$panel.stderr.log)"
  log "$point $panel read: $(wc -l < "$P/$panel/$panel.predictions.jsonl") rows"
}
readout() { local p; for p in dev css-pilot ht-dev2 score5t-dev; do infer "$1" "$p"; done; }

arm_spec() { echo "$1=$L/$1/dev/dev.predictions.jsonl,$L/$1/css-pilot/css-pilot.predictions.jsonl"; }

dev_readout() {  # <out name> <line spec> <point ...>
  local name=$1 spec=$2
  shift 2
  local out=$L/readout/$name.json args=(--arm "$(arm_spec "$TIER-I")") p
  for p in "$@"; do args+=(--arm "$(arm_spec "$p")" --compare "$TIER-I:$p"); done
  [ -f "$out" ] && mv "$out" "$L/readout/$name.$(date -u +%Y%m%dT%H%M%SZ).json"
  (cd "$S" && PYTHONPATH=$S python3 -B -m v2.dec.dev_readout --typed-gold "$GOLD/typed-dev.gold.jsonl" \
    --css-gold "$GOLD/css-pilot.gold.jsonl" "${args[@]}" --output "$out" > "$L/readout/$name.log" 2>&1) \
    || die "$name dev_readout FAILED (see $L/readout/$name.log)"
  printf '%s\n' "$spec" > "$L/readout/$name.line"
  log "$name readout: $(sha "$out") (${*})"
}

do_line() {
  local line=$1 art
  [[ " $ARMS " == *" $line "* ]] || die "unknown arm $line (arms: $ARMS)"
  [ ! -f "$M/status/$line.FAILED" ] || die "L-$line: arm soup FAILED ($(cat "$M/status/$line.FAILED")); declare the line dropped"
  [ -f "$M/status/$line.DONE" ] || die "L-$line: arm soup not done yet"
  art=$M/soup/$line/build/$line-soup
  [ -f "$art/decision_config.json" ] || die "L-$line: artifact $art missing"
  [ -f "$L/$TIER-I/dev/dev.predictions.jsonl" ] || die "reference $TIER-I not read yet (m8-lines.sh refs)"
  mkdir "$L/L-$line.lock" 2>/dev/null || die "L-$line already running"
  LOCK=$L/L-$line.lock
  log "L-$line start: artifact $art"
  alias_point "$TIER-$line-a1" "L-$line" alpha 1 A "$art"
  build "$TIER-$line-a2_3" "L-$line" 2/3 "I=$I" "A=$art" -- I A A
  build "$TIER-$line-a1_3" "L-$line" 1/3 "I=$I" "A=$art" -- I I A
  for p in "$TIER-$line-a1" "$TIER-$line-a2_3" "$TIER-$line-a1_3"; do readout "$p"; done
  dev_readout "L-$line" "L-$line:1=$TIER-$line-a1,2/3=$TIER-$line-a2_3,1/3=$TIER-$line-a1_3" \
    "$TIER-$line-a1" "$TIER-$line-a2_3" "$TIER-$line-a1_3"
  log "L-$line done"
}

do_diag() {
  local point=$1 P=$L/$1 ref=$L/$TIER-I
  [ -f "$P/weights.json" ] || die "$point is not a recorded point"
  [ -f "$DIAG_GOLD/hs1-dev.gold.jsonl" ] || die "missing $DIAG_GOLD/hs1-dev.gold.jsonl"
  infer "$TIER-I" hs1-dev
  infer "$point" hs1-dev
  [ "$point" = "$TIER-I" ] && return 0
  (cd "$S" && PYTHONPATH=$S python3 -B -m v2.data.hs1.validity --gold "$DIAG_GOLD/hs1-dev.gold.jsonl" \
    --left "$P/hs1-dev/hs1-dev.predictions.jsonl" --left-name "$point" \
    --right "$ref/hs1-dev/hs1-dev.predictions.jsonl" --right-name "$TIER-I" --output "$L/diag/$point.hs1.json") \
    > "$L/diag/$point.hs1.log" 2>&1 || die "$point hs1-dev score FAILED"
  log "$point hs1-dev scored (report only)"
}

case $CMD in
  refs)
    [ "$(tree_manifest "$I" | sha256sum | cut -d' ' -f1)" = "$I_LIST" ] || die "incumbent $I differs from its list $I_LIST"
    alias_point "$TIER-I" ref ref 0 I "$I"
    [ "$(sha "$M7I/files.sha256")" = "$I_LIST" ] || die "M7 reference readout is not of these weights"
    for panel in dev css-pilot ht-dev2 hs1-dev; do
      if [ ! -f "$L/$TIER-I/$panel/$panel.predictions.jsonl" ] && [ -f "$M7I/$panel/$panel.predictions.jsonl" ]; then
        cp -a "$M7I/$panel" "$L/$TIER-I/$panel" || die "copy of the M7 $panel readout FAILED"
        sha256sum "$M7I/$panel/$panel.predictions.jsonl" "$L/$TIER-I/$panel/$panel.predictions.jsonl" >> "$L/$TIER-I/reused-from-m7.sha256"
        log "$TIER-I: $panel reused from $M7I (same weights, node, image, limit)"
      fi
    done
    readout "$TIER-I"
    ;;
  line) [ $# -eq 2 ] || { sed -n '2,20p' "$0"; exit 2; }; do_line "$2" ;;
  diag) [ $# -eq 2 ] || { sed -n '2,20p' "$0"; exit 2; }; do_diag "$2" ;;
  status)
    for P in "$L"/"$TIER"-*/; do
      p=$(basename "$P")
      printf '%-16s' "$p"
      for panel in dev css-pilot ht-dev2 score5t-dev hs1-dev; do
        printf ' %s=%s' "$panel" "$([ -f "$P/$panel/$panel.predictions.jsonl" ] && echo y || echo n)"
      done
      echo
    done
    [ -f "$L/GPU-SECONDS.jsonl" ] && python3 -c 'import json,sys; r=[json.loads(l) for l in open(sys.argv[1])]; print(f"{len(r)} GPU jobs, {sum(x[\"gpu_seconds\"] for x in r)/3600:.3f} GPU-h")' "$L/GPU-SECONDS.jsonl"
    ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
