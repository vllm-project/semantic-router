#!/usr/bin/env bash
# Decoder M6 development lines (prereg dec-m6-prereg-2026-09-29.md, "Candidates" and "Development readouts"), run
# on the tier's node from an exact mirror under /data/dev2/src. Outputs under /data/dev2/runs/dec/m6/lines/<tier>.
#
#   m6-lines.sh <tier> refs                      16K readouts of the references <tier>-I (incumbent) and <tier>-O
#                                                (own 1.0 FP32 zero-step), then readout/refs.json (O vs I)
#   m6-lines.sh <tier> line <LINE> [<artifact>]  build the line's points on CPU (v2.dec.soup by member repetition),
#                                                hash them (files.sha256 + weights.json), 16K readouts of every point,
#                                                then readout/L-<LINE>.json (dev_readout, --compare <tier>-I:<point>)
#   m6-lines.sh <tier> readout <LINE>            re-run only the line's dev_readout (the previous file is kept)
#   m6-lines.sh 4b mlx <point>                   MLX-DEV readout (report only; M5 settings) + paired vs 4b-I
#   m6-lines.sh <tier> status                    points, readouts and GPU-seconds so far
#
# Tiers: 4b and 2b on node B, 08b on node A. LINE: an arm (4b N6D N6A N6P, 2b S6X S6D, 08b E6K) whose artifact
# defaults to the arm soup /data/dev2/runs/dec/m6/soup/<ARM>/build/<ARM>-soup once m6-soup.sh wrote soup/<ARM>/DONE
# (pass <artifact> to override); N5BN (4b only, beta 1/3 and 1/2); or the own 1.0 line Nox / Sol / Eos (gamma 1/6
# and 1/3).
# Arm lines: beta 1/3, 1/2, 2/3 = members [I,I,A], [I,A], [I,A,A]; beta 1 = A itself (no build). Own 1.0 lines:
# [I x5, O] and [I x2, O]. Points are named <tier>-<LINE>-b1_3 / -b1_2 / -b2_3 / -b1 / -g1_6 / -g1_3.
# The references must be read first (refs); a line's readout needs the reference readout of I.
#
# Readouts (infer_dec, typed DEV + CSS pilot, --max-length 16384, no calibration) run on GPU M6_GPU (node B 3|4,
# node A 5) as co-tenants next to a training job: lease entry /data/dev2/leases/gpu<N>.lock/owner.dec-readout,
# refused below 60 GB free VRAM; every GPU job's wall seconds are appended to GPU-SECONDS.jsonl.
set -u
S=$(cd "$(dirname "$0")/../../../.." && pwd)
MIRROR=${S%/src/training/decision2}
SRC=${MIRROR##*/}
[ -f "$MIRROR/.dev2-mirror.json" ] || { echo "run from an exact mirror under /data/dev2/src" >&2; exit 2; }
TIER=${1:-} CMD=${2:-}
# M6_DEC_ROOT / M6_LAUNCH / M6_GOLD / M6_LEASES / M6_DEV_READOUT exist for the CPU tests only.
R=${M6_DEC_ROOT:-/data/dev2/runs/dec}
M=$R/m6
LAUNCH=${M6_LAUNCH:-$S/v2/dec/launch.sh}
GOLD=${M6_GOLD:-/data/dev2/private/panels/gold}
LEASES=${M6_LEASES:-/data/dev2/leases}
DEV_READOUT=${M6_DEV_READOUT:-v2.dec.dev_readout}
MIN_FREE_GB=${M6_MIN_FREE_GB:-60}
case $TIER in
  4b)
    NODE=B IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
    DATA=$R/m3/data-sel700-cal698
    I=$R/m4/soup/N4XF/build/N4XF-soup
    O=$R/m4/arms/pre/m4-N4XF-s1-zero/checkpoint-0000000
    SOURCE=/hf/models--llm-semantic-router--Decision-1.0-Nox-4B/snapshots/cde2a68dbaa557ea65dc458104d410a0802ee259
    OWN=Nox ARMS="N6D N6A N6P" N5BN=$R/m5/soup/N5BN/build/N5BN-soup GPU=${M6_GPU:-} ;;
  2b)
    NODE=B IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
    DATA=$R/m3/data-sel700-cal698
    I=$R/m3/soup/S2T/build/S2T-soup
    O=$R/m3/arms/pre/m3-S2T-s1-zero/checkpoint-0000000
    SOURCE=/hf/models--llm-semantic-router--Decision-1.0-Sol-2B/snapshots/ce0c018a28de16d6639b1cd203b761bf643b89e6
    OWN=Sol ARMS="S6X S6D" N5BN='' GPU=${M6_GPU:-} ;;
  08b)
    NODE=A IMAGE=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
    DATA=/data/decision20-20260926/data/hf-private-decision20-clean-v2
    # node-A copy of the released E8F soup, verified file by file against node B's m2/hf-staging/batch2.sha256
    I=$R/m2/formal-candidates/staging-16c0929a/m2/E8F-soup/checkpoint
    O=$R/m6/arms/pre/m6-E6K-s1-zero/checkpoint-0000000
    SOURCE=/hf/models--llm-semantic-router--Decision-1.0-Eos-0.8B/snapshots/363c4a5e56afc115b1c78c837633956d0bbb63ab
    OWN=Eos ARMS="E6K" N5BN='' GPU=${M6_GPU:-5} ;;
  *) sed -n '2,24p' "$0"; exit 2 ;;
esac
declare -A RENDER=([B3]=/dev/dri/renderD153 [B4]=/dev/dri/renderD161 [A5]=/dev/dri/renderD169)
L=$M/lines/$TIER
mkdir -p "$L/readout"

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

# gpu_seconds <launch receipt> <point> <purpose>: one JSON line per GPU job.
gpu_seconds() {
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

# gpu_begin <purpose>: co-tenant guard and lease entry (removed on exit).
gpu_begin() {
  [ -n "$GPU" ] || die "set M6_GPU (node B 3|4, node A 5)"
  [ -n "${RENDER[$NODE$GPU]:-}" ] || die "GPU$GPU is not a node-$NODE decoder GPU for tier $TIER"
  local free
  free=$(vram_free_gb "$GPU") || die "rocm-smi failed on GPU$GPU"
  [ "$free" -ge "$MIN_FREE_GB" ] || die "GPU$GPU has $free GB free VRAM (< $MIN_FREE_GB); readout not started"
  if [ -z "$LEASE" ]; then
    LEASE=$LEASES/gpu$GPU.lock/owner.dec-readout
    [ ! -e "$LEASE" ] || die "$LEASE exists (another readout on GPU$GPU?)"
    mkdir -p "$(dirname "$LEASE")"
    printf 'track=dec\nstatus=busy (co-tenant)\npurpose=decoder M6 %s\nstart_utc=%s\nexpected_end_utc=%s\n' "$1" \
      "$(date -u +%FT%TZ)" "$(date -u -d '+90 min' +%FT%TZ)" > "$LEASE"
  fi
  export DEC_RENDER=${RENDER[$NODE$GPU]} DEC_GPU_LABEL="node $NODE GPU$GPU"
  log "GPU$GPU $1 ($free GB free)"
}

# model_of <point dir>: the checkpoint a point reads out.
model_of() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$1/weights.json"; }

# record <point> <line> <kind beta|gamma|ref> <step> <checkpoint> <anchor name=path ...> -- <member name ...>
# weights.json: members in order, effective rational weights, per-file list and its hash, soup output identity.
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

# build <point> <line> <kind> <step> <anchor name=path ...> -- <member name ...>
build() {
  local point=$1 line=$2 kind=$3 step=$4
  shift 4
  local P=$L/$point args=() m cut=0 anchors=() names=()
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
    case ${path[$m]} in "$R"/*) ;; *) die "$point: member $m is not under $R (not visible as /runs)" ;; esac
    args+=(--member "$(incontainer "${path[$m]}")")
  done
  if [ ! -d "$P/build/$point" ]; then
    [ ! -e "$P/build.launch.json" ] || die "$point: earlier build failed (receipt $P/build.launch.json); not rerun"
    log "$point build start (${names[*]})"
    bash "$LAUNCH" "m6-$point-build" "$SRC" "$P/build" --cpu -- -m v2.dec.soup "${args[@]}" --output "/out/$point" \
      || die "$point build FAILED (see $P/build.stderr.log)"
  fi
  record "$point" "$line" "$kind" "$step" "$P/build/$point" "${anchors[@]}" -- "${names[@]}" >> "$L/OPERATIONS.log" \
    || die "$point record FAILED"
  log "$point built: $(jq_weights "$P")"
}
jq_weights() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1]+"/weights.json")); print(d["effective_weights"], d["files_sha256_list_sha256"][:12])' "$1"; }

# alias_point <point> <line> <kind> <step> <name> <path>: a point that is an existing checkpoint (beta 1, I, O).
alias_point() {
  local point=$1 line=$2 kind=$3 step=$4 name=$5 ck=$6 P=$L/$1
  [ -f "$P/weights.json" ] && return 0
  [ -f "$ck/decision_config.json" ] || die "$point: $ck is not a checkpoint"
  mkdir -p "$P"
  record "$point" "$line" "$kind" "$step" "$ck" "$name=$ck" -- "$name" >> "$L/OPERATIONS.log" || die "$point record FAILED"
  log "$point recorded (no build): $ck"
}

# readout <point>: typed DEV + CSS pilot at 16,384 tokens, no calibration.
readout() {
  local point=$1 P=$L/$1 ck panel
  ck=$(model_of "$P")
  for panel in dev css-pilot; do
    [ -f "$P/$panel/$panel.predictions.jsonl" ] && continue
    [ ! -e "$P/$panel.launch.json" ] || die "$point $panel: earlier readout failed (receipt $P/$panel.launch.json); not rerun"
    gpu_begin "readout $point $panel"
    bash "$LAUNCH" "m6-$point-$panel" "$SRC" "$P/$panel" -- -m v2.dec.infer_dec --checkpoint "$(incontainer "$ck")" \
      --source-path "$SOURCE" --input "/panels/$panel.prompts.jsonl" --output "/out/$panel.predictions.jsonl" \
      --model-id "decision2-dec-m6-$point" --model-revision "$(sha "$P/files.sha256")" --max-length 16384
    local rc=$?
    gpu_seconds "$P/$panel.launch.json" "$point" "readout $panel"
    [ $rc = 0 ] || die "$point $panel readout FAILED (see $P/$panel.stderr.log)"
    log "$point $panel read: $(wc -l < "$P/$panel/$panel.predictions.jsonl") rows"
  done
}

arm_spec() {  # <point> -> name=typed,css
  echo "$1=$L/$1/dev/dev.predictions.jsonl,$L/$1/css-pilot/css-pilot.predictions.jsonl"
}

# dev_readout <out name> <line spec or -> <point ...>: arms I + points, --compare I:point.
dev_readout() {
  local name=$1 spec=$2
  shift 2
  local out=$L/readout/$name.json args=(--arm "$(arm_spec "$TIER-I")") p
  for p in "$@"; do
    for f in dev/dev css-pilot/css-pilot; do
      [ -f "$L/$p/$f.predictions.jsonl" ] || die "$name: $p has no ${f%%/*} readout yet"
    done
    args+=(--arm "$(arm_spec "$p")" --compare "$TIER-I:$p")
  done
  [ -f "$out" ] && mv "$out" "$L/readout/$name.$(date -u +%Y%m%dT%H%M%SZ).json"
  (cd "$S" && PYTHONPATH=$S python3 -B -m "$DEV_READOUT" --typed-gold "$GOLD/typed-dev.gold.jsonl" \
    --css-gold "$GOLD/css-pilot.gold.jsonl" "${args[@]}" --output "$out" > "$L/readout/$name.log" 2>&1) \
    || die "$name dev_readout FAILED (see $L/readout/$name.log)"
  [ "$spec" = - ] || printf '%s\n' "$spec" > "$L/readout/$name.line"
  log "$name readout: $(sha "$out") (${*})"
}

# line_points <LINE>: prints "point step kind" for every point of the line.
line_points() {
  case $1 in
    "$OWN") printf '%s\n' "$TIER-$1-g1_6 1/6 gamma" "$TIER-$1-g1_3 1/3 gamma" ;;
    N5BN) printf '%s\n' "$TIER-N5BN-b1_3 1/3 beta" "$TIER-N5BN-b1_2 1/2 beta" ;;
    *) printf '%s\n' "$TIER-$1-b1_3 1/3 beta" "$TIER-$1-b1_2 1/2 beta" "$TIER-$1-b2_3 2/3 beta" "$TIER-$1-b1 1 beta" ;;
  esac
}

do_line() {
  local line=$1 art=${2:-} point step kind spec="L-$1:" pts=()
  if [ "$line" = "$OWN" ]; then
    art=$O
  elif [ "$line" = N5BN ]; then
    [ -n "$N5BN" ] || die "N5BN line is 4B only"
    art=$N5BN
  elif [[ " $ARMS " == *" $line "* ]]; then
    if [ -z "$art" ]; then
      [ ! -f "$M/soup/$line/FAILED" ] || die "L-$line: arm soup FAILED ($(cat "$M/soup/$line/FAILED")); declare the line dropped"
      [ -f "$M/soup/$line/DONE" ] || die "L-$line: arm soup not done yet ($M/soup/$line/DONE)"
      art=$M/soup/$line/build/$line-soup
    fi
  else
    die "unknown line $line for tier $TIER (arms: $ARMS, N5BN, $OWN)"
  fi
  art=${art%/}
  [ -f "$art/decision_config.json" ] || die "L-$line: artifact $art missing (arm soup not built yet?)"
  [ ! -e "$art.pending" ] || die "L-$line: $art.pending exists (soup build in progress)"
  [ -f "$L/$TIER-I/dev/dev.predictions.jsonl" ] || die "reference $TIER-I not read yet (m6-lines.sh $TIER refs)"
  case $art in "$R"/*) ;; *) die "L-$line: artifact $art is not under $R" ;; esac
  mkdir "$L/L-$line.lock" 2>/dev/null || die "L-$line already running (lock $L/L-$line.lock)"
  LOCK=$L/L-$line.lock
  log "L-$line start: artifact $art"
  local X=A
  [ "$line" = "$OWN" ] && X=O
  while read -r point step kind; do
    case $step in
      1/3) if [ $X = O ]; then build "$point" "L-$line" "$kind" "$step" "I=$I" "O=$art" -- I I O
           else build "$point" "L-$line" "$kind" "$step" "I=$I" "A=$art" -- I I A; fi ;;
      1/2) build "$point" "L-$line" "$kind" "$step" "I=$I" "A=$art" -- I A ;;
      2/3) build "$point" "L-$line" "$kind" "$step" "I=$I" "A=$art" -- I A A ;;
      1/6) build "$point" "L-$line" "$kind" "$step" "I=$I" "O=$art" -- I I I I I O ;;
      1) alias_point "$point" "L-$line" "$kind" "$step" A "$art" ;;
    esac
    spec+="$step=$point,"
    pts+=("$point")
  done < <(line_points "$line")
  for point in "${pts[@]}"; do readout "$point"; done
  dev_readout "L-$line" "${spec%,}" "${pts[@]}"
  log "L-$line done"
}

case $CMD in
  refs)
    alias_point "$TIER-I" ref ref 0 I "$I"
    [ -f "$O/decision_config.json" ] || die "own 1.0 zero-step $O not there yet"
    alias_point "$TIER-O" ref ref 0 O "$O"
    readout "$TIER-I"
    readout "$TIER-O"
    dev_readout refs - "$TIER-O"
    ;;
  line)
    [ $# -ge 3 ] || { sed -n '2,24p' "$0"; exit 2; }
    do_line "$3" "${4:-}"
    ;;
  readout)
    [ $# -eq 3 ] || { sed -n '2,24p' "$0"; exit 2; }
    [ -f "$L/readout/L-$3.line" ] || die "L-$3 has no line spec yet"
    spec=$(cat "$L/readout/L-$3.line")
    mapfile -t pts < <(line_points "$3" | cut -d' ' -f1)
    dev_readout "L-$3" "$spec" "${pts[@]}"
    ;;
  mlx)
    [ "$TIER" = 4b ] && [ $# -eq 3 ] || { sed -n '2,24p' "$0"; exit 2; }
    P=$L/$3 X=$R/m5/mlxdev
    [ -f "$P/weights.json" ] || die "$3 is not a recorded point"
    [ -f "$X/READY" ] || die "MLX-DEV panel not READY ($X/READY)"
    if [ ! -f "$P/mlxdev/mlxdev-predictions.jsonl" ]; then
      [ ! -e "$P/mlxdev.launch.json" ] || die "$3 MLX-DEV: earlier readout failed; not rerun"
      gpu_begin "MLX-DEV $3"
      bash "$LAUNCH" "m6-$3-mlxdev" "$SRC" "$P/mlxdev" -- -m v2.dec.eval_rows --checkpoint "$(incontainer "$(model_of "$P")")" \
        --rows /runs/m5/mlxdev/build/panel.jsonl --tag mlxdev --output /out
      rc=$?
      gpu_seconds "$P/mlxdev.launch.json" "$3" "MLX-DEV"
      [ $rc = 0 ] || die "$3 MLX-DEV FAILED"
    fi
    IDX=$X/build/panel.jsonl.index.jsonl
    py() { (cd "$S" && PYTHONPATH=$S python3 -B -m v2.dec.mlx_dev "$@"); }
    [ -f "$P/mlxdev/score.json" ] || py score --index "$IDX" --predictions "$P/mlxdev/mlxdev-predictions.jsonl" \
      --output "$P/mlxdev/score.json" >> "$L/OPERATIONS.log" 2>&1 || die "$3 MLX-DEV score FAILED"
    REF=$L/$TIER-I/mlxdev/mlxdev-predictions.jsonl
    if [ "$3" != "$TIER-I" ] && [ -f "$REF" ] && [ ! -f "$P/mlxdev/vs-$TIER-I.json" ]; then
      py compare --index "$IDX" --a "$REF" --b "$P/mlxdev/mlxdev-predictions.jsonl" --output "$P/mlxdev/vs-$TIER-I.json" \
        >> "$L/OPERATIONS.log" 2>&1 || die "$3 MLX-DEV compare FAILED"
    fi
    log "$3 MLX-DEV done (report only)"
    ;;
  status)
    for P in "$L"/"$TIER"-*/; do
      p=$(basename "$P")
      printf '%-18s weights=%s dev=%s css=%s\n' "$p" "$([ -f "$P/weights.json" ] && echo y || echo n)" \
        "$([ -f "$P/dev/dev.predictions.jsonl" ] && echo y || echo n)" "$([ -f "$P/css-pilot/css-pilot.predictions.jsonl" ] && echo y || echo n)"
    done
    ls "$L/readout"/*.json 2>/dev/null
    [ -f "$L/GPU-SECONDS.jsonl" ] && python3 -c 'import json,sys; r=[json.loads(l) for l in open(sys.argv[1])]; print(f"{len(r)} GPU jobs, {sum(x[\"gpu_seconds\"] for x in r)/3600:.3f} GPU-h")' "$L/GPU-SECONDS.jsonl"
    ;;
  *) sed -n '2,24p' "$0"; exit 2 ;;
esac
