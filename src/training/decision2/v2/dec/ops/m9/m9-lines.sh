#!/usr/bin/env bash
# Decoder M9 development readouts, early-rule reads, lines, diagnostics and scoring on node A (prereg
# dec-m9-prereg-2026-10-01.md, "Early stop", "Development readout path", "Lines and the development gates"), from an
# exact mirror under /data/dev2/src. Outputs under /data/dev2/runs/dec/m9/lines/4b (scores in diag/, rules in
# m9/select/).
#
#   m9-lines.sh refs            4b-I (the N4XF soup staged on node A, list df602ca9 = DEV2.0-4B's weights) on every
#                               panel, then its parity with the stored node-B readouts (report only)
#   m9-lines.sh early <ARM>     HT-DEV v2 of <ARM>-s1's BEST (point 4b-<ARM>-s1), scored against 4b-I
#   m9-lines.sh e1              the early rule once both seed-1 reads exist -> m9/early/E1.json (m9_rules.py early)
#   m9-lines.sh control         amendment 1: verify the N7C soup copied to node A (m9/control/N7C-soup, list 9bcc0d10)
#   m9-lines.sh line <ARM>      alpha 1 (the arm soup) and alpha 1/2 ([I, soup], v2.dec.soup, CPU); typed DEV, CSS pilot,
#                               HT-DEV v2, Score5-typed-DEV, HR2 DEV; hs1-dev for alpha 1; scored on node A. <ARM> is
#                               H9, or the control N7C (amendment 1; C9 stopped by its seed-1 preflight)
#   m9-lines.sh score           (re)score every read point not yet scored, the H9 - control pairs and the line readouts
#   m9-lines.sh rules           m9_rules.py gates -> m9/select/4b-pick.json
#   m9-lines.sh status
#
# Every read uses v2.dec.infer_dec through launch.sh on node A GPU M9_GPU (6|7), image dbe5f32b,
# HIP_FORCE_DEV_KERNARG=1, 16,384 tokens, T = 1, and one persistent readout autotune cache (a copy of node B's
# decoder cache for dbe5f32b). Reads are co-tenants (lease entry owner.dec-readout, refused below 60 GB free VRAM);
# wall seconds go to GPU-SECONDS.jsonl beside the launch receipts.
set -u
S=$(cd "$(dirname "$0")/../../../.." && pwd)
MIRROR=${S%/src/training/decision2}
SRC=${MIRROR##*/}
[ -f "$MIRROR/.dev2-mirror.json" ] || { echo "run from an exact mirror under /data/dev2/src" >&2; exit 2; }
CMD=${1:-}
R=/data/dev2/runs/dec
M=$R/m9
L=$M/lines/4b
LAUNCH=$S/v2/dec/launch.sh
OPS=$S/v2/dec/ops
GOLD=/data/dev2/private/panels/gold
PANEL_ROOT=/data/dev2/private/panels
HT_GOLD=$GOLD/ht-dev2.gold.jsonl HT_GOLD_SHA=659c92b45d2a5b37e250ccb728cdbf5580ba361b3fc4b2f2ab505a13e0a556cc
HS1_GOLD=/data/dev2/private/dec/m7/hs1-dev.gold.jsonl
HR2_GOLD=/data/dev2/private/dec/m9/hr2-dev.gold.jsonl
EVAL_HT=/data/dev2/runs/eval/htdev2/collect/dec-m4-N4XF-soup/output/ht-dev2.predictions.jsonl
LEASES=/data/dev2/leases
MIN_FREE_GB=${M9_MIN_FREE_GB:-60}
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
CACHE_MASTER=$M/caches/readout-master/dbe5f32b2263
CACHE_MASTER_LIST=a4be2c239b54d3e125a793f06e2a72b251d355e0f1268268bc0650ee46604b92
CACHE=$M/triton-readout
I=$R/m4/formal-candidates/staging-e8656221/m4/N4XF-soup/checkpoint
I_LIST=df602ca9f2574bcf127322c9b232ac94b2470dd00cfbefd8320b3f5fca5351f4
SOURCE=/hf/models--llm-semantic-router--Decision-1.0-Nox-4B/snapshots/cde2a68dbaa557ea65dc458104d410a0802ee259
GPU=${M9_GPU:-}
TIER=4b
# Amendment 1 (dec-m9-amendment-1-2026-10-01.md): C9 stopped by its seed-1 preflight; the control line is M7's N7C.
CONTROL=${M9_CONTROL:-N7C}
N7C_DIR=$M/control N7C_SOUP=$M/control/N7C-soup
N7C_LIST=9bcc0d10054b142dfc2497119cf5a8d282a1a0f435f621a075a637a70dd7e826
N7C_HALF_LIST=d568829ac619d8bbc6cae70ee1496fea07ec3f6df77cad1feb04e7b40b1eb932
declare -A RENDER=([6]=/dev/dri/renderD177 [7]=/dev/dri/renderD185)
declare -A PROMPTS=([dev]=dev.prompts.jsonl [css-pilot]=css-pilot.prompts.jsonl [ht-dev2]=ht-dev2.prompts.jsonl
  [score5t-dev]=score5t-dev.prompts.jsonl [hs1-dev]=hs1-dev.prompts.jsonl [hr2-dev]=hr2-dev.prompts.jsonl)
mkdir -p "$L/readout" "$L/diag" "$L/parity" "$M/early" "$M/select"

log() { echo "$(date -u +%FT%TZ) $TIER $*" >> "$L/OPERATIONS.log"; }
die() { log "$*"; echo "$*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
tree_manifest() { (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum); }
incontainer() { echo "/runs/${1#"$R"/}"; }
py() { (cd "$S" && PYTHONPATH=$S python3 -B "$@"); }
vram_free_gb() {
  rocm-smi -d "$1" --showmeminfo vram | awk -F': ' '/Total Memory/ {t=$NF} /Total Used Memory/ {u=$NF} END {printf "%d\n", (t-u)/1e9}'
}
export DEC_IMAGE=$IMAGE DEC_DATA=$R/m3/data-sel700-cal698 DEC_KERNARG=1 DEC_TRITON_CACHE=$CACHE
LEASE='' LOCKS=()
cleanup() {
  [ -n "$LEASE" ] && rm -f "$LEASE"
  local l
  for l in "${LOCKS[@]}"; do rmdir "$l" 2> /dev/null; done
  return 0
}
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

wait_lock() {  # <lock dir> [<done file>]: take the lock, or wait while another process holds it (at most 60 min)
  local n=0
  until mkdir "$1" 2> /dev/null; do
    [ -n "${2:-}" ] && [ -e "$2" ] && return 1
    n=$((n + 1))
    [ $n -le 180 ] || die "lock $1 held for over 60 min"
    sleep 20
  done
  LOCKS+=("$1")
  return 0
}

cache_ready() {  # the persistent readout cache: one cp -a copy of the node-B master, made once
  [ -d "$CACHE" ] && return 0
  wait_lock "$CACHE.lock" "$CACHE" || return 0
  if [ ! -d "$CACHE" ]; then
    [ "$(tree_manifest "$CACHE_MASTER" | sha256sum | cut -d' ' -f1)" = "$CACHE_MASTER_LIST" ] \
      || die "readout cache master differs from its list $CACHE_MASTER_LIST"
    if ! { cp -a "$CACHE_MASTER" "$CACHE.tmp" && mv -T "$CACHE.tmp" "$CACHE"; }; then die "readout cache copy FAILED"; fi
    log "readout cache $CACHE copied from $CACHE_MASTER (list $CACHE_MASTER_LIST)"
  fi
  rmdir "$CACHE.lock"
}

gpu_begin() {  # <purpose>: co-tenant guard and lease entry (removed on exit)
  [ -n "$GPU" ] || die "set M9_GPU (node A 6|7)"
  [ -n "${RENDER[$GPU]:-}" ] || die "GPU$GPU is not an M9 GPU"
  local free
  free=$(vram_free_gb "$GPU") || die "rocm-smi failed on GPU$GPU"
  [ "$free" -ge "$MIN_FREE_GB" ] || die "GPU$GPU has $free GB free VRAM (< $MIN_FREE_GB); readout not started"
  if [ -z "$LEASE" ]; then
    LEASE=$LEASES/gpu$GPU.lock/owner.dec-readout-$$
    [ ! -e "$LEASE" ] || die "$LEASE exists"
    mkdir -p "$(dirname "$LEASE")"
    printf 'track=dec\nstatus=busy (co-tenant)\npurpose=decoder M9 %s\nstart_utc=%s\nexpected_end_utc=%s\n' "$1" \
      "$(date -u +%FT%TZ)" "$(date -u -d '+90 min' +%FT%TZ)" > "$LEASE"
  fi
  export DEC_RENDER=${RENDER[$GPU]} DEC_GPU_LABEL="node A GPU$GPU"
  log "GPU$GPU $1 ($free GB free)"
}

model_of() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$1/weights.json"; }

# record <point> <line> <kind alpha|ref|seed> <step> <checkpoint> <anchor name=path ...> -- <member name ...>
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
    "milestone": "M9",
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
    bash "$LAUNCH" "m9-$point-build" "$SRC" "$P/build" --cpu -- -m v2.dec.soup "${args[@]}" --output "/out/$point" \
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
  local point=$1 panel=$2 P=$L/$1 ck lock
  ck=$(model_of "$P")
  [ -f "$P/$panel/$panel.predictions.jsonl" ] && return 0
  [ ! -e "$P/$panel.launch.json" ] || die "$point $panel: earlier readout failed (receipt $P/$panel.launch.json); not rerun"
  [ -f "$R/panels/${PROMPTS[$panel]}" ] || die "missing gold-free prompts $R/panels/${PROMPTS[$panel]}"
  lock=$P/$panel.lock
  wait_lock "$lock" "$P/$panel/$panel.predictions.jsonl" || return 0
  if [ -f "$P/$panel/$panel.predictions.jsonl" ]; then rmdir "$lock"; return 0; fi
  [ ! -e "$P/$panel.launch.json" ] || die "$point $panel: earlier readout failed (receipt $P/$panel.launch.json); not rerun"
  cache_ready
  gpu_begin "readout $point $panel"
  bash "$LAUNCH" "m9-$point-$panel" "$SRC" "$P/$panel" -- -m v2.dec.infer_dec --checkpoint "$(incontainer "$ck")" \
    --source-path "$SOURCE" --input "/panels/${PROMPTS[$panel]}" --output "/out/$panel.predictions.jsonl" \
    --model-id "decision2-dec-m9-$point" --model-revision "$(sha "$P/files.sha256")" --max-length 16384
  local rc=$?
  gpu_seconds "$P/$panel.launch.json" "$point" "readout $panel"
  rmdir "$lock"
  [ $rc = 0 ] || die "$point $panel readout FAILED (see $P/$panel.stderr.log)"
  log "$point $panel read: $(wc -l < "$P/$panel/$panel.predictions.jsonl") rows"
}
readout() { local p; for p in dev css-pilot ht-dev2 score5t-dev hr2-dev; do infer "$1" "$p"; done; }

# ---- scoring (node A host, CPU) ----
ht() {  # <point>: HT-DEV v2 vs 4b-I
  local out=$L/diag/$1.htdev2.json left=$L/$1/ht-dev2/ht-dev2.predictions.jsonl right=$L/$TIER-I/ht-dev2/ht-dev2.predictions.jsonl
  [ -f "$out" ] && return 0
  [ -f "$left" ] && [ -f "$right" ] || return 0
  py "$OPS/m9/m9_rules.py" htdev2 --gold "$HT_GOLD" --left "$left" --left-name "$1" --right "$right" --right-name "$TIER-I" \
    --output "$out" > "$L/diag/$1.htdev2.log" 2>&1 || die "$1 HT-DEV v2 score FAILED (see $L/diag/$1.htdev2.log)"
  log "$1 ht-dev2 vs $TIER-I: $(tail -1 "$L/diag/$1.htdev2.log")"
}
s5() {  # <point>: Score5-typed-DEV blocks
  local out=$L/diag/$1.score5t.json preds=$L/$1/score5t-dev/score5t-dev.predictions.jsonl
  [ -f "$out" ] || [ ! -f "$preds" ] && return 0
  py "$OPS/m9/m9_rules.py" score5t --panel-root "$PANEL_ROOT" --predictions "$preds" --output "$out" > "$L/diag/$1.score5t.log" 2>&1 \
    || die "$1 Score5-typed-DEV score FAILED (see $L/diag/$1.score5t.log)"
  log "$1 score5t: $(tail -1 "$L/diag/$1.score5t.log")"
}
hr2() {  # <point>: HR2 DEV slice, against 4b-I when present
  local out=$L/diag/$1.hr2dev.json preds=$L/$1/hr2-dev/hr2-dev.predictions.jsonl ref=$L/$TIER-I/hr2-dev/hr2-dev.predictions.jsonl
  [ -f "$out" ] || [ ! -f "$preds" ] && return 0
  local args=(--gold "$HR2_GOLD" --predictions "$preds" --name "$1" --output "$out")
  [ "$1" != "$TIER-I" ] && { [ -f "$ref" ] || return 0; args+=(--reference "$ref" --reference-name "$TIER-I"); }
  py "$OPS/m9/m9_hr2dev.py" score "${args[@]}" > "$L/diag/$1.hr2dev.log" 2>&1 || die "$1 HR2 DEV score FAILED (see $L/diag/$1.hr2dev.log)"
  log "$1 hr2-dev: $(tail -1 "$L/diag/$1.hr2dev.log")"
}
hs1() {  # <point>: hs1-dev against 4b-I (report only)
  local out=$L/diag/$1.hs1.json left=$L/$1/hs1-dev/hs1-dev.predictions.jsonl right=$L/$TIER-I/hs1-dev/hs1-dev.predictions.jsonl
  [ "$1" = "$TIER-I" ] || [ -f "$out" ] || [ ! -f "$left" ] || [ ! -f "$right" ] && return 0
  py -m v2.data.hs1.validity --gold "$HS1_GOLD" --left "$left" --left-name "$1" --right "$right" --right-name "$TIER-I" \
    --output "$out" > "$L/diag/$1.hs1.log" 2>&1 || die "$1 hs1-dev score FAILED"
  log "$1 hs1-dev scored (report only)"
}
pair() {  # <tag a1|a1_2>: H9 - control on HT-DEV v2 at one alpha
  local X=$CONTROL
  local out=$L/diag/pair-H9-$X-$1.htdev2.json h=$L/$TIER-H9-$1/ht-dev2/ht-dev2.predictions.jsonl c=$L/$TIER-$X-$1/ht-dev2/ht-dev2.predictions.jsonl
  [ -f "$out" ] || [ ! -f "$h" ] || [ ! -f "$c" ] && return 0
  py "$OPS/m9/m9_rules.py" htdev2 --gold "$HT_GOLD" --left "$h" --left-name "$TIER-H9-$1" --right "$c" \
    --right-name "$TIER-$X-$1" --output "$out" > "$L/diag/pair-H9-$X-$1.log" 2>&1 || die "pair $1 FAILED"
  log "pair H9 - $X $1: $(tail -1 "$L/diag/pair-H9-$X-$1.log")"
}
arm_spec() { echo "$1=$L/$1/dev/dev.predictions.jsonl,$L/$1/css-pilot/css-pilot.predictions.jsonl"; }
dev_readout() {  # <line>: typed DEV + CSS pilot readout of the line's points against 4b-I
  local line=$1 X=${1#L-} out=$L/readout/$1.json args=() p points=()
  for p in "$TIER-$X-a1" "$TIER-$X-a1_2"; do
    [ -f "$L/$p/dev/dev.predictions.jsonl" ] && [ -f "$L/$p/css-pilot/css-pilot.predictions.jsonl" ] && points+=("$p")
  done
  [ ${#points[@]} = 2 ] || return 0
  [ -f "$out" ] && return 0
  args=(--arm "$(arm_spec "$TIER-I")")
  for p in "${points[@]}"; do args+=(--arm "$(arm_spec "$p")" --compare "$TIER-I:$p"); done
  py -m v2.dec.dev_readout --typed-gold "$GOLD/typed-dev.gold.jsonl" --css-gold "$GOLD/css-pilot.gold.jsonl" "${args[@]}" \
    --output "$out" > "$L/readout/$line.log" 2>&1 || die "$line dev_readout FAILED (see $L/readout/$line.log)"
  printf '%s\n' "$line:1=$TIER-$X-a1,1/2=$TIER-$X-a1_2" > "$L/readout/$line.line"
  log "$line readout: $(sha "$out")"
}
score_all() {
  [ "$(sha "$HT_GOLD")" = "$HT_GOLD_SHA" ] || die "HT-DEV v2 gold is not $HT_GOLD_SHA"
  local P p
  for P in "$L"/"$TIER"-*/; do
    p=$(basename "$P")
    [ "$p" = "$TIER-I" ] || ht "$p"
    s5 "$p"
    hr2 "$p"
    hs1 "$p"
  done
  pair a1
  pair a1_2
  dev_readout L-H9
  dev_readout "L-$CONTROL"
}

parity() {  # 4b-I on the M9 path vs the stored node-B readouts (report only)
  local panel stored out
  for panel in dev css-pilot ht-dev2 hs1-dev score5t-dev; do
    case $panel in
      score5t-dev) stored=$M/parity/m8/score5t-dev/score5t-dev.predictions.jsonl ;;
      *) stored=$M/parity/m7/4b-I/$panel/$panel.predictions.jsonl ;;
    esac
    out=$L/parity/$panel.json
    [ -f "$out" ] || [ ! -f "$stored" ] || [ ! -f "$L/$TIER-I/$panel/$panel.predictions.jsonl" ] && continue
    python3 - "$L/$TIER-I/$panel/$panel.predictions.jsonl" "$stored" "$panel" > "$out" <<'EOF'
import json, sys
def load(p):
    return {r["id"]: r for r in (json.loads(x) for x in open(p) if x.strip())}
a, b = load(sys.argv[1]), load(sys.argv[2])
def point(ans):
    if not isinstance(ans, dict) or "error" in ans:
        return None, {}
    if "noul" in ans:
        return ans["noul"] > 0.5, {"true": ans["noul"]}
    probs = ans.get("probabilities") or {}
    return ans.get("choice", ans.get("score")), probs
diff, drift, n = 0, 0.0, 0
for i in sorted(set(a) | set(b)):
    qa, qb = (a.get(i) or {}).get("answers", {}), (b.get(i) or {}).get("answers", {})
    for q in sorted(set(qa) | set(qb)):
        n += 1
        pa, pra = point(qa.get(q))
        pb, prb = point(qb.get(q))
        diff += pa != pb
        for k in set(pra) | set(prb):
            drift = max(drift, abs(float(pra.get(k, 0)) - float(prb.get(k, 0))))
print(json.dumps({"panel": sys.argv[3], "m9_path": sys.argv[1], "stored_node_b": sys.argv[2], "answers": n,
                  "answer_differences": diff, "max_probability_drift": drift,
                  "ids_only_one_side": len(set(a) ^ set(b))}, indent=1))
EOF
    log "parity $panel: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["answer_differences"], "of", d["answers"], "differ; drift", d["max_probability_drift"])' "$out")"
  done
  out=$L/parity/ht-dev2-vs-eval-reference.json
  if [ ! -f "$out" ] && [ -f "$L/$TIER-I/ht-dev2/ht-dev2.predictions.jsonl" ]; then
    py "$OPS/m9/m9_rules.py" htdev2 --gold "$HT_GOLD" --left "$L/$TIER-I/ht-dev2/ht-dev2.predictions.jsonl" \
      --left-name "$TIER-I-m9" --right "$EVAL_HT" --right-name eval-dec-m4-N4XF-soup --output "$out" \
      > "$L/parity/ht-dev2-vs-eval-reference.log" 2>&1 || die "eval-reference parity FAILED"
    log "parity ht-dev2 vs the eval reference: $(tail -1 "$L/parity/ht-dev2-vs-eval-reference.log")"
  fi
}

ref_point() {
  [ -f "$L/$TIER-I/weights.json" ] && return 0
  wait_lock "$L/ref.lock" "$L/$TIER-I/weights.json" || return 0
  if [ ! -f "$L/$TIER-I/weights.json" ]; then
    [ "$(tree_manifest "$I" | sha256sum | cut -d' ' -f1)" = "$I_LIST" ] || die "incumbent $I differs from its list $I_LIST"
    alias_point "$TIER-I" ref ref 0 I "$I"
  fi
  rmdir "$L/ref.lock"
}

seed_point() {  # <ARM>: 4b-<ARM>-s1 = BEST of m9-<ARM>-s1
  local g=$1 run=$M/arms/full/m9-$1-s1 best
  [ -f "$M/status/m9-$g-s1.DONE" ] || die "m9-$g-s1 is not DONE"
  best=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['checkpoint'])" "$run/BEST.json") || die "no BEST for m9-$g-s1"
  alias_point "$TIER-$g-s1" "E1" seed 1 "$g-s1" "$run/$best"
}

do_e1() {
  local out=$M/early/E1.json lock=$M/early/E1.lock
  [ -f "$out" ] && return 0
  for f in "$L/$TIER-H9-s1/ht-dev2/ht-dev2.predictions.jsonl" "$L/$TIER-C9-s1/ht-dev2/ht-dev2.predictions.jsonl" \
    "$L/$TIER-I/ht-dev2/ht-dev2.predictions.jsonl"; do
    [ -f "$f" ] || { echo "E1 waits for $f"; return 0; }
  done
  mkdir "$lock" 2> /dev/null || return 0
  LOCKS+=("$lock")
  py "$OPS/m9/m9_rules.py" early --h-run "$M/arms/full/m9-H9-s1" --c-run "$M/arms/full/m9-C9-s1" --gold "$HT_GOLD" \
    --h-ht "$L/$TIER-H9-s1/ht-dev2/ht-dev2.predictions.jsonl" --c-ht "$L/$TIER-C9-s1/ht-dev2/ht-dev2.predictions.jsonl" \
    --i-ht "$L/$TIER-I/ht-dev2/ht-dev2.predictions.jsonl" --output "$out" > "$M/early/E1.log" 2>&1 \
    || die "E1 rule FAILED (see $M/early/E1.log)"
  log "E1: $(tail -1 "$M/early/E1.log")"
}

control_ready() {  # amendment 1: M7's N7C soup copied to node A, checked against its per-file list
  [ -f "$N7C_DIR/VERIFIED" ] && return 0
  [ -f "$N7C_SOUP/decision_config.json" ] || die "no N7C soup at $N7C_SOUP (copy it from node B over the direct link)"
  [ "$(tree_manifest "$N7C_SOUP" | sha256sum | cut -d' ' -f1)" = "$N7C_LIST" ] || die "N7C soup differs from its list $N7C_LIST"
  printf 'soup=%s\nsha256_list_sha256=%s\nverified_utc=%s\nsource=node B /data/dev2/runs/dec/m7/soup/N7C/build/N7C-soup\n' \
    "$N7C_SOUP" "$N7C_LIST" "$(date -u +%FT%TZ)" > "$N7C_DIR/VERIFIED"
  log "N7C soup verified on node A (list $N7C_LIST)"
}

do_line() {
  local X=$1 art
  case $X in
    H9 | C9)
      [ ! -f "$M/status/$X.FAILED" ] || die "L-$X: arm soup FAILED ($(cat "$M/status/$X.FAILED")); the line is empty"
      [ -f "$M/status/$X.DONE" ] || die "L-$X: arm soup not done yet"
      art=$M/soup/$X/build/$X-soup ;;
    N7C) control_ready; art=$N7C_SOUP ;;
    *) die "unknown arm $X" ;;
  esac
  [ -f "$art/decision_config.json" ] || die "L-$X: artifact $art missing"
  ref_point
  mkdir "$L/L-$X.lock" 2> /dev/null || die "L-$X already running"
  LOCKS+=("$L/L-$X.lock")
  log "L-$X start: artifact $art"
  alias_point "$TIER-$X-a1" "L-$X" alpha 1 A "$art"
  build "$TIER-$X-a1_2" "L-$X" 1/2 "I=$I" "A=$art" -- I A
  if [ "$X" = N7C ]; then
    log "4b-N7C-a1_2 file list $(sha "$L/$TIER-N7C-a1_2/files.sha256") (M7 finalist 4b-N7C-b1_2: $N7C_HALF_LIST)"
  fi
  for p in "$TIER-$X-a1" "$TIER-$X-a1_2"; do readout "$p"; done
  infer "$TIER-$X-a1" hs1-dev
  score_all
  log "L-$X done"
}

case $CMD in
  refs)
    ref_point
    for panel in ht-dev2 dev css-pilot score5t-dev hr2-dev hs1-dev; do infer "$TIER-I" "$panel"; done
    score_all
    parity
    ;;
  early)
    [ $# -eq 2 ] || { sed -n '2,24p' "$0"; exit 2; }
    ref_point
    infer "$TIER-I" ht-dev2
    seed_point "$2"
    infer "$TIER-$2-s1" ht-dev2
    ht "$TIER-$2-s1"
    ;;
  e1) do_e1 ;;
  line) [ $# -eq 2 ] || { sed -n '2,24p' "$0"; exit 2; }; do_line "$2" ;;
  score) score_all ;;
  control) control_ready ;;
  rules)
    py "$OPS/m9/m9_rules.py" gates --lines-root "$L" --control "$CONTROL" --output "$M/select/4b-pick.json" \
      > "$M/select/rules.log" 2>&1 || die "rules FAILED (see $M/select/rules.log)"
    log "pick: $(tail -1 "$M/select/rules.log")"
    ;;
  status)
    for P in "$L"/"$TIER"-*/; do
      p=$(basename "$P")
      printf '%-16s' "$p"
      for panel in dev css-pilot ht-dev2 score5t-dev hr2-dev hs1-dev; do
        printf ' %s=%s' "$panel" "$([ -f "$P/$panel/$panel.predictions.jsonl" ] && echo y || echo n)"
      done
      echo
    done
    [ -f "$L/GPU-SECONDS.jsonl" ] && python3 -c 'import json,sys; r=[json.loads(l) for l in open(sys.argv[1])]; print(f"{len(r)} GPU jobs, {sum(x[\"gpu_seconds\"] for x in r)/3600:.3f} GPU-h")' "$L/GPU-SECONDS.jsonl"
    ;;
  *) sed -n '2,24p' "$0"; exit 2 ;;
esac
