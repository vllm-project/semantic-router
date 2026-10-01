#!/usr/bin/env bash
# 9B M9 stage-4 chain on node A (amendment 4): the alpha 1/2 point K-a12IB = the uniform FP32 soup of [KIB soup,
# Lux 1.0] from the inputs stage 3 already holds on node A (soup/KIB/build/KIB-soup, pulled with its content
# manifest; the Lux 1.0 zero-step checkpoint of m9-KIB-s1, byte-identical to K-a13's base), on CPU, recorded in
# soup/K-a12IB/DONE and <point>.built.json. Then the eight development panels plus the IB1 / IB2 DEV diagnostics on
# this GPU (lines.sh), scoring vs C0 (score.sh points C0 K-a12IB), the typed readout readout/m9-s4.json (C0, K-a12IB,
# K-a13IB), the report-only contrast K-a12IB:K-a13IB and the rules (select/9b-finalists-s4.json; m9_rules.py
# unchanged). Runs once (chains/post-a4.lock); a failed step stops the chain (never rerun).
#
# usage: M9_NODE=a post-a4.sh launch|run <mirror-dir> <gpu>
set -u
MODE=$1 SRC=$2 GPU=$3
NODE=${M9_NODE:?set M9_NODE=a}
M=/data/dev2/runs/9b/m9
ST=$M/status
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/9b/lux9b/m9
POINT=K-a12IB
ART=$M/soup/KIB/build/KIB-soup
ARTCK=/runs/m9/soup/KIB/build/KIB-soup
LUXHOST=$M/arms/pre/m9-KIB-s1-zero/checkpoint-0000000
LUXCK=/runs/m9/arms/pre/m9-KIB-s1-zero/checkpoint-0000000
PANELS="dev css-pilot ht-dev2 score5t-dev hs1-dev pn1-dev m9-probes mlxdev"
export M9_READOUT=m9-s4 M9_RULES=9b-finalists-s4
mkdir -p "$M/chains" "$M/logs" "$ST"
case $GPU in 6 | 7) ;; *) echo "GPU $GPU is not an M9 node-A GPU" >&2; exit 2 ;; esac
if [ "$MODE" = launch ]; then
  mkdir "$M/chains/post-a4.lock" 2> /dev/null || { echo "stage-4 chain already launched"; exit 0; }
  M9_NODE=$NODE setsid nohup bash "$0" run "$SRC" "$GPU" > "$M/logs/post-a4.log" 2>&1 < /dev/null &
  echo $! > "$M/chains/post-a4.pid"
  echo "$(date -u +%FT%TZ) M9 stage-4 chain ($POINT) launched on node A GPU$GPU from $SRC (pid $(cat "$M/chains/post-a4.pid"))" \
    | tee -a "$M/OPERATIONS.log"
  exit 0
fi
log() { echo "$(date -u +%FT%TZ) post-a4 $*" | tee -a "$M/OPERATIONS.log"; }
fail() { echo "$*" > "$ST/failed-$POINT"; log "FAILED: $*"; exit 1; }
manifest='find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -d" " -f1'

[ -f "$ART.pulled.json" ] && [ -f "$LUXHOST.pulled.json" ] || fail "stage-3 inputs missing on node A"
want=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_manifest"])' "$ART.pulled.json")
[ "$(cd "$ART" && eval "$manifest")" = "$want" ] || fail "KIB soup content differs from its pull record"
want=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_manifest"])' "$LUXHOST.pulled.json")
[ "$(cd "$LUXHOST" && eval "$manifest")" = "$want" ] || fail "Lux 1.0 member content differs from its pull record"
P=$M/soup/$POINT
if [ ! -f "$P/DONE" ]; then
  [ -e "$P/build" ] && fail "a partial build of $POINT exists; not rebuilt"
  mkdir -p "$P"
  M9_NODE=$NODE bash "$OPS/launch.sh" "soup-$POINT" "$SRC" "$P/build" --cpu -- -m v2.dec.soup \
    --member "$ARTCK" --member "$LUXCK" --output "/out/$POINT" \
    || fail "the alpha 1/2 soup of $POINT failed (see $P/build.stderr.log)"
  pt=$P/build/$POINT
  python3 - "$pt" "$POINT" "$ART" "$LUXHOST" "$(cd "$pt" && eval "$manifest")" << 'EOF'
import json, sys, time
pt, point, art, lux, content = sys.argv[1:]
json.dump({
    "point": point,
    "construction": "uniform FP32 soup of [KIB soup, Lux 1.0]: alpha 1/2 toward the KIB soup (amendment 4); K-a13IB "
                    "is [KIB soup, Lux 1.0, Lux 1.0]",
    "members": [art, lux],
    "arm_artifact": json.load(open(art + ".pulled.json")),
    "lux_member": json.load(open(lux + ".pulled.json")),
    "content_manifest": content,
    "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}, open(pt + ".built.json", "w"), indent=1)
EOF
  echo "$pt" > "$P/DONE"
  log "built $POINT = [KIB soup, Lux 1.0]: $(tail -c 300 "$P/build.stdout.log" | tr '\n' ' ')"
fi
pt=$(cat "$P/DONE")
# shellcheck disable=SC2086
M9_NODE=$NODE bash "$OPS/lines.sh" read "$SRC" "$GPU" "$POINT" "$pt" lux $PANELS ib1dev ib2dev \
  || fail "readouts of $POINT failed"
for panel in $PANELS; do
  grep -qs '"exit_status": 0' "$M/lines/$POINT/$panel.launch.json" || fail "$POINT $panel readout missing or failed"
done
bash "$OPS/score.sh" points C0 "$POINT" || fail "scoring of $POINT failed"
date -u +%FT%TZ > "$ST/scored-$POINT"
log "$POINT scored"
mkdir "$ST/rules-s4.lock" 2> /dev/null || { log "stage-4 rules already ran"; exit 0; }
bash "$OPS/score.sh" readout C0 "$POINT" K-a13IB || fail "typed readout failed"
cons=()
bash "$OPS/score.sh" contrast "$POINT" K-a13IB && cons+=("$POINT:K-a13IB")
bash "$OPS/score.sh" rules "$POINT=C0" -- "${cons[@]}" || fail "rules failed"
log "stage-4 rules ran: $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["finalists"])' "$M/select/9b-finalists-s4.json" 2> /dev/null)"
