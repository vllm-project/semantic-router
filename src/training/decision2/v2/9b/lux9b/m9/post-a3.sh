#!/usr/bin/env bash
# 9B M9 stage-3 post-training chain on node A for one arm (amendment 3). Waits for node C's arm soup (soup/<ARM>/DONE),
# pulls it over the temporary transfer key (content manifest checked on both sides) and builds the development point
# the way K-a13 was built from the K soup: the uniform FP32 soup of [arm soup, Lux 1.0, Lux 1.0] (alpha 1/3 toward the
# arm soup) with K-a13's own checkpoint-form Lux 1.0 (m3/pf-D-s1-zero/run/checkpoint-0000000), on CPU, recorded in
# soup/<POINT>/DONE and <point>.built.json. Then it reads every development panel plus the IB1 / IB2 DEV diagnostics
# on this GPU (lines.sh), scores the point against C0 (score.sh points C0 <POINT>), marks status/scored-<POINT> and
# reads the IB DEV panels of the arm soup itself (alpha 1; report only). The chain that finds both points scored (or
# failed) runs once (status/rules-s3.lock): C0's IB DEV, the typed readout readout/m9-s3.json (with L9IB / L9IBX when
# read), the contrasts K-a13IB:K-a13IBX, K-a13IB:L9IB, K-a13IBX:L9IBX and the rules (select/9b-finalists-s3.json).
# A failed step stops the chain (never rerun).   KIB -> K-a13IB   KIBX -> K-a13IBX
#
# usage: M9_NODE=a post-a3.sh launch|run <mirror-dir> <ARM> <gpu> <user@node-c>
set -u
MODE=$1 SRC=$2 ARM=$3 GPU=$4 FROM=$5
NODE=${M9_NODE:?set M9_NODE=a}
M=/data/dev2/runs/9b/m9
ST=$M/status
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/9b/lux9b/m9
case $ARM in
  KIB) POINT=K-a13IB ;;
  KIBX) POINT=K-a13IBX ;;
  *) echo "no stage-3 arm $ARM" >&2; exit 2 ;;
esac
POINTS="K-a13IB K-a13IBX" LOCK=rules-s3.lock
LUXCK=/runs/m3/pf-D-s1-zero/run/checkpoint-0000000
PANELS="dev css-pilot ht-dev2 score5t-dev hs1-dev pn1-dev m9-probes mlxdev"
export M9_READOUT=m9-s3 M9_RULES=9b-finalists-s3
mkdir -p "$M/chains" "$M/logs" "$ST"
if [ "$MODE" = launch ]; then
  mkdir "$M/chains/post-a-$ARM.lock" 2> /dev/null || { echo "post chain $ARM already launched"; exit 0; }
  M9_NODE=$NODE setsid nohup bash "$0" run "$SRC" "$ARM" "$GPU" "$FROM" > "$M/logs/post-a-$ARM.log" 2>&1 < /dev/null &
  echo $! > "$M/chains/post-a-$ARM.pid"
  echo "$(date -u +%FT%TZ) M9 stage-3 post chain $ARM ($POINT) launched on node A GPU$GPU from $SRC (pid $(cat "$M/chains/post-a-$ARM.pid"))" \
    | tee -a "$M/OPERATIONS.log"
  exit 0
fi
log() { echo "$(date -u +%FT%TZ) post-a-$ARM $*" | tee -a "$M/OPERATIONS.log"; }
on() { ssh -i /root/.ssh/d2_temp_cd -o BatchMode=yes "$FROM" "$@"; }
manifest='find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -d" " -f1'
finish() {  # both points scored or failed -> C0 IB DEV, typed readout, contrasts, rules (once)
  local p q pts=() scored=() extra=() cons=()
  for p in $POINTS; do [ -f "$ST/scored-$p" ] || [ -f "$ST/failed-$p" ] || return 0; done
  mkdir "$ST/$LOCK" 2> /dev/null || return 0
  for p in $POINTS; do [ -f "$ST/scored-$p" ] && scored+=("$p") && pts+=("$p=C0"); done
  [ ${#scored[@]} -gt 0 ] || { log "no stage-3 point was scored; no rules"; return 0; }
  M9_NODE=$NODE bash "$OPS/lines.sh" read "$SRC" "$GPU" C0 /data/dev2/runs/9b/m4/K-a13-build/soup lux ib1dev ib2dev \
    || log "C0 IB DEV diagnostics failed (report only)"
  for q in L9IB L9IBX; do
    [ -s "$M/lines/$q/dev/dev.predictions.jsonl" ] && [ -s "$M/lines/$q/css-pilot/css-pilot.predictions.jsonl" ] \
      && extra+=("$q")
  done
  bash "$OPS/score.sh" readout C0 "${scored[@]}" "${extra[@]}" || { log "typed readout failed"; return 1; }
  [ ${#scored[@]} = 2 ] && bash "$OPS/score.sh" contrast K-a13IB K-a13IBX && cons+=(K-a13IB:K-a13IBX)
  for p in "${scored[@]}"; do
    q=L9${p#K-a13}
    [[ " ${extra[*]} " == *" $q "* ]] && bash "$OPS/score.sh" contrast "$p" "$q" && cons+=("$p:$q")
  done
  bash "$OPS/score.sh" rules "${pts[@]}" -- "${cons[@]}"
  log "stage-3 rules ran: $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["finalists"])' "$M/select/9b-finalists-s3.json" 2> /dev/null)"
}
fail() { echo "$*" > "$ST/failed-$POINT"; log "FAILED: $*"; finish; exit 1; }

n=0
log "waiting for node C's $ARM soup"
until on "test -f $M/soup/$ARM/DONE -o -f $M/soup/$ARM/FAILED"; do
  n=$((n + 1))
  [ $((n % 15)) = 0 ] && log "still waiting for node C's $ARM soup"
  sleep 120
done
on "test -f $M/soup/$ARM/DONE" || fail "node C soup of $ARM failed"
art=$(on "cat $M/soup/$ARM/DONE")
case $art in "$M"/soup/"$ARM"/build/"$ARM"-soup | "$M"/arms/full/m9-"$ARM"-s[123]/checkpoint-*) ;;
  *) fail "unexpected artifact path $art" ;; esac
if [ ! -f "$art.pulled.json" ]; then
  mb=$(on "cd '$art' && $manifest")
  mkdir -p "$art"
  rsync -a -e 'ssh -i /root/.ssh/d2_temp_cd -o BatchMode=yes' "$FROM:$art/" "$art/" || fail "pull of $art failed"
  ma=$(cd "$art" && eval "$manifest")
  [ "$ma" = "$mb" ] || fail "$art manifest differs after the pull ($mb vs $ma)"
  rsync -a -e 'ssh -i /root/.ssh/d2_temp_cd -o BatchMode=yes' --exclude '*.safetensors' "$FROM:$M/soup/$ARM/" "$M/soup/$ARM/" \
    || log "side files of soup/$ARM not pulled (report only)"
  printf '{"artifact": "%s", "content_manifest": "%s", "pulled_utc": "%s"}\n' "$art" "$ma" "$(date -u +%FT%TZ)" \
    > "$art.pulled.json"
  log "pulled $art ($ma)"
fi
P=$M/soup/$POINT
if [ ! -f "$P/DONE" ]; then
  [ -e "$P/build" ] && fail "a partial build of $POINT exists; not rebuilt"
  mkdir -p "$P"
  M9_NODE=$NODE bash "$OPS/launch.sh" "soup-$POINT" "$SRC" "$P/build" --cpu -- -m v2.dec.soup \
    --member "/runs/${art#/data/dev2/runs/9b/}" --member "$LUXCK" --member "$LUXCK" --output "/out/$POINT" \
    || fail "the alpha 1/3 soup of $POINT failed (see $P/build.stderr.log)"
  pt=$P/build/$POINT
  python3 - "$pt" "$POINT" "$art" "/data/dev2/runs/9b/${LUXCK#/runs/}" "$(cd "$pt" && eval "$manifest")" << 'EOF'
import json, sys, time
pt, point, art, lux, content = sys.argv[1:]
json.dump({
    "point": point,
    "construction": "uniform FP32 soup of [arm soup, Lux 1.0, Lux 1.0]: alpha 1/3 toward the arm soup, as K-a13 = "
                    "[K soup, Lux 1.0, Lux 1.0] (m4/K-a13-build)",
    "members": [art, lux, lux],
    "arm_artifact": json.load(open(art + ".pulled.json")),
    "content_manifest": content,
    "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}, open(pt + ".built.json", "w"), indent=1)
EOF
  echo "$pt" > "$P/DONE"
  log "built $POINT = [$ARM soup, Lux 1.0, Lux 1.0]: $(tail -c 300 "$P/build.stdout.log" | tr '\n' ' ')"
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
M9_NODE=$NODE bash "$OPS/lines.sh" read "$SRC" "$GPU" "$ARM" "$art" lux ib1dev ib2dev \
  || log "$ARM (alpha 1) IB DEV diagnostics failed (report only)"
finish
