#!/usr/bin/env bash
# 9B M9 post-training chain on node A for one arm: wait for node C's arm artifact (soup/<ARM>/DONE), pull it over the
# temporary transfer key (content manifest checked on both sides), read every development panel on this GPU
# (lines.sh read), score it against C0 (score.sh points C0 <ARM>) and mark status/scored-<ARM>. The chain that finds
# every arm scored (or failed) runs the typed readout, the L9 - L9L contrast and the rules once (status/rules.lock).
# A failed step stops the chain (never rerun).
#
# usage: M9_NODE=a post-a.sh launch <mirror-dir> <ARM> <gpu> <user@node-c>
#        M9_NODE=a post-a.sh run <mirror-dir> <ARM> <gpu> <user@node-c>
set -u
MODE=$1 SRC=$2 ARM=$3 GPU=$4 FROM=$5
NODE=${M9_NODE:?set M9_NODE=a}
M=/data/dev2/runs/9b/m9
ST=$M/status
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/9b/lux9b/m9
ARMS="L9 L9L"
mkdir -p "$M/chains" "$M/logs" "$ST"
if [ "$MODE" = launch ]; then
  mkdir "$M/chains/post-a-$ARM.lock" 2> /dev/null || { echo "post chain $ARM already launched"; exit 0; }
  M9_NODE=$NODE setsid nohup bash "$0" run "$SRC" "$ARM" "$GPU" "$FROM" > "$M/logs/post-a-$ARM.log" 2>&1 < /dev/null &
  echo $! > "$M/chains/post-a-$ARM.pid"
  echo "$(date -u +%FT%TZ) M9 post chain $ARM launched on node A GPU$GPU from $SRC (pid $(cat "$M/chains/post-a-$ARM.pid"))" \
    | tee -a "$M/OPERATIONS.log"
  exit 0
fi
log() { echo "$(date -u +%FT%TZ) post-a-$ARM $*" | tee -a "$M/OPERATIONS.log"; }
on() { ssh -i /root/.ssh/d2_temp_cd -o BatchMode=yes "$FROM" "$@"; }
manifest='find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -d" " -f1'
finish() {  # every arm scored or failed -> typed readout, contrast, rules (once)
  local a pts=() scored=()
  for a in $ARMS; do [ -f "$ST/scored-$a" ] || [ -f "$ST/failed-$a" ] || return 0; done
  mkdir "$ST/rules.lock" 2> /dev/null || return 0
  for a in $ARMS; do [ -f "$ST/scored-$a" ] && scored+=("$a") && pts+=("$a=C0"); done
  if [ ! -f "$M/lines/B0/m9-probes/m9-probes.predictions.jsonl" ]; then
    mkdir -p "$M/lines/B0"
    rsync -a -e 'ssh -i /root/.ssh/d2_temp_cd -o BatchMode=yes' --include='*/' --include='*.predictions.jsonl' \
      --include='*.manifest.json' --include='*.launch.json' --exclude='*' "$FROM:$M/lines/B0/" "$M/lines/B0/" \
      || log "B0 predictions not pulled (report only)"
  fi
  bash "$OPS/score.sh" points C0 LUX B0 || log "LUX / B0 report screens failed (report only)"
  [ ${#scored[@]} -gt 0 ] || { log "no arm was scored; no rules"; return 0; }
  bash "$OPS/score.sh" readout C0 "${scored[@]}" LUX || { log "typed readout failed"; return 1; }
  [ ${#scored[@]} = 2 ] && bash "$OPS/score.sh" contrast L9 L9L
  if [ ${#scored[@]} = 2 ]; then
    bash "$OPS/score.sh" rules "${pts[@]}" -- L9:L9L
  else
    bash "$OPS/score.sh" rules "${pts[@]}"
  fi
  log "rules ran: $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["finalists"])' "$M/select/9b-finalists.json" 2> /dev/null)"
}
fail() { echo "$*" > "$ST/failed-$ARM"; log "FAILED: $*"; finish; exit 1; }

n=0
log "waiting for node C's $ARM artifact"
until on "test -f $M/soup/$ARM/DONE -o -f $M/soup/$ARM/FAILED"; do
  n=$((n + 1))
  [ $((n % 15)) = 0 ] && log "still waiting for node C's $ARM artifact"
  sleep 120
done
on "test -f $M/soup/$ARM/DONE" || fail "node C soup of $ARM failed"
art=$(on "cat $M/soup/$ARM/DONE")
case $art in "$M"/soup/"$ARM"/*) ;; *) fail "unexpected artifact path $art" ;; esac
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
case $ARM in
  L9) source=/data/dev2/models/Qwen--Qwen3.5-9B-Base/68c46c4b3498877f3ef123c856ecfde50c39f404 ;;
  *) source=lux ;;
esac
M9_NODE=$NODE bash "$OPS/lines.sh" read "$SRC" "$GPU" "$ARM" "$art" "$source" || fail "readouts of $ARM failed"
for panel in dev css-pilot ht-dev2 score5t-dev hs1-dev pn1-dev m9-probes mlxdev; do
  grep -qs '"exit_status": 0' "$M/lines/$ARM/$panel.launch.json" || fail "$ARM $panel readout missing or failed"
done
c0_ready() {
  local p
  for p in dev css-pilot ht-dev2 score5t-dev hs1-dev pn1-dev m9-probes mlxdev; do
    [ -f "$M/lines/C0/$p.launch.json" ] || return 1
  done
}
until c0_ready; do sleep 60; done
bash "$OPS/score.sh" points C0 "$ARM" || fail "scoring of $ARM failed"
date -u +%FT%TZ > "$ST/scored-$ARM"
log "$ARM scored"
finish
