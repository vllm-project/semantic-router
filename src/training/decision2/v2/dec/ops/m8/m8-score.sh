#!/usr/bin/env bash
# Decoder M8 node-A scoring (prereg dec-m8-prereg-2026-09-30.md, "Lines and the development rule"), CPU only, from an
# exact mirror. Inputs are the node-B readouts relayed by m8-relay.sh lines (same paths under m8/lines/4b).
#
#   m8-score.sh points <point> [<point> ...]   HT-DEV v2 vs 4b-I (ops/m7/m7_htdev2.py: FLAG / TIE / GAIN at +-0.02)
#                                              -> diag/<point>.htdev2.json; Score5-typed-DEV blocks (m8_rules.py
#                                              score5t) -> diag/<point>.score5t.json (also for 4b-I)
#   m8-score.sh rules [--dropped L-X=why ...]  m8_rules.py alpha -> m8/select/4b-finalists.json
set -u
S=$(cd "$(dirname "$0")/../../../.." && pwd)
MIRROR=${S%/src/training/decision2}
[ -f "$MIRROR/.dev2-mirror.json" ] || { echo "run from an exact mirror under /data/dev2/src" >&2; exit 2; }
M=/data/dev2/runs/dec/m8
L=$M/lines/4b
OPS=$S/v2/dec/ops
PANELS=/data/dev2/private/panels
HT_GOLD=$PANELS/gold/ht-dev2.gold.jsonl HT_GOLD_SHA=659c92b45d2a5b37e250ccb728cdbf5580ba361b3fc4b2f2ab505a13e0a556cc
log() { echo "$(date -u +%FT%TZ) score $*" | tee -a "$L/OPERATIONS-nodeA.log"; }
die() { log "$*"; exit 1; }
py() { (cd "$S" && PYTHONPATH=$S python3 -B "$@"); }
mkdir -p "$L/diag" "$M/select"

s5() {  # <point>
  local out=$L/diag/$1.score5t.json preds=$L/$1/score5t-dev/score5t-dev.predictions.jsonl
  [ -f "$out" ] && return 0
  [ -f "$preds" ] || die "$1: no Score5-typed-DEV predictions (relay them)"
  py "$OPS/m8/m8_rules.py" score5t --panel-root "$PANELS" --predictions "$preds" --output "$out" > "$L/diag/$1.score5t.log" 2>&1 \
    || die "$1 Score5-typed-DEV score FAILED (see $L/diag/$1.score5t.log)"
  log "$1 score5t: $(tail -1 "$L/diag/$1.score5t.log")"
}

ht() {  # <point>
  local out=$L/diag/$1.htdev2.json left=$L/$1/ht-dev2/ht-dev2.predictions.jsonl right=$L/4b-I/ht-dev2/ht-dev2.predictions.jsonl
  [ -f "$out" ] && return 0
  for f in "$left" "$right"; do [ -f "$f" ] || die "missing $f (relay it)"; done
  py "$OPS/m7/m7_htdev2.py" --gold "$HT_GOLD" --left "$left" --left-name "$1" --right "$right" --right-name 4b-I \
    --output "$out" > "$L/diag/$1.htdev2.log" 2>&1 || die "$1 HT-DEV v2 score FAILED (see $L/diag/$1.htdev2.log)"
  log "$1 ht-dev2 vs 4b-I: $(tail -1 "$L/diag/$1.htdev2.log")"
}

case ${1:-} in
  points)
    shift
    [ "$(sha256sum "$HT_GOLD" | cut -d' ' -f1)" = "$HT_GOLD_SHA" ] || die "HT-DEV v2 gold is not $HT_GOLD_SHA"
    s5 4b-I
    for p in "$@"; do
      [ "$p" = 4b-I ] && continue
      ht "$p"
      s5 "$p"
    done
    ;;
  rules)
    shift
    [ ! -e "$M/select/4b-finalists.json" ] || die "4b-finalists.json exists; the rules run once"
    py "$OPS/m8/m8_rules.py" alpha --lines-root "$L" --output "$M/select/4b-finalists.json" "$@" > "$M/select/rules.log" 2>&1 \
      || die "rules FAILED (see $M/select/rules.log)"
    log "finalists: $(tail -1 "$M/select/rules.log")"
    ;;
  *) sed -n '2,9p' "$0"; exit 2 ;;
esac
