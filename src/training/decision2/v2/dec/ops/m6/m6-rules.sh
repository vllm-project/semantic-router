#!/usr/bin/env bash
# Decoder M6 selection for one tier (prereg dec-m6-prereg-2026-09-29.md, "Selection rule"), run from an exact
# mirror on the tier's node once every line is read out (m6-lines.sh): the 9B alpha rule (v2/9b/lux9b/m4_rules.py,
# unchanged, --lux <tier>-I, --h-field H_mean) over lines/<tier>/readout/L-*.json, then the slot table ->
# /data/dev2/runs/dec/m6/select/<tier>-finalists.json (+ <tier>-rules.json). CPU only, stdlib only.
#
#   m6-rules.sh <tier 4b|2b|08b> [--dropped L-X=reason ...]
#
# A preregistered line without a readout must be declared dropped (an arm stopped by a stop rule); an L-N6P line
# (PN1 amendment) is used if its readout exists and takes 4B slot 3.
set -u
S=$(cd "$(dirname "$0")/../../../.." && pwd)
MIRROR=${S%/src/training/decision2}
[ -f "$MIRROR/.dev2-mirror.json" ] || { echo "run from an exact mirror under /data/dev2/src" >&2; exit 2; }
TIER=${1:-}
case $TIER in 4b|2b|08b) ;; *) sed -n '2,11p' "$0"; exit 2 ;; esac
shift
M=${M6_ROOT:-/data/dev2/runs/dec/m6}
mkdir -p "$M/select"
python3 -B "$S/v2/dec/ops/m6/m6_finalists.py" --tier "$TIER" --lines-root "$M/lines/$TIER" \
  --rules "$S/v2/9b/lux9b/m4_rules.py" --output "$M/select/$TIER-finalists.json" "$@" \
  || { echo "$(date -u +%FT%TZ) $TIER rules FAILED" >> "$M/select/OPERATIONS.log"; exit 1; }
echo "$(date -u +%FT%TZ) $TIER finalists: $(sha256sum "$M/select/$TIER-finalists.json" | cut -d' ' -f1) (mirror ${MIRROR##*/})" \
  >> "$M/select/OPERATIONS.log"
