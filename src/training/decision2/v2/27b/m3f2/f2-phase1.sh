#!/usr/bin/env bash
# ~27B M3 F2 completion, phase 1 (node B host; preregistered steps that never ran after the 14:25 broken chain).
#   f2-phase1.sh s2    M3-S-s2 kernel readout at 32,768 on GPU7 (the exact F1 readout queue m3-queue.sh)
#   f2-phase1.sh soup  M3-S soup (CPU container, run_finalist.sh STAGES=soup), then its kernel readout on GPU5
# Same mirror (35fa052d2) and frozen cache (583241fb, run_finalist.sh / m3-queue.sh defaults) as F1's stages.
set -uo pipefail
L=/data/dev2/runs/27b/m3-logs
R=/data/dev2/runs/27b
SRC=35fa052d2b7c3ad0f9b2ee9bd1529e6b28076029-src_training_decision2
S=/data/dev2/src/$SRC/src/training/decision2
export TMPDIR=/data/dev2/tmp PYTHONDONTWRITEBYTECODE=1
case "${1:-}" in
  s2)
    exec bash "$L/m3-queue.sh" 7 readout:M3-S-s2:20260928
    ;;
  soup)
    echo "=== $(date -u +%FT%TZ) soup M3-S-soup (members M3-S-s1, M3-S-s2)"
    (cd "$S" && STAGES=soup bash v2/27b/run_finalist.sh M3-S-soup 5 "$SRC" M3-S-s1 M3-S-s2) \
      >> "$R/M3-S-soup.soup.log" 2>&1
    rc=$?
    echo "=== $(date -u +%FT%TZ) soup exit=$rc"
    if [ "$rc" != 0 ] || [ ! -f "$R/M3-S-soup/soup/checkpoint/soup_manifest.json" ]; then
      echo "=== soup failed or has no soup_manifest.json; soup readout not started"
      exit 1
    fi
    exec bash "$L/m3-queue.sh" 5 soupreadout:M3-S-soup:M3-S-s1,M3-S-s2
    ;;
  *)
    echo "usage: f2-phase1.sh s2|soup" >&2
    exit 2
    ;;
esac
