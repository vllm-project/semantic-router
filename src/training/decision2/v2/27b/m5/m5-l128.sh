#!/usr/bin/env bash
# ~27B M5 branch B1 only (preregistration "Early-stop and branch rules"): one L128 arm-seed, M4's LoRA template
# (run_lora_arm.sh: stages admit, onestep, reload, full) at rank 128 / alpha 256 on the a20 mixture (M5 copy, SHA-256
# checked), the training cache seeded from the frozen T0, SAVE_EVERY 446 (M4's a20 cadence) and a 12 GPU-hour cap per
# arm-seed. Refuses to start unless /data/dev2/runs/27b/m5/BRANCH-B1 exists (written when B1 triggers). Detached.
# Usage: m5-l128.sh NODE GPU SEED       (NODE a|b; GPU b:5-7 / a:2-4; SEED s1 | s2)
set -euo pipefail
NODE=${1:?NODE} GPU=${2:?GPU} SEED=${3:?SEED}
S=$(cd "$(dirname "$0")/../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
[ -f /data/dev2/runs/27b/m5/BRANCH-B1 ] || { echo "branch B1 has not triggered (no /data/dev2/runs/27b/m5/BRANCH-B1)" >&2; exit 2; }
case "$NODE:$GPU" in b:5 | b:6 | b:7 | a:2 | a:3 | a:4) ;; *) echo "node $NODE GPU$GPU is outside M5's lanes" >&2; exit 2 ;; esac
case "$SEED" in s1) SEED_VALUE=20260926 ;; s2) SEED_VALUE=20260928 ;; *) echo "SEED is s1 or s2" >&2; exit 2 ;; esac
ARM=M5-L128-$SEED
MIX=/data/dev2/private/27b/m5-data/mixtures-m5-1/a20.train.jsonl
MIX_SHA=4aa0dc964505682b2840fa5167a7ec14983e5a0cea427480bed1996f1befc0d4
T0=/data/dev2/runs/27b/m4-train-cache-T0 T0_SHA=1933eb36d746a3bf3b716967ce97f977620eab0f7a390f751e79bfd4f8b2e21f
BASE=/data/decision20-20260926/models/Qwen3.8-27B REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
[ "$(sha256sum < "$MIX" | cut -c1-64)" = "$MIX_SHA" ] || { echo "a20.train.jsonl changed" >&2; exit 2; }
[ ! -e "/data/dev2/runs/27b/$ARM" ] || { echo "/data/dev2/runs/27b/$ARM exists: one attempt per arm-seed" >&2; exit 66; }
export TMPDIR=/data/dev2/tmp DEV2_NODE=$NODE
mkdir -p "/data/dev2/runs/27b/$ARM"
cd "$S"
echo "=== $(date -u +%FT%TZ) $ARM node $NODE GPU$GPU a20 ($MIX_SHA) seed $SEED_VALUE rank 128 alpha 256 save-every 446" \
  "mirror $SRC" >> "/data/dev2/runs/27b/$ARM/driver.log"
TRAIN_FILE=$MIX SAVE_EVERY=446 ARM_CAP=12.0 FULL_CAP=12.0 LORA_RANK=128 LORA_ALPHA=256 \
TRAIN_CACHE_FROZEN=$T0 TRAIN_CACHE_SHA=$T0_SHA TRITON_AUTOTUNE_CACHE=1 \
TRAIN_PYTHONPATH=/pipeline:/code:/opt/decision-fla STAGES=admit,onestep,reload,full \
setsid nohup bash v2/27b/run_lora_arm.sh "$ARM" "$GPU" "$BASE" . "$REV" "$SEED_VALUE" "$SRC" \
  >> "/data/dev2/runs/27b/$ARM/driver.log" 2>&1 < /dev/null &
echo "launched $ARM on node $NODE GPU$GPU (pid $!)"
