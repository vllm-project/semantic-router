#!/usr/bin/env bash
# usage: lux_readout.sh SHA GPU [NAME]
# Same-runtime Lux 1.0 reference: the Lux full checkpoint /m3/pf-D-s1-zero/run/checkpoint-0000000
# read exactly like a soup (readout.sh: CAL698 via v2.dec.calibrate_ckpt, typed DEV + CSS pilot
# at 16,384 tokens) into /data/dev2/runs/9b/m4/NAME-* (default NAME lux-16k).
set -uo pipefail
sha=$1; gpu=$2; name=${3:-lux-16k}
exec "/data/dev2/src/$sha-src_training_decision2/src/training/decision2/v2/9b/lux9b/m4/readout.sh" \
  "$sha" "$gpu" "$name" /m3/pf-D-s1-zero/run/checkpoint-0000000
