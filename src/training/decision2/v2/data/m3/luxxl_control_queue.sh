#!/usr/bin/env bash
# M3b amendment 3 §4, node B GPU7: own-Lux on the XL control-only waves, one at a time.
#
# Usage: luxxl_control_queue.sh SHA256_C_W1 SHA256_C_W2
#
# Starts after LUX_XL_DONE (waves w1-w5) is in teach.log. Each wave starts once its prompt file
# under /data/dev2/private/data/teachers-v2/m3b/ has the expected SHA-256, and runs with the
# Milestone 2 launcher (teach.sh lux: image, mirror, node-B Triton cache of own-Lux waves 1-4).
# Appends LUX_XL_C_W<k>_DONE to teach.log after each wave, then LUX_XL_C_DONE. Writes only
# under /data.
set -u
[[ $# -eq 2 ]] || { sed -n '2,10p' "$0" >&2; exit 2; }
P=/data/dev2/private/data/teachers-v2/m3b
R=/data/dev2/runs/data/m3b-lux
L=/data/dev2/logs/data
declare -A SHA=([1]=$1 [2]=$2)
mkdir -p "$R"
until grep -qx LUX_XL_DONE "$L/teach.log"; do sleep 60; done
for k in 1 2; do
  f=$P/lux-xl-c-w$k.prompts.jsonl
  until [[ -f "$f" && "$(sha256sum "$f" | cut -d' ' -f1)" == "${SHA[$k]}" ]]; do sleep 60; done
  "$L/teach.sh" lux "$f" "$R/lux-xl-c-w$k.jsonl"
  echo "LUX_XL_C_W${k}_DONE" >> "$L/teach.log"
done
echo LUX_XL_C_DONE >> "$L/teach.log"
