#!/usr/bin/env bash
# Own-Lux wave h-w1 (prereg m3b-lux-h-prereg-2026-09-29.md), node B GPU7: the H7 / H8 wave, then its
# 256-prompt repeat run in a fresh process, with the Milestone 2 launcher (teach.sh lux: image,
# mirror, node-B Triton cache of own-Lux waves 1-4 and of the XL waves).
#
# Usage: luxxl_gap_queue.sh SHA256_WAVE SHA256_REPEAT
#
# Starts once both prompt files have the expected SHA-256 under
# /data/dev2/private/data/teachers-v2/m3b/. Nothing starts unless the image, the launcher and the
# Triton cache (file count, tree digest) equal those of the earlier XL waves; the cache state is
# appended to /data/dev2/runs/data/m3b-lux/lux-xl-h.triton-cache.jsonl before the wave, after it
# and after the repeat run. Appends LUX_XL_H_W1_DONE, LUX_XL_H_R256_DONE and LUX_XL_H_DONE to
# teach.log, or LUX_XL_H_STOPPED <reason> on the first failed check. Writes only under /data.
set -u
[[ $# -eq 2 ]] || { sed -n '2,13p' "$0" >&2; exit 2; }
export TMPDIR=/data/dev2/tmp
P=/data/dev2/private/data/teachers-v2/m3b
R=/data/dev2/runs/data/m3b-lux
L=/data/dev2/logs/data
TC=/data/dev2/runs/data/triton-cache-lux-nodeB
IMAGE=sha256:ce895822fc48bb6864911d4488a3946f3a18fd3dd2ec90c8a0a49b259145f2fb
LAUNCHER=49e50c6e5a257716ddc1e0123cadbdb475d0355147b65ec0c20f578a4b986598
CACHE_FILES=1762
CACHE_TREE=299151ab4ce121459dac89670b0b756a0e880b9f451908886f8bcc744f0b28d7
mkdir -p "$R" "$TMPDIR"

stop() { echo "LUX_XL_H_STOPPED $1" >> "$L/teach.log"; exit 1; }
cache_ok() {
  local n t
  n=$(cd "$TC" && find . -type f | wc -l)
  t=$(cd "$TC" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
  printf '{"stage":"%s","files":%s,"tree_sha256":"%s","utc":"%s"}\n' "$1" "$n" "$t" \
    "$(date -u +%FT%TZ)" >> "$R/lux-xl-h.triton-cache.jsonl"
  [[ "$n" == "$CACHE_FILES" && "$t" == "$CACHE_TREE" ]]
}
wait_for() { # name expected-sha256
  local f=$P/$1.prompts.jsonl
  until [[ -f "$f" && "$(sha256sum "$f" | cut -d' ' -f1)" == "$2" ]]; do sleep 60; done
}
teach() { # name expected-sha256
  local f=$P/$1.prompts.jsonl
  wait_for "$1" "$2"
  "$L/teach.sh" lux "$f" "$R/$1.jsonl"
  grep -F "\"input\":\"$f\"" "$L/teach.log" | tail -1 | grep -q '"rc":0,' || stop "rc_$1"
}

wait_for lux-xl-h-w1 "$1"
wait_for lux-xl-h-w1-r256 "$2"
[[ "$(docker image inspect --format '{{.Id}}' decision20-lux-runtime:latest)" == "$IMAGE" ]] || stop image
[[ "$(sha256sum "$L/teach.sh" | cut -d' ' -f1)" == "$LAUNCHER" ]] || stop launcher
cache_ok before || stop cache_before
teach lux-xl-h-w1 "$1"
cache_ok after_wave || stop cache_after_wave
echo LUX_XL_H_W1_DONE >> "$L/teach.log"
teach lux-xl-h-w1-r256 "$2"
cache_ok after_repeat || stop cache_after_repeat
echo LUX_XL_H_R256_DONE >> "$L/teach.log"
echo LUX_XL_H_DONE >> "$L/teach.log"
