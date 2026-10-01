#!/usr/bin/env bash
# ~27B M6 guard for the shared gates directory on node B (node side; detached). The four chains share
# /data/dev2/runs/27b/m6/gates, and each chain's `m6-gates.sh gates` ends with m4_contrast --output gates/contrast.json,
# which refuses an existing file: the second chain with a sealed finalist would stop there, before overlap and
# verdicts. While any listed chain PID is alive, this moves every complete contrast.json (it parses as JSON) to
# contrast-<finalists>-<UTC>.json within 5 s of its writing, and copies contrast.log beside it 15 s later. Nothing
# reads contrast.json downstream (the verdicts read gates/<NAME>/, gates/mlx and gates/overlap); the contrasts over
# all finalists are a separate final run.
# Usage: m6-contrast-guard.sh CHAIN_PID...
set -euo pipefail
G=/data/dev2/runs/27b/m6/gates
[ $# -ge 1 ] || { echo "at least one chain PID" >&2; exit 2; }
for p in "$@"; do [[ "$p" =~ ^[0-9]+$ ]] || { echo "bad PID $p" >&2; exit 2; }; done
alive() {
  local p
  for p in "$@"; do kill -0 "$p" 2> /dev/null && return 0; done
  return 1
}
names() {  # contrast JSON -> its finalists joined by +, or an empty string when it is not complete JSON
  python3 -c 'import json,sys; print("+".join(sorted(json.load(open(sys.argv[1]))["finalists"])))' "$1" 2> /dev/null
}
echo "m6 contrast guard $*: start $(date -u +%FT%TZ)"
while alive "$@"; do
  if [ -f "$G/contrast.json" ] && n=$(names "$G/contrast.json") && [ -n "$n" ]; then
    out=contrast-$n-$(date -u +%Y%m%dT%H%M%SZ)
    mv -n "$G/contrast.json" "$G/$out.json"
    echo "$(date -u +%FT%TZ) gates/contrast.json -> $out.json"
    sleep 15
    [ ! -f "$G/contrast.log" ] || cp -p "$G/contrast.log" "$G/$out.log"
  fi
  sleep 5
done
echo "m6 contrast guard complete (no listed chain alive): $(date -u +%FT%TZ)"
