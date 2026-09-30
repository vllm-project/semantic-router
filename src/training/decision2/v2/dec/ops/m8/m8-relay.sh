#!/usr/bin/env bash
# Decoder M8 relays through the workstation (small files only; node aliases resolve through
# ${DEV2_NODES_FILE:-~/.config/decision2/nodes.env} as in mirror_to_node.sh). Nothing is overwritten; every file is
# checked by SHA-256 on both sides, directories by a content manifest.
#
#   m8-relay.sh panels          Score5-typed-DEV gold-free prompts: node A's installed panel -> node B
#                               /data/dev2/runs/dec/panels/score5t-dev.prompts.jsonl (the frozen panel hash)
#   m8-relay.sh label-prompts   node-B m8/data/build/label/shard-*.prompts.jsonl + manifest.json -> node-A m8/label/
#                               (+ SHARDS.sha256 written from node B's hashes)
#   m8-relay.sh teacher         node-A m8/teacher-a20r/{D1,D2}/teacher.jsonl + manifest.json -> node B (same paths)
#   m8-relay.sh lines <point> [<point> ...]
#                               node-B m8/lines/4b/<point>/{weights.json,files.sha256} + gold-free predictions of dev,
#                               css-pilot, ht-dev2, score5t-dev -> node A (same paths); readout/*.json|.line too
#   m8-relay.sh select          node-A m8/select/4b-finalists.json -> node B
#   m8-relay.sh gpuh            node-A m8/gpuh-node-a.json -> node B (the milestone-total stop rule reads it)
#   m8-relay.sh early           node-B m8/early/*.json and m8/status/* -> node A m8/relayed-status/ (for records)
set -euo pipefail
nodes_file=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
resolve() { local d=""; [ -f "$nodes_file" ] && d=$(awk -F= -v k="$1" '$1 == k { print substr($0, length(k) + 2); exit }' "$nodes_file"); echo "${d:-$1}"; }
A=$(resolve node-a) B=$(resolve node-b)
on_a() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$A" "$@"; }
on_b() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$B" "$@"; }
R=/data/dev2/runs/dec
M=$R/m8

copy_file() {  # <from a|b> <path> [<dest path>]: no overwrite unless identical, hash-checked
  local from=$1 src=$2 dst=${3:-$2} to h
  to=$([ "$from" = b ] && echo a || echo b)
  h=$("on_$from" "sha256sum '$src' | cut -d' ' -f1")
  if "on_$to" "test -f '$dst'"; then
    [ "$("on_$to" "sha256sum '$dst' | cut -d' ' -f1")" = "$h" ] || { echo "node $to: $dst exists with other content" >&2; exit 1; }
    return 0
  fi
  "on_$from" "cat '$src'" | "on_$to" "set -o noclobber; mkdir -p '$(dirname "$dst")'; cat > '$dst'"
  [ "$("on_$to" "sha256sum '$dst' | cut -d' ' -f1")" = "$h" ] || { echo "node $to: $dst hash differs after the copy" >&2; exit 1; }
  echo "node $from -> $to: $dst ($h)"
}

case ${1:-} in
  panels)
    want=$(cd "$(dirname "$0")/../../../.." && python3 -c 'from v2.eval import panels; print(panels.expected_files()["goldfree/score5t-dev.prompts.jsonl"])')
    [ "$(on_a "sha256sum /data/dev2/private/panels/goldfree/score5t-dev.prompts.jsonl | cut -d' ' -f1")" = "$want" ] \
      || { echo "node A Score5-typed-DEV prompts differ from the frozen panel $want" >&2; exit 1; }
    copy_file a /data/dev2/private/panels/goldfree/score5t-dev.prompts.jsonl "$R/panels/score5t-dev.prompts.jsonl"
    ;;
  label-prompts)
    for f in $(on_b "cd '$M/data/build/label' && ls shard-*.prompts.jsonl"); do
      copy_file b "$M/data/build/label/$f" "$M/label/$f"
    done
    copy_file b "$M/data/build/manifest.json" "$M/label/data-manifest.json"
    on_b "cd '$M/data/build/label' && sha256sum shard-*.prompts.jsonl" | on_a "set -o noclobber; cat > '$M/label/SHARDS.sha256'"
    on_a "cd '$M/label' && sha256sum -c SHARDS.sha256"
    ;;
  teacher)
    for f in D1/teacher.jsonl D2/teacher.jsonl manifest.json; do copy_file a "$M/teacher-a20r/$f"; done
    ;;
  lines)
    shift
    [ $# -gt 0 ] || { echo "name the points" >&2; exit 2; }
    for point in "$@"; do
      P=$M/lines/4b/$point
      for f in weights.json files.sha256; do copy_file b "$P/$f"; done
      for panel in dev css-pilot ht-dev2 score5t-dev; do
        if on_b "test -f '$P/$panel/$panel.predictions.jsonl'"; then copy_file b "$P/$panel/$panel.predictions.jsonl"; fi
      done
    done
    for f in $(on_b "cd '$M/lines/4b/readout' 2>/dev/null && ls L-*.json L-*.line 2>/dev/null | grep -v '\.[0-9]\{8\}T'" || true); do
      copy_file b "$M/lines/4b/readout/$f"
    done
    ;;
  select) copy_file a "$M/select/4b-finalists.json" ;;
  gpuh)
    h=$(on_a "sha256sum '$M/gpuh-node-a.json' | cut -d' ' -f1")
    on_a "cat '$M/gpuh-node-a.json'" | on_b "cat > '$M/gpuh-node-a.json.tmp' && mv '$M/gpuh-node-a.json.tmp' '$M/gpuh-node-a.json'"
    [ "$(on_b "sha256sum '$M/gpuh-node-a.json' | cut -d' ' -f1")" = "$h" ] && echo "gpuh-node-a.json relayed ($h)"
    ;;
  early)
    for f in $(on_b "cd '$M' && ls early/*.json status/* 2>/dev/null" || true); do
      on_b "cat '$M/$f'" | on_a "mkdir -p '$M/relayed-status/$(dirname "$f")' && cat > '$M/relayed-status/$f'"
    done
    echo "status relayed"
    ;;
  *) sed -n '2,19p' "$0"; exit 2 ;;
esac
