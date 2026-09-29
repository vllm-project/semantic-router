#!/usr/bin/env bash
# Decoder M7 relays, run on the workstation (the nodes do not reach each other). Node aliases resolve through
# ${DEV2_NODES_FILE:-~/.config/decision2/nodes.env} as in mirror_to_node.sh. Nothing is overwritten; directories
# are checked by a content manifest (sha256 of the sorted per-file sha256 list) on both sides, files by SHA-256.
#
#   m7-relay.sh data-2b      node-B m7/data/2b/mix, m7/teacher/m7-2b-*, m7/lock-2b.json -> node A (same paths)
#   m7-relay.sh ref-2b       node-B m3/soup/S2T/build/S2T-soup -> node-A m7/ref/S2T-soup (the 2B line reference)
#   m7-relay.sh panels       hs1-dev / PN1-dev: gold-free prompts into /data/dev2/runs/dec/panels (mounted as
#                            /panels) and gold into /data/dev2/private/dec/m7 (mode 700) on both nodes, from node A's
#                            installed panels (hs1-dev) and each node's PN1 copy, every file hash-checked
#   m7-relay.sh soup <ARM>   node-A m7/soup/<ARM> -> node B (not planned; for a 2B formal fallback only)
set -euo pipefail
nodes_file=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
resolve() { local d=""; [ -f "$nodes_file" ] && d=$(awk -F= -v k="$1" '$1 == k { print substr($0, length(k) + 2); exit }' "$nodes_file"); echo "${d:-$1}"; }
A=$(resolve node-a) B=$(resolve node-b)
on_a() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$A" "$@"; }
on_b() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$B" "$@"; }
manifest='find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -d" " -f1'
R=/data/dev2/runs/dec
HS1P=49f192a700242efe46265c4377a3cedb44dd635e5c5d23db1fc2d1e6fac3f072
HS1G=808dfc01c825acdb7039f81e65ace5bae3ccdbe16e0036ec2258aebb6004d48f
PN1P=79dbf9996ad09caa622c4a9ea2f28075ddf80a5ae1ba9db8cd633573621bc14a
PN1G=e0d3e57cbba88139ecd326f7837bff1374090960a3d6d21e340140d491d4be5e
PN1D=/data/dev2/private/data/m4-pn1/e7bedd642175

relay_dir() {  # <src node a|b> <src parent> <name> <dst parent>
  local from=$1 src=$2 name=$3 dst=$4 to ms md
  to=$([ "$from" = b ] && echo a || echo b)
  "on_$to" "test ! -e '$dst/$name'" || { echo "$dst/$name exists on node $to" >&2; exit 1; }
  ms=$("on_$from" "cd '$src/$name' && $manifest")
  "on_$to" "mkdir -p '$dst'"
  "on_$from" "tar -C '$src' -cf - '$name'" | "on_$to" "tar -C '$dst' -xf -"
  md=$("on_$to" "cd '$dst/$name' && $manifest")
  [ "$ms" = "$md" ] || { echo "manifest mismatch for $name: $ms vs $md" >&2; exit 1; }
  echo "relayed $name ($ms)"
}
put() {  # <node a|b> <dst> <sha256> <mode>: stdin -> file, no overwrite, hash-checked
  "on_$1" "set -o noclobber; mkdir -p '$(dirname "$2")'; umask 077; cat > '$2'; chmod $4 '$2'; test \"\$(sha256sum '$2' | cut -d' ' -f1)\" = '$3'" \
    || { echo "node $1: $2 missing its hash $3" >&2; exit 1; }
}
have() { "on_$1" "test -f '$2' && test \"\$(sha256sum '$2' | cut -d' ' -f1)\" = '$3'"; }

case ${1:-} in
  data-2b)
    relay_dir b "$R/m7/data/2b" mix "$R/m7/data/2b"
    for arm in H C P; do relay_dir b "$R/m7/teacher" "m7-2b-$arm" "$R/m7/teacher"; done
    h=$(on_b "sha256sum '$R/m7/lock-2b.json' | cut -d' ' -f1")
    on_b "cat '$R/m7/lock-2b.json'" | put a "$R/m7/lock-2b.json" "$h" 644
    echo "2B data on node A"
    ;;
  ref-2b)
    relay_dir b "$R/m3/soup/S2T/build" S2T-soup "$R/m7/ref"
    ;;
  panels)
    on_a "mkdir -p -m 700 /data/dev2/private/dec/m7"
    on_b "mkdir -p -m 700 /data/dev2/private/dec/m7"
    for n in a b; do
      have "$n" "$R/panels/hs1-dev.prompts.jsonl" "$HS1P" \
        || on_a "cat /data/dev2/private/panels/goldfree/hs1-dev.prompts.jsonl" | put "$n" "$R/panels/hs1-dev.prompts.jsonl" "$HS1P" 444
      have "$n" /data/dev2/private/dec/m7/hs1-dev.gold.jsonl "$HS1G" \
        || on_a "cat /data/dev2/private/panels/gold/hs1-dev.gold.jsonl" | put "$n" /data/dev2/private/dec/m7/hs1-dev.gold.jsonl "$HS1G" 600
      have "$n" "$R/panels/pn1-dev.prompts.jsonl" "$PN1P" \
        || "on_$n" "cat $PN1D/pn1.dev.prompts.jsonl" | put "$n" "$R/panels/pn1-dev.prompts.jsonl" "$PN1P" 444
      have "$n" /data/dev2/private/dec/m7/pn1-dev.gold.jsonl "$PN1G" \
        || "on_$n" "cat $PN1D/pn1.dev.gold.jsonl" | put "$n" /data/dev2/private/dec/m7/pn1-dev.gold.jsonl "$PN1G" 600
    done
    echo "diagnostic panels in place on both nodes"
    ;;
  soup)
    [ $# -eq 2 ] || { sed -n '2,13p' "$0"; exit 2; }
    relay_dir a "$R/m7/soup/$2/build" "$2-soup" "$R/m7/soup-from-a/$2"
    ;;
  *) sed -n '2,13p' "$0"; exit 2 ;;
esac
