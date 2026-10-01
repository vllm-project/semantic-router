#!/usr/bin/env bash
# Decoder M14 relay between nodes A and B, run on the workstation (A and B do not reach each other, and the
# workstation link is too slow for model directories). Data moves through node E over the temporary transfer key:
# the source node pushes into an M14-only transit directory on node E (/data/dev2/runs/dec/m14-relay/<stamp>), the
# target node pulls it, and the transit copy is removed. Node aliases resolve through
# ${DEV2_NODES_FILE:-~/.config/decision2/nodes.env} as in m6-relay.sh. Nothing is overwritten; the target's files are
# hashed on both sides in the same order (sha256 of the sorted per-file sha256 list).
#
#   m14-relay.sh lines <point> [...]   node-B m14/lines/<point> -> node A (predictions, manifests, launch receipts and
#                                      parity files only; refuses gold-named files)
#   m14-relay.sh dir <a|b> <path>      one directory from node A to node B (a) or node B to node A (b), same path,
#                                      e.g. a finalist soup for the node-B formal path (weights allowed; no gold)
set -euo pipefail
nodes_file=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
resolve() { local d=""; [ -f "$nodes_file" ] && d=$(awk -F= -v k="$1" '$1 == k { print substr($0, length(k) + 2); exit }' "$nodes_file"); echo "${d:-$1}"; }
A=$(resolve node-a) B=$(resolve node-b) E=$(resolve node-e)
on() { local h=$1; shift; ssh -o BatchMode=yes -o ConnectTimeout=20 "$h" "$@"; }
KEY="-i /root/.ssh/d2_temp_cd -o BatchMode=yes -o ConnectTimeout=20"
T=/data/dev2/runs/dec/m14-relay
L=/data/dev2/runs/dec/m14/lines
LINES_FILTER="--include=*/ --include=*.predictions.jsonl --include=*.manifest.json --include=*.launch.json --include=parity-*.json --exclude=*"
usage() { sed -n '2,15p' "$0"; exit 2; }

# hop <from-host> <to-host> <parent> <name> <lines|all>
hop() {
  local from=$1 to=$2 parent=$3 name=$4 filter="" stamp mf mt list
  [ "$5" = lines ] && filter=$LINES_FILTER
  stamp=$(date -u +%Y%m%dT%H%M%SZ)-$$
  on "$to" "test ! -e '$parent/$name'" || { echo "$parent/$name exists on the target" >&2; exit 1; }
  on "$from" "test -d '$parent/$name' && ! find '$parent/$name' -iname '*gold*' | grep -q ." \
    || { echo "$parent/$name is missing or contains gold-named files" >&2; exit 1; }
  on "$from" "ssh $KEY '$E' 'mkdir -p $T/$stamp' && rsync -a $filter -e 'ssh $KEY' '$parent/$name/' '$E:$T/$stamp/$name/'"
  on "$to" "mkdir -p '$parent' && rsync -a -e 'ssh $KEY' '$E:$T/$stamp/$name/' '$parent/$name/' && ssh $KEY '$E' 'rm -rf $T/$stamp'"
  mt=$(on "$to" "cd '$parent/$name' && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -d' ' -f1")
  list=$(on "$to" "cd '$parent/$name' && find . -type f -print | LC_ALL=C sort")
  mf=$(printf '%s\n' "$list" | on "$from" "cd '$parent/$name' && xargs -d '\n' -r sha256sum | sha256sum | cut -d' ' -f1")
  [ "$mf" = "$mt" ] || { echo "manifest mismatch for $name: source $mf, target $mt" >&2; exit 1; }
  if [ -z "$filter" ]; then
    [ "$(on "$from" "find '$parent/$name' -type f | wc -l")" = "$(printf '%s\n' "$list" | wc -l)" ] \
      || { echo "file count differs for $name" >&2; exit 1; }
  fi
  echo "relayed $parent/$name ($mt, $(printf '%s\n' "$list" | wc -l) files)"
}

case ${1:-} in
  lines)
    shift
    [ $# -ge 1 ] || usage
    for p in "$@"; do hop "$B" "$A" "$L" "$p" lines; done
    ;;
  dir)
    [ $# -eq 3 ] || usage
    case $2 in a) from=$A to=$B ;; b) from=$B to=$A ;; *) usage ;; esac
    hop "$from" "$to" "$(dirname "$3")" "$(basename "$3")" all
    ;;
  *) usage ;;
esac
