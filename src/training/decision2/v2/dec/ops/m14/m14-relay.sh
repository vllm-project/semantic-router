#!/usr/bin/env bash
# Decoder M14 relay, run on the workstation (nodes A and B do not reach each other). Node aliases node-a / node-b
# resolve through ${DEV2_NODES_FILE:-~/.config/decision2/nodes.env} as in m6-relay.sh. Nothing is overwritten; every
# copy is checked by a content manifest (sha256 of the sorted per-file sha256 list) on both sides.
#
#   m14-relay.sh lines <point> [...]   node-B m14/lines/<point> -> node A (predictions, manifests, launch receipts and
#                                      parity files only; refuses gold-named files)
#   m14-relay.sh dir <a|b> <path>      one directory from node A to node B (a) or node B to node A (b), same path,
#                                      e.g. a finalist soup for the node-B formal path (weights allowed; no gold)
set -euo pipefail
nodes_file=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
resolve() { local d=""; [ -f "$nodes_file" ] && d=$(awk -F= -v k="$1" '$1 == k { print substr($0, length(k) + 2); exit }' "$nodes_file"); echo "${d:-$1}"; }
A=$(resolve node-a) B=$(resolve node-b)
on() { local h=$1; shift; ssh -o BatchMode=yes -o ConnectTimeout=20 "$h" "$@"; }
manifest='find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -d" " -f1'
L=/data/dev2/runs/dec/m14/lines
usage() { sed -n '2,10p' "$0"; exit 2; }

# copy <from-host> <to-host> <parent> <name> [tar exclude args]: tar stream, refuse to overwrite; the target's files
# are hashed on both sides in the same order (without excludes the file counts must also agree).
copy() {
  local from=$1 to=$2 parent=$3 name=$4 mf mt list
  shift 4
  on "$to" "test ! -e '$parent/$name'" || { echo "$parent/$name exists on the target" >&2; exit 1; }
  on "$from" "! find '$parent/$name' -iname '*gold*' | grep -q ." || { echo "$name contains gold-named files" >&2; exit 1; }
  on "$to" "mkdir -p '$parent'"
  on "$from" "tar -C '$parent' -cf - $* '$name'" | on "$to" "tar -C '$parent' -xf -"
  mt=$(on "$to" "cd '$parent/$name' && $manifest")
  list=$(on "$to" "cd '$parent/$name' && find . -type f -print | LC_ALL=C sort")
  mf=$(printf '%s\n' "$list" | on "$from" "cd '$parent/$name' && xargs -d '\n' -r sha256sum | sha256sum | cut -d' ' -f1")
  [ "$mf" = "$mt" ] || { echo "manifest mismatch for $name: source $mf, target $mt" >&2; exit 1; }
  if [ $# -eq 0 ]; then
    [ "$(on "$from" "find '$parent/$name' -type f | wc -l")" = "$(printf '%s\n' "$list" | wc -l)" ] \
      || { echo "file count differs for $name" >&2; exit 1; }
  fi
  echo "relayed $parent/$name ($mt)"
}

case ${1:-} in
  lines)
    shift
    [ $# -ge 1 ] || usage
    for p in "$@"; do
      on "$B" "test -d '$L/$p'" || { echo "no node-B readouts $p" >&2; exit 1; }
      copy "$B" "$A" "$L" "$p" --exclude='*.stdout.log' --exclude='*.stderr.log'
    done
    ;;
  dir)
    [ $# -eq 3 ] || usage
    case $2 in a) from=$A to=$B ;; b) from=$B to=$A ;; *) usage ;; esac
    copy "$from" "$to" "$(dirname "$3")" "$(basename "$3")"
    ;;
  *) usage ;;
esac
