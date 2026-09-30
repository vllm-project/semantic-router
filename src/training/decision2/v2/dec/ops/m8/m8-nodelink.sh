#!/usr/bin/env bash
# Decoder M8 direct node-B -> node-A copy (COORDINATION 2026-09-30 08:40: use the direct node link, not the workstation
# relay, for large files). Run on the workstation. A temporary ed25519 key is made on node B, authorized on node A with
# a marker comment, used for one tar stream (ssh, AES-GCM), and removed from both nodes on exit (also on failure).
# Nothing is overwritten on node A; the copy is checked by a content manifest (sha256 of the sorted per-file list).
#
#   m8-nodelink.sh <node-B parent dir> <name> <node-A parent dir>
set -euo pipefail
[ $# -eq 3 ] || { sed -n '2,8p' "$0"; exit 2; }
SRC=$1 NAME=$2 DST=$3
nodes_file=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
resolve() { local d=""; [ -f "$nodes_file" ] && d=$(awk -F= -v k="$1" '$1 == k { print substr($0, length(k) + 2); exit }' "$nodes_file"); echo "${d:-$1}"; }
A=$(resolve node-a) B=$(resolve node-b)
on_a() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$A" "$@"; }
on_b() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$B" "$@"; }
manifest='find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -d" " -f1'
MARK="dev2-m8-temp-link-$(date -u +%Y%m%dT%H%M%SZ)"
KEY=/root/.ssh/$MARK
cleanup() {
  on_a "sed -i '/$MARK/d' /root/.ssh/authorized_keys" || echo "WARNING: remove the $MARK line from node A authorized_keys" >&2
  on_b "rm -f '$KEY' '$KEY.pub' '$KEY.known'" || echo "WARNING: remove $KEY* on node B" >&2
}
trap cleanup EXIT
on_a "test ! -e '$DST/$NAME'" || { echo "node A $DST/$NAME exists" >&2; exit 1; }
want=$(on_b "cd '$SRC/$NAME' && $manifest")
on_b "umask 077; ssh-keygen -q -t ed25519 -N '' -C '$MARK' -f '$KEY'"
on_b "cat '$KEY.pub'" | on_a "umask 077; cat >> /root/.ssh/authorized_keys"
on_a "mkdir -p '$DST'"
target=${A#*@}
t0=$(date +%s)
on_b "tar -C '$SRC' -cf - '$NAME' | ssh -i '$KEY' -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile='$KEY.known' -c aes128-gcm@openssh.com root@$target 'tar -C \"$DST\" -xf -'"
got=$(on_a "cd '$DST/$NAME' && $manifest")
[ "$got" = "$want" ] || { echo "manifest mismatch: node B $want, node A $got" >&2; exit 1; }
echo "copied $NAME node B -> node A in $(($(date +%s) - t0)) s; content manifest $want"
