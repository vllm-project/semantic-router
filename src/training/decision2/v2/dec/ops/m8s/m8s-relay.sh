#!/usr/bin/env bash
# Decoder M8-small relays through this workstation (node aliases from the private nodes.env; no address is written
# anywhere). Gold-free files only; Triton caches are never relayed. Every relayed file is re-hashed on both sides.
#   m8s-relay.sh pull <run> [<run> ...]   node B m8s/formal/<run> -> node A /data/dev2/runs/dec/formal/m8s/<run>
#   m8s-relay.sh back <file> [<file> ...] node A formal/m8s/<file> -> node B m8s/formal/nodeA/<file>
set -euo pipefail
ENVF=${DEV2_NODES_ENV:-$HOME/.config/decision2/nodes.env}
NA=$(grep '^node-a=' "$ENVF" | cut -d= -f2-) NB=$(grep '^node-b=' "$ENVF" | cut -d= -f2-)
FB=/data/dev2/runs/dec/m8s/formal FA=/data/dev2/runs/dec/formal/m8s
MODE=${1:-}
shift || true
manifest() {  # <host> <dir> <entry>
  ssh -o ConnectTimeout=20 "$1" "cd '$2' && find '$3' -type f -not -path '*/triton-cache/*' -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum"
}
case $MODE in
  pull)
    [ $# -ge 1 ] || { sed -n '2,6p' "$0"; exit 2; }
    for run in "$@"; do
      [[ $run =~ ^m8s-[A-Za-z0-9_.-]+$ ]] || { echo "bad run name $run" >&2; exit 2; }
      ssh -o ConnectTimeout=20 "$NB" "cd '$FB' && tar --exclude='triton-cache' -cf - '$run'" \
        | ssh -o ConnectTimeout=20 "$NA" "mkdir -p '$FA' && cd '$FA' && tar -xf -"
      if diff -q <(manifest "$NB" "$FB" "$run") <(manifest "$NA" "$FA" "$run") > /dev/null; then
        echo "relayed $run: $(manifest "$NA" "$FA" "$run" | wc -l) files, hashes equal"
      else
        echo "relay of $run: hash lists differ" >&2
        exit 1
      fi
    done
    ;;
  back)
    [ $# -ge 1 ] || { sed -n '2,6p' "$0"; exit 2; }
    for f in "$@"; do
      [[ $f != *..* ]] || { echo "bad path $f" >&2; exit 2; }
      ssh -o ConnectTimeout=20 "$NA" "cd '$FA' && tar -cf - '$f'" \
        | ssh -o ConnectTimeout=20 "$NB" "mkdir -p '$FB/nodeA' && cd '$FB/nodeA' && tar -xf -"
      a=$(ssh -o ConnectTimeout=20 "$NA" "sha256sum '$FA/$f'" | cut -d' ' -f1)
      b=$(ssh -o ConnectTimeout=20 "$NB" "sha256sum '$FB/nodeA/$f'" | cut -d' ' -f1)
      [ "$a" = "$b" ] || { echo "relay back of $f: hashes differ" >&2; exit 1; }
      echo "relayed back $f ($a)"
    done
    ;;
  *) sed -n '2,6p' "$0"; exit 2 ;;
esac
