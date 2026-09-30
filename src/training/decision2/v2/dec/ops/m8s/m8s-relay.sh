#!/usr/bin/env bash
# Decoder M8-small relays through this workstation (node aliases from the private nodes.env; no address is written
# anywhere). Gold-free files only; Triton caches are never relayed. Every relayed file is re-hashed on both sides.
#   m8s-relay.sh pull <run> [<run> ...]   node B m8s/formal/<run> -> node A /data/dev2/runs/dec/formal/m8s/<run>
#   m8s-relay.sh back <file> [<file> ...] node A formal/m8s/<file> -> node B m8s/formal/nodeA/<file>
#   m8s-relay.sh s5 <tier> <point> ...    Score5-typed-DEV (amendment 2): node B lines/<tier>/<point>/score5t-dev
#                                         predictions -> node A formal/m8s/s5/<tier>/, scored there with
#                                         m8s_rules.py score5t (mirror M8S_MIRROR_A) -> node B lines/<tier>/diag/
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
  s5)
    [ $# -ge 2 ] || { sed -n '2,9p' "$0"; exit 2; }
    tier=$1
    shift
    [[ $tier =~ ^(2b|08b)$ ]] || { echo "bad tier $tier" >&2; exit 2; }
    MA=/data/dev2/src/${M8S_MIRROR_A:?set M8S_MIRROR_A to the node-A mirror directory name}/src/training/decision2
    LB=/data/dev2/runs/dec/m8s/lines/$tier D=$FA/s5/$tier
    for point in "$@"; do
      [[ $point =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "bad point $point" >&2; exit 2; }
      src=$LB/$point/score5t-dev/score5t-dev.predictions.jsonl out=$LB/diag/$point.score5t.json
      if ssh -o ConnectTimeout=20 "$NB" "test -f '$out'"; then echo "$point: already scored"; continue; fi
      ssh -o ConnectTimeout=20 "$NB" "cat '$src'" \
        | ssh -o ConnectTimeout=20 "$NA" "mkdir -p '$D' && cat > '$D/$point.score5t-dev.predictions.jsonl'"
      a=$(ssh -o ConnectTimeout=20 "$NB" "sha256sum '$src'" | cut -d' ' -f1)
      b=$(ssh -o ConnectTimeout=20 "$NA" "sha256sum '$D/$point.score5t-dev.predictions.jsonl'" | cut -d' ' -f1)
      [ "$a" = "$b" ] || { echo "$point: prediction relay hashes differ" >&2; exit 1; }
      ssh -o ConnectTimeout=20 "$NA" "cd /tmp && PYTHONPATH='$MA' python3 -B '$MA/v2/dec/ops/m8s/m8s_rules.py' score5t \
        --panel-root /data/dev2/private/panels --predictions '$D/$point.score5t-dev.predictions.jsonl' \
        --output '$D/$point.score5t.json'" || { echo "$point: scoring FAILED" >&2; exit 1; }
      ssh -o ConnectTimeout=20 "$NA" "cat '$D/$point.score5t.json'" | ssh -o ConnectTimeout=20 "$NB" "cat > '$out'"
      a=$(ssh -o ConnectTimeout=20 "$NA" "sha256sum '$D/$point.score5t.json'" | cut -d' ' -f1)
      b=$(ssh -o ConnectTimeout=20 "$NB" "sha256sum '$out'" | cut -d' ' -f1)
      [ "$a" = "$b" ] || { echo "$point: score relay hashes differ" >&2; exit 1; }
      echo "$point: scored, flags $(ssh -o ConnectTimeout=20 "$NB" "python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"check\"][\"flags\"])' '$out'")"
    done
    ;;
  *) sed -n '2,9p' "$0"; exit 2 ;;
esac
