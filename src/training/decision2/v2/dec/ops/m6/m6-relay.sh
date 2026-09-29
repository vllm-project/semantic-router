#!/usr/bin/env bash
# Decoder M6 formal relay, run on the workstation (the nodes do not reach each other). Node aliases node-a / node-b
# resolve through ${DEV2_NODES_FILE:-~/.config/decision2/nodes.env} as in mirror_to_node.sh. Nothing is overwritten;
# every copy is checked by a content manifest (sha256 of the sorted per-file sha256 list) on both sides.
#
#   m6-relay.sh pull <run>          node-B formal/m6/<run> (gold-free predictions, receipts, seal, header stubs; no
#                                   weights, no cache) -> node-A formal/m6/<run>; refuses gold-named files
#   m6-relay.sh mark <run>          node-A REPORT.json / SEAL.json hashes -> node-B <run>/V3-SEALED.json (gates
#                                   m6-formal.sh mlx on node B; no score leaves node A)
#   m6-relay.sh mark-ref            node-A formal/m6/m6-ref-S2T-soup/REF-EXACT.json -> node B (gates the 2B finalists on
#                                   node B; the seal hashes must agree)
#   m6-relay.sh pkg <point>         2B node-A fallback: node-B formal/m6/pkg/m6-<point> (+ its SHA-256 list and the
#                                   staging parameter stubs) -> node A, then every file checked against the list
#   m6-relay.sh cal698              node-B m3/data-sel700-cal698 (CAL698 19cc1a8c... + SELECT700) -> node A, for the
#                                   0.8B 16K CAL698 fits; checked against its SHA256SUMS on arrival
#   m6-relay.sh test                round trip of a small throwaway directory
set -euo pipefail
FB=${M6_FORMAL_ROOT:-/data/dev2/runs/dec/formal/m6}
FA=$FB
CAL=/data/dev2/runs/dec/m3/data-sel700-cal698
CAL698_SHA=19cc1a8c4ebe6fd13031079f6b7d131046ce503f435f894e3ea672f91c2ed41f
nodes_file=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
resolve() { local d=""; [ -f "$nodes_file" ] && d=$(awk -F= -v k="$1" '$1 == k { print substr($0, length(k) + 2); exit }' "$nodes_file"); echo "${d:-$1}"; }
A=$(resolve node-a) B=$(resolve node-b)
on_a() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$A" "$@"; }
on_b() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$B" "$@"; }
manifest='find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -d" " -f1'
usage() { sed -n '2,20p' "$0"; exit 2; }

# relay <node-B parent> <name> <node-A parent>: tar stream B -> A, refuse to overwrite, compare content manifests.
relay() {
  local src=$1 name=$2 dst=$3 mb ma
  on_a "test ! -e '$dst/$name'" || { echo "node A $dst/$name exists" >&2; exit 1; }
  mb=$(on_b "cd '$src/$name' && $manifest")
  on_a "mkdir -p '$dst'"
  on_b "tar -C '$src' -cf - '$name'" | on_a "tar -C '$dst' -xf -"
  ma=$(on_a "cd '$dst/$name' && $manifest")
  [ "$ma" = "$mb" ] || { echo "manifest mismatch for $name: node B $mb, node A $ma" >&2; exit 1; }
  echo "relayed $name ($mb)"
}
# relay_file <node-B path> <node-A path>: one file, hash-checked, no overwrite.
relay_file() {
  local hb ha
  on_a "test ! -e '$2'" || { echo "node A $2 exists" >&2; exit 1; }
  hb=$(on_b "sha256sum '$1' | cut -d' ' -f1")
  on_a "mkdir -p '$(dirname "$2")'"
  on_b "cat '$1'" | on_a "set -o noclobber; cat > '$2'"
  ha=$(on_a "sha256sum '$2' | cut -d' ' -f1")
  [ "$ha" = "$hb" ] || { echo "hash mismatch for $1: node B $hb, node A $ha" >&2; exit 1; }
  echo "relayed $(basename "$1") ($hb)"
}

case ${1:-} in
  pull)
    [ $# -eq 2 ] || usage
    on_b "test -f '$FB/$2/M6-RECEIPT.json'" || { echo "$2 has no M6 receipt on node B" >&2; exit 1; }
    on_b "test ! -e '$FB/$2/gold' && ! find '$FB/$2' -iname '*gold*' | grep -q ." || { echo "$2 contains gold-named files" >&2; exit 1; }
    on_b "! find '$FB/$2' -name '*.safetensors' -size +1M | grep -q ." || { echo "$2 contains weights" >&2; exit 1; }
    relay "$FB" "$2" "$FA"
    ;;
  mark)
    [ $# -eq 2 ] || usage
    hashes=$(on_a "cd '$FA/$2' && sha256sum REPORT.json SEAL.json")
    json=$(python3 -c 'import json,sys,datetime as d; h=dict(l.split()[::-1] for l in sys.argv[1].splitlines()); print(json.dumps({"run": sys.argv[2], "report_sha256": h["REPORT.json"], "seal_sha256": h["SEAL.json"], "marked_utc": d.datetime.now(d.timezone.utc).isoformat()}))' "$hashes" "$2")
    seal_b=$(on_b "sha256sum '$FB/$2/SEAL.json' | cut -d' ' -f1")
    [ "$seal_b" = "$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["seal_sha256"])' "$json")" ] \
      || { echo "$2: node-A seal differs from node B" >&2; exit 1; }
    printf '%s\n' "$json" | on_b "set -o noclobber; cat > '$FB/$2/V3-SEALED.json'"
    echo "marked $2"
    ;;
  mark-ref)
    R=m6-ref-S2T-soup
    doc=$(on_a "cat '$FA/$R/REF-EXACT.json'")
    seal_b=$(on_b "sha256sum '$FB/$R/SEAL.json' | cut -d' ' -f1")
    [ "$seal_b" = "$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["seal_sha256"])' "$doc")" ] \
      || { echo "$R: node-A seal differs from node B" >&2; exit 1; }
    printf '%s\n' "$doc" | on_b "set -o noclobber; cat > '$FB/$R/REF-EXACT.json'"
    echo "marked $R exact=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["exact"])' "$doc")"
    ;;
  pkg)
    [ $# -eq 2 ] || usage
    P=m6-$2
    on_b "test -f '$FB/pkg/$P.sha256' && test -f '$FB/stage-params/$P/PARAMS.json'" || { echo "$P is not staged on node B" >&2; exit 1; }
    on_b "cd '$FB/pkg/$P' && sha256sum -c --quiet '$FB/pkg/$P.sha256'" || { echo "$P changed on node B since staging" >&2; exit 1; }
    relay_file "$FB/pkg/$P.sha256" "$FA/pkg/$P.sha256"
    relay "$FB/pkg" "$P" "$FA/pkg"
    on_a "cd '$FA/pkg/$P' && sha256sum -c --quiet '$FA/pkg/$P.sha256'" || { echo "$P: node-A copy fails its list" >&2; exit 1; }
    relay "$FB/stage-params" "$P" "$FA/stage-params"
    on_a "test -e /data/dev2/runs/dec/m6/select/2b-finalists.json" \
      || relay_file /data/dev2/runs/dec/m6/select/2b-finalists.json /data/dev2/runs/dec/m6/select/2b-finalists.json
    echo "package $P on node A verified against its list"
    ;;
  cal698)
    [ "$(on_b "sha256sum '$CAL/cal.jsonl' | cut -d' ' -f1")" = "$CAL698_SHA" ] || { echo "node-B CAL698 is not $CAL698_SHA" >&2; exit 1; }
    relay "$(dirname "$CAL")" "$(basename "$CAL")" "$(dirname "$CAL")"
    on_a "cd '$CAL' && sha256sum -c --quiet SHA256SUMS" || { echo "node-A CAL698 copy fails SHA256SUMS" >&2; exit 1; }
    echo "CAL698 on node A: $(on_a "sha256sum '$CAL/cal.jsonl'")"
    ;;
  test)
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    on_b "mkdir -p /data/dev2/tmp/m6-relay-test/rt-$ts/output && echo '{\"id\": \"t\"}' > /data/dev2/tmp/m6-relay-test/rt-$ts/output/x.predictions.jsonl && date -u > /data/dev2/tmp/m6-relay-test/rt-$ts/M6-RECEIPT.json"
    relay /data/dev2/tmp/m6-relay-test "rt-$ts" "$FA/dryrun"
    on_a "rm -rf '$FA/dryrun/rt-$ts'"
    on_b "rm -rf /data/dev2/tmp/m6-relay-test/rt-$ts"
    ;;
  *) usage ;;
esac
