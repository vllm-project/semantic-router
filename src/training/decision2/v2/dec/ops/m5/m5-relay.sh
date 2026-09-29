#!/usr/bin/env bash
# Decoder M5 formal relay, run on the workstation (the nodes do not reach each other). Node aliases node-a /
# node-b resolve through ${DEV2_NODES_FILE:-~/.config/decision2/nodes.env} as in mirror_to_node.sh.
#
#   m5-relay.sh pull <run>              node-B formal/m5/<run> (gold-free predictions, receipts, seal, header stubs;
#                                       no weights, no cache) -> node-A formal/m5/<run>; content manifests must match
#   m5-relay.sh mark <run>              node-A REPORT.json / SEAL.json hashes -> node-B <run>/V3-SEALED.json
#                                       (gates m5-formal.sh mlx; no score leaves node A)
#   m5-relay.sh dev                     node-B development readouts (soup/<ARM>/readout.json, mlxdev/readouts/*/
#                                       {score,vs-n4xf-soup,vs-nox1}.json) -> node-A formal/m5/dev (m5-results.py --dev-root)
#   m5-relay.sh mlx-prompts <mirror>    node-A goldfree/mlx-diag.prompts.jsonl -> node-B panel root via the eval
#                                       track's hash-verified `v2.eval.panels install` (gold-free file only)
#   m5-relay.sh test                    round trip of a small throwaway directory
set -euo pipefail
FB=/data/dev2/runs/dec/formal/m5
FA=/data/dev2/runs/dec/formal/m5
nodes_file=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
resolve() { local d=""; [ -f "$nodes_file" ] && d=$(awk -F= -v k="$1" '$1 == k { print substr($0, length(k) + 2); exit }' "$nodes_file"); echo "${d:-$1}"; }
A=$(resolve node-a) B=$(resolve node-b)
on_a() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$A" "$@"; }
on_b() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$B" "$@"; }
manifest='find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -d" " -f1'

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

case ${1:-} in
  pull)
    [ $# -eq 2 ] || { sed -n '2,17p' "$0"; exit 2; }
    on_b "test -f '$FB/$2/M5-RECEIPT.json'" || { echo "$2 has no receipt on node B" >&2; exit 1; }
    on_b "test ! -e '$FB/$2/gold' && ! find '$FB/$2' -name '*gold*' | grep -q ." || { echo "$2 contains gold-named files" >&2; exit 1; }
    relay "$FB" "$2" "$FA"
    ;;
  mark)
    [ $# -eq 2 ] || { sed -n '2,17p' "$0"; exit 2; }
    hashes=$(on_a "cd '$FA/$2' && sha256sum REPORT.json SEAL.json")
    json=$(python3 -c 'import json,sys,datetime as d; h=dict(l.split()[::-1] for l in sys.argv[1].splitlines()); print(json.dumps({"run": sys.argv[2], "report_sha256": h["REPORT.json"], "seal_sha256": h["SEAL.json"], "marked_utc": d.datetime.now(d.timezone.utc).isoformat()}))' "$hashes" "$2")
    seal_b=$(on_b "sha256sum '$FB/$2/SEAL.json' | cut -d' ' -f1")
    [ "$seal_b" = "$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["seal_sha256"])' "$json")" ] \
      || { echo "$2: node-A seal differs from node B" >&2; exit 1; }
    printf '%s\n' "$json" | on_b "set -o noclobber; cat > '$FB/$2/V3-SEALED.json'"
    echo "marked $2"
    ;;
  dev)
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    on_b "cd /data/dev2/runs/dec/m5 && find soup -maxdepth 2 -name readout.json; find mlxdev/readouts -maxdepth 2 \\( -name score.json -o -name vs-n4xf-soup.json -o -name vs-nox1.json \\)" > "/tmp/m5-dev-files-$ts"
    on_b "cd /data/dev2/runs/dec/m5 && tar -cf - -T -" < "/tmp/m5-dev-files-$ts" | on_a "mkdir -p '$FA/dev-$ts' && tar -C '$FA/dev-$ts' -xf - && ln -sfn 'dev-$ts' '$FA/dev'"
    echo "relayed $(wc -l < "/tmp/m5-dev-files-$ts") development files to node A $FA/dev-$ts (symlink $FA/dev)"
    rm -f "/tmp/m5-dev-files-$ts"
    ;;
  mlx-prompts)
    [ $# -eq 2 ] || { sed -n '2,17p' "$0"; exit 2; }
    t=/data/dev2/tmp/m5-mlx-diag.prompts.$$.jsonl
    on_a "cat /data/dev2/private/panels/goldfree/mlx-diag.prompts.jsonl" | on_b "cat > '$t'"
    on_b "cd /data/dev2/src/$2/src/training/decision2 && PYTHONPATH=. python3 -B -m v2.eval.panels install --root /data/dev2/private/panels --source 'mlx-diag=$t'; rc=\$?; rm -f '$t'; exit \$rc"
    ;;
  test)
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    on_b "mkdir -p /data/dev2/tmp/m5-relay-test/rt-$ts/output && echo '{\"id\": \"t\"}' > /data/dev2/tmp/m5-relay-test/rt-$ts/output/x.predictions.jsonl && date -u > /data/dev2/tmp/m5-relay-test/rt-$ts/M5-RECEIPT.json"
    relay /data/dev2/tmp/m5-relay-test "rt-$ts" "$FA/dryrun"
    on_a "rm -rf '$FA/dryrun/rt-$ts'"
    on_b "rm -rf /data/dev2/tmp/m5-relay-test/rt-$ts"
    ;;
  *) sed -n '2,17p' "$0"; exit 2 ;;
esac
