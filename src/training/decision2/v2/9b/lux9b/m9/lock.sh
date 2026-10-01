#!/usr/bin/env bash
# 9B M9 finalist lock facts on node A (prereg "Formal and successor"; amendment 2), CPU, before any formal run of the
# finalists: for each finalist NAME, soup/NAME/SHA256SUMS over its artifact directory (written once, then verified),
# the readout manifests' model_sha256, and the hashes of the rules output and typed readout that chose it. Prints one
# JSON object for the lock record; it never starts a GPU job.
#
# usage: lock.sh <rules-json> <readout-json> NAME [NAME ...]
set -euo pipefail
M=/data/dev2/runs/9b/m9
rules=$1 readout=$2
shift 2
for name in "$@"; do
  art=$(cat "$M/soup/$name/DONE")
  sums=$M/soup/$name/SHA256SUMS
  if [ -f "$sums" ]; then
    (cd "$art" && sha256sum -c --quiet "$sums") || { echo "$name: SHA256SUMS mismatch" >&2; exit 1; }
  else
    (cd "$art" && find . -type f ! -name '*.pulled.json' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > "$sums"
  fi
done
python3 - "$M" "$rules" "$readout" "$@" << 'EOF'
import hashlib, json, sys
m, rules, readout, *names = sys.argv[1:]
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
out = {"rules": {"path": rules, "sha256": sha(rules)}, "readout": {"path": readout, "sha256": sha(readout)},
       "finalists": {}}
for name in names:
    art = open(f"{m}/soup/{name}/DONE").read().strip()
    manifest = json.load(open(f"{m}/lines/{name}/dev/dev.predictions.jsonl.manifest.json"))
    out["finalists"][name] = {
        "artifact": art,
        "sha256sums": sha(f"{m}/soup/{name}/SHA256SUMS"),
        "files": sum(1 for _ in open(f"{m}/soup/{name}/SHA256SUMS")),
        "model_sha256": manifest["model_sha256"],
        "pulled": json.load(open(f"{art}.pulled.json")),
    }
print(json.dumps(out, indent=1))
EOF
