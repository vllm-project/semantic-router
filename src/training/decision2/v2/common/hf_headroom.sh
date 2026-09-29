#!/usr/bin/env bash
# Report the llm-semantic-router organization's PRIVATE Hugging Face storage and headroom;
# exit non-zero when the headroom is below a threshold.
#
# Usage:
#   hf_headroom.sh [--node <alias>] [--min-free-gb <GB>] [--cap-gb <GB>] [--top <N>] [--json]
#   hf_headroom.sh --from-json <repos.json> [--min-free-gb <GB>] [--cap-gb <GB>] [--top <N>] [--json]
#
# Before uploading N GB, run it with --min-free-gb N (or more): exit 1 means the upload would not fit.
# Private usage is the sum of the Hub's usedStorage over the organization's private models, datasets
# and spaces (LFS objects and history included, as the Hub counts them; 1 GB = 10^9 bytes). The cap
# defaults to 100 GB, the private tier of an organization without a paid plan.
#
# --node       query the Hub on that experiment node over ssh (the HF token lives on the nodes); the
#              alias is looked up in ${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env} as in
#              mirror_to_node.sh, and the node runs ${DEV2_HF_REMOTE_PYTHON:-/data/dev2/tools/hf-cli/bin/python}.
#              Without --node the query runs here with ${DEV2_HF_PYTHON:-/data/dev2/tools/hf-cli/bin/python}
#              (python3 if that is missing) and its default token.
# --from-json  evaluate a saved query result instead ({"org": ..., "repos": [{kind, id, private, used}]}).
# --top        how many of the largest private repositories to list (default 8).
# --json       print one JSON object instead of the text report.
# Exit status: 0 headroom >= threshold (default threshold 0), 1 below it, 2 usage or query error.
set -euo pipefail

usage() {
  sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//' >&2
  exit 2
}

org="llm-semantic-router"
node=""
from_json=""
min_free="0"
cap="100"
top="8"
as_json=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --node) node="${2:?}"; shift 2 ;;
    --from-json) from_json="${2:?}"; shift 2 ;;
    --min-free-gb) min_free="${2:?}"; shift 2 ;;
    --cap-gb) cap="${2:?}"; shift 2 ;;
    --top) top="${2:?}"; shift 2 ;;
    --json) as_json=1; shift ;;
    -h|--help) usage ;;
    *) echo "hf_headroom: unknown argument: $1" >&2; usage ;;
  esac
done
number='^[0-9]+([.][0-9]+)?$'
if ! [[ "$min_free" =~ $number && "$cap" =~ $number && "$top" =~ ^[0-9]+$ ]]; then
  echo "hf_headroom: --min-free-gb and --cap-gb take a non-negative number, --top an integer" >&2
  exit 2
fi
if [[ -n "$node" && -n "$from_json" ]]; then
  echo "hf_headroom: --node and --from-json are exclusive" >&2
  exit 2
fi

IFS= read -r -d '' query_py <<'PY' || true
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi

org = sys.argv[1]
api = HfApi()
rows, private = [], []
for kind, lister, getter in (("model", api.list_models, api.model_info),
                             ("dataset", api.list_datasets, api.dataset_info),
                             ("space", api.list_spaces, api.space_info)):
    for info in lister(author=org, expand=["private"]):
        if info.private:
            private.append((kind, info.id, getter))
        else:
            rows.append({"kind": kind, "id": info.id, "private": False, "used": None})


def used(job):
    kind, repo_id, getter = job
    info = getter(repo_id, expand=["usedStorage"])
    return {"kind": kind, "id": repo_id, "private": True, "used": int(getattr(info, "used_storage", 0) or 0)}


with ThreadPoolExecutor(8) as pool:
    rows.extend(pool.map(used, private))
json.dump({"org": org, "repos": rows}, sys.stdout)
PY

IFS= read -r -d '' report_py <<'PY' || true
import json
import sys
from datetime import datetime, timezone

min_free, cap, top, as_json = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1"
try:
    data = json.load(sys.stdin)
    private = sorted((r for r in data["repos"] if r.get("private")), key=lambda r: -int(r.get("used") or 0))
    used = sum(int(r.get("used") or 0) for r in private)
except Exception as exc:  # exit 1 is reserved for "below the threshold"
    print(f"hf_headroom: unreadable query result ({type(exc).__name__})", file=sys.stderr)
    sys.exit(2)
cap_b, need_b = int(cap * 1e9), int(min_free * 1e9)
free_b = cap_b - used
ok = free_b >= need_b
if as_json:
    print(json.dumps({"org": data.get("org"), "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "private_repos": len(private), "private_bytes": used, "cap_bytes": cap_b,
                      "headroom_bytes": free_b, "min_free_bytes": need_b, "ok": ok,
                      "largest": [{"id": r["id"], "kind": r["kind"], "bytes": int(r.get("used") or 0)}
                                  for r in private[:top]]}))
else:
    print(f"{data.get('org')} private storage: {used / 1e9:.2f} GB used of {cap:g} GB, headroom {free_b / 1e9:.2f} GB "
          f"({len(private)} private repos; threshold {min_free:g} GB: {'OK' if ok else 'BELOW'})")
    for r in private[:top]:
        print(f"  {int(r.get('used') or 0) / 1e9:8.2f} GB  {r['kind']:7s}  {r['id']}")
sys.exit(0 if ok else 1)
PY

query_out="$(mktemp)"
trap 'rm -f "$query_out"' EXIT
if [[ -n "$from_json" ]]; then
  cp -- "$from_json" "$query_out" || exit 2
elif [[ -n "$node" ]]; then
  nodes_file="${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}"
  dest=""
  if [[ -f "$nodes_file" ]]; then
    dest="$(awk -F= -v k="$node" '$1 == k { print substr($0, length(k) + 2); exit }' "$nodes_file")"
  fi
  dest="${dest:-$node}"
  remote_py="${DEV2_HF_REMOTE_PYTHON:-/data/dev2/tools/hf-cli/bin/python}"
  query_err="$(mktemp)"
  trap 'rm -f "$query_out" "$query_err"' EXIT
  status=0
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$dest" "$remote_py - $org" <<<"$query_py" >"$query_out" 2>"$query_err" \
    || status=$?
  # ssh and remote errors may name the node's address: print them with the alias instead.
  if [[ "$dest" == "$node" ]]; then
    cat "$query_err" >&2
  else
    DEST="$dest" HOST="${dest#*@}" TAG="<$node>" awk '
      function redact(line, s,   out, i) {
        out = ""
        while (s != "" && (i = index(line, s)) > 0) {
          out = out substr(line, 1, i - 1) ENVIRON["TAG"]
          line = substr(line, i + length(s))
        }
        return out line
      }
      { print redact(redact($0, ENVIRON["DEST"]), ENVIRON["HOST"]) }' "$query_err" >&2
  fi
  if [[ $status -ne 0 ]]; then
    echo "hf_headroom: Hub query on $node failed (exit $status)" >&2
    exit 2
  fi
else
  py="${DEV2_HF_PYTHON:-/data/dev2/tools/hf-cli/bin/python}"
  [[ -x "$py" ]] || py=python3
  "$py" - "$org" <<<"$query_py" >"$query_out" || { echo "hf_headroom: Hub query failed" >&2; exit 2; }
fi

status=0
python3 -c "$report_py" "$min_free" "$cap" "$top" "$as_json" <"$query_out" || status=$?
if [[ $status -gt 1 ]]; then
  exit 2
fi
exit "$status"
