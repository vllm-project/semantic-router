#!/usr/bin/env bash
# Exact-mirror one pushed commit of this repository onto an experiment node.
#
# Usage:
#   mirror_to_node.sh <node-alias> <commit> [<target-root>]
#   mirror_to_node.sh --verify <node-alias> <commit> [<target-root>]
#
# The tree lands in <target-root>/<full-sha>/ (default target root /data/dev2/src)
# with <full-sha>/.dev2-mirror.json recording commit, tree, archive and content
# manifest SHA-256; the mirrored files are made read-only. Re-running for an
# existing mirror only re-verifies it. A directory created earlier by a plain
# `git archive | tar -x` is adopted only if its content manifest matches exactly.
#
# <node-alias> is looked up in ${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
# (private, never committed; lines "node-a=user@host"). An alias absent from that
# file is passed to ssh unchanged, so ssh-config host aliases also work.
# Only commits contained in a remote-tracking branch are mirrored; run
# `git fetch origin` first if a just-pushed commit is reported as unpushed.
set -euo pipefail

usage() {
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//' >&2
  exit 2
}

verify_only=0
if [[ "${1:-}" == "--verify" ]]; then
  verify_only=1
  shift
fi
[[ $# -eq 2 || $# -eq 3 ]] || usage
alias_name="$1"
commit_arg="$2"
target_root="${3:-/data/dev2/src}"
[[ "$target_root" == /* ]] || { echo "target root must be absolute" >&2; exit 2; }

nodes_file="${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}"
dest=""
if [[ -f "$nodes_file" ]]; then
  dest="$(awk -F= -v k="$alias_name" '$1 == k { print substr($0, length(k) + 2); exit }' "$nodes_file")"
fi
dest="${dest:-$alias_name}"

repo_root="$(git rev-parse --show-toplevel)"
sha="$(git -C "$repo_root" rev-parse --verify "${commit_arg}^{commit}")"
tree="$(git -C "$repo_root" rev-parse "${sha}^{tree}")"
if [[ -z "$(git -C "$repo_root" branch -r --contains "$sha" 2>/dev/null | head -n 1)" ]]; then
  echo "commit $sha is not contained in any remote-tracking branch; push (and fetch) first" >&2
  exit 1
fi
remote_dir="$target_root/$sha"
remote() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$dest" "$@"; }

manifest_cmd='find . -type f ! -name .dev2-mirror.json -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d" " -f1'

scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
git -C "$repo_root" archive --format=tar "$sha" > "$scratch/src.tar"
archive_sha="$(sha256sum "$scratch/src.tar" | cut -d' ' -f1)"
mkdir "$scratch/tree"
tar -x -C "$scratch/tree" -f "$scratch/src.tar"
local_manifest="$(cd "$scratch/tree" && eval "$manifest_cmd")"
file_count="$(cd "$scratch/tree" && find . -type f | wc -l | tr -d ' ')"
receipt() {
  printf '{"schema":"dev2-mirror/1","commit":"%s","tree":"%s","archive_sha256":"%s","content_manifest_sha256":"%s","files":%s,"created_utc":"%s"}' \
    "$sha" "$tree" "$archive_sha" "$local_manifest" "$file_count" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

state="$(remote "if [ -f '$remote_dir/.dev2-mirror.json' ]; then echo receipt; elif [ -e '$remote_dir' ]; then echo stray; else echo absent; fi")"

if [[ "$state" != "receipt" && "$verify_only" -eq 1 ]]; then
  echo "no verified mirror of $sha on $alias_name ($state)" >&2
  exit 1
fi

if [[ "$state" == "stray" ]]; then
  stray_manifest="$(remote "cd '$remote_dir' && $manifest_cmd")"
  if [[ "$stray_manifest" != "$local_manifest" ]]; then
    echo "$remote_dir exists but differs from commit $sha; refusing to touch it" >&2
    exit 1
  fi
  remote "set -e; cd '$remote_dir'; chmod u+w .; printf '%s\n' '$(receipt)' > .dev2-mirror.json; chmod -R a-w ."
elif [[ "$state" == "absent" ]]; then
  staging="$target_root/.incoming-$sha-$$"
  transit_sha="$(remote "set -e; mkdir -p '$target_root'; rm -rf '$staging' '$staging.tar'; cat > '$staging.tar'; sha256sum '$staging.tar' | cut -d' ' -f1" < "$scratch/src.tar")"
  if [[ "$transit_sha" != "$archive_sha" ]]; then
    remote "rm -f '$staging.tar'"
    echo "archive digest mismatch in transit" >&2
    exit 1
  fi
  remote "set -e; mkdir '$staging'; tar -x -C '$staging' -f '$staging.tar'; rm -f '$staging.tar'; cd '$staging'; m=\$($manifest_cmd); if [ \"\$m\" != '$local_manifest' ]; then cd /; rm -rf '$staging'; echo 'content manifest mismatch' >&2; exit 1; fi; printf '%s\n' '$(receipt)' > .dev2-mirror.json; chmod -R a-w .; mv '$staging' '$remote_dir'"
fi

remote_manifest="$(remote "cd '$remote_dir' && $manifest_cmd")"
recorded="$(remote "cat '$remote_dir/.dev2-mirror.json'" | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["commit"], r["tree"], r["content_manifest_sha256"])')"
if [[ "$remote_manifest" != "$local_manifest" || "$recorded" != "$sha $tree $local_manifest" ]]; then
  echo "mirror of $sha on $alias_name does not match the commit" >&2
  exit 1
fi
printf 'mirror ok: %s commit=%s tree=%s content_manifest=%s files=%s dir=%s\n' \
  "$alias_name" "$sha" "$tree" "$local_manifest" "$file_count" "$remote_dir"
