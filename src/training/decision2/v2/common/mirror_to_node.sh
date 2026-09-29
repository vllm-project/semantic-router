#!/usr/bin/env bash
# Exact-mirror one pushed commit of this repository onto an experiment node.
#
# Usage:
#   mirror_to_node.sh [--verify] [--path <repo-subtree>] <node-alias> <commit> [<target-root>]
#
# The tree lands in <target-root>/<full-sha>/ (default target root /data/dev2/src)
# with <full-sha>/.dev2-mirror.json recording commit, tree, archive and content
# manifest SHA-256; the mirrored files are made read-only. Re-running for an
# existing mirror only re-verifies it. A directory created earlier by a plain
# `git archive | tar -x` is adopted only if its content manifest matches exactly.
# --path mirrors only one subtree (for example src/training/decision2, ~21 MB
# instead of ~444 MB) into <full-sha>-<subtree with / replaced by _>/, keeping
# repository-relative paths inside it.
#
# Concurrent runs for the same mirror are safe. The node-side steps that create
# or adopt a mirror hold an exclusive flock on <target-root>/.locks/<mirror>.lock
# (waiting at most ${DEV2_MIRROR_LOCK_TIMEOUT:-900} s). An existing mirror is
# used only if its receipt names this commit and tree and the content manifest
# recomputed over every file on the node matches; otherwise the run fails and
# leaves the directory untouched. A new copy is extracted into a private staging
# directory next to the target and renamed into place with `mv -T`, which fails
# rather than moving the copy into a directory that already exists.
#
# <node-alias> is looked up in ${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
# (private, never committed; lines "node-a=user@host"). An alias absent from that
# file is passed to ssh unchanged, so ssh-config host aliases also work.
# Only commits contained in a remote-tracking branch are mirrored; run
# `git fetch origin` first if a just-pushed commit is reported as unpushed.
set -euo pipefail

usage() {
  sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//' >&2
  exit 2
}

verify_only=0
subtree=""
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --verify) verify_only=1; shift ;;
    --path) subtree="${2%/}"; shift 2 ;;
    *) usage ;;
  esac
done
[[ $# -eq 2 || $# -eq 3 ]] || usage
alias_name="$1"
commit_arg="$2"
target_root="${3:-/data/dev2/src}"
[[ "$target_root" == /* ]] || { echo "target root must be absolute" >&2; exit 2; }
lock_timeout="${DEV2_MIRROR_LOCK_TIMEOUT:-900}"
[[ "$lock_timeout" =~ ^[0-9]+$ ]] || { echo "DEV2_MIRROR_LOCK_TIMEOUT must be whole seconds" >&2; exit 2; }

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
mirror_name="$sha${subtree:+-${subtree//\//_}}"
remote_dir="$target_root/$mirror_name"
remote() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$dest" "$@"; }

# Runs a bash script on the node; each NAME=VALUE argument becomes a quoted assignment above it.
remote_script() {
  local body="$1" header="" kv
  shift
  for kv in "$@"; do
    header+="${kv%%=*}=$(printf '%q' "${kv#*=}")"$'\n'
  done
  printf '%s%s\n' "$header" "$body" | remote bash -s
}

manifest_cmd='find . -type f ! -name .dev2-mirror.json -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d" " -f1'

# Node side, under the lock: reuse a mirror another run finished meanwhile, adopt an identical
# plain extraction, or extract, verify and rename a new copy. The uploaded tarball is always removed.
node_place_script() {
  cat <<'NODE'
set -euo pipefail
mkdir -p "$root/.locks"
exec 9>"$root/.locks/$name.lock"
if ! flock -w "$timeout" 9; then
  rm -f "$tar"
  echo "timed out after ${timeout} s waiting for $root/.locks/$name.lock" >&2
  exit 75
fi
manifest() { (cd "$1" && find . -type f ! -name .dev2-mirror.json -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d" " -f1); }
dir="$root/$name"
if [ -f "$dir/.dev2-mirror.json" ]; then
  rm -f "$tar"
  echo reused
elif [ -e "$dir" ] || [ -L "$dir" ]; then
  rm -f "$tar"
  if [ -L "$dir" ] || [ ! -d "$dir" ] || [ "$(manifest "$dir")" != "$want" ]; then
    echo "$dir exists but differs from commit $sha; refusing to touch it" >&2
    exit 1
  fi
  chmod u+w "$dir"
  printf '%s\n' "$receipt" > "$dir/.dev2-mirror.json"
  chmod -R a-w "$dir"
  echo adopted
else
  staging="$(mktemp -d "$root/.incoming-$name.XXXXXX")"
  trap 'chmod -R u+w "$staging" 2>/dev/null || true; rm -rf "$staging"' EXIT
  chmod 755 "$staging"
  tar -x -C "$staging" -f "$tar"
  rm -f "$tar"
  if [ "$(manifest "$staging")" != "$want" ]; then
    echo "content manifest mismatch after extraction" >&2
    exit 1
  fi
  printf '%s\n' "$receipt" > "$staging/.dev2-mirror.json"
  chmod -R a-w "$staging"
  mv -T "$staging" "$dir"
  trap - EXIT
  echo created
fi
NODE
}

# Node side, read-only: "<content manifest> <nested .incoming-* entries>", then the receipt.
node_check_script() {
  cat <<'NODE'
set -euo pipefail
cd "$dir"
m="$(find . -type f ! -name .dev2-mirror.json -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d" " -f1)"
printf '%s %s\n' "$m" "$(find . -mindepth 1 -name '.incoming-*' | wc -l)"
cat .dev2-mirror.json
NODE
}

remote_tar=""
scratch="$(mktemp -d)"
cleanup() {
  rm -rf "$scratch"
  if [[ -n "$remote_tar" ]]; then
    remote "rm -f '$remote_tar'" || true
  fi
}
trap cleanup EXIT
if [[ -n "$subtree" ]]; then
  git -C "$repo_root" cat-file -e "$sha:$subtree" || { echo "$subtree is not in $sha" >&2; exit 1; }
  git -C "$repo_root" archive --format=tar "$sha" "$subtree" > "$scratch/src.tar"
else
  git -C "$repo_root" archive --format=tar "$sha" > "$scratch/src.tar"
fi
archive_sha="$(sha256sum "$scratch/src.tar" | cut -d' ' -f1)"
mkdir "$scratch/tree"
tar -x -C "$scratch/tree" -f "$scratch/src.tar"
local_manifest="$(cd "$scratch/tree" && eval "$manifest_cmd")"
file_count="$(cd "$scratch/tree" && find . -type f | wc -l | tr -d ' ')"
receipt() {
  printf '{"schema":"dev2-mirror/1","commit":"%s","tree":"%s","path":"%s","archive_sha256":"%s","content_manifest_sha256":"%s","files":%s,"created_utc":"%s"}' \
    "$sha" "$tree" "${subtree:-.}" "$archive_sha" "$local_manifest" "$file_count" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

state="$(remote "if [ -f '$remote_dir/.dev2-mirror.json' ]; then echo receipt; elif [ -e '$remote_dir' ] || [ -L '$remote_dir' ]; then echo stray; else echo absent; fi")"

if [[ "$state" != "receipt" && "$verify_only" -eq 1 ]]; then
  echo "no verified mirror of $sha on $alias_name ($state)" >&2
  exit 1
fi

action=verified
if [[ "$state" != "receipt" ]]; then
  upload="$(remote "set -e; mkdir -p '$target_root'; t=\$(mktemp '$target_root/.incoming-$mirror_name.tar.XXXXXX'); cat > \"\$t\"; printf '%s %s\n' \"\$t\" \"\$(sha256sum < \"\$t\" | cut -d' ' -f1)\"" < "$scratch/src.tar")"
  remote_tar="${upload% *}"
  if [[ "${upload##* }" != "$archive_sha" ]]; then
    echo "archive digest mismatch in transit" >&2
    exit 1
  fi
  action="$(remote_script "$(node_place_script)" \
    "root=$target_root" "name=$mirror_name" "tar=$remote_tar" "timeout=$lock_timeout" \
    "sha=$sha" "want=$local_manifest" "receipt=$(receipt)")"
  action="${action##*$'\n'}"
  remote_tar=""
fi

check="$(remote_script "$(node_check_script)" "dir=$remote_dir")" \
  || { echo "cannot read the mirror of $sha on $alias_name" >&2; exit 1; }
read -r remote_manifest nested <<< "${check%%$'\n'*}"
recorded="$(python3 -c 'import json,sys; r=json.loads(sys.stdin.read()); print(r["commit"], r["tree"], r["content_manifest_sha256"])' <<< "${check#*$'\n'}")"
if [[ "$nested" != 0 ]]; then
  echo "$remote_dir holds $nested nested staging entries (.incoming-*); refusing to use it" >&2
  exit 1
fi
if [[ "$remote_manifest" != "$local_manifest" || "$recorded" != "$sha $tree $local_manifest" ]]; then
  echo "mirror of $sha on $alias_name does not match the commit; refusing to use it" >&2
  exit 1
fi
printf 'mirror ok: %s commit=%s tree=%s path=%s content_manifest=%s files=%s dir=%s action=%s\n' \
  "$alias_name" "$sha" "$tree" "${subtree:-.}" "$local_manifest" "$file_count" "$remote_dir" "$action"
