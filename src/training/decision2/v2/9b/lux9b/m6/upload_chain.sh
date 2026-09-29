#!/usr/bin/env bash
# usage: upload_chain.sh NODE LOCAL_CHAIN_FILE REMOTE_PATH
# Local side, 16:00 chain rule step 1: copies a non-empty chain file to NODE (an ssh
# destination such as root@<node A>) with scp, then in a separate ssh command checks that the remote size and
# SHA-256 equal the local ones. Never launches; prints the launch.sh arguments to use next.
set -euo pipefail
node=$1; local_file=$2; remote=$3
SSH=(-o ConnectTimeout=15)
[ -s "$local_file" ] || { echo "$local_file is missing or empty" >&2; exit 3; }
bash -n "$local_file" || { echo "$local_file does not parse" >&2; exit 3; }
size=$(stat -c %s "$local_file")
sha=$(sha256sum "$local_file" | cut -c1-64)
q=$(printf %q "$remote")
qd=$(printf %q "$(dirname "$remote")")
# shellcheck disable=SC2029  # the quoted remote path is meant to expand locally
ssh "${SSH[@]}" "$node" "mkdir -p $qd"
scp "${SSH[@]}" -q "$local_file" "$node:$remote"
# shellcheck disable=SC2029
read -r rsize rsha < <(ssh "${SSH[@]}" "$node" "echo \$(stat -c %s $q) \$(sha256sum $q | cut -c1-64)")
if [ "$rsize" != "$size" ] || [ "$rsha" != "$sha" ]; then
  echo "MISMATCH: local $size $sha, remote $rsize $rsha" >&2; exit 3
fi
echo "verified $remote on $node: $size bytes sha256 $sha"
echo "launch next (separate step): bash \$L/launch.sh CHAIN $remote $size $sha"
