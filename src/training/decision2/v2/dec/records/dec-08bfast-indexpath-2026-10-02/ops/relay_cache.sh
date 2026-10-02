#!/usr/bin/env bash
# The persisted Triton autotune caches of an M16 point's formal runs (node B: formal/m16/m16-<point>-cache for typed
# FINAL / CSS15 / public 231 and m16-<point>-mlx-cache for mlx-diag) -> node A at the same paths, for the release
# parity in the scored runtime. Run on the workstation: nodes A and B do not reach each other, so the copy goes
# through a transit directory on node E over the temporary transfer key (as m16-relay.sh). Each copy must equal the
# after-manifest the formal run recorded (M6-RECEIPT.json cache.after_manifest_sha256); nothing is overwritten.
# Usage: relay_cache.sh <point>      e.g. 08b-RA-a75, 2b-RASD-a25
set -euo pipefail
POINT=${1:?point}
[[ "$POINT" =~ ^(08b|2b)-[A-Za-z0-9-]+$ ]] || { echo "bad point $POINT" >&2; exit 2; }
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
A=$(grep '^node-a=' "$NODES" | cut -d= -f2-) B=$(grep '^node-b=' "$NODES" | cut -d= -f2-) E=$(grep '^node-e=' "$NODES" | cut -d= -f2-)
on() { local h=$1; shift; ssh -o BatchMode=yes -o ConnectTimeout=20 "$h" "$@"; }
KEY="-i /root/.ssh/d2_temp_cd -o BatchMode=yes -o ConnectTimeout=20"
F=/data/dev2/runs/dec/formal/m16
T=/data/dev2/runs/dec/ixf-relay
tm() { echo "cd $1 && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -d' ' -f1"; }
for run in "m16-$POINT" "m16-$POINT-mlx"; do
  dir=$F/$run-cache
  if [[ "$run" == *-mlx ]] && ! on "$A" "test -f $F/$run/M6-RECEIPT.json"; then echo "$run: no mlx-diag run"; continue; fi
  want=$(on "$A" "python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"cache\"][\"after_manifest_sha256\"])' $F/$run/M6-RECEIPT.json")
  [[ "$want" =~ ^[0-9a-f]{64}$ ]] || { echo "no after-manifest for $run on node A" >&2; exit 3; }
  if on "$A" "test -d $dir"; then
    [ "$(on "$A" "$(tm "$dir")")" = "$want" ] || { echo "node A $dir exists and is not the formal run's cache" >&2; exit 3; }
    echo "$run cache already on node A ($want)"; continue
  fi
  [ "$(on "$B" "$(tm "$dir")")" = "$want" ] || { echo "node B $dir is not its after-manifest $want" >&2; exit 3; }
  stamp=$(date -u +%Y%m%dT%H%M%SZ)-$$
  on "$B" "ssh $KEY '$E' 'mkdir -p $T/$stamp' && rsync -a -e 'ssh $KEY' '$dir/' '$E:$T/$stamp/cache/'"
  on "$A" "mkdir -p $dir.part && rsync -a -e 'ssh $KEY' '$E:$T/$stamp/cache/' '$dir.part/' && ssh $KEY '$E' 'rm -rf $T/$stamp'"
  got=$(on "$A" "$(tm "$dir.part")")
  [ "$got" = "$want" ] || { echo "node A copy of $run cache is $got, not $want" >&2; exit 3; }
  on "$A" "mv -T $dir.part $dir && (cd $dir && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > $dir.sha256"
  echo "$run cache relayed to node A: $(on "$A" "find $dir -type f | wc -l") files, manifest $want"
done
