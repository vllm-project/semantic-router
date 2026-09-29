#!/usr/bin/env bash
# Stage the ~27B event-3 assets on node A (node plan (b)); run on the workstation from the worktree.
# Usage: event3-stage-nodeA.sh --src MIRROR --peers27 K,K [--c27 f1|f2] [--dry-run]
# Executes the table's stage_node_a steps for the selected 27B rows: "copy" streams a node-B file or
# directory to node A through this workstation (tar over the two SSH sessions; only small items:
# package, frozen autotune caches, T = 1 calibration, stored formal typed-FINAL predictions, runtime
# source); "hf" downloads a Hub repo at its pinned revision on node A. Existing destinations are left
# as they are. Then `event3.sh --verify-only` on node A checks every pinned hash (CPU only).
# Node addresses come from ~/.config/decision2/nodes.env and are never printed. Nothing here reads
# or writes the sealed directory, the C1 event directory or any GPU.
set -euo pipefail

SRC="" C27=f1 PEERS27="" DRY=0
while [ $# -gt 0 ]; do
  case $1 in
  --src) SRC=$2; shift 2 ;;
  --c27) C27=$2; shift 2 ;;
  --peers27) PEERS27=$2; shift 2 ;;
  --dry-run) DRY=1; shift ;;
  *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$SRC" ] && [ -n "$PEERS27" ] || { echo "--src MIRROR and --peers27 K,K are required" >&2; exit 2; }
here=$(cd "$(dirname "$0")/../../.." && pwd)
head=$(git -C "$here" rev-parse HEAD)
[ "${SRC%%-*}" = "$head" ] || { echo "the worktree HEAD $head is not the mirror $SRC" >&2; exit 1; }
nodes="$HOME/.config/decision2/nodes.env"
NA=$(grep '^node-a=' "$nodes" | cut -d= -f2-)
NB=$(grep '^node-b=' "$nodes" | cut -d= -f2-)
ssh_a() { ssh -o BatchMode=yes "$NA" "$@"; }
ssh_b() { ssh -o BatchMode=yes "$NB" "$@"; }

plan=$(mktemp)
trap 'rm -f "$plan"' EXIT
(cd "$here" && python3 -m v2.eval.sealed.event3 plan --table v2/eval/sealed/event3-models.json \
  --models cand27 --c27 "$C27" --peers27 "$PEERS27" --output "$plan" >/dev/null)
mapfile -d '' -t steps < <(cd "$here" && python3 -m v2.eval.sealed.event3 stage --plan "$plan")
for ((i = 0; i < ${#steps[@]}; i += 6)); do
  kind=${steps[i]} key=${steps[i + 1]} from=${steps[i + 2]} to=${steps[i + 3]}
  repo=${steps[i + 4]} rev=${steps[i + 5]}
  if ssh_a "test -e $(printf %q "$to")"; then
    echo "$key: $to exists on node A (left as is)"
    continue
  fi
  if [ "$kind" = copy ]; then
    echo "$key: copy node B $from -> node A $to"
    [ "$DRY" = 1 ] && continue
    ssh_b "tar -C $(printf %q "$(dirname "$from")") -cf - $(printf %q "$(basename "$from")")" |
      ssh_a "set -e; mkdir -p $(printf %q "$(dirname "$to")"); t=\$(mktemp -d $(printf %q "$(dirname "$to")")/.stage.XXXXXX); tar -C \"\$t\" -xf -; mv \"\$t\"/$(printf %q "$(basename "$from")") $(printf %q "$to"); rmdir \"\$t\""
  elif [ "$kind" = hf ]; then
    echo "$key: download $repo@$rev on node A -> $to"
    [ "$DRY" = 1 ] && continue
    ssh_a "set -e; mkdir -p $(printf %q "$(dirname "$to")"); HF_HUB_CACHE=/data/dev2/hf-cache /usr/local/bin/hf download $(printf %q "$repo") --revision $(printf %q "$rev") --local-dir $(printf %q "$to") >/dev/null"
  else
    echo "$key: unknown staging kind $kind" >&2
    exit 1
  fi
done
[ "$DRY" = 1 ] && exit 0
ssh_a "bash /data/dev2/src/$SRC/src/training/decision2/v2/eval/sealed/event3.sh --verify-only --src $SRC --c27 $C27 --peers27 $PEERS27"
