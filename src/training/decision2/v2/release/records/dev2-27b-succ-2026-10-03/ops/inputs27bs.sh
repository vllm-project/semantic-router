#!/usr/bin/env bash
# Decision-2.0-Vega-27B successor over M6-IBxIB2-m50 (27B owner, continuation #7; workstation side, control only): the
# private Index files of ARM -> node A /data/dev2/private/release/27bs/ARM/ (mode 700; values stay private), pulled by node A
# from node D over the temporary transfer key (never through the workstation), each file's SHA-256 equal at both ends:
#   boot-vs-current.json   the full-panel paired bootstrap vs the current revision's run, M6-IBxIB2-m50 (m6-index.sh
#                          score with M6_INDEX_BASE=M6-IBxIB2-m50)
#   receipt.json           the ARM's IX1 run receipt; base-receipt.json: M6-IBxIB2-m50's
#   kit-index.json         the kit index.json of the ARM's IX1 run (card_index's 27B point)
#   package-manifest.json  the restaged package's MODEL_MANIFEST.json (identity and loaded count of these weights)
#   audit.json             the row-level contamination audit of the ARM's training files (m6-index.sh audit-arm)
# The formal run, gates and mlx-diag collection are on node A already (m6-stage-a.sh). Prints digests' prefixes only.
# Usage: inputs27bs.sh ARM
set -euo pipefail
ARM=${1:?ARM}
[[ "$ARM" =~ ^(M7-(IB124ML|IB14ML)|M8-(IB14|IB124)|X7-(IBxIB2xIB14ML|4ARM)|X8-(IBxIB2-8|ML)|X9-(LRH2|LRH2xM50|LRH|LRHxM50|ML0|IBxIB2-10|IBLRH|IBLRHxM50|LRHALL))$ ]] || { echo "bad ARM $ARM" >&2; exit 2; }
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
addr() { awk -F= -v k="node-$1" '$1 == k { print substr($0, length(k) + 2); exit }' "$NODES"; }
on() { local n=$1; shift; ssh -o BatchMode=yes -o ConnectTimeout=30 "$(addr "$n")" "$@" < /dev/null; }
KEY="-i /root/.ssh/d2_temp_cd -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes"
IX=/data/dev2/private/eval/index021/ix1
PRIV=/data/dev2/private/release/27bs/$ARM
D=$(addr d)
on a "umask 077; mkdir -p $PRIV && chmod 700 /data/dev2/private/release/27bs $PRIV"
for pair in "$IX/runs/$ARM/paired-boot-vs-m6-ibxib2-m50.json:boot-vs-current.json" "$IX/runs/$ARM/merged/receipt.json:receipt.json" \
  "$IX/runs/M6-IBxIB2-m50/merged/receipt.json:base-receipt.json" "$IX/runs/$ARM/merged/kit/index.json:kit-index.json" \
  "/data/dev2/models/ix1/m6/$ARM-re876fbe/MODEL_MANIFEST.json:package-manifest.json" \
  "$IX/runs/m6-audit-$ARM/audit.json:audit.json"; do
  src=${pair%%:*} dst=${pair#*:}
  on d "test -f $src" || { echo "node D has no $src" >&2; exit 3; }
  on a "test ! -e $PRIV/$dst" || { echo "$PRIV/$dst exists" >&2; exit 3; }
  on a "umask 077; rsync -a -e 'ssh $KEY' root@${D#*@}:$src $PRIV/$dst && chmod 600 $PRIV/$dst"
  [ "$(on d "sha256sum < $src")" = "$(on a "sha256sum < $PRIV/$dst")" ] || { echo "$dst differs" >&2; exit 3; }
  echo "$dst $(on a "sha256sum < $PRIV/$dst | cut -c1-16")"
done
