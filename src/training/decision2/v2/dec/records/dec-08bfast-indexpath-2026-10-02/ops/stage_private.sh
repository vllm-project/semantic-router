#!/usr/bin/env bash
# Private release inputs of a 0.8B / 2B Index-first successor on node A (/data/dev2/private/release/ixf-<key>, mode
# 700), run on the workstation after ix.sh boot. Prints hashes, counts and equal / differs only, never a value.
#   1. from node C: the full-panel and transfer-only bootstraps of the BF16 release weights vs the current release,
#      both IX1 run receipts and the row-level contamination audit of the arm's TRAIN file
#   2. kit-run records (index_runs.py) of every tier's current release weights: node C for 0.6B / 0.8B / 2B / 4B /
#      9B, node D for 27B; this tier's point is the BF16 run, a tier released by this continuation earlier its new run
#   3. node A: MODEL_MANIFEST.json of every tier's current main from the Hub (this tier: the IX1-scored package),
#      the board snapshot (2026-09-28), then python -m v2.release.card_index (the default generator: board-served
#      parameter counts, the audited footnote) -> decision-index-card.json; each family point is compared with the
#      latest earlier input (the 9B release's) and only "equal" / "differs" is printed
# Usage: stage_private.sh MIRROR_SHA 0p8b|2b BOARD_JSON
set -euo pipefail
SHA=${1:?MIRROR_SHA} KEY=${2:?0p8b|2b} BOARD=${3:?board snapshot}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
A=$(grep '^node-a=' "$NODES" | cut -d= -f2-) C=$(grep '^node-c=' "$NODES" | cut -d= -f2-) D=$(grep '^node-d=' "$NODES" | cut -d= -f2-)
ona() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$A" "$@"; }
onc() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$C" "$@"; }
ond() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$D" "$@"; }
KEYF="-i /root/.ssh/d2_temp_cd -o BatchMode=yes -o ConnectTimeout=20"
M=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2
OPS=$M/v2/dec/records/dec-08bfast-indexpath-2026-10-02/ops
R=/data/dev2/private/eval/index021/ix1
P=/data/dev2/private/release/ixf-$KEY
case "$KEY" in
  0p8b) TIER=0.8B NAME=M16-08b-RA-a75-bf16 REF=DEV2.0-0.8B AUDIT=out-08bRA PKG=/data/dev2/models/ix1/dec-indexpath/08b-RA-a75-bf16-rbede7938 ;;
  2b) TIER=2B NAME=M16-2b-RASD-bf16 REF=DEV2.0-2B AUDIT=out-2bRA PKG=/data/dev2/models/ix1/dec-indexpath/2b-RASD-bf16-ra53cf66a ;;
  *) echo "tier key 0p8b or 2b" >&2; exit 2 ;;
esac
declare -A REPO=([0.6B]=Decision-2.0-Kai-0.6B [0.8B]=Decision-2.0-Eos-0.8B [2B]=Decision-2.0-Sol-2B
  [4B]=Decision-2.0-Nox-4B [9B]=Decision-2.0-Lux-9B [27B]=Decision-2.0-Vega-27B)
ona "test -f $OPS/index_runs.py" || { echo "mirror $SHA is not on node A" >&2; exit 2; }
onc "test -f $OPS/index_runs.py" || { echo "mirror $SHA is not on node C" >&2; exit 2; }
ond "test -f $OPS/index_runs.py" || { echo "mirror $SHA is not on node D" >&2; exit 2; }
ona "umask 077; mkdir -p $P/manifests && chmod 700 $P"
pull_c() { # node-C path, node-A name
  ona "ssh $KEYF $C 'cat $1' > $P/$2.part && mv $P/$2.part $P/$2 && chmod 600 $P/$2"
  [ "$(onc "sha256sum < $1")" = "$(ona "sha256sum < $P/$2")" ] || { echo "copy of $1 differs" >&2; exit 3; }
  echo "$2 $(ona "sha256sum < $P/$2 | cut -c1-12")"
}
pull_c "$R/runs/$NAME/paired-boot-full-vs-ref.json" paired-boot-full-vs-current.json
pull_c "$R/runs/$NAME/paired-boot-vs-ref.json" paired-boot-transfer-vs-current.json
pull_c "$R/runs/$NAME/merged/receipt.json" ix1-receipt.json
pull_c "$R/runs/$REF/merged/receipt.json" ix1-reference-receipt.json
pull_c "$R/audit/$AUDIT/audit.json" contamination-audit.json
# The current main's identity of every tier, from its MODEL_MANIFEST.json on the Hub (this tier: the scored package).
for t in 0.6B 0.8B 2B 4B 9B 27B; do
  if [ "$t" = "$TIER" ]; then
    ona "mkdir -p $P/manifests/$t && cp $PKG/MODEL_MANIFEST.json $P/manifests/$t/MODEL_MANIFEST.json"
  else
    ona "mkdir -p $P/manifests/$t && /data/dev2/tools/hf-cli/bin/python -c 'import shutil, sys; from huggingface_hub import HfApi, hf_hub_download; r = sys.argv[1]; rev = HfApi().model_info(r).sha; shutil.copy(hf_hub_download(r, \"MODEL_MANIFEST.json\", revision=rev), sys.argv[2]); print(r, rev[:8])' llm-semantic-router/${REPO[$t]} $P/manifests/$t/MODEL_MANIFEST.json"
  fi
done
ids=$(ona "for t in 0.6B 0.8B 2B 4B 9B 27B; do python3 -c 'import json,sys; print(sys.argv[2], json.load(open(sys.argv[1]))[\"identity\"][\"model_sha256\"])' $P/manifests/\$t/MODEL_MANIFEST.json \$t; done")
echo "$ids" | cut -c1-20
# Kit runs: every node C / node D IX1 run whose weights are one of those identities.
LIST="for r in $R/runs/*/merged/receipt.json; do python3 -c 'import json,sys; print(sys.argv[1].split(\"/\")[-3], json.load(open(sys.argv[1]))[\"model_source\"][\"model_sha256\"])' \$r; done"
for node in c d; do
  listing=$(if [ $node = c ]; then onc "$LIST"; else ond "$LIST"; fi)
  args=""
  while read -r run id; do
    [ -n "$id" ] && grep -q " $id\$" <<< "$ids" && args="$args --run $run=$R/runs/$run"
  done <<< "$listing"
  [ -n "$args" ] || continue
  out=/data/dev2/private/release/ixf-$KEY-runs-$node.json
  if [ $node = c ]; then onc "umask 077; mkdir -p /data/dev2/private/release; python3 $OPS/index_runs.py $args --out $out" | cut -c1-400;
  else ond "umask 077; mkdir -p /data/dev2/private/release; python3 $OPS/index_runs.py $args --out $out" | cut -c1-400; fi
  if [ $node = c ]; then ona "ssh $KEYF $C 'cat $out' > $P/runs-$node.json"; else ona "ssh $KEYF $D 'cat $out' > $P/runs-$node.json"; fi
done
ona "umask 077; cat > $P/index-latest.json" < "$BOARD"
[ "$(sha256sum < "$BOARD")" = "$(ona "sha256sum < $P/index-latest.json")" ] || { echo "board copy differs" >&2; exit 3; }
ona "cd $M && PYTHONPATH=$M python3 -m v2.release.card_index --runs $P/runs-c.json $P/runs-d.json --board $P/index-latest.json \
  --snapshot 2026-09-28 --manifests $P/manifests --out $P/decision-index-card.json" | sed -E 's/"out": "[^"]*", //'
ona "chmod 600 $P/decision-index-card.json; python3 - $P/decision-index-card.json /data/dev2/private/release/ka13ib/decision-index-card.json" << 'EOF'
import json, sys
new, old = (json.load(open(p)) for p in sys.argv[1:3])
fields = ("parameters", "loaded_parameters", "balanced_skill", "areas", "model_sha256", "kit_index_sha256")
old_family = {p["tier"]: p for p in old["family"]}
for p in new["family"]:
    o = old_family[p["tier"]]
    same = all(p.get(k) == o.get(k) for k in fields)
    print(p["tier"], "equal to the 9B release's input" if same else "differs (new weights)", p["model_sha256"][:12])
print("decision1 / entrants equal:", new["decision1"] == old["decision1"] and new["entrants"] == old["entrants"],
      "footnote equal:", new["footnote"] == old["footnote"])
EOF
