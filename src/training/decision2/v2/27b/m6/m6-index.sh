#!/usr/bin/env bash
# ~27B M6 private Index run of one frozen formal finalist (workstation side; prereg amendments 4-5, the Index path of
# COORDINATION 2026-10-02 02:05 / 03:40: Index runs for Index-path candidates and the chosen successor, item 1' needs a
# significantly positive paired Index delta vs A20r; runs use the eval allowance on node C GPU1-7 and node D GPU4-7).
# IX1's harness (v2/eval/ix1: image host2, kit 87d4650b, panel-8, the 86-request parity gate, dual scoring) set up as
# IX1 ran the M5-L128 diagnostic: the frozen soup checkpoint restaged into DEV2.0-27B 4e89288d with the forward-budget
# runtime (fix2 package, e876fbe; T = 1), A20r's frozen autotune cache, the same panel. The mirror carries the ARM's
# DIAGNOSTIC entry in v2/eval/ix1/launch.sh and is on node D (and node C when it takes shards; mirror_to_node.sh ...
# node-d / node-c). Index values stay in the nodes' /data/dev2/private/eval/index021/ix1/ and the local private
# folder; this script prints none.
# Usage: m6-index.sh MIRROR_SHA ARM STAGE
#   plan     prints which shards run on which node and GPU for M6_INDEX_GPUS (no node is touched)
#   stage    node B m6/ARM/checkpoint -> node D /data/dev2/models/ix1/m6/ARM-ckpt over node B's transfer key (SHA-256
#            lists equal), then v2.eval.ix1.restage -> /data/dev2/models/ix1/m6/ARM-re876fbe with the model SHA-256 of
#            m6/ARM/package/PACKAGE.json; checks the loaded count (27,497,508,864) and the identity
#   stage-c  after stage: the same on node C (the fix2 template relayed node D -> node C through node B once, its
#            SHA-256 list equal to node D's), and node C's restaged package must equal node D's file for file
#   control  once for M6 (any ARM): A20r's own package through the same runtime (DEV2.0-27B-budget) read by its entry
#            point over the 86 compatibility requests on node D GPU4, vs IX1's A20r kit results -> must pass
#   parity   launch.sh parity on node D GPU4 (detached; waits for it, ~0.1 GPU-h); parity/ARM/parity.json must pass
#   run      m6-index-run.sh detached on each node with shards (log ix1/logs/m6-index-ARM-NODE.log): 8 shards on the
#            entries of M6_INDEX_GPUS (default "d4 d5 d6 d7"; dN = node D GPU N in 0-7 (GPU0-3 are M6's own leases, used
#            once its seeds ended, under an eval-ix1 owner naming the arm), cN = node C GPU N in 1-7, a bare
#            N = node D), shard k on entry k mod n; node C needs stage-c, and gets node D's parity record (SHA-256 equal)
#   status   per node and shard: records written, ended, exit code; GPU-h so far
#   collect  after node C's shards ended 0: their result files (no Triton cache, no home) and launcher records ->
#            node D's run directory, SHA-256 lists equal; never overwrites a node D shard
#   score    after all 8 shards ended 0 on node D: score.sh (merge, port + kit scoring, compare incl. the frontier peer),
#            family_delta vs A20r's IX1 run (merged-budget) and vs M5-L128's, and the paired bootstrap vs A20r
#            (paired_boot: 2,000 replicates, seed 20261002; item 1' (b) = its 95% lower bound > 0); private outputs
#            copied to ~/code/decision2-program/private/m6/ARM/ (mode 700)
#   audit    once for M6 (any ARM; CPU): v2.eval.ix1.contamination of a20ib12pn (it contains every arm's rows) and
#            a20ib1x against the panel -> ix1/runs/m6-audit (backs the card's "audited at row level" footnote)
#   release  owner files that launch.sh wrote for ARM on node D GPU4-7 and node C GPU1-7 -> status released (no ARM
#            container running)
set -euo pipefail
SHA=${1:?MIRROR_SHA} ARM=${2:?ARM} STAGE=${3:?STAGE}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
[[ "$ARM" =~ ^M6-(IB|IBX|IB2|IB2PN)$ ]] || { echo "bad ARM $ARM" >&2; exit 2; }
placement() {  # one line per node with shards: "NODE SHARDS GPU..." (shard k on entry k mod n of M6_INDEX_GPUS)
  local entries=() norm=() e k node g
  local -A seen=() shards=() gpus=()
  read -r -a entries <<< "${M6_INDEX_GPUS:-d4 d5 d6 d7}"
  (( ${#entries[@]} >= 1 && ${#entries[@]} <= 8 )) || { echo "M6_INDEX_GPUS: 1-8 entries" >&2; return 2; }
  for e in "${entries[@]}"; do
    [[ "$e" =~ ^[4-7]$ ]] && e=d$e
    [[ "$e" =~ ^(d[0-7]|c[1-7])$ ]] ||
      { echo "M6_INDEX_GPUS: node D GPU0-7 (d0-d7) or node C GPU1-7 (c1-c7), not '$e'" >&2; return 2; }
    [ -z "${seen[$e]:-}" ] || { echo "M6_INDEX_GPUS lists $e twice" >&2; return 2; }
    seen[$e]=1
    norm+=("$e")
  done
  for k in 0 1 2 3 4 5 6 7; do
    e=${norm[$((k % ${#norm[@]}))]} node=${e:0:1} g=${e:1}
    shards[$node]+="${shards[$node]:+,}$k"
    [[ " ${gpus[$node]:-} " == *" $g "* ]] || gpus[$node]+="${gpus[$node]:+ }$g"
  done
  for node in d c; do
    [ -z "${shards[$node]:-}" ] || echo "$node ${shards[$node]} ${gpus[$node]}"
  done
}
if [ "$STAGE" = plan ]; then
  plan=$(placement) || exit 2
  while read -r node shards gpus; do
    echo "node $node: shards $shards on GPU $gpus"
    read -r -a gl <<< "$gpus"
    M6_INDEX_DRY=1 bash "$(dirname "$0")/m6-index-run.sh" "$SHA" "$ARM" "$node" "$shards" "${gl[@]}" | sed 's/^/  /'
  done <<< "$plan"
  exit 0
fi
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
B=$(grep '^node-b=' "$NODES" | cut -d= -f2-) C=$(grep '^node-c=' "$NODES" | cut -d= -f2-)
D=$(grep '^node-d=' "$NODES" | cut -d= -f2-)
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=4)
onb() { "${SSH[@]}" "$B" "$@"; }
onc() { "${SSH[@]}" "$C" "$@"; }
ond() { "${SSH[@]}" "$D" "$@"; }
XFER="ssh -i /root/.ssh/d2_temp_cd -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes"
M=/data/dev2/src/$SHA-src_training_decision2
S=$M/src/training/decision2
R=/data/dev2/private/eval/index021/ix1
R6=/data/dev2/runs/27b/m6
MD=/data/dev2/models/ix1/m6
FIX2=/data/dev2/models/ix1/fix2/DEV2.0-27B-4e89288d-re876fbe
PKG=$MD/$ARM-re876fbe CK=$MD/$ARM-ckpt
LOADED=27497508864
LOCAL=${M6_INDEX_LOCAL:-$HOME/code/decision2-program/private/m6}/$ARM
TAG=ix1-$(tr 'A-Z.' 'a-z_' <<< "$ARM")-
has_mirror() {  # NODE: the mirror with ARM's DIAGNOSTIC entry is on that node
  "on$1" "test -f $S/v2/eval/ix1/launch.sh" || { echo "mirror $SHA is not on node ${1^^}" >&2; return 2; }
  "on$1" "grep -q '^  \[$ARM\]=\"DEV2.0-27B [0-9a-f]* $PKG\"' $S/v2/eval/ix1/launch.sh" ||
    { echo "mirror $SHA has no DIAGNOSTIC entry $ARM -> $PKG" >&2; return 2; }
}
sums() { echo "cd $1 && find . -type f | sort | xargs -P 8 -n 4 sha256sum | sort -k2"; }
model_sha() {  # the frozen package's identity; M6_INDEX_FROM_SOUP=1: the soup's, for a candidate without a formal run
  local model soup
  onb "test -f $R6/$ARM/checkpoint/soup_manifest.json" || { echo "no frozen soup for $ARM on node B" >&2; return 3; }
  soup=$(onb "python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"output\"][\"model_sha256\"])' $R6/$ARM/checkpoint/soup_manifest.json")
  [[ "$soup" =~ ^[0-9a-f]{64}$ ]] || { echo "bad model SHA-256 in soup_manifest.json" >&2; return 3; }
  if onb "test -f $R6/$ARM/package/PACKAGE.json"; then
    model=$(onb "python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"model_sha256\"])' $R6/$ARM/package/PACKAGE.json")
    [ "$model" = "$soup" ] || { echo "PACKAGE.json and the soup name different identities" >&2; return 3; }
  elif [ "${M6_INDEX_FROM_SOUP:-0}" != 1 ]; then
    echo "no frozen package for $ARM on node B (M6_INDEX_FROM_SOUP=1 stages its soup at T = 1; amendment 7)" >&2
    return 3
  fi
  echo "$soup"
}
copy_checkpoint() {  # NODE: node B soup checkpoint -> that node's $CK over node B's transfer key, SHA-256 lists equal
  local addr b t
  [ "$1" = c ] && addr=${C#*@} || addr=${D#*@}
  "on$1" "umask 077; mkdir -p $MD"
  echo "$(date -u +%FT%TZ) $ARM: copying the soup checkpoint node B -> node ${1^^}"
  onb "rsync -a -e '$XFER' $R6/$ARM/checkpoint/ root@$addr:$CK/"
  b=$(onb "$(sums "$R6/$ARM/checkpoint")") t=$("on$1" "$(sums "$CK")")
  [ -n "$b" ] && [ "$b" = "$t" ] || { echo "node ${1^^} checkpoint differs from node B's" >&2; return 3; }
  echo "checkpoint: $(wc -l <<< "$b") files, SHA-256 lists equal"
}
restage() {  # NODE MODEL_SHA256
  "on$1" "cd $S && PYTHONPATH=$S python3 -m v2.eval.ix1.restage --package $FIX2 --out $PKG --checkpoint $CK --model-sha256 $2"
  "on$1" "python3 - $PKG/MODEL_MANIFEST.json $2 $LOADED" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
assert m["identity"]["model_sha256"] == sys.argv[2], "identity"
assert m["parameters"]["loaded"] == int(sys.argv[3]), f"loaded {m['parameters']['loaded']}"
assert m["calibration"] is None
print(f"restaged: identity {sys.argv[2][:12]}, loaded {m['parameters']['loaded']:,}, T = 1")
EOF
  echo "manifest $("on$1" "sha256sum < $PKG/MODEL_MANIFEST.json | cut -c1-64")"
}
case "$STAGE" in
  stage)
    has_mirror d
    model=$(model_sha)
    ond "test ! -e $PKG" || { echo "$PKG exists: refusing to overwrite" >&2; exit 3; }
    copy_checkpoint d
    restage d "$model" ;;
  stage-c)
    has_mirror c
    model=$(model_sha)
    ond "test -f $PKG/MODEL_MANIFEST.json" || { echo "run stage (node D) first" >&2; exit 3; }
    onc "test ! -e $PKG" || { echo "node C $PKG exists: refusing to overwrite" >&2; exit 3; }
    if ! onc "test -e $FIX2"; then
      echo "$(date -u +%FT%TZ) relaying the fix2 template node D -> node C through node B"
      onb "$XFER root@${D#*@} 'tar -C ${FIX2%/*} -cf - ${FIX2##*/}' | \
        $XFER root@${C#*@} 'umask 077; mkdir -p ${FIX2%/*} && tar -C ${FIX2%/*} -xf -'"
    fi
    t=$(ond "$(sums "$FIX2")") c=$(onc "$(sums "$FIX2")")
    [ -n "$t" ] && [ "$t" = "$c" ] || { echo "node C's fix2 template differs from node D's" >&2; exit 3; }
    echo "fix2 template: $(wc -l <<< "$t") files, SHA-256 list equal to node D's"
    copy_checkpoint c
    restage c "$model"
    t=$(ond "$(sums "$PKG")") c=$(onc "$(sums "$PKG")")
    [ -n "$t" ] && [ "$t" = "$c" ] || { echo "node C's restaged package differs from node D's" >&2; exit 3; }
    echo "node C package: $(wc -l <<< "$c") files, SHA-256 list equal to node D's" ;;
  control)
    has_mirror d
    C0=$R/runs/DEV2.0-27B-budget-control
    ond "test ! -e $C0/control.json" || { echo "the restage control already ran ($C0)" >&2; exit 3; }
    ond "cd $S && bash v2/eval/ix1/launch.sh ref --src $M --model DEV2.0-27B-budget --gpu 4 --run $C0 \
      --rows $R/panel-8/compat-86.gold-free.jsonl.gz --cache $R/parity/DEV2.0-27B/cache-frozen > $R/logs/m6-control.log 2>&1 && \
      PYTHONPATH=$S python3 -m v2.eval.ix1.parity --kit $R/parity/DEV2.0-27B/kit/results.jsonl --ref $C0/ref.jsonl \
        --out $C0/control.json > /dev/null"
    ond "python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k: d[k] for k in (\"pass\", \"requests\", \"statuses\", \"max_abs_dp\")})); sys.exit(0 if d[\"pass\"] else 1)' $C0/control.json" ;;
  audit)
    has_mirror d
    A6=$R/runs/m6-audit X6=/data/dev2/private/27b/m6-data
    ond "test ! -e $R/logs/m6-audit.exit" || { echo "the M6 audit already ran ($A6)" >&2; exit 3; }
    ond "mkdir -p $R/logs; setsid nohup bash -c 'cd $S && PYTHONHASHSEED=0 PYTHONPATH=$S python3 -m v2.eval.ix1.contamination \
      --panel $R/panel-8 --train a20ib12pn=$X6/mixtures-m6pn-1/a20ib12pn.train.jsonl \
      --train a20ib1x=$X6/mixtures-m6-1/a20ib1x.train.jsonl --workers 24 --out $A6; echo \$? > $R/logs/m6-audit.exit' \
      > $R/logs/m6-audit.log 2>&1 < /dev/null &"
    echo "$(date -u +%FT%TZ) M6 contamination audit started on node D (CPU)"
    until ond "test -f $R/logs/m6-audit.exit"; do sleep 60; done
    ond "echo exit \$(cat $R/logs/m6-audit.exit); tail -n 3 $R/logs/m6-audit.log"
    (umask 077 && mkdir -p "${LOCAL%/*}/audit")
    ond "cat $A6/audit.json" > "${LOCAL%/*}/audit/audit.json"
    chmod 600 "${LOCAL%/*}/audit/audit.json" ;;
  parity)
    has_mirror d
    ond "test -f $PKG/MODEL_MANIFEST.json" || { echo "run stage first" >&2; exit 3; }
    ond "test ! -e $R/logs/m6-parity-$ARM.exit" || { echo "parity of $ARM already ran" >&2; exit 3; }
    ond "mkdir -p $R/logs; setsid nohup bash -c 'cd $S && bash v2/eval/ix1/launch.sh parity --src $M --model $ARM --gpu 4 \
      --run $R/parity/$ARM --rows $R/panel-8/compat-86.gold-free.jsonl.gz; echo \$? > $R/logs/m6-parity-$ARM.exit' \
      > $R/logs/m6-parity-$ARM.log 2>&1 < /dev/null &"
    echo "$(date -u +%FT%TZ) $ARM parity gate started on node D GPU4 (two 27B passes, ~10 min)"
    until ond "test -f $R/logs/m6-parity-$ARM.exit"; do sleep 60; done
    ond "cat $R/logs/m6-parity-$ARM.exit; python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k: d[k] for k in (\"pass\", \"requests\", \"statuses\", \"max_abs_dp\")})); sys.exit(0 if d[\"pass\"] else 1)' $R/parity/$ARM/parity.json" ;;
  run)
    plan=$(placement) || exit 2
    has_mirror d
    ond "python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))[\"pass\"] else 1)' $R/parity/$ARM/parity.json" ||
      { echo "parity gate missing or failed" >&2; exit 3; }
    if grep -q '^c ' <<< "$plan"; then
      has_mirror c
      t=$(ond "$(sums "$PKG")") c=$(onc "test -f $PKG/MODEL_MANIFEST.json && $(sums "$PKG")" || true)
      [ -n "$t" ] && [ "$t" = "$c" ] || { echo "node C has no package equal to node D's: run stage-c first" >&2; exit 3; }
      if ! onc "test -f $R/parity/$ARM/parity.json"; then
        ond "cat $R/parity/$ARM/parity.json" | onc "umask 077; mkdir -p $R/parity/$ARM && cat > $R/parity/$ARM/parity.json"
      fi
      [ "$(ond "sha256sum < $R/parity/$ARM/parity.json")" = "$(onc "sha256sum < $R/parity/$ARM/parity.json")" ] ||
        { echo "node C's parity record differs from node D's" >&2; exit 3; }
    fi
    while read -r node shards gpus; do  # ssh reads stdin: without < /dev/null it eats the plan's other lines
      "on$node" "mkdir -p $R/logs; setsid nohup bash $S/v2/27b/m6/m6-index-run.sh $SHA $ARM $node $shards $gpus \
        > $R/logs/m6-index-$ARM-$node.log 2>&1 < /dev/null & echo node $node m6-index-run \$!: shards $shards on GPU $gpus" \
        < /dev/null
    done <<< "$plan"
    sleep 20
    while read -r node _; do "on$node" "head -n 3 $R/logs/m6-index-$ARM-$node.log" < /dev/null; done <<< "$plan" ;;
  status)
    for node in d c; do
      "on$node" "test -d $R/runs/$ARM" || continue
      "on$node" "python3 - $R/runs/$ARM $node" <<'EOF'
import glob, os, sys, time
run, node, total = sys.argv[1], sys.argv[2], 0.0
for w in sorted(glob.glob(f"{run}/shard-*")):
    if not os.path.exists(f"{w}/start_epoch"):
        print(f"node {node} {os.path.basename(w)}: waiting")
        continue
    rows = sum(1 for _ in open(f"{w}/results.jsonl")) if os.path.exists(f"{w}/results.jsonl") else 0
    start = float(open(f"{w}/start_epoch").read())
    end = float(open(f"{w}/end_epoch").read()) if os.path.exists(f"{w}/end_epoch") else None
    code = open(f"{w}/exit_code").read().strip() if os.path.exists(f"{w}/exit_code") else "-"
    total += ((end or time.time()) - start) / 3600
    print(f"node {node} {os.path.basename(w)}: {rows} records, {'ended exit ' + code if end else 'running'}")
print(f"node {node}: GPU-h so far (current intervals) {total:.2f}")
EOF
      "on$node" "tail -n 4 $R/logs/m6-index-$ARM-$node.log 2> /dev/null || echo 'node $node: no run log'"
    done ;;
  collect)
    ks=$(onc "cd $R/runs/$ARM 2> /dev/null && for k in 0 1 2 3 4 5 6 7; do test -d shard-\$k && echo \$k; done" || true)
    [ -n "$ks" ] || { echo "node C ran no shard of $ARM" >&2; exit 3; }
    for k in $ks; do
      if ! onc "test -f $R/runs/$ARM/shard-$k/end_epoch && test \"\$(cat $R/runs/$ARM/shard-$k/exit_code)\" = 0"; then
        echo "node C shard $k has not ended with exit code 0" >&2; exit 3
      fi
      ond "test ! -e $R/runs/$ARM/shard-$k" || { echo "node D already has shard-$k of $ARM" >&2; exit 3; }
    done
    ond "umask 077; mkdir -p $R/runs/$ARM"
    for k in $ks; do
      onc "tar -C $R/runs/$ARM --exclude=shard-$k/triton --exclude=shard-$k/home -cf - shard-$k" |
        ond "tar -C $R/runs/$ARM -xf -"
      c=$(onc "cd $R/runs/$ARM/shard-$k && find . \\( -path ./triton -o -path ./home \\) -prune -o -type f -print | sort | xargs sha256sum")
      t=$(ond "cd $R/runs/$ARM/shard-$k && find . -type f | sort | xargs sha256sum")
      [ -n "$c" ] && [ "$c" = "$t" ] || { echo "node D's copy of shard $k differs from node C's" >&2; exit 3; }
      echo "shard $k: $(wc -l <<< "$t") files copied node C -> node D, SHA-256 lists equal"
    done
    onc "cd $R/runs/$ARM && tar -cf - launcher-run-only-*.json" | ond "tar -C $R/runs/$ARM --keep-old-files -xf -"
    echo "node C launcher records copied" ;;
  score)
    has_mirror d
    ond "for k in 0 1 2 3 4 5 6 7; do test \"\$(cat $R/runs/$ARM/shard-\$k/exit_code 2>/dev/null)\" = 0 || exit 1; done" ||
      { echo "not every shard ended with exit code 0 on node D (collect node C's shards first)" >&2; exit 3; }
    ond "cd $S && bash v2/eval/ix1/score.sh --src $M --model $ARM --size 27B --panel $R/panel-8 > $R/logs/m6-score-$ARM.log 2>&1 && \
      PYTHONPATH=$S python3 -m v2.eval.ix1.family_delta --base A20r=$R/runs/DEV2.0-27B/merged-budget/compare.json \
        --new $ARM=$R/runs/$ARM/merged/compare.json --out $R/runs/$ARM/family-delta-vs-a20r.json > /dev/null && \
      PYTHONPATH=$S python3 -m v2.eval.ix1.family_delta --base M5-L128=$R/runs/M5-L128/merged/compare.json \
        --new $ARM=$R/runs/$ARM/merged/compare.json --out $R/runs/$ARM/family-delta-vs-m5-l128.json > /dev/null && \
      cd $R/.. && PYTHONPATH=$S:\$PWD/kit-19ad28ec venv/bin/python -m v2.eval.ix1.paired_boot --suite-dir suite-0.2 \
        --base $R/runs/DEV2.0-27B/merged-budget/results.jsonl --new $R/runs/$ARM/merged/results.jsonl \
        --external $R/../external/index021-frontier-gap-2026-10-01.json --replicates 2000 --seed 20261002 --workers 24 \
        --out $R/runs/$ARM/paired-boot-vs-a20r.json > $R/logs/m6-boot-$ARM.log 2>&1"
    (umask 077 && mkdir -p "$LOCAL")
    for f in merged/compare.json merged/receipt.json merged/port.json family-delta-vs-a20r.json \
      family-delta-vs-m5-l128.json paired-boot-vs-a20r.json; do
      ond "cat $R/runs/$ARM/$f" > "$LOCAL/$(basename "$f")"
    done
    ond "cat $R/runs/DEV2.0-27B/merged-budget/compare.json" > "$LOCAL/compare-a20r.json"
    ond "cat $R/parity/$ARM/parity.json" > "$LOCAL/parity.json"
    ond "cat $R/runs/DEV2.0-27B-budget-control/control.json 2> /dev/null" > "$LOCAL/control-a20r.json" || true
    chmod 600 "$LOCAL"/*.json
    python3 - "$LOCAL/receipt.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
print(json.dumps({k: r[k] for k in ("rows", "statuses", "gpu_hours", "results_sha256", "panel_run_ids_sha256")}))
EOF
    echo "private outputs in $LOCAL (never copy a value into a commit, record, gist or card)" ;;
  release)  # a GPU whose M6 owner was set aside (owner.m6-set-aside-<UTC>, node D GPU0-3) gets that owner back
    for node in d c; do
      [ "$node" = d ] && gs="0 1 2 3 4 5 6 7" || gs="1 2 3 4 5 6 7"
      "on$node" "docker ps --format '{{.Names}}' | grep -q '^$TAG'" &&
        { echo "an $ARM Index container is still running on node ${node^^}" >&2; exit 3; }
      "on$node" "for g in $gs; do f=/data/dev2/leases/gpu\$g.lock/owner; grep -q 'IX1 .* $ARM\$' \$f 2>/dev/null || continue; \
        m=\$(ls -t /data/dev2/leases/gpu\$g.lock/owner.m6-set-aside-* 2> /dev/null | head -n 1); \
        if [ -n \"\$m\" ]; then cp -p \"\$m\" \$f; echo node $node gpu\$g back to its 27b M6 owner; continue; fi; \
        printf 'track=eval-ix1\nstatus=released (27B M6 Index run of $ARM done)\nlast_job_end_utc=%s\n' \
        \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\" > \$f; echo node $node gpu\$g released; done"
    done ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
