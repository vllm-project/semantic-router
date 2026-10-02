#!/usr/bin/env bash
# 4B Index-first release (workstation side): the formal typed-FINAL panel, collected with CSS15 and public 231 in one
# run, of the candidate without one (M14 4b-LHA10UP), for the integrity check "no decision type collapsed" (formal
# item 3). The decoder formal library as M16 ran its 4B finalists: node B GPU3 / GPU4 (image dbe5f32b, master cache
# formal/m5/cache-frozen f6d0f920..., the CAL698 16K fit and the 23:15 rule), outputs under
# /data/dev2/runs/dec/formal/4bif (prefix 4bif). M16 showed this path answer-identical to the released LH's stored
# formal run, so no parity run is repeated. Relayed gold-free node B -> node C -> node A (node A and node B do not reach
# each other), scored on node A (m6-score.sh: report, compares, TYPES.json), plus, as references, the paired v3
# comparison and the public-231 guard against bar-lh (the released LH's stored run formal/m10/m10-4b-LH-t1-derived).
# Usage: formal4b.sh MIRROR_SHA STAGE [GPU]
#   select   node B, CPU: m10_formal_select.py for 4b-LHA10UP (the M14 soup, its M14 16K typed DEV / CSS pilot readouts)
#   launch   GPU (3 | 4): node B's owner file -> track=dec-4bif (only when absent or released, and the GPU idle); smoke
#            (8 items) then the collection, detached (formal/4bif/logs/formal-4b-LHA10UP.log)
#   status   the chain's log
#   relay    node B -> node C transit -> node A: the run directory (no weights, no gold-named file), content manifests
#            equal on both ends; the transit copy is removed
#   score    node A, CPU: m6-score.sh 4b, then same_panel compare and gates public231 vs bar-lh; prints the verdicts
#   release  node B's owner file -> released (no formal container on the GPU)
set -euo pipefail
SHA=${1:?MIRROR_SHA} STAGE=${2:?STAGE} GPU=${3:-}
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
addr() { awk -F= -v k="node-$1" '$1 == k { print substr($0, length(k) + 2); exit }' "$NODES"; }
on() { local n=$1; shift; ssh -o BatchMode=yes -o ConnectTimeout=30 "$(addr "$n")" "$@"; }
KEY="-i /root/.ssh/d2_temp_cd -o BatchMode=yes -o ConnectTimeout=30"
S=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2
F=/data/dev2/runs/dec/formal/4bif
POINT=4b-LHA10UP RUN=4bif-4b-LHA10UP
CKPT=/data/dev2/runs/dec/m14/soup/4b-LHA10UP/build/4b-LHA10UP-soup
LINES=/data/dev2/runs/dec/m14/lines/4b-LHA10UP
BAR=/data/dev2/runs/dec/formal/m10/m10-4b-LH-t1-derived
manifest="find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -d' ' -f1"
case "$STAGE" in
  select)
    on b "test -f $S/v2/dec/ops/m10/m10_formal_select.py" || { echo "mirror $SHA is not on node B" >&2; exit 2; }
    on b "test ! -e $F/select/4b-finalists.json" || { echo "select file exists" >&2; exit 3; }
    on b "mkdir -p $F/select $F/logs && cd $S && PYTHONPATH=$S python3 -B v2/dec/ops/m10/m10_formal_select.py --tier 4b \
      --point $POINT=$CKPT,$LINES/dev/dev.predictions.jsonl,$LINES/css-pilot/css-pilot.predictions.jsonl --output $F/select && \
      python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d[\"finalists\"][0][\"files_sha256_list_sha256\"]))' \
      $F/select/4b-finalists.json" ;;
  launch)
    [[ "$GPU" == 3 || "$GPU" == 4 ]] || { echo "node B GPU3 or GPU4 (the library's node-B render map)" >&2; exit 2; }
    on b "test -f $F/select/4b-finalists.json && test ! -e $F/logs/formal-$POINT.lock" || { echo "no select file, or launched" >&2; exit 3; }
    on b "f=/data/dev2/leases/gpu$GPU.lock/owner; if [ -s \$f ] && ! grep -q '^status=released' \$f && ! grep -qx 'track=dec-4bif' \$f; then \
      echo 'gpu$GPU is leased by another owner' >&2; exit 1; fi; \
      used=\$(rocm-smi -d $GPU --showmeminfo vram --json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(int(list(d.values())[0][\"VRAM Total Used Memory (B)\"]) >> 30)'); \
      [ \"\$used\" -le 2 ] || { echo \"gpu$GPU holds \$used GiB\" >&2; exit 1; }; \
      [ -f \$f ] && cp \$f \$f.prev-\$(date -u +%Y%m%dT%H%M%SZ); \
      printf 'track=dec-4bif\nstatus=busy\npurpose=4B Index-first: formal typed-FINAL panel of $POINT (smoke + collection)\nstart_utc=%s\nexpected_end_utc=%s\n' \
        \"\$(date -u +%FT%TZ)\" \"\$(date -u -d '+90 min' +%FT%TZ)\" > \$f && mkdir $F/logs/formal-$POINT.lock"
    on b "setsid nohup bash -c 'export M6_FORMAL_ROOT=$F M6_SELECT=$F/select M6_PREFIX=4bif M6_GPU=$GPU; \
      bash $S/v2/dec/ops/m6/m6-formal.sh 4b smoke $POINT 8 && bash $S/v2/dec/ops/m6/m6-formal.sh 4b finalist $POINT; \
      echo \$? > $F/logs/formal-$POINT.exit' > $F/logs/formal-$POINT.log 2>&1 < /dev/null &"
    echo "$(date -u +%FT%TZ) formal chain for $POINT launched on node B GPU$GPU" ;;
  status)
    on b "cat $F/logs/formal-$POINT.exit 2>/dev/null || echo running; tail -n 5 $F/logs/formal-$POINT.log; tail -n 5 $F/OPERATIONS.log" ;;
  relay)
    on b "test \"\$(cat $F/logs/formal-$POINT.exit)\" = 0 && test -f $F/$RUN/M6-RECEIPT.json" || { echo "collection not finished" >&2; exit 3; }
    on b "! find $F/$RUN -iname '*gold*' | grep -q . && ! find $F/$RUN -name '*.safetensors' -size +1M | grep -q ." ||
      { echo "$RUN holds gold-named files or weights" >&2; exit 3; }
    on a "test ! -e $F/$RUN" || { echo "node A already has $RUN" >&2; exit 3; }
    T=/data/dev2/runs/dec/4bif-relay/$(date -u +%Y%m%dT%H%M%SZ)-$$
    on c "mkdir -p $T"
    on b "rsync -a -e 'ssh $KEY' $F/$RUN/ $(addr c):$T/$RUN/"
    on a "mkdir -p $F && rsync -a -e 'ssh $KEY' $(addr c):$T/$RUN/ $F/$RUN/"
    on c "rm -rf $T"
    mb=$(on b "cd $F/$RUN && $manifest") ma=$(on a "cd $F/$RUN && $manifest")
    [ "$mb" = "$ma" ] || { echo "relay manifest mismatch" >&2; exit 3; }
    echo "relayed $RUN to node A (manifest $ma)" ;;
  score)
    on a "test -f $S/v2/dec/ops/m6/m6-score.sh" || { echo "mirror $SHA is not on node A" >&2; exit 2; }
    on a "mkdir -p $F/logs && cd $S && M6_FORMAL_ROOT=$F bash v2/dec/ops/m6/m6-score.sh 4b $RUN > $F/logs/score-$RUN.log 2>&1" ||
      { echo "m6-score FAILED (node A $F/logs/score-$RUN.log)" >&2; exit 3; }
    on a "cd $S && export PYTHONPATH=$S && \
      { [ -f $F/$RUN/PAIRED-vs-bar-lh.json ] || python3 -B -m v2.eval.same_panel compare --run-dir $F/$RUN --comparator-run-dir $BAR \
          --left-name $RUN --right-name bar-lh > $F/$RUN.compare-bar-lh.log 2>&1; } && \
      { [ -f $F/$RUN.public231-bar-lh.json ] || python3 -B -m v2.eval.gates public231 --left $F/$RUN --right $BAR --left-name $RUN \
          --right-name bar-lh --output $F/$RUN.public231-bar-lh.json > $F/$RUN.public231-bar-lh.log 2>&1; }"
    on a "python3 - $F/$RUN" << 'EOF'
import json, sys
r = sys.argv[1]
t = json.load(open(f"{r}/TYPES.json"))
p = json.load(open(f"{r}/PAIRED-vs-bar-lh.json"))
g = json.load(open(f"{r}.public231-bar-lh.json"))
print(json.dumps({"types": {k: v["verdict"] for k, v in t["types"].items()},
                  "v3_vs_bar_lh": [round(p["point"]["delta"]["score"], 3), round(p["ci95"]["low"], 3), round(p["ci95"]["high"], 3)],
                  "H_vs_bar_lh": [round(p["axis_ci95"]["H"]["delta"]["low"], 4), round(p["axis_ci95"]["H"]["delta"]["high"], 4)],
                  "public231": [g["left_correct"], g["right_correct"], g["verdict"]]}))
EOF
    ;;
  release)
    [[ "$GPU" == 3 || "$GPU" == 4 ]] || { echo "GPU 3 or 4" >&2; exit 2; }
    on b "docker ps --format '{{.Names}}' | grep -q '^dev2-dec-gpu$GPU-' && { echo 'a decoder container is running on gpu$GPU' >&2; exit 1; }; \
      f=/data/dev2/leases/gpu$GPU.lock/owner; grep -qx 'track=dec-4bif' \$f && \
      printf 'track=dec-4bif\nstatus=released (4B Index-first formal panel of $POINT done)\nlast_job_end_utc=%s\n' \"\$(date -u +%FT%TZ)\" > \$f && echo released" ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
