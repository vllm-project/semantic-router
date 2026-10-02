#!/usr/bin/env bash
# Decoder M17b formal path of wave-6 candidates (release R3, the card's reports), workstation side: ssh control only;
# the nodes run M17's formal tools from the exact mirror MIRROR_SHA. Runs land where M17's do (node F collects under
# /data/dev2/runs/dec/formal/m17, prefix m17; node A scores with m17-fscore.sh).
#
# Usage: m17b-formal.sh MIRROR_SHA STAGE ARGS...
#   link NAME...              node F: hard links of the arm factory's soup soup/NAME/build/NAME as
#                             /data/dev2/runs/dec/m17/soup/NAME/build/NAME-soup (+ MODEL_SHA256, DONE, LINKED.json;
#                             inode lists equal), so M17's readout and formal tools read it under /data/dev2/runs/dec
#   readouts GPU NAME...      node F, detached, co-tenant (M17_COTENANT=1): the 16K typed DEV and CSS pilot readouts
#                             (m17-lines.sh read dev css-pilot) the formal library's 23:15 calibration rule needs
#   select TAG NAME...        node F: m10_formal_select.py --tier 4b over NAME... -> /data/dev2/runs/dec/m17b/select/TAG
#   formal TAG GPU NAME...    node F, detached: m17-formal.sh run with M17_SELECT=that directory (smoke, collection)
#   mlx TAG GPU NAME...       node F, detached: m17-formal.sh mlx-run (after node A marked the v3 report)
#   score NAME                node A: m17-fscore.sh formal-pull, run (report, paired vs both bars), formal-mark
#   mlxscore NAME             node A: m17-fscore.sh formal-pull of the mlx-diag run, then mlx
#   status NAME...            node F: readout, formal and mlx markers
set -euo pipefail
SHA=${1:?MIRROR_SHA} STAGE=${2:?STAGE}
shift 2
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
addr() { awk -F= -v k="node-$1" '$1 == k { print substr($0, length(k) + 2); exit }' "$NODES"; }
on() { local n=$1; shift; ssh -o BatchMode=yes -o ConnectTimeout=30 "$(addr "$n")" "$@"; }
MIR=$SHA-src_training_decision2
S=/data/dev2/src/$MIR/src/training/decision2
OPS=$S/v2/dec/ops/m17
AF=/data/dev2/runs/af/4b/soup
M=/data/dev2/runs/dec/m17
F=/data/dev2/runs/dec/formal/m17
SEL=/data/dev2/runs/dec/m17b/select
BASE=/data/dev2/models/Qwen--Qwen3.5-4B-Base/1001bb4d826a52d1f399e183466143f4da7b741b
ck() { echo "$M/soup/$1/build/$1-soup"; }
case "$STAGE" in
  link)
    for n in "$@"; do
      on f "set -e; src=\$(cat $AF/$n/DONE); test -f $AF/$n/MODEL_SHA256; test ! -e $M/soup/$n; \
        mkdir -p $M/soup/$n/build; cp -al \"\$src\" $(ck "$n"); cp $AF/$n/MODEL_SHA256 $M/soup/$n/MODEL_SHA256; \
        a=\$(cd \"\$src\" && find . -type f -printf '%P %i\n' | LC_ALL=C sort); b=\$(cd $(ck "$n") && find . -type f -printf '%P %i\n' | LC_ALL=C sort); \
        [ -n \"\$a\" ] && [ \"\$a\" = \"\$b\" ]; \
        printf '{\"from\": \"%s\", \"model_sha256\": \"%s\", \"files\": %s}\n' \"\$src\" \"\$(cat $AF/$n/MODEL_SHA256)\" \"\$(wc -l <<< \"\$a\")\" > $M/soup/$n/LINKED.json; \
        echo $(ck "$n") > $M/soup/$n/DONE; echo \"$n linked: \$(cat $M/soup/$n/LINKED.json)\""
    done ;;
  readouts)
    G=${1:?GPU}; shift
    cmds=""
    for n in "$@"; do
      on f "test -f $M/soup/$n/DONE" || { echo "$n is not linked" >&2; exit 3; }
      cmds+="M17_COTENANT=1 M17_NODE=f bash $OPS/m17-lines.sh read $MIR $G $n $(ck "$n") $BASE dev css-pilot; "
    done
    on f "mkdir -p $M/logs && setsid nohup bash -c '$cmds' > $M/logs/m17b-readouts-g$G-$(date -u +%H%M%S).log 2>&1 < /dev/null &"
    echo "readouts of $* on node F GPU$G started" ;;
  select)
    TAG=${1:?TAG}; shift
    pts=()
    for n in "$@"; do pts+=(--point "$n=$(ck "$n"),$M/lines/$n/dev/dev.predictions.jsonl,$M/lines/$n/css-pilot/css-pilot.predictions.jsonl"); done
    on f "test ! -e $SEL/$TAG && mkdir -p $SEL && cd $S && PYTHONPATH=$S python3 -B $S/v2/dec/ops/m10/m10_formal_select.py --tier 4b ${pts[*]} --output $SEL/$TAG" ;;
  formal | mlx)
    TAG=${1:?TAG} G=${2:?GPU}; shift 2
    mode=run; [ "$STAGE" = mlx ] && mode=mlx-run
    on f "test -f $SEL/$TAG/4b-finalists.json" || { echo "no select directory $TAG" >&2; exit 3; }
    on f "mkdir -p $F/logs && M17_SELECT=$SEL/$TAG M17_NODE=f setsid nohup bash $OPS/m17-formal.sh $mode $MIR $G $* \
      > $F/logs/m17b-$STAGE-$TAG-g$G-$(date -u +%H%M%S).log 2>&1 < /dev/null &"
    echo "$STAGE of $* on node F GPU$G started (select $TAG)" ;;
  score)
    n=${1:?NAME}
    on a "set -e; bash $OPS/m17-fscore.sh formal-pull $(addr f) m17-$n && bash $OPS/m17-fscore.sh run m17-$n && \
      bash $OPS/m17-fscore.sh formal-mark $(addr f) m17-$n" ;;
  mlxscore)
    n=${1:?NAME}
    on a "set -e; bash $OPS/m17-fscore.sh formal-pull $(addr f) m17-$n-mlx && bash $OPS/m17-fscore.sh mlx m17-$n" ;;
  status)
    for n in "$@"; do
      on f "echo \"$n: dev \$(grep -c . $M/lines/$n/dev/dev.predictions.jsonl 2>/dev/null) css \$(grep -c . $M/lines/$n/css-pilot/css-pilot.predictions.jsonl 2>/dev/null) status \$(ls $F/status 2>/dev/null | grep -E '^m17-${n}[.]' | tr '\n' ' ')\""
    done ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
