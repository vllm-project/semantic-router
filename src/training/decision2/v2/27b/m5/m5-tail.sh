#!/usr/bin/env bash
# ~27B M5 post-training stages on node B (host side): M4b's verified drivers with Milestone 5's allocation and roots.
# Usage: m5-tail.sh STAGE MIRROR_SHA ARGS...
#   lease GPU                 take an auxiliary GPU (node B GPU0-2) for track 27b (launch3 lease; a foreign idle owner
#                             moves to owner.prev-<UTC>); refused while another track's job holds it
#   pull ARM-SEED             m5-pull.sh: a node A relay (BEST checkpoint + SHA-256 list) over the private link
#   soup NAME CKPT CKPT...    combine_full.sh soup -> /data/dev2/runs/27b/m5/NAME/checkpoint (+ verify.json)
#   reference GPU             A20r's typed DEV + CSS pilot + HT-DEV v2 readout (m5-htdev2.sh, htdev2-refs-nodeB.json)
#                             -> /data/dev2/runs/27b/m5/readouts/A20r-ref
#   readout NAME CKPT GPU     run_readout.sh (CAL698 kernel fit, then typed DEV + CSS pilot + HT-DEV v2 with it, score,
#                             summary) -> /data/dev2/runs/27b/m5/readouts/NAME
#   devgates NAME...          m5_devgates.py: the four development gates vs A20r-ref -> readouts/DEVGATES-<UTC>.json
#   formal NAME CKPT GPU      run_formal.sh (CAL698 release fit, adoption, frozen package, smoke, typed FINAL + CSS15 +
#                             public 231, seal, report, paired compares) -> /data/dev2/runs/27b/m5/NAME
#   mlx NAME GPU              run_mlx.sh on NAME's frozen package -> /data/dev2/runs/27b/m5/mlx-diag/NAME
#   mlx-push NAME             NAME's mlx-diag collection (no cache) -> node A relay mlx/NAME with a SHA-256 list
#   mlx-pull NAME             node A's mlx-paired output (m5-mlx-nodeA.sh) -> gates/mlx/NAME-vs-A20r.json
# Every inference job: 32,768 tokens, kernel path, a fresh verified copy of DEV2.0-27B's scored cache 03b172f1
# (formal runs: no autotune entry may be added). Readouts, CAL698 fits and formal runs never run on a training GPU.
set -euo pipefail
echo "m5 tail $*: start $(date -u +%FT%TZ)"
STAGE=${1:?STAGE} SHA=${2:?MIRROR_SHA}
shift 2
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
S=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2
[ -f "$S/v2/27b/m5/m5-tail.sh" ] || { echo "missing mirror $SHA" >&2; exit 2; }
M4B=$S/v2/27b/m4b M5=$S/v2/27b/m5
R=/data/dev2/runs/27b/m5
F1_CACHE=/data/dev2/runs/27b/m3-f2/f1-scored-cache
F1_CACHE_SHA=03b172f1a6adeef6c6a6c491d04389b355c9d8579480008023f408c8659b502b
export DEV2_27B_LAUNCH_ALLOC=m5-b M4B_ROOT=$R PYTHONPATH=$S PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/dev2/tmp
aux() {  # GPU: only node B GPU0-2 serve single-GPU M5 inference jobs
  case "$1" in 0 | 1 | 2) ;; *) echo "GPU$1 is not an M5 auxiliary GPU (node B GPU0-2)" >&2; exit 2 ;; esac
}
case "$STAGE" in
  lease)
    GPU=${1:?GPU}; aux "$GPU"
    (cd "$S" && python3 -m v2.27b.m4b.launch3 lease --gpus "$GPU" --purpose "27b M5 auxiliary inference" \
      --status reserved-idle) ;;
  pull) bash "$M5/m5-pull.sh" "${1:?ARM-SEED}" ;;
  soup)
    NAME=${1:?NAME}; shift
    COMBINE_ROOT=$R COMBINE_PREFIX=d2-27b-m5 bash "$M4B/combine_full.sh" "$SHA" "$NAME" soup "$@" ;;
  reference)
    GPU=${1:?GPU}; aux "$GPU"
    ROOT=$R/readouts PANELS=typed-dev,css-pilot,ht-dev2 STAGES=collect \
      bash "$M5/m5-htdev2.sh" "$M5/htdev2-refs-nodeB.json" m4-a20r-soup "$GPU" "$SHA"
    mkdir -p "$R/readouts"
    [ -e "$R/readouts/A20r-ref" ] || ln -s m4-a20r-soup "$R/readouts/A20r-ref"
    (cd "$S" && python3 -m v2.eval.dev_readout --run-dir "$R/readouts/m4-a20r-soup" --label "27b M5 A20r reference" \
      --output "$R/readouts/m4-a20r-soup/READOUT.json") ;;
  readout)
    NAME=${1:?NAME} CKPT=${2:?CKPT} GPU=${3:?GPU}; aux "$GPU"
    FROZEN=$F1_CACHE CACHE_SHA=$F1_CACHE_SHA PANELS=typed-dev,css-pilot,ht-dev2 \
      STAGES=${STAGES:-verify,cal,collect,score,summary} LABEL="27b M5 $NAME" \
      bash "$M4B/run_readout.sh" "$NAME" "$CKPT" "$GPU" "$SHA" ;;
  devgates)
    [ $# -ge 1 ] || { echo "devgates NAME..." >&2; exit 2; }
    cands=()
    for NAME in "$@"; do cands+=(--candidate "$NAME=$R/readouts/$NAME"); done
    (cd "$S" && python3 -m v2.27b.m5.m5_devgates --reference "A20r=$R/readouts/A20r-ref" "${cands[@]}" \
      --output "$R/readouts/DEVGATES-$(date -u +%Y%m%dT%H%M%SZ).json") ;;
  formal)
    NAME=${1:?NAME} CKPT=${2:?CKPT} GPU=${3:?GPU}; aux "$GPU"
    FROZEN=$F1_CACHE CACHE_SHA=$F1_CACHE_SHA READOUT=$R/readouts/$NAME LABEL="DEV2.0-27B (M5 $NAME)" \
      EXTRA_COMPARATOR="M4-A20r-soup=/data/dev2/runs/27b/M4-A20r-soup/formal M4-A20-soup=/data/dev2/runs/27b/M4-A20-soup/formal F-b=/data/dev2/runs/27b/m4b/F-b/formal ${EXTRA_COMPARATOR:-}" \
      bash "$M4B/run_formal.sh" "$NAME" "$CKPT" "$GPU" "$SHA" ;;
  mlx)
    NAME=${1:?NAME} GPU=${2:?GPU}; aux "$GPU"
    bash "$M4B/run_mlx.sh" "$NAME" "$R/$NAME/package/PACKAGE.json" "$GPU" "$SHA" ;;
  mlx-push | mlx-pull)
    NAME=${1:?NAME} KEY=/data/dev2/tmp/27b-m5-xfer
    X="ssh -i $KEY/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=$KEY/known_hosts -o StrictHostKeyChecking=yes -o BatchMode=yes"
    PEER=$(cat "$KEY/peer") D=$R/mlx-diag/$NAME
    if [ "$STAGE" = mlx-push ]; then
      [ -f "$D/COLLECT.json" ] || { echo "no finished mlx-diag collection $D" >&2; exit 2; }
      (cd "$D" && find output -type f | sort | xargs sha256sum > SHA256SUMS)
      rsync -a --mkpath -e "$X" --exclude triton-cache/ "$D/" "root@$PEER:mlx/$NAME/"
    else
      mkdir -p "$R/gates/mlx"
      rsync -a -e "$X" "root@$PEER:mlx/$NAME-vs-A20r.json" "$R/gates/mlx/$NAME-vs-A20r.json"
      python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({'R4': d['R4']['pass'], 'card_macro_ci95': d['bootstrap']['card_macro_ci95'], 'delta': d['delta']}))" \
        "$R/gates/mlx/$NAME-vs-A20r.json"
    fi ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
echo "m5 tail $STAGE complete: $(date -u +%FT%TZ)"
