#!/usr/bin/env bash
# ~27B M6 post-training and reference stages on node B (host side): M4b / M5's verified drivers with Milestone 6's
# allocation (launch3 m6-b: node B GPU0, GPU1, GPU5; track 27b) and root /data/dev2/runs/27b/m6.
# Usage: m6-tail.sh STAGE MIRROR_SHA ARGS...
#   lease GPU                 take an M6 GPU for track 27b (launch3 lease; refused while another track's job holds it)
#   slices NAME CKPT GPU SLICE...   run_slices.sh (SLICE = NAME=HOST_ROWS=SHA256) -> m6/slices/NAME/probs
#   pn1 NAME REF ROWS         m6_slices pn1 (gate G5) of slices/NAME vs slices/REF on the PN1 rows -> slices/NAME/pn1-vs-REF.json
#   breadth NAME REF ROWS [FAMILY...]   m6_slices breadth (gate G6; FAMILY = in-distribution families)
#                             -> slices/NAME/breadth-vs-REF.json
#   pull ARM-SEED             a node A relay (BEST checkpoint + SHA-256 list) over the M6 node link -> m6/relay/ARM-SEED
#   pull-d ARM-SEED           the same from node D's relay over the node B -> node D transfer key (/root/.ssh/d2_temp_cd)
#   pull-e / pull-f ARM-SEED  the same from node E's / F's relay (peer-e / peer-f, written by m6-stage-d.sh
#                             M6_STAGE_NODE=e / f)
#   lsoup NAME CKPT CKPT...   v2.27b.lora_soup (exact rank concatenation) in a CPU-only container -> m6/NAME/checkpoint;
#                             members with a relay list (RELAY_SUMS="CKPT=SHA256SUMS ...") must match it file by file;
#                             SOUP_WEIGHTS="W1 W2 ..." (one per CKPT) makes a weighted soup, SOUP_CPUS (default 16)
#   readout NAME CKPT GPU     m4b/run_readout.sh (CAL698 kernel fit, typed DEV + CSS pilot + HT-DEV v2) -> m6/readouts/NAME
#   devgates NAME...          m6_devgates.py (gates G1-G6 vs A20r and M5-L128) -> m6/readouts/DEVGATES-<UTC>.json
#   formal NAME CKPT GPU      m4b/run_formal.sh -> m6/NAME (package, formal, paired compares incl. A20r and M5-L128)
#   mlx NAME GPU              m4b/run_mlx.sh on NAME's frozen package -> m6/mlx-diag/NAME
#   mlx-push NAME / mlx-pull NAME   node A scoring of NAME's mlx-diag collection over the M6 node link
# Every inference job: 32,768 tokens, kernel path, a fresh verified copy of DEV2.0-27B's scored cache 03b172f1. LoRA
# checkpoints (CHECKPOINT_FORMAT=peft-lora/1, the default here); formal runs take LOADED_PARAMETERS (27,497,508,864
# for a rank-256 soup).
set -euo pipefail
echo "m6 tail $*: start $(date -u +%FT%TZ)"
STAGE=${1:?STAGE} SHA=${2:?MIRROR_SHA}
shift 2
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
S=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2
[ -f "$S/v2/27b/m6/m6-tail.sh" ] || { echo "missing mirror $SHA" >&2; exit 2; }
M4B=$S/v2/27b/m4b M6=$S/v2/27b/m6
R=/data/dev2/runs/27b/m6
F1_CACHE=/data/dev2/runs/27b/m3-f2/f1-scored-cache
F1_CACHE_SHA=03b172f1a6adeef6c6a6c491d04389b355c9d8579480008023f408c8659b502b
KEY=/data/dev2/tmp/27b-m6-xfer
export DEV2_27B_LAUNCH_ALLOC=m6-b M4B_ROOT=$R PYTHONPATH=$S PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/dev2/tmp
export CHECKPOINT_FORMAT=${CHECKPOINT_FORMAT:-peft-lora/1}
aux() {  # GPU: node B GPU0 / GPU1 / GPU5 (the M6 allocation on node B)
  case "$1" in 0 | 1 | 5) ;; *) echo "GPU$1 is not an M6 node B GPU (0, 1, 5)" >&2; exit 2 ;; esac
}
link() {  # the M6 node link (node B -> node A rrsync root /data/dev2/xfer/27b-m6)
  [ -f "$KEY/id_ed25519" ] && [ -f "$KEY/peer" ] || { echo "no M6 node link at $KEY" >&2; exit 2; }
  X="ssh -i $KEY/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=$KEY/known_hosts -o StrictHostKeyChecking=yes -o BatchMode=yes"
  PEER=$(cat "$KEY/peer")
}
mkdir -p "$R"
case "$STAGE" in
  lease)
    GPU=${1:?GPU}; aux "$GPU"
    (cd "$S" && python3 -m v2.27b.m4b.launch3 lease --gpus "$GPU" --purpose "27b M6 (stage jobs)" \
      --status reserved-idle) ;;
  slices)
    NAME=${1:?NAME} CKPT=${2:?CKPT} GPU=${3:?GPU}; shift 3; aux "$GPU"
    bash "$M6/run_slices.sh" "$NAME" "$CKPT" "$GPU" "$SHA" "$@" ;;
  pn1 | breadth)
    NAME=${1:?NAME} REF=${2:?REF} ROWS=${3:?ROWS}; shift 3
    extra=()
    for family in "$@"; do extra+=(--in-distribution "$family"); done
    for n in "$NAME" "$REF"; do [ -f "$R/slices/$n/probs/slices.json" ] || { echo "no slices for $n" >&2; exit 2; }; done
    slice=$( [ "$STAGE" = pn1 ] && echo pn1 || echo "${IB_SLICE:-ib}" )
    (cd "$S" && python3 -m v2.27b.m6.m6_slices "$STAGE" --rows "$ROWS" \
      --candidate "$NAME=$R/slices/$NAME/probs/$slice.probs.jsonl" \
      --reference "$REF=$R/slices/$REF/probs/$slice.probs.jsonl" "${extra[@]}" \
      --output "$R/slices/$NAME/$STAGE-vs-$REF.json") ;;
  pull | pull-d | pull-e | pull-f)
    NAME=${1:?ARM-SEED}; link
    [[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ARM-SEED must be one directory name" >&2; exit 2; }
    DEST=$R/relay/$NAME
    mkdir -p "$DEST"
    if [ "$STAGE" = pull ]; then
      rsync -a -e "$X" "root@$PEER:relay/$NAME/" "$DEST/"
    else
      peer=$KEY/peer-${STAGE#pull-}
      [ -f "$peer" ] || { echo "no node address in $peer" >&2; exit 2; }
      rsync -a -e "ssh -i /root/.ssh/d2_temp_cd -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes" \
        "root@$(cat "$peer"):/data/dev2/xfer/27b-m6/relay/$NAME/" "$DEST/"
    fi
    (cd "$DEST/checkpoint" && find . -type f | sort | xargs -P 16 -n 4 sha256sum | sort -k2) > "$DEST/SHA256SUMS.nodeB"
    diff "$DEST/SHA256SUMS" "$DEST/SHA256SUMS.nodeB"
    echo "m6 pull $NAME: $(wc -l < "$DEST/SHA256SUMS") files, SHA-256 lists equal" ;;
  lsoup)
    NAME=${1:?NAME}; shift
    [ $# -ge 2 ] || { echo "lsoup NAME CKPT CKPT..." >&2; exit 2; }
    [[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "NAME must be one directory name" >&2; exit 2; }
    OUT=$R/$NAME BASE=/data/decision20-20260926/models/Qwen3.8-27B
    IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
    [ ! -e "$OUT/checkpoint" ] || { echo "$OUT/checkpoint exists: refusing to overwrite" >&2; exit 66; }
    args=() mounts=()
    for c in "$@"; do
      (cd "$S" && python3 -m v2.27b.m4b.ckpt_format check --checkpoint "$c" --format peft-lora/1)
      args+=(--member "$c")
      mounts+=(--mount "type=bind,src=$c,dst=$c,readonly")
    done
    read -r -a weights <<< "${SOUP_WEIGHTS:-}"
    if [ ${#weights[@]} -gt 0 ]; then
      [ ${#weights[@]} = $# ] || { echo "SOUP_WEIGHTS: one weight per member ($# members)" >&2; exit 2; }
      for w in "${weights[@]}"; do
        [[ "$w" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "SOUP_WEIGHTS: positive numbers, not '$w'" >&2; exit 2; }
        args+=(--weight "$w")
      done
    fi
    cpus=${SOUP_CPUS:-16}
    [[ "$cpus" =~ ^[1-9][0-9]?$ ]] || { echo "SOUP_CPUS: 1-99, not '$cpus'" >&2; exit 2; }
    mkdir -p "$OUT/soup"
    docker run --rm --name "d2-27b-M6-$NAME-soup" --network none --cpus "$cpus" -e OMP_NUM_THREADS="$cpus" \
      -e HIP_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES= -e PYTHONPATH=/code -e PYTHONDONTWRITEBYTECODE=1 \
      --mount "type=bind,src=$S,dst=/code,readonly" --mount "type=bind,src=$BASE,dst=$BASE,readonly" "${mounts[@]}" \
      --mount "type=bind,src=$OUT,dst=$OUT" -w /code --entrypoint python3 "$IMAGE" \
      -m v2.27b.lora_soup "${args[@]}" --source-path "$BASE" --output "$OUT/checkpoint" 2>&1 | tee "$OUT/soup/soup.log"
    (cd "$S" && python3 -m v2.27b.m6.m6_data soup-check --manifest "$OUT/checkpoint/soup_manifest.json" \
      --relay-sums "${RELAY_SUMS:-}" "$@") | tee "$OUT/soup/check.json" ;;
  readout)
    NAME=${1:?NAME} CKPT=${2:?CKPT} GPU=${3:?GPU}; aux "$GPU"
    FROZEN=$F1_CACHE CACHE_SHA=$F1_CACHE_SHA PANELS=typed-dev,css-pilot,ht-dev2 \
      STAGES=${STAGES:-verify,cal,collect,score,summary} LABEL="27b M6 $NAME" \
      bash "$M4B/run_readout.sh" "$NAME" "$CKPT" "$GPU" "$SHA" ;;
  devgates)
    [ $# -ge 1 ] || { echo "devgates NAME... (env PN1_ROWS, IB_ROWS, IN_DIST)" >&2; exit 2; }
    : "${PN1_ROWS:?devgates needs PN1_ROWS}" "${IB_ROWS:?devgates needs IB_ROWS}"
    extra=()
    for family in ${IN_DIST:-}; do extra+=(--in-distribution "$family"); done
    (cd "$S" && python3 -m v2.27b.m6.m6_devgates --root "$R" --pn1-rows "$PN1_ROWS" --ib-rows "$IB_ROWS" \
      "${extra[@]}" --pn1-validation "$R/slices/M5-L128/pn1-vs-A20r.json" --ref-ib "${REF_IB:-A20r-ib1}" \
      --ib-slice "${IB_SLICE:-ib}" "$@" \
      --output "$R/readouts/DEVGATES-$(date -u +%Y%m%dT%H%M%SZ).json") ;;
  formal)
    NAME=${1:?NAME} CKPT=${2:?CKPT} GPU=${3:?GPU}; aux "$GPU"
    : "${LOADED_PARAMETERS:?formal needs LOADED_PARAMETERS}"
    FROZEN=$F1_CACHE CACHE_SHA=$F1_CACHE_SHA READOUT=$R/readouts/$NAME LABEL="DEV2.0-27B (M6 $NAME)" \
      EXTRA_COMPARATOR="M4-A20r-soup=/data/dev2/runs/27b/M4-A20r-soup/formal M5-L128=/data/dev2/runs/27b/m5/M5-L128/formal ${EXTRA_COMPARATOR:-}" \
      bash "$M4B/run_formal.sh" "$NAME" "$CKPT" "$GPU" "$SHA" ;;
  mlx)
    NAME=${1:?NAME} GPU=${2:?GPU}; aux "$GPU"
    bash "$M4B/run_mlx.sh" "$NAME" "$R/$NAME/package/PACKAGE.json" "$GPU" "$SHA" ;;
  mlx-push | mlx-pull)
    NAME=${1:?NAME}; link
    D=$R/mlx-diag/$NAME
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
echo "m6 tail $STAGE complete: $(date -u +%FT%TZ)"
