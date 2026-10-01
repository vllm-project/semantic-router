#!/usr/bin/env bash
# Decoder M14 data preparation on node A / B (prereg dec-m14-prereg-2026-10-01.md, "Data"), CPU only, from an exact
# mirror. Idempotent; a failed build is not rerun. Sources are pulled over the temporary transfer key from node E
# (M14_FROM_E) and node F (M14_FROM_F, node B only); the addresses come from the operator's environment and are not
# logged.
#   1. inputs (both nodes): M12's three locked TRAIN files reused by M14 (08b-RA, 2b-RA, 4b-LHA10) with their
#      train.ids.jsonl, each against M12's data-lock hash; the ib-dev / m10-probes prompt panels (gold-free) where
#      the node lacks them;
#   2. references and teachers: node A the DEV2.0-0.8B C0 package (node E); node B the DEV2.0-2B C0 package (node E),
#      M10's LH soup (node F) and the 2B own-Sol / 4B own-Lux teachers (node E); directories are copied whole and
#      checked by a content manifest on both sides;
#   3. M14 Triton caches <tier>-train / <tier>-read for the node's tiers: cp -a copies of the node's decoder cache
#      (triton-cache/dbe5f32b2263); the pre-warm seed and the first reference read fill them;
#   4. TRAIN: m14/data/<tier>/<ARM>/train.jsonl = hard link of the staged M12 file; weights.jsonl from m14_weights.py
#      (the preregistered weights), all three arms on both nodes (the data lock compares the builds).
#
# usage: M14_NODE=a|b M14_FROM_E=<user@node-e> [M14_FROM_F=<user@node-f>] m14-prep.sh <mirror-dir>
set -euo pipefail
SRC=$1
NODE=${M14_NODE:?set M14_NODE=a or b}
FROM_E=${M14_FROM_E:?set M14_FROM_E}
R=/data/dev2/runs/dec
M=$R/m14
CODE=/data/dev2/src/$SRC/src/training/decision2
OPS=$CODE/v2/dec/ops/m14
mkdir -p "$M/inputs/m12" "$M/inputs/refs" "$M/inputs/teachers" "$M/data" "$M/triton-cache" "$M/lines"
log() { echo "$(date -u +%FT%TZ) prep-$NODE $*" | tee -a "$M/OPERATIONS.log"; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
check() { [ "$(sha "$1")" = "$2" ] || { log "FAILED: $1 is not $2"; exit 1; }; }
KEY=(-i /root/.ssh/d2_temp_cd -o BatchMode=yes -o ConnectTimeout=20)
manifest='find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -d" " -f1'
pull_file() {  # <host> <remote path> <local path> <sha256>
  [ -f "$3" ] && { check "$3" "$4"; return 0; }
  mkdir -p "$(dirname "$3")"
  scp -q "${KEY[@]}" "$1:$2" "$3.part" && mv "$3.part" "$3"
  check "$3" "$4"
  log "pulled $(basename "$3") ($(echo "$4" | cut -c1-16))"
}
# shellcheck disable=SC2029  # the remote paths are expanded here on purpose
pull_dir() {  # <host> <remote dir> <local dir>: whole directory, content manifest equal on both sides
  local want got
  want=$(ssh "${KEY[@]}" "$1" "cd '$2' && $manifest")
  if [ ! -d "$3" ]; then
    mkdir -p "$(dirname "$3")"
    ssh "${KEY[@]}" "$1" "tar -C '$(dirname "$2")' -cf - '$(basename "$2")'" | tar -C "$(dirname "$3")" -xf -
    [ "$(basename "$2")" = "$(basename "$3")" ] || mv "$(dirname "$3")/$(basename "$2")" "$3"
  fi
  got=$(cd "$3" && eval "$manifest")
  [ "$got" = "$want" ] || { log "FAILED: $3 manifest $got != source $want"; exit 1; }
  log "pulled $(basename "$3") (manifest $(echo "$got" | cut -c1-16), $(find "$3" -type f | wc -l) files)"
}

# M12 TRAIN files and ids (M12 data lock d506595c2)
declare -A TRAIN_SHA=([08b-RA]=12bd63d8b215877b6f471c12652e534e0f0be794c15bff74ad5ac50e98018bf3
  [2b-RA]=08140409a6afc439e35bc4e0aefc9b5812162c4a2972e5e1da5b137a99d0e592
  [4b-LHA10]=d41cdd1aa1e63b47394ed8fa61a87bf461cb4d2cea08cc7ff12f2e58b36dc9d5)
declare -A IDS_SHA=([08b-RA]=5b83116183c9cecd8217c38b9e8c6832370b385a6f9108031059e4246c77428e
  [2b-RA]=22854cfca61cb61f48619d33a4d9e968b59b4b5d8bfea5c5ebceff882d2826f7
  [4b-LHA10]=e429ce9b7506e3e39340e6c2f8dd7804ea65cb7469367e71f8257bc5b4ac064e)
declare -A TIER_OF=([08b-RA]=08b [2b-RA]=2b [4b-LHA10]=4b)
for a in 08b-RA 2b-RA 4b-LHA10; do
  t=${TIER_OF[$a]}
  pull_file "$FROM_E" "$R/m12/data/$t/$a/train.jsonl" "$M/inputs/m12/$t/$a/train.jsonl" "${TRAIN_SHA[$a]}"
  pull_file "$FROM_E" "$R/m12/data/$t/$a/train.ids.jsonl" "$M/inputs/m12/$t/$a/train.ids.jsonl" "${IDS_SHA[$a]}"
done
pull_file "$FROM_E" "$R/panels/ib-dev.prompts.jsonl" "$R/panels/ib-dev.prompts.jsonl" \
  7cf53e4722397a1fde97240872b40dfe51e587fb64592deba9f195847edac196
pull_file "$FROM_E" "$R/panels/m10-probes.prompts.jsonl" "$R/panels/m10-probes.prompts.jsonl" \
  a324f1e292a232decffa775ca04052a96d769ebb500c8768685cbef1ea952891
log "M12 inputs and panels verified"

case $NODE in
  a)
    pull_dir "$FROM_E" /data/dev2/models/DEV2.0-0.8B/bede7938a8c209c09f27400b79eed57948d6b75e \
      "$M/inputs/refs/DEV2.0-0.8B/bede7938a8c209c09f27400b79eed57948d6b75e"
    TIERS="08b" ;;
  b)
    FROM_F=${M14_FROM_F:?set M14_FROM_F on node B}
    pull_dir "$FROM_E" /data/dev2/models/DEV2.0-2B/a53cf66a0d9d492a84b6617b61e7ce35fcd03af0 \
      "$M/inputs/refs/DEV2.0-2B/a53cf66a0d9d492a84b6617b61e7ce35fcd03af0"
    pull_dir "$FROM_F" "$R/m10/soup/LH/build/LH-soup" "$M/inputs/refs/LH-soup"
    pull_file "$FROM_E" "$R/m11/data/2b/teacher.jsonl" "$M/inputs/teachers/2b/teacher.jsonl" \
      3f92e8c108c5a9949fda4b08638907fcc80175f68efec18bece912d0edc8eea1
    pull_file "$FROM_E" "$R/m10/inputs/n4xf-teacher/m4-xl-full-29m/lux-teacher.jsonl" \
      "$M/inputs/teachers/4b/lux-teacher.jsonl" e2ff27ce2fc397c8c203e37133dcfac52ac3d1f172c6f3e9412ec28c2ffe296c
    TIERS="2b 4b" ;;
  *) echo "unknown node $NODE" >&2; exit 2 ;;
esac

for t in $TIERS; do
  for k in train read; do
    c=$t-$k
    [ -d "$M/triton-cache/$c" ] && continue
    [ -d "$R/triton-cache/dbe5f32b2263" ] || { log "FAILED: no decoder cache triton-cache/dbe5f32b2263 on this node"; exit 1; }
    cp -a "$R/triton-cache/dbe5f32b2263" "$M/triton-cache/$c.tmp" && mv "$M/triton-cache/$c.tmp" "$M/triton-cache/$c"
    log "Triton cache $c copied from triton-cache/dbe5f32b2263 ($(find "$M/triton-cache/$c" -type f | wc -l) files)"
  done
done

# arm: <ARM> <tier> <M12 arm> <released rows> <released TRAIN sha256> <typed weights>
build() {
  local arm=$1 t=$2 src=$3 rows=$4 rsha=$5 w=$6 d=$M/data/$2/$1
  if [ ! -f "$d/train.jsonl" ]; then
    mkdir -p "$d"
    ln "$M/inputs/m12/$t/$src/train.jsonl" "$d/train.jsonl"
    log "TRAIN $arm = hard link of the staged M12 $src"
  fi
  [ -f "$d/weights.jsonl" ] && return 0
  [ ! -e "$M/data/weights-$arm.launch.json" ] || { log "$arm weights build failed earlier; not rerun"; exit 1; }
  M14_NODE=$NODE bash "$OPS/m14-launch.sh" "weights-$arm" "$SRC" "$M/data/weights-$arm" --cpu -- \
    v2/dec/ops/m14/m14_weights.py --train "/runs/m14/inputs/m12/$t/$src/train.jsonl" --train-sha "${TRAIN_SHA[$src]}" \
    --ids "/runs/m14/inputs/m12/$t/$src/train.ids.jsonl" --ids-sha "${IDS_SHA[$src]}" --released-rows "$rows" \
    --released-sha "$rsha" --weight "$w" --ib-weight 1.0 --name "$arm" --output /out \
    || { log "$arm weights build FAILED (see $M/data/weights-$arm.stderr.log)"; exit 1; }
  mv "$M/data/weights-$arm/$arm/weights.jsonl" "$M/data/weights-$arm/$arm/report.json" "$d/"
  log "weights $arm: $(tail -1 "$M/data/weights-$arm.stdout.log" | cut -c1-400)"
}
build 08b-RAUP 08b 08b-RA 162696 f9f3c0229551357ad1cb8e6923c64c636ee697e71d162f059548ae223e78d8ae \
  choice=1.5,noul=1.5,score=1.5
build 2b-RAUP 2b 2b-RA 56141 1527b38b1ba888695fe48dd43e92827d1719d57674009cfc29d5ab759b08dd2c \
  choice=1.5,noul=1.5,score=2.0
build 4b-LHA10UP 4b 4b-LHA10 58739 c385406e8f78a2ae257cf2479b7088a523be18009e55e16caf59327c34260f09 \
  choice=1.5,noul=1.5,score=1.5
log "prep finished"
