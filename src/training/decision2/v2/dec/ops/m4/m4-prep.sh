#!/usr/bin/env bash
# Decoder Milestone 4 data preparation on node B (prereg dec-m4-prereg-2026-09-29.md):
#   1. pinned HF downloads (XL r2 ids, pool files and registries, own-Lux XL waves, rp-v2 wave3);
#   2. the r2-clean M3-line mixture (arms LR / LR2), then the own-Lux gap: retention rows that no
#      published own-Lux file covers, labeled with Lux 1.0 on one GPU (plus a 300-row cross-check);
#   3. the A7q/k/s swap mixture (LRQ) and the two XL token-matched subsamples (XA, XF);
#   4. one composed own-Lux teacher file per mixture (XF: the gold-only H7 / H8 rows stay uncovered).
# A finished step is skipped on a rerun; each job leaves its launch receipt. Nothing trains here.
# usage: m4-prep.sh <mirror-dir> <label-gpu 0-4>
set -uo pipefail
SRC=$1 GPU=$2
M=/data/dev2/runs/dec/m4
mkdir -p "$M/data" "$M/teacher"
L=/data/dev2/src/$SRC/src/training/decision2/v2/dec/launch.sh
log() { echo "$(date -u +%FT%TZ) prep $*" | tee -a "$M/OPERATIONS.log"; }
declare -A RENDER=([0]=/dev/dri/renderD129 [1]=/dev/dri/renderD137 [2]=/dev/dri/renderD145 [3]=/dev/dri/renderD153 [4]=/dev/dri/renderD161)
export DEC_IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
export DEC_DATA=/data/dev2/runs/dec/m3/data-sel700-cal698
REPO=llm-semantic-router/decision-2.0-training-data
R2=100536133e192c54ec57c2599a5e4706f6d334ff
W3=002e5b422bc74fc3a276be99daad1b7ad793520c
H=/hf/datasets--llm-semantic-router--decision-2.0-training-data/snapshots
TOK=/hf/models--llm-semantic-router--Decision-1.0-Eos-0.8B/snapshots/363c4a5e56afc115b1c78c837633956d0bbb63ab
LUX1=/hf/models--llm-semantic-router--Decision-1.0-Lux-9B/snapshots/bd45a30aee8c84032791c245c70f86dee5389cc8

if [ ! -f "$M/data/DOWNLOADED" ]; then
  files=(m2/registry.json v2/registry.json v2/a7/registry.json m3/arms/registry.json
    m3/mixtures/xl-r2/cx-xl-r2-a7v1-full.ids.jsonl m3/mixtures/xl-r2/mx-xl-full-r2.ids.jsonl)
  for p in A7g A7h A7i A7k A7m A7o A7p A7q A7r A7s; do files+=("v2/a7/arms/$p/train.jsonl"); done
  for p in A1 A2 A3 A4v2h A5 A6g A6h; do files+=("v2/arms/$p/train.jsonl"); done
  for p in E11 G2 G4h G6 H1 H3 H5 H6; do files+=("m2/arms/$p/train.jsonl"); done
  for p in H7 H8; do files+=("m3/arms/$p/train.jsonl"); done
  for w in w1 w2 w3 w4 w5 c-w1; do files+=("m3/teachers/lux1/xl/$w.targets.jsonl"); done
  export HF_HUB_CACHE=/data/dev2/hf-cache HF_XET_CACHE=$M/xet-cache
  if hf download "$REPO" "${files[@]}" --repo-type dataset --revision "$R2" --quiet \
    && hf download "$REPO" m2/teachers/lux1/rp-v2/wave3.targets.jsonl --repo-type dataset --revision "$W3" --quiet; then
    date -u +%FT%TZ > "$M/data/DOWNLOADED"
    log "downloads done"
  else
    log "downloads FAILED"
    exit 1
  fi
fi

SRCS=(--source "pk1=$H/d8eae3e4fb5b91871c5aa7c13f0d94ea96e86ea7,m3/registry.json,"
  --source "v2=$H/ed87a03ab80ca5b9560780bba51a83a77ff47d14,m2/registry.json,"
  --source "v1=$H/5c0255ed6a05d08b2fcb863e8045289eedd996cd,v2/registry.json,"
  --source "mx=$H/5c602c5ae76c3930d5ed191facef4ac710863b6d,-,"
  --source "a7=$H/3a4efc7768e3458e392f44e83391313fc25e49df,v2/a7/registry.json,v2/"
  --source "xl=$H/$R2,-,"
  --source "xa7=$H/$R2,v2/a7/registry.json,v2/"
  --source "xv1=$H/$R2,v2/registry.json,"
  --source "xv2=$H/$R2,m2/registry.json,"
  --source "xh=$H/$R2,m3/arms/registry.json,m3/"
  --source "spec=/code/v2/dec/specs,-,")
build() {  # <mixture> <spec file>
  [ -f "$M/data/$1/train.jsonl" ] && return 0
  if bash "$L" "m4-build-$1" "$SRC" "$M/data/$1" --cpu -- -m v2.dec.build_mixture --spec "/code/v2/dec/specs/$2" \
    "${SRCS[@]}" --tokenizer "$TOK" --workers 40 --output /out/train.jsonl; then
    log "built $1: $(tail -1 "$M/data/$1.stdout.log" 2>/dev/null)"
  else
    log "build $1 FAILED"
    return 1
  fi
}
LUX=(--source "$H/d8eae3e4fb5b91871c5aa7c13f0d94ea96e86ea7/m3/pk1/lux1/A0-train.canonical.jsonl"
  --source "$H/7885baf6ea3805cc0e00405bde5061ad5b9d0aa9/m2/teachers/lux1/rp-v2/wave1.targets.jsonl"
  --source "$H/6bd8eb4d4fe0bc2c47f517c1017f7422510b09f0/m2/teachers/lux1/rp-v2/wave2.targets.jsonl"
  --source "$H/$W3/m2/teachers/lux1/rp-v2/wave3.targets.jsonl"
  --source "$H/03b1e72d4b4a4526b2e7f5db3659adb07f200a71/m2/teachers/lux1/rp-v2/wave4.targets.jsonl")
for w in w1 w2 w3 w4 w5 c-w1; do LUX+=(--source "$H/$R2/m3/teachers/lux1/xl/$w.targets.jsonl"); done
GAP=/runs/m4/teacher/ret-gap-lux/labels.jsonl
compose() {  # <mixture> [compose args]
  local mix=$1
  shift
  [ -f "$M/teacher/$mix/lux-teacher.jsonl" ] && return 0
  if bash "$L" "m4-teacher-$mix" "$SRC" "$M/teacher/$mix" --cpu -- -m v2.dec.compose_teacher compose \
    --train "/runs/m4/data/$mix/train.jsonl" "${LUX[@]}" --source "$GAP" "$@" --output /out/lux-teacher.jsonl; then
    log "composed $mix: $(tail -1 "$M/teacher/$mix.stdout.log" 2>/dev/null)"
  else
    log "compose $mix FAILED"
    return 1
  fi
}

build m4-v2m-ret-r2 m4-v2m-ret-r2.json || exit 1
if [ ! -f "$M/teacher/ret-gap-rows/rows.jsonl" ]; then
  if bash "$L" m4-retgap-rows "$SRC" "$M/teacher/ret-gap-rows" --cpu -- -m v2.dec.compose_teacher missing \
    --train /runs/m4/data/m4-v2m-ret-r2/train.jsonl "${LUX[@]}" --check-sample 300 --seed dec-m4-retgap \
    --output /out/rows.jsonl; then
    log "gap rows: $(tail -1 "$M/teacher/ret-gap-rows.stdout.log" 2>/dev/null)"
  else
    log "gap rows FAILED"
    exit 1
  fi
fi
if [ ! -f "$M/teacher/ret-gap-lux/labels.jsonl" ]; then
  export DEC_RENDER=${RENDER[$GPU]} DEC_GPU_LABEL="node B GPU$GPU"
  if bash "$L" m4-retgap-lux "$SRC" "$M/teacher/ret-gap-lux" -- -m v2.dec.teacher_label --teacher-path "$LUX1" \
    --teacher-repo llm-semantic-router/Decision-1.0-Lux-9B --teacher-revision bd45a30aee8c84032791c245c70f86dee5389cc8 \
    --train /runs/m4/teacher/ret-gap-rows/rows.jsonl --output /out/labels.jsonl; then
    log "gap labels done"
  else
    log "gap labels FAILED"
    exit 1
  fi
fi
compose m4-v2m-ret-r2 || exit 1
date -u +%FT%TZ > "$M/data/m4-v2m-ret-r2/BUILT"

build m4-v2m-ret-r2-q20 m4-v2m-ret-r2-q20.json &
build m4-xl-a7v1-29m m4-xl-a7v1-29m.json &
build m4-xl-full-29m m4-xl-full-29m.json &
wait
IDS=$H/$R2/m3/mixtures/xl-r2/mx-xl-full-r2.ids.jsonl
for mix in m4-v2m-ret-r2-q20 m4-xl-a7v1-29m m4-xl-full-29m; do
  [ -f "$M/data/$mix/train.jsonl" ] || continue
  extra=()
  [ "$mix" = m4-xl-full-29m ] && extra=(--allow-missing-pool "$IDS:H7" --allow-missing-pool "$IDS:H8")
  if compose "$mix" "${extra[@]}"; then date -u +%FT%TZ > "$M/data/$mix/BUILT"; fi
done
log "prep finished"
