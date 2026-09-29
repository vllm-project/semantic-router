#!/usr/bin/env bash
# Decoder Milestone 7 data preparation on node B, CPU only (prereg dec-m7-prereg-2026-09-30.md, "Arms and data").
# Nothing trains here and nothing is uploaded.
#   0. the pinned HS1 / PN1 TRAIN files from the private dataset (hf download of one path per revision, read back
#      against the published SHA-256);
#   1. m7_compose.py per tier: the three matched-token TRAIN files (4b: base m4-xl-full-29m, pool m6-xl-full-59m,
#      filler = more of the recipe; 2b: base m4-v2m-ret-r2, LP from the same pool, filler = replay of the base);
#   2. one composed teacher per TRAIN file (v2.dec.compose_teacher; gold-only pools allowed);
#   3. an overlap exposure receipt per TRAIN file against the r2 payload (2194716a);
#   4. m7_lock.py per tier (lock-<tier>.json). READY files are written by hand only after the lock record is
#      committed.
# A finished step is skipped on a rerun; a failed step is not rerun. Each job leaves its launch receipt.
# M7_TAG (e.g. -r2) builds into data/<tier><TAG>, teacher/m7-<tier>-<ARM><TAG>, exposure/... and lock-<tier><TAG>.json,
# for the arms in M7_ARMS (default "H C P"); M7_PN1_REV / M7_PN1_SHA pick another PN1 revision (the prereg's PN1
# revision rule), M7_HS1_CLEARED_REV / _SHA the cleared HS1 revision the lock compares the HS1 rows with.
# usage: m7-prep.sh <mirror-dir> [4b|2b ...]
set -uo pipefail
SRC=$1
shift
TIERS=${*:-4b 2b}
TAG=${M7_TAG:-}
ARMS=${M7_ARMS:-H C P}
M=/data/dev2/runs/dec/m7
mkdir -p "$M/data" "$M/teacher" "$M/exposure" "$M/logs"
CODE=/data/dev2/src/$SRC/src/training/decision2
L=$CODE/v2/dec/launch.sh
log() { echo "$(date -u +%FT%TZ) prep $*" | tee -a "$M/OPERATIONS.log"; }
export DEC_IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
export DEC_DATA=/data/dev2/runs/dec/m3/data-sel700-cal698
PAYLOAD_DIR=/data/dev2/runs/eval/m5/overlap-effects/final
PAYLOAD_SHA=2194716a179b2e6c3ba529dee3952c5dd62c0f3281638b7236029d5d542f4914
HOST_H=/data/dev2/hf-cache/datasets--llm-semantic-router--decision-2.0-training-data/snapshots
H=/hf/datasets--llm-semantic-router--decision-2.0-training-data/snapshots
HS1_REV=171e6f0c0913d5f0d8f7304928da4f14dca6e127
HS1_SHA=c90ef3164d90d3fd6a1ab0397529faec172760b1756de7d4d8cbf2677a131a71
PN1_REV=${M7_PN1_REV:-5ad362872f57034c7fada2c2990cfc6c36398d73}
PN1_SHA=${M7_PN1_SHA:-f6f6a5315a0aebc9d83b656ff90d378a0e1f2083f95bbcafdc18399969a64616}
PN1_PATH=${M7_PN1_PATH:-m4/pn1/arms/pn1.train.jsonl}
R2=100536133e192c54ec57c2599a5e4706f6d334ff
XL_IDS=$H/$R2/m3/mixtures/xl-r2/mx-xl-full-r2.ids.jsonl
TOK=/hf/models--llm-semantic-router--Decision-1.0-Eos-0.8B/snapshots/363c4a5e56afc115b1c78c837633956d0bbb63ab
POOL=/runs/m6/data/m6-xl-full-59m/train.jsonl POOL_SHA=160812e2c3d39bd09fcbe6c98e2b4b1633f6680bd435241d3a83e791d5dfe33b
B4=/runs/m4/data/m4-xl-full-29m/train.jsonl B4_SHA=c7d51219e9f0fdd107b914df5c7e9f81f2b53496bb43a94796857b8ea3fcfa60
B2=/runs/m4/data/m4-v2m-ret-r2/train.jsonl B2_SHA=1527b38b1ba888695fe48dd43e92827d1719d57674009cfc29d5ab759b08dd2c
LUX59=/runs/m6/teacher/lux-all-59m/teacher.jsonl
N4XF_T=/runs/m4/teacher/m4-xl-full-29m/lux-teacher.jsonl
SOL59=/runs/m6/teacher/sol-59m/teacher.jsonl
S2T_T=/runs/m3/teacher/sol/sol-teacher.jsonl
QUAR=/code/v2/dec/ops/m7/specs/m7-quarantine-groups.json
DEFECT=" nights consecutive nights"
HS1C_REV=${M7_HS1_CLEARED_REV:-27b1d2f130292268b43a618584bebab5d4e4a6b5}
HS1C_SHA=${M7_HS1_CLEARED_SHA:-0dfaa6ebff1b614e5f1c432f2c7432c3fd217f7c344d402c7a7b37fe507de5e7}

cpu() {  # <job> <out dir under m7> <python args...>: skipped when <out>.launch.json exists with exit 0
  local job=$1 out=$M/$2
  shift 2
  if [ -f "$out.launch.json" ]; then
    grep -q '"exit_status": 0' "$out.launch.json" && return 0
    log "$job failed earlier (receipt $out.launch.json); not rerun"
    return 1
  fi
  if bash "$L" "m7-$job" "$SRC" "$out" --cpu -- "$@"; then
    log "$job done: $(tail -c 400 "$out.stdout.log" | tr '\n' ' ')"
  else
    log "$job FAILED: $(tail -c 600 "$out.stderr.log" | tr '\n' ' ')"
    return 1
  fi
}
fetch() {  # <revision> <path> <sha256>: one file of the private dataset at a pinned revision
  local f=$HOST_H/$1/$2
  if [ ! -f "$f" ]; then
    HF_HUB_CACHE=/data/dev2/hf-cache hf download llm-semantic-router/decision-2.0-training-data "$2" --repo-type dataset \
      --revision "$1" > "$M/logs/fetch-$(basename "$2").log" 2>&1 || { log "download of $2@${1:0:8} FAILED"; return 1; }
  fi
  [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$3" ] || { log "$2@${1:0:8} sha256 differs from $3"; return 1; }
  log "input $2@${1:0:8} verified ($3)"
}
[ "$(sha256sum "$PAYLOAD_DIR/excluded-groups.json" | cut -d' ' -f1)" = "$PAYLOAD_SHA" ] || { log "payload hash MISMATCH"; exit 1; }
fetch "$HS1_REV" m4/hs1/train.jsonl "$HS1_SHA" || exit 1
fetch "$PN1_REV" "$PN1_PATH" "$PN1_SHA" || exit 1
fetch "$HS1C_REV" m4/hs1/train.jsonl "$HS1C_SHA" || exit 1

for tier in $TIERS; do
  if [ "$tier" = 4b ]; then
    args=(--tier 4b --base "$B4" --base-sha "$B4_SHA" --pool-teacher "$LUX59" --filler pool --gold-pools "H7,H8")
  else
    args=(--tier 2b --base "$B2" --base-sha "$B2_SHA" --pool-teacher "$SOL59" --filler replay --base-teacher "$S2T_T"
      --gold-pools '')
  fi
  cpu "compose-$tier$TAG" "data/$tier$TAG" v2/dec/ops/m7/m7_compose.py "${args[@]}" --pool "$POOL" --pool-sha "$POOL_SHA" \
    --pool-ids "$XL_IDS" --hs1 "$H/$HS1_REV/m4/hs1/train.jsonl" --hs1-sha "$HS1_SHA" \
    --pn1 "$H/$PN1_REV/$PN1_PATH" --pn1-sha "$PN1_SHA" --quarantine "$QUAR" --hs1-drop-substring "$DEFECT" \
    --tokenizer "$TOK" --workers 40 --output /out/mix || exit 1
  for arm in $ARMS; do
    n=m7-$tier-$arm d=/runs/m7/data/$tier$TAG/mix/$n
    if [ "$tier" = 4b ]; then
      src=(--source "$N4XF_T" --source "$d/teacher-new.jsonl")
      allow=(base-gold lp-gold fill-gold hs1 pn1)
    else
      src=(--source "$S2T_T")
      [ -f "$M/data/$tier$TAG/mix/$n/teacher-new.jsonl" ] && src+=(--source "$d/teacher-new.jsonl")
      [ -f "$M/data/$tier$TAG/mix/$n/teacher-replay.jsonl" ] && src+=(--source "$d/teacher-replay.jsonl")
      allow=(hs1 pn1)
    fi
    am=()
    for pool in "${allow[@]}"; do am+=(--allow-missing-pool "$d/train.ids.jsonl:$pool"); done
    cpu "teacher-$n$TAG" "teacher/$n$TAG" -m v2.dec.compose_teacher compose --train "$d/train.jsonl" "${src[@]}" "${am[@]}" \
      --output /out/teacher.jsonl || exit 1
    sha=$(sha256sum "$M/data/$tier$TAG/mix/$n/train.jsonl" | cut -d' ' -f1)
    DEC_DATA=$PAYLOAD_DIR cpu "exposure-$n$TAG" "exposure/$n$TAG" -m v2.eval.overlap_effects exposure \
      --groups /data/excluded-groups.json --train "$d/train.jsonl" --expect-sha256 "$sha" \
      --label "decoder M7 $n" --output "/out/exposure-$n.json" || exit 1
  done
  python3 -B "$CODE/v2/dec/ops/m7/m7_lock.py" --tier "$tier" --tag "$TAG" --arms "$ARMS" \
    --hs1-cleared "$HOST_H/$HS1C_REV/m4/hs1/train.jsonl" --hs1-cleared-sha "$HS1C_SHA" \
    --out "$M/lock-$tier$TAG.json" > "$M/lock-$tier$TAG.log" 2>&1
  log "lock $tier$TAG: $(tail -1 "$M/lock-$tier$TAG.log")"
done
log "prep finished ($TIERS$TAG: $ARMS)"
