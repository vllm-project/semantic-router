#!/usr/bin/env bash
# Decoder Milestone 9 data preparation on node A, CPU only (prereg dec-m9-prereg-2026-10-01.md, "Data").
# Nothing trains here and nothing is uploaded.
#   0. the 4B recipe inputs copied from node B over the direct link (m9/inputs/...), checked against their pinned
#      SHA-256; the pinned HR2 TRAIN / DEV files and the XL r2 recipe id list from the private dataset (hf download of
#      one path per revision, read back against the published SHA-256);
#   1. m9_compose.py: the two matched-token TRAIN files (H9 = base + HR2, C9 = base + recipe filler);
#   2. one composed teacher per TRAIN file (v2.dec.compose_teacher; gold-only pools allowed);
#   3. an overlap exposure receipt per TRAIN file against the r2 payload (2194716a);
#   4. the HR2 DEV prompts (m9_hr2dev.py prompts) into the decoder panel directory, gold under /data/dev2/private;
#      the Score5-typed-DEV gold-free prompts copied into the decoder panel directory;
#   5. m9_lock.py (lock.json). READY files are written by hand only after the lock record is committed.
# A finished step is skipped on a rerun; a failed step is not rerun. Each job leaves its launch receipt.
# usage: m9-prep.sh <mirror-dir>
set -uo pipefail
SRC=$1
M=/data/dev2/runs/dec/m9
mkdir -p "$M/data" "$M/teacher" "$M/exposure" "$M/logs"
CODE=/data/dev2/src/$SRC/src/training/decision2
L=$CODE/v2/dec/launch.sh
log() { echo "$(date -u +%FT%TZ) prep $*" | tee -a "$M/OPERATIONS.log"; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
export DEC_IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
export DEC_DATA=/data/dev2/runs/dec/m3/data-sel700-cal698
PANELS=/data/dev2/runs/dec/panels
PRIV=/data/dev2/private/dec/m9
PAYLOAD_DIR=/data/dev2/runs/eval/m5/overlap-effects/final
PAYLOAD_SHA=2194716a179b2e6c3ba529dee3952c5dd62c0f3281638b7236029d5d542f4914
HOST_H=/data/dev2/hf-cache/datasets--llm-semantic-router--decision-2.0-training-data/snapshots
H=/hf/datasets--llm-semantic-router--decision-2.0-training-data/snapshots
HR2_REV=afc3bc1e1d6849058f6fafdfbc3dfe007d067400
HR2_SHA=0fd9b2dbe68f863dc8acd13b3466b7e9aaf32d2308bc740f012e5c9717502f3a
HR2_DEV_SHA=697c31428ae821d7c0238a64ced8b997cc5671be47307f88a273d082bc5aaebe
R2=100536133e192c54ec57c2599a5e4706f6d334ff
XL_IDS=m3/mixtures/xl-r2/mx-xl-full-r2.ids.jsonl XL_IDS_SHA=7843afb7b2bbb315902b6748d6559532f384283d8efd836b935daac5f55dbd4a
TOK=/hf/models--llm-semantic-router--Decision-1.0-Eos-0.8B/snapshots/363c4a5e56afc115b1c78c837633956d0bbb63ab
I=/runs/m9/inputs
BASE=$I/base/m4-xl-full-29m/train.jsonl BASE_SHA=c7d51219e9f0fdd107b914df5c7e9f81f2b53496bb43a94796857b8ea3fcfa60
POOL=$I/pool/m6-xl-full-59m/train.jsonl POOL_SHA=160812e2c3d39bd09fcbe6c98e2b4b1633f6680bd435241d3a83e791d5dfe33b
N4XF_T=$I/n4xf-teacher/m4-xl-full-29m/lux-teacher.jsonl N4XF_T_SHA=e2ff27ce2fc397c8c203e37133dcfac52ac3d1f172c6f3e9412ec28c2ffe296c
LUX59=$I/pool-teacher/lux-all-59m/teacher.jsonl LUX59_SHA=a1bafad59411901cd8a13451b741a186fece445ab78ce9b0466fc5adc7f1f793
N7C_IDS=$I/n7c/m7-4b-C/train.ids.jsonl N7C_IDS_SHA=0dcede5cdce3ff567b9a2fb94d9a42dac5e59d57514461ad6506391e7ab17711
QUAR=/code/v2/dec/ops/m7/specs/m7-quarantine-groups.json
S5T_SRC=/data/dev2/private/panels/goldfree/score5t-dev.prompts.jsonl

cpu() {  # <job> <out dir under m9> <python args...>: skipped when <out>.launch.json exists with exit 0
  local job=$1 out=$M/$2
  shift 2
  if [ -f "$out.launch.json" ]; then
    grep -q '"exit_status": 0' "$out.launch.json" && return 0
    log "$job failed earlier (receipt $out.launch.json); not rerun"
    return 1
  fi
  if bash "$L" "m9-$job" "$SRC" "$out" --cpu -- "$@"; then
    log "$job done: $(tail -c 400 "$out.stdout.log" | tr '\n' ' ')"
  else
    log "$job FAILED: $(tail -c 600 "$out.stderr.log" | tr '\n' ' ')"
    return 1
  fi
}
host() { echo "/data/dev2/runs/dec/${1#/runs/}"; }
check() {  # <container path> <sha256>
  [ "$(sha "$(host "$1")")" = "$2" ] || { log "$1 sha256 differs from $2"; return 1; }
  log "input $1 verified ($2)"
}
fetch() {  # <revision> <path> <sha256>: one file of the private dataset at a pinned revision
  local f=$HOST_H/$1/$2
  if [ ! -f "$f" ]; then
    HF_HUB_CACHE=/data/dev2/hf-cache hf download llm-semantic-router/decision-2.0-training-data "$2" --repo-type dataset \
      --revision "$1" > "$M/logs/fetch-$(basename "$2").log" 2>&1 || { log "download of $2@${1:0:8} FAILED"; return 1; }
  fi
  [ "$(sha "$f")" = "$3" ] || { log "$2@${1:0:8} sha256 differs from $3"; return 1; }
  log "input $2@${1:0:8} verified ($3)"
}

[ "$(sha "$PAYLOAD_DIR/excluded-groups.json")" = "$PAYLOAD_SHA" ] || { log "payload hash MISMATCH"; exit 1; }
for pair in "$BASE $BASE_SHA" "$POOL $POOL_SHA" "$N4XF_T $N4XF_T_SHA" "$LUX59 $LUX59_SHA" "$N7C_IDS $N7C_IDS_SHA"; do
  # shellcheck disable=SC2086
  check $pair || exit 1
done
fetch "$HR2_REV" m5/hr2/hr2.train.jsonl "$HR2_SHA" || exit 1
fetch "$HR2_REV" m5/hr2/hr2.dev.jsonl "$HR2_DEV_SHA" || exit 1
fetch "$R2" "$XL_IDS" "$XL_IDS_SHA" || exit 1

cpu compose data/4b v2/dec/ops/m9/m9_compose.py --base "$BASE" --base-sha "$BASE_SHA" --pool "$POOL" \
  --pool-sha "$POOL_SHA" --pool-ids "$H/$R2/$XL_IDS" --pool-ids-sha "$XL_IDS_SHA" --pool-teacher "$LUX59" \
  --pool-teacher-sha "$LUX59_SHA" --hr2 "$H/$HR2_REV/m5/hr2/hr2.train.jsonl" --hr2-sha "$HR2_SHA" --quarantine "$QUAR" \
  --tokenizer "$TOK" --workers 40 --output /out/mix || exit 1
for arm in H9 C9; do
  n=m9-4b-$arm d=/runs/m9/data/4b/mix/$n
  src=(--source "$N4XF_T")
  [ -f "$M/data/4b/mix/$n/teacher-new.jsonl" ] && src+=(--source "$d/teacher-new.jsonl")
  am=()
  for pool in base-gold fill-gold hr2; do am+=(--allow-missing-pool "$d/train.ids.jsonl:$pool"); done
  cpu "teacher-$n" "teacher/$n" -m v2.dec.compose_teacher compose --train "$d/train.jsonl" "${src[@]}" "${am[@]}" \
    --output /out/teacher.jsonl || exit 1
  tsha=$(sha "$M/data/4b/mix/$n/train.jsonl")
  DEC_DATA=$PAYLOAD_DIR cpu "exposure-$n" "exposure/$n" -m v2.eval.overlap_effects exposure \
    --groups /data/excluded-groups.json --train "$d/train.jsonl" --expect-sha256 "$tsha" \
    --label "decoder M9 $n" --output "/out/exposure-$n.json" || exit 1
done

cpu hr2dev data/hr2dev v2/dec/ops/m9/m9_hr2dev.py prompts --dev "$H/$HR2_REV/m5/hr2/hr2.dev.jsonl" \
  --dev-sha "$HR2_DEV_SHA" --output /out/build || exit 1
B=$M/data/hr2dev/build
if [ ! -f "$PANELS/hr2-dev.prompts.jsonl" ]; then
  cp "$B/hr2-dev.prompts.jsonl" "$PANELS/hr2-dev.prompts.jsonl.tmp" && mv "$PANELS/hr2-dev.prompts.jsonl.tmp" "$PANELS/hr2-dev.prompts.jsonl"
fi
[ "$(sha "$PANELS/hr2-dev.prompts.jsonl")" = "$(sha "$B/hr2-dev.prompts.jsonl")" ] || { log "panel hr2-dev prompts differ from the build"; exit 1; }
if [ -f "$B/hr2-dev.gold.jsonl" ]; then
  mkdir -p "$PRIV" && chmod 700 "$PRIV"
  [ ! -e "$PRIV/hr2-dev.gold.jsonl" ] || { log "$PRIV/hr2-dev.gold.jsonl exists"; exit 1; }
  install -m 600 "$B/hr2-dev.gold.jsonl" "$PRIV/hr2-dev.gold.jsonl" && rm -f "$B/hr2-dev.gold.jsonl"
  log "hr2-dev gold moved to $PRIV ($(sha "$PRIV/hr2-dev.gold.jsonl"))"
fi
if [ ! -f "$PANELS/score5t-dev.prompts.jsonl" ]; then
  cp "$S5T_SRC" "$PANELS/score5t-dev.prompts.jsonl.tmp" && mv "$PANELS/score5t-dev.prompts.jsonl.tmp" "$PANELS/score5t-dev.prompts.jsonl"
  log "score5t-dev prompts copied to the decoder panel directory ($(sha "$PANELS/score5t-dev.prompts.jsonl"))"
fi
[ "$(sha "$PANELS/score5t-dev.prompts.jsonl")" = "$(sha "$S5T_SRC")" ] || { log "panel score5t-dev prompts differ"; exit 1; }

panels=(--panel select=/data/select.jsonl --panel cal=/data/cal.jsonl)
for p in dev css-pilot ht-dev2 score5t-dev hs1-dev hr2-dev; do panels+=(--panel "$p=/panels/$p.prompts.jsonl"); done
if [ ! -f "$M/lock.json" ]; then
  bash "$L" m9-lock "$SRC" "$M/lock" --cpu -- v2/dec/ops/m9/m9_lock.py --mix /runs/m9/data/4b/mix \
    --hr2 "$H/$HR2_REV/m5/hr2/hr2.train.jsonl" --hr2-sha "$HR2_SHA" --teacher-root /runs/m9/teacher \
    --exposure-root /runs/m9/exposure "${panels[@]}" --n7c-ids "$N7C_IDS" --out /out/lock.json
  rc=$?
  [ -f "$M/lock/lock.json" ] && cp "$M/lock/lock.json" "$M/lock.json"
  log "lock (exit $rc): $(tail -c 600 "$M/lock.stdout.log" | tr '\n' ' ')"
  [ $rc = 0 ] || exit 1
fi
log "prep finished"
