#!/usr/bin/env bash
# Decoder Milestone 6 data preparation on node B, CPU only (prereg dec-m6-prereg-2026-09-29.md). Nothing trains here.
#   1. m6-xl-full-59m: the XL r2 full recipe at 28.84% per pool (spec ops/m6/specs/m6-xl-full-59m.json = M4's
#      m4-xl-full-29m spec with every budget fraction doubled; same builder, sampler seed, exclusions and A0s rows);
#   2. own-Lux 1.0 teachers on all rows: lux-all-29m (N4XF's composed file e2ff27ce, then h-w1 for H7 / H8) and
#      lux-all-59m (N4XF's file, then the M4 precedence list: A0 canonical, RP-v2 waves 1-4, XL w1-w5 and c-w1,
#      then h-w1); no --allow-missing-pool, so a composed file covers every row;
#   3. m6-e8f-r2clean: E8F's d1dc33fc minus the r2-excluded groups, the A7 v3 removals and the A7 quarantine;
#   4. overlap exposure receipts for both new TRAIN files against the r2 payload (2194716a).
# A finished step is skipped on a rerun; a failed step is not rerun. Each job leaves its launch receipt.
# usage: m6-prep.sh <mirror-dir>
set -uo pipefail
SRC=$1
M=/data/dev2/runs/dec/m6
mkdir -p "$M/data" "$M/teacher" "$M/exposure" "$M/logs"
L=/data/dev2/src/$SRC/src/training/decision2/v2/dec/launch.sh
log() { echo "$(date -u +%FT%TZ) prep $*" | tee -a "$M/OPERATIONS.log"; }
export DEC_IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
export DEC_DATA=/data/dev2/runs/dec/m3/data-sel700-cal698
PAYLOAD_DIR=/data/dev2/runs/eval/m5/overlap-effects/final
PAYLOAD_SHA=2194716a179b2e6c3ba529dee3952c5dd62c0f3281638b7236029d5d542f4914
R2=100536133e192c54ec57c2599a5e4706f6d334ff
HW1REV=75e557f170979bdbc428b6ea698a2047e2d2a5cd
A7V3=3a4efc7768e3458e392f44e83391313fc25e49df
H=/hf/datasets--llm-semantic-router--decision-2.0-training-data/snapshots
TOK=/hf/models--llm-semantic-router--Decision-1.0-Eos-0.8B/snapshots/363c4a5e56afc115b1c78c837633956d0bbb63ab
N4=/runs/m4/data/m4-xl-full-29m/train.jsonl
LUXT=/runs/m4/teacher/m4-xl-full-29m/lux-teacher.jsonl
HW1=$H/$HW1REV/m3/teachers/lux1/xl/h-w1.targets.jsonl
E8F=/runs/m2/data/m2-full-a7-v1/train.jsonl
E8F_SHA=d1dc33fcb7a49fa9df9519545b1b0debeac38f86cbd760b481eb37334c9bd6f6

cpu() {  # <job> <out dir under m6> <python args...>: skipped when <out>.launch.json exists with exit 0
  local job=$1 out=$M/$2
  shift 2
  if [ -f "$out.launch.json" ]; then
    grep -q '"exit_status": 0' "$out.launch.json" && return 0
    log "$job failed earlier (receipt $out.launch.json); not rerun"
    return 1
  fi
  if bash "$L" "m6-$job" "$SRC" "$out" --cpu -- "$@"; then
    log "$job done: $(tail -c 400 "$out.stdout.log" | tr '\n' ' ')"
  else
    log "$job FAILED: $(tail -c 400 "$out.stderr.log" | tr '\n' ' ')"
    return 1
  fi
}
[ "$(sha256sum "$PAYLOAD_DIR/excluded-groups.json" | cut -d' ' -f1)" = "$PAYLOAD_SHA" ] || { log "payload hash MISMATCH"; exit 1; }

SRCS=(--source "pk1=$H/d8eae3e4fb5b91871c5aa7c13f0d94ea96e86ea7,m3/registry.json,"
  --source "mx=$H/5c602c5ae76c3930d5ed191facef4ac710863b6d,-,"
  --source "xl=$H/$R2,-,"
  --source "xa7=$H/$R2,v2/a7/registry.json,v2/"
  --source "xv1=$H/$R2,v2/registry.json,"
  --source "xv2=$H/$R2,m2/registry.json,"
  --source "xh=$H/$R2,m3/arms/registry.json,m3/"
  --source "spec=/code/v2/dec/specs,-,")
cpu build-59m data/m6-xl-full-59m -m v2.dec.build_mixture --spec /code/v2/dec/ops/m6/specs/m6-xl-full-59m.json \
  "${SRCS[@]}" --tokenizer "$TOK" --workers 40 --output /out/train.jsonl || exit 1

LUX=(--source "$LUXT"
  --source "$H/d8eae3e4fb5b91871c5aa7c13f0d94ea96e86ea7/m3/pk1/lux1/A0-train.canonical.jsonl"
  --source "$H/7885baf6ea3805cc0e00405bde5061ad5b9d0aa9/m2/teachers/lux1/rp-v2/wave1.targets.jsonl"
  --source "$H/6bd8eb4d4fe0bc2c47f517c1017f7422510b09f0/m2/teachers/lux1/rp-v2/wave2.targets.jsonl"
  --source "$H/002e5b422bc74fc3a276be99daad1b7ad793520c/m2/teachers/lux1/rp-v2/wave3.targets.jsonl"
  --source "$H/03b1e72d4b4a4526b2e7f5db3659adb07f200a71/m2/teachers/lux1/rp-v2/wave4.targets.jsonl")
for w in w1 w2 w3 w4 w5 c-w1; do LUX+=(--source "$H/$R2/m3/teachers/lux1/xl/$w.targets.jsonl"); done
LUX+=(--source "$HW1")
ok=1
if ! cpu lux-all-29m teacher/lux-all-29m -m v2.dec.compose_teacher compose --train "$N4" --source "$LUXT" \
  --source "$HW1" --output /out/teacher.jsonl; then
  ok=0
  cpu lux-all-29m-missing teacher/lux-all-29m-missing -m v2.dec.compose_teacher missing --train "$N4" \
    --source "$LUXT" --source "$HW1" --output /out/uncovered.jsonl
fi
if ! cpu lux-all-59m teacher/lux-all-59m -m v2.dec.compose_teacher compose \
  --train /runs/m6/data/m6-xl-full-59m/train.jsonl "${LUX[@]}" --output /out/teacher.jsonl; then
  ok=0
  cpu lux-all-59m-missing teacher/lux-all-59m-missing -m v2.dec.compose_teacher missing \
    --train /runs/m6/data/m6-xl-full-59m/train.jsonl "${LUX[@]}" --output /out/uncovered.jsonl
fi

A7=()
for p in A7g A7p A7m A7o A7i A7h; do A7+=(--a7v3 "$H/$A7V3/v2/a7/arms/$p/train.jsonl"); done
DEC_DATA=$PAYLOAD_DIR cpu e8f-r2clean data/m6-e8f-r2clean v2/dec/ops/m6/m6-e8f-clean.py --train "$E8F" \
  --expect-sha256 "$E8F_SHA" --excluded-groups /data/excluded-groups.json "${A7[@]}" \
  --quarantine /runs/m3/e8f-a7v3-removed-in-mixture.ids.json --tokenizer "$TOK" --workers 16 \
  --output /out/train.jsonl || exit 1

for mix in m6-xl-full-59m m6-e8f-r2clean; do
  sha=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['output_sha256'])" "$M/data/$mix/train.jsonl.manifest.json")
  DEC_DATA=$PAYLOAD_DIR cpu "exposure-$mix" "exposure/$mix" -m v2.eval.overlap_effects exposure \
    --groups /data/excluded-groups.json --train "/runs/m6/data/$mix/train.jsonl" --expect-sha256 "$sha" \
    --label "decoder M6 $mix" --output "/out/exposure-$mix.json" || exit 1
done
[ $ok = 1 ] || { log "a teacher composition left rows uncovered (see *-missing); prep finished with gaps"; exit 1; }
log "prep finished"
