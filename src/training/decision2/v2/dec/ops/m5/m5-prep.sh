#!/usr/bin/env bash
# Decoder Milestone 5 data preparation on node B (prereg dec-m5-prereg-2026-09-29.md). Nothing trains here.
#   phase n5n:   own-Lux h-w1 download (pinned revision), own-Nox replay labels on N4XF's multilingual-block
#                rows (one GPU) + the whole-batch re-label check + the M3 own-Nox overlap check, the N5N teacher
#                (Nox on block rows, then N4XF's Lux file, then h-w1 on English H8 rows; H7 gold-only) and
#                the training-data diagnostic.
#   phase block: the five-pool superset build, the MLX-DEV holdout + N5B mixture (m5_block; the segment rule's
#                template threshold is an amendment parameter), the MLX-DEV panel, own-Nox labels on N5B's
#                added block rows, the N5B and N5BN teachers and the N5B diagnostic.
# A finished step is skipped on a rerun; each job leaves its launch receipt.
# usage: m5-prep.sh <mirror-dir> <label-gpu 3|4> n5n
#        m5-prep.sh <mirror-dir> <label-gpu 3|4> block <template-max-groups>
set -uo pipefail
SRC=$1 GPU=$2 PHASE=$3
M=/data/dev2/runs/dec/m5
mkdir -p "$M/data" "$M/teacher" "$M/mlxdev" "$M/logs"
L=/data/dev2/src/$SRC/src/training/decision2/v2/dec/launch.sh
log() { echo "$(date -u +%FT%TZ) prep-$PHASE $*" | tee -a "$M/OPERATIONS.log"; }
declare -A RENDER=([3]=/dev/dri/renderD153 [4]=/dev/dri/renderD161)
[ -n "${RENDER[$GPU]:-}" ] || { echo "GPU $GPU is not a decoder M5 GPU" >&2; exit 2; }
export DEC_IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
export DEC_DATA=/data/dev2/runs/dec/m3/data-sel700-cal698
REPO=llm-semantic-router/decision-2.0-training-data
R2=100536133e192c54ec57c2599a5e4706f6d334ff
HW1REV=75e557f170979bdbc428b6ea698a2047e2d2a5cd
H=/hf/datasets--llm-semantic-router--decision-2.0-training-data/snapshots
HH=/data/dev2/hf-cache/datasets--llm-semantic-router--decision-2.0-training-data/snapshots
TOK=/hf/models--llm-semantic-router--Decision-1.0-Eos-0.8B/snapshots/363c4a5e56afc115b1c78c837633956d0bbb63ab
NOX=/hf/models--llm-semantic-router--Decision-1.0-Nox-4B/snapshots/cde2a68dbaa557ea65dc458104d410a0802ee259
NOXREPO=(--teacher-repo llm-semantic-router/Decision-1.0-Nox-4B --teacher-revision cde2a68dbaa557ea65dc458104d410a0802ee259)
N4=/runs/m4/data/m4-xl-full-29m/train.jsonl
LUXT=/runs/m4/teacher/m4-xl-full-29m/lux-teacher.jsonl
IDS=$H/$R2/m3/mixtures/xl-r2/mx-xl-full-r2.ids.jsonl
HW1=$H/$HW1REV/m3/teachers/lux1/xl/h-w1.targets.jsonl
M3NOX=/runs/m3/teacher/nox/nox-teacher.jsonl
T=/runs/m5/teacher

cpu() {  # <job> <out dir under m5> <python args...>: skipped when <out>.launch.json exists with exit 0
  local job=$1 out=$M/$2
  shift 2
  if [ -f "$out.launch.json" ]; then
    grep -q '"exit_status": 0' "$out.launch.json" && return 0
    log "$job failed earlier (receipt $out.launch.json); not rerun"
    return 1
  fi
  if bash "$L" "m5-$job" "$SRC" "$out" --cpu -- "$@"; then
    log "$job done: $(tail -c 400 "$out.stdout.log" | tr '\n' ' ')"
  else
    log "$job FAILED"
    return 1
  fi
}
gpu() {  # <job> <out dir under m5> <python args...>
  local job=$1 out=$M/$2
  shift 2
  if [ -f "$out.launch.json" ]; then
    grep -q '"exit_status": 0' "$out.launch.json" && return 0
    log "$job failed earlier (receipt $out.launch.json); not rerun"
    return 1
  fi
  printf 'track=dec\nstatus=busy\npurpose=decoder M5 prep %s (own-Nox labels / readouts)\nstart_utc=%s\nexpected_end_utc=%s\n' \
    "$job" "$(date -u +%FT%TZ)" "$(date -u -d '+30 min' +%FT%TZ)" > "/data/dev2/leases/gpu$GPU.lock/owner"
  if DEC_RENDER=${RENDER[$GPU]} DEC_GPU_LABEL="node B GPU$GPU" bash "$L" "m5-$job" "$SRC" "$out" -- "$@"; then
    log "$job done: $(tail -c 400 "$out.stdout.log" | tr '\n' ' ')"
  else
    log "$job FAILED"
    return 1
  fi
}
idle() {
  printf 'track=dec\nstatus=idle (decoder M5 prep %s finished)\nend_utc=%s\n' "$PHASE" "$(date -u +%FT%TZ)" > "/data/dev2/leases/gpu$GPU.lock/owner"
}
LUXW=() LUXC=$LUXT
for w in w1 w2 w3 w4 w5 c-w1; do
  LUXW+=(--source "$H/$R2/m3/teachers/lux1/xl/$w.targets.jsonl")
  LUXC+=,$H/$R2/m3/teachers/lux1/xl/$w.targets.jsonl
done

if [ "$PHASE" = n5n ]; then
  if [ ! -f "$M/data/hw1.sha256" ]; then
    export HF_HUB_CACHE=/data/dev2/hf-cache HF_XET_CACHE=/data/dev2/runs/dec/m4/xet-cache
    hf download "$REPO" --repo-type dataset --revision "$HW1REV" --quiet \
      --include 'm3/teachers/lux1/xl/h-w1*' --include 'm3/teachers/lux1/xl/coverage-r2.json' > /dev/null || { log "h-w1 download FAILED"; exit 1; }
    (cd "$HH/$HW1REV" && sha256sum -b m3/teachers/lux1/xl/h-w1* m3/teachers/lux1/xl/coverage-r2.json) > "$M/data/hw1.sha256.pending"
    if ! grep -q '^679ef009.*h-w1.targets.jsonl$' "$M/data/hw1.sha256.pending" \
      || ! grep -q '^ecb6dc36.*coverage-r2.json$' "$M/data/hw1.sha256.pending"; then
      log "h-w1 hash MISMATCH"
      exit 1
    fi
    mv "$M/data/hw1.sha256.pending" "$M/data/hw1.sha256"
    log "h-w1 verified: $(tr '\n' ' ' < "$M/data/hw1.sha256")"
  fi
  cpu rows-n5n teacher/nox-rows-n5n -m v2.dec.m5_labels rows --train "$N4" --output /out/rows.jsonl || exit 1
  cpu check-n5n teacher/nox-check-n5n -m v2.dec.m5_labels check-sample --rows "$T/nox-rows-n5n/rows.jsonl" \
    --tokenizer "$NOX" --output /out/check.jsonl || exit 1
  gpu nox-n5n teacher/nox-n5n -m v2.dec.teacher_label --teacher-path "$NOX" "${NOXREPO[@]}" \
    --train "$T/nox-rows-n5n/rows.jsonl" --output /out/labels.jsonl || { idle; exit 1; }
  gpu nox-n5n-relabel teacher/nox-n5n-relabel -m v2.dec.teacher_label --teacher-path "$NOX" "${NOXREPO[@]}" \
    --train "$T/nox-check-n5n/check.jsonl" --output /out/labels.jsonl || { idle; exit 1; }
  idle
  cpu bitwise-n5n teacher/checks-n5n-bitwise -m v2.dec.m5_labels bitwise --full "$T/nox-n5n/labels.jsonl" \
    --check "$T/nox-n5n-relabel/labels.jsonl" --report /out/bitwise.json || exit 1
  cpu overlap-n5n teacher/checks-n5n-overlap -m v2.dec.m5_labels overlap --labels "$T/nox-n5n/labels.jsonl" \
    --reference "$M3NOX" --report /out/overlap.json || exit 1
  cpu hw1-n4xf-en teacher/hw1-n4xf-en -m v2.dec.m5_labels subset --targets "$HW1" --train "$N4" --component H8 \
    --language en --output /out/targets.jsonl || exit 1
  cpu hw1-n4xf-h8 teacher/hw1-n4xf-h8 -m v2.dec.m5_labels subset --targets "$HW1" --train "$N4" --component H8 \
    --output /out/targets.jsonl || exit 1
  cpu teacher-n5n teacher/n5n -m v2.dec.compose_teacher compose --train "$N4" --source "$T/nox-n5n/labels.jsonl" \
    --source "$LUXT" --source "$T/hw1-n4xf-en/targets.jsonl" --allow-missing-pool "$IDS:H7" \
    --output /out/teacher.jsonl || exit 1
  cpu diag-n4xf teacher/diag-n4xf -m v2.dec.m5_labels diag --train "$N4" \
    --teacher "lux=$LUXT,$T/hw1-n4xf-h8/targets.jsonl" --teacher "nox=$T/nox-n5n/labels.jsonl" --report /out/diag.json || exit 1
  mkdir -p "$M/data/n5n"
  [ -f "$M/data/n5n/BUILT" ] || { echo "train=$N4" > "$M/data/n5n/BUILT"; date -u +%FT%TZ >> "$M/data/n5n/BUILT"; }
  log "phase n5n finished"
  exit 0
fi

[ "$PHASE" = block ] || { echo "unknown phase $PHASE" >&2; exit 2; }
TMG=${4:?block phase needs the amended template-max-groups value}
SRCS=(--source "xl=$H/$R2,-,"
  --source "xa7=$H/$R2,v2/a7/registry.json,v2/"
  --source "xv2=$H/$R2,m2/registry.json,"
  --source "xh=$H/$R2,m3/arms/registry.json,m3/"
  --source "spec=/code/v2/dec/specs,-,")
[ -f "$M/data/n5n/BUILT" ] || { log "phase n5n has not finished"; exit 1; }
cpu build-superset data/superset -m v2.dec.build_mixture --spec /code/v2/dec/specs/m5-block-superset.json "${SRCS[@]}" \
  --tokenizer "$TOK" --workers 40 --output /out/train.jsonl || exit 1
cpu block-n5b data/n5b -m v2.dec.m5_block --n4xf "$N4" --superset /runs/m5/data/superset/train.jsonl \
  --excluded-groups /code/v2/dec/specs/m4-r2-excluded-groups.json --tokenizer "$TOK" --workers 40 \
  --template-max-groups "$TMG" --output-dir /out || exit 1
cpu mlxdev-panel mlxdev/build -m v2.dec.mlx_dev build --superset /runs/m5/data/superset/train.jsonl \
  --selection /runs/m5/data/n5b/mlxdev.selection.json --output /out/panel.jsonl || exit 1
N5B=/runs/m5/data/n5b/train.jsonl
cpu rows-n5b-add teacher/nox-rows-n5b-add -m v2.dec.m5_labels rows --train "$N5B" \
  --exclude-labeled "$T/nox-n5n/labels.jsonl" --output /out/rows.jsonl || exit 1
cpu check-n5b-add teacher/nox-check-n5b-add -m v2.dec.m5_labels check-sample --rows "$T/nox-rows-n5b-add/rows.jsonl" \
  --tokenizer "$NOX" --output /out/check.jsonl || exit 1
gpu nox-n5b-add teacher/nox-n5b-add -m v2.dec.teacher_label --teacher-path "$NOX" "${NOXREPO[@]}" \
  --train "$T/nox-rows-n5b-add/rows.jsonl" --output /out/labels.jsonl || { idle; exit 1; }
gpu nox-n5b-add-relabel teacher/nox-n5b-add-relabel -m v2.dec.teacher_label --teacher-path "$NOX" "${NOXREPO[@]}" \
  --train "$T/nox-check-n5b-add/check.jsonl" --output /out/labels.jsonl || { idle; exit 1; }
idle
cpu bitwise-n5b-add teacher/checks-n5b-add-bitwise -m v2.dec.m5_labels bitwise --full "$T/nox-n5b-add/labels.jsonl" \
  --check "$T/nox-n5b-add-relabel/labels.jsonl" --report /out/bitwise.json || exit 1
cpu overlap-n5b-add teacher/checks-n5b-add-overlap -m v2.dec.m5_labels overlap --labels "$T/nox-n5b-add/labels.jsonl" \
  --reference "$M3NOX" --report /out/overlap.json || exit 1
cpu hw1-n5b-h8 teacher/hw1-n5b-h8 -m v2.dec.m5_labels subset --targets "$HW1" --train "$N5B" --component H8 \
  --output /out/targets.jsonl || exit 1
# A failed teacher compose stops only its own arm (preflight 3): the other arm's teacher is still built.
built=()
if cpu teacher-n5b teacher/n5b -m v2.dec.compose_teacher compose --train "$N5B" --source "$LUXT" "${LUXW[@]}" \
  --source "$T/hw1-n5b-h8/targets.jsonl" --allow-missing-pool "$IDS:H7" --output /out/teacher.jsonl; then
  built+=(n5b)
else
  log "N5B teacher not built: N5B stopped (preflight 3)"
  cpu missing-n5b teacher/missing-n5b -m v2.dec.compose_teacher missing --train "$N5B" --source "$LUXT" "${LUXW[@]}" \
    --source "$T/hw1-n5b-h8/targets.jsonl" --output /out/uncovered.jsonl
fi
if cpu teacher-n5bn teacher/n5bn -m v2.dec.compose_teacher compose --train "$N5B" --source "$T/nox-n5n/labels.jsonl" \
  --source "$T/nox-n5b-add/labels.jsonl" --source "$LUXT" "${LUXW[@]}" --source "$T/hw1-n5b-h8/targets.jsonl" \
  --allow-missing-pool "$IDS:H7" --output /out/teacher.jsonl; then
  built+=(n5bn)
else
  log "N5BN teacher not built: N5BN stopped (preflight 3)"
fi
cpu diag-n5b-sources teacher/diag-n5b-sources -m v2.dec.m5_labels diag --train "$N5B" \
  --teacher "lux=$LUXC,$T/hw1-n5b-h8/targets.jsonl" \
  --teacher "nox=$T/nox-n5n/labels.jsonl,$T/nox-n5b-add/labels.jsonl" --report /out/diag.json || exit 1
for mix in "${built[@]}"; do
  mkdir -p "$M/data/$mix"
  [ -f "$M/data/$mix/BUILT" ] || { echo "train=$N5B" > "$M/data/$mix/BUILT"; date -u +%FT%TZ >> "$M/data/$mix/BUILT"; }
done
log "phase block finished (template-max-groups $TMG)"
