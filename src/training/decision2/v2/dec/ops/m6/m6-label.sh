#!/usr/bin/env bash
# Decoder M6 own-1.0 teacher labels (prereg dec-m6-prereg-2026-09-29.md), the M3 own-Sol procedure:
# v2.dec.teacher_label with the teacher package's published per-type temperatures (Sol 1.30036), one shard,
# then v2.dec.merge_labels into the trainer file (every row exactly once), then the lock check.
#   sol (node B GPU4): all rows of m6-xl-full-59m -> teacher/sol-59m; m6-xl-full-59m is nested over
#                      m4-xl-full-29m (data lock part 1), so sol-29m is the same labels composed on c7d51219's rows.
#   eos (node A GPU5): all rows of m6-e8f-r2clean -> teacher/eos-e8f-r2clean.
# The chains wait for teacher/<name>/READY (written only after the lock part 2 naming the hash is committed).
# usage: m6-label.sh <mirror-dir> sol|eos
set -uo pipefail
SRC=$1 KIND=$2
M=/data/dev2/runs/dec/m6
L=/data/dev2/src/$SRC/src/training/decision2/v2/dec/launch.sh
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m6
log() { echo "$(date -u +%FT%TZ) label-$KIND $*" | tee -a "$M/OPERATIONS.log"; }
job() {  # <job> <out dir under m6> [--cpu] <python args...>: a job with a receipt is never rerun
  local name=$1 out=$M/$2
  shift 2
  if [ -f "$out.launch.json" ]; then
    grep -q '"exit_status": 0' "$out.launch.json" && return 0
    log "$name failed earlier (receipt $out.launch.json); not rerun"
    return 1
  fi
  if bash "$L" "m6-$name" "$SRC" "$out" "$@"; then
    log "$name done: $(tail -c 300 "$out.stdout.log" | tr '\n' ' ')"
  else
    log "$name FAILED: $(tail -c 300 "$out.stderr.log" | tr '\n' ' ')"
    return 1
  fi
}
case $KIND in
  sol)
    export DEC_IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
    export DEC_DATA=/data/dev2/runs/dec/m3/data-sel700-cal698 DEC_RENDER=/dev/dri/renderD161 DEC_GPU_LABEL="node B GPU4"
    T=/hf/models--llm-semantic-router--Decision-1.0-Sol-2B/snapshots/ce0c018a28de16d6639b1cd203b761bf643b89e6
    ID=(--teacher-repo llm-semantic-router/Decision-1.0-Sol-2B --teacher-revision ce0c018a28de16d6639b1cd203b761bf643b89e6)
    TRAIN=/runs/m6/data/m6-xl-full-59m/train.jsonl
    ;;
  eos)
    export DEC_IMAGE=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
    export DEC_DATA=/data/decision20-20260926/data/hf-private-decision20-clean-v2 DEC_RENDER=/dev/dri/renderD169 DEC_GPU_LABEL="node A GPU5"
    T=/hf/models--llm-semantic-router--Decision-1.0-Eos-0.8B/snapshots/363c4a5e56afc115b1c78c837633956d0bbb63ab
    ID=(--teacher-repo llm-semantic-router/Decision-1.0-Eos-0.8B --teacher-revision 363c4a5e56afc115b1c78c837633956d0bbb63ab)
    TRAIN=/runs/m6/data/m6-e8f-r2clean/train.jsonl
    ;;
  *) echo "unknown label kind $KIND" >&2; exit 2 ;;
esac
mkdir -p "$M/teacher" "$M/status"
job "label-$KIND" "teacher/$KIND-labels" -- -m v2.dec.teacher_label --teacher-path "$T" "${ID[@]}" --train "$TRAIN" \
  --output /out/labels.jsonl || exit 1
if [ "$KIND" = sol ]; then
  job sol-59m teacher/sol-59m --cpu -- -m v2.dec.merge_labels --train "$TRAIN" \
    --part /runs/m6/teacher/sol-labels/labels.jsonl --output /out/teacher.jsonl || exit 1
  job sol-29m teacher/sol-29m --cpu -- -m v2.dec.compose_teacher compose --train /runs/m4/data/m4-xl-full-29m/train.jsonl \
    --source /runs/m6/teacher/sol-labels/labels.jsonl --output /out/teacher.jsonl || exit 1
else
  job eos-e8f-r2clean teacher/eos-e8f-r2clean --cpu -- -m v2.dec.merge_labels --train "$TRAIN" \
    --part /runs/m6/teacher/eos-labels/labels.jsonl --output /out/teacher.jsonl || exit 1
fi
if PYTHONPATH=/data/dev2/src/$SRC/src/training/decision2 python3 "$OPS/m6-lockcheck.py" "$KIND" > "$M/lock-$KIND.log" 2>&1; then
  log "lock check $KIND PASS: $(tail -1 "$M/lock-$KIND.log")"
  date -u +%FT%TZ > "$M/status/label-$KIND.LABELED"
else
  log "lock check $KIND FAIL: $(tail -1 "$M/lock-$KIND.log")"
  exit 1
fi
