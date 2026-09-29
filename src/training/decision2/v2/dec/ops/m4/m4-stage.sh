#!/usr/bin/env bash
# Decoder M4 finalist staging on node B (prereg dec-m4-prereg-2026-09-29.md, formal section): CAL698
# temperatures at 16,384 tokens, checkpoint + calibrations and a per-file SHA-256 list, then one upload of
# the folder m4/<name>/ to the private staging repo (soups / finalists only; storage policy of 02:45).
# usage: m4-stage.sh <mirror-dir> <name e.g. N4XA-soup> <gpu 0-4> <checkpoint dir relative to /data/dev2/runs/dec> <dev calibration.json (host path)>
set -u
SRC=$1 NAME=$2 GPU=$3 CKREL=$4 DEVCAL=$5
M=/data/dev2/runs/dec/m4
U=$M/hf-staging/$NAME
mkdir -p "$U/m4/$NAME" "$M/stage-cal"
L=/data/dev2/src/$SRC/src/training/decision2/v2/dec/launch.sh
log() { echo "$(date -u +%FT%TZ) stage $NAME $*" >> "$M/soup/OPERATIONS.log"; }
declare -A RENDER=([0]=/dev/dri/renderD129 [1]=/dev/dri/renderD137 [2]=/dev/dri/renderD145 [3]=/dev/dri/renderD153 [4]=/dev/dri/renderD161)
export DEC_IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
export DEC_DATA=/data/dev2/runs/dec/m3/data-sel700-cal698
export DEC_RENDER=${RENDER[$GPU]} DEC_GPU_LABEL="node B GPU$GPU"
C16=$M/stage-cal/$NAME-cal698-16k
if [ ! -f "$C16/calibration.json" ]; then
  bash "$L" "m4-$NAME-cal698-16k" "$SRC" "$C16" -- -m v2.dec.calibrate_ckpt --checkpoint "/runs/$CKREL" \
    --cal /data/cal.jsonl --max-length 16384 --output /out/calibration.json || { log "16K calibration FAILED"; exit 1; }
fi
[ -d "$U/m4/$NAME/checkpoint" ] || cp -al "/data/dev2/runs/dec/$CKREL" "$U/m4/$NAME/checkpoint"
mkdir -p "$U/m4/$NAME/cal698-16k" "$U/m4/$NAME/cal698-8k-dev"
cp "$C16/calibration.json" "$U/m4/$NAME/cal698-16k/calibration.json"
cp "$DEVCAL" "$U/m4/$NAME/cal698-8k-dev/calibration.json"
(cd "$U" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "../$NAME.sha256")
HF=/data/dev2/tools/hf-cli/bin
"$HF/hf" upload llm-semantic-router/dev2-dec-staging "$U" . --repo-type model \
  --commit-message "M4 decoder finalist $NAME: checkpoint + CAL698 calibrations (16K formal, 8K development)" \
  > "$M/hf-staging/$NAME.upload.log" 2>&1
rc=$?
commit=$("$HF/python" -c "from huggingface_hub import HfApi; print(HfApi().list_repo_commits('llm-semantic-router/dev2-dec-staging')[0].commit_id)")
log "upload exit $rc, staging head $commit, files $(wc -l < "$M/hf-staging/$NAME.sha256")"
echo "$commit" > "$M/hf-staging/$NAME.commit"
[ $rc -eq 0 ] || exit 1
