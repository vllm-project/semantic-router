#!/usr/bin/env bash
# Private push of a Decision 2.5 export: build the package, upload it to a PRIVATE repository, read it back, and
# (optionally) submit the RTX PRO 6000 job for the uploaded revision. Run on the node that holds the export, in a
# pod with no GPU; everything is resumable (finished steps are skipped).
#
#   EXPORT=<export dir> REPO=vllm-sr/Decision-2.5-Vega-27B NAME=Decision-2.5-Vega-27B CARD=<card dir> \
#   TEACHERS=<teachers.json> WORK=<work dir> [WAIT_HOURS=6] [CUDA_JOB=1 KIT=<kit dir> RUN=<job run name>] \
#   [EXPECT_IDENTITY=<model_sha256 verified on staging>] [CUDA_STEPS="smoke latency parity"]  (add "variants" only to re-characterise batch size / fla / conv) \
#   bash private_push.sh
#
# A dry run is the same command with REPO=vllm-sr/d25-vega-staging. hub.py creates the repository PRIVATE when
# it does not exist and refuses to upload to a public one; making a repository public is not part of this script.
set -uo pipefail
: "${EXPORT:?}" "${REPO:?}" "${NAME:?}" "${CARD:?}" "${TEACHERS:?}" "${WORK:?}"
WAIT_HOURS=${WAIT_HOURS:-0}
mkdir -p "$WORK"
step() { echo; echo "=== $(date -u +%H:%M:%S) $*"; }
fail() { echo "FAILED: $*"; exit 1; }

step wait-for-export
deadline=$(( $(date +%s) + ${WAIT_HOURS%.*} * 3600 ))
# Exports are written to <dir>.partial and renamed when complete.
until [ -f "$EXPORT/decision_config.json" ] && [ -f "$EXPORT/readout.safetensors" ]; do
  [ "$(date +%s)" -lt "$deadline" ] || fail "no complete export at $EXPORT"
  sleep 120
done

step build
read -r -a BUILD_ARG_LIST <<< "${BUILD_ARGS:-}"
[ -d "$WORK/package" ] || python -m d25.vega.release.build --export "$EXPORT" --out "$WORK/package" --model-name "$NAME" "${BUILD_ARG_LIST[@]}" \
  --repo-id "$REPO" --card "$CARD" --teachers "$TEACHERS" > "$WORK/build.json" || fail build
cat "$WORK/build.json"
if [ -n "${EXPECT_IDENTITY:-}" ]; then
  got=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['model_sha256'])" "$WORK/build.json")
  [ "$got" = "$EXPECT_IDENTITY" ] || fail "model identity $got != expected $EXPECT_IDENTITY"
fi

step headroom
need=$(python -c "import json,sys; print(int(json.load(open(sys.argv[1]))['bytes'] / 1e9) + 5)" "$WORK/build.json")
python -m d25.vega.release.hub headroom --need-gb "$need" --receipt "$WORK/headroom.json" > /dev/null || fail "storage headroom"

step upload
[ -f "$WORK/upload.json" ] || python -m d25.vega.release.hub upload --repo "$REPO" --package "$WORK/package" \
  --message "$NAME from $(basename "$EXPORT")" --receipt "$WORK/upload.json" || fail upload
REV=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['revision'])" "$WORK/upload.json")
echo "revision $REV"

step readback
python -m d25.vega.release.hub readback --repo "$REPO" --revision "$REV" --package "$WORK/package" \
  --receipt "$WORK/readback.json" > /dev/null || fail "readback (see $WORK/readback.json)"

if [ "${CUDA_JOB:-0}" = 1 ]; then
  step cuda-job
  read -r -a CUDA_ARG_LIST <<< "${CUDA_ARGS:-}"
  [ -f "$WORK/cuda-job.json" ] || python -m d25.vega.release.hf_job --model "$REPO" --revision "$REV" --run "${RUN:?}" \
    --kit "${KIT:?}" --steps "${CUDA_STEPS:-smoke latency parity}" "${CUDA_ARG_LIST[@]}" > "$WORK/cuda-job.json" || fail "cuda job submission"
  cat "$WORK/cuda-job.json"
fi
echo ALL-DONE
