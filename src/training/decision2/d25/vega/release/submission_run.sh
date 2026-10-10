#!/usr/bin/env bash
# Decision Index 0.3 public run of a Decision 2.5 package for a submission: the kit's runner with the package's
# engine, one request at a time, one runner process per GPU over deterministic shards, then merge, kit scoring
# and the staged run directory (see submission.py). Resumable: rerun the same script (the kit runner skips
# finished requests and retries errors; finished steps are skipped).
#
#   PKG=<package dir> KIT=<kit dir> SUITE=<suite-0.3 dir> OUT=<work dir> NAME=<run name> GPUS=<n> \
#   REPO=vllm-sr/Decision-2.5-Vega-27B REVISION=<sha> HARDWARE="<n> x <GPU>, one request at a time per process" \
#   bash submission_run.sh
set -uo pipefail
: "${PKG:?}" "${KIT:?}" "${SUITE:?}" "${OUT:?}" "${NAME:?}" "${GPUS:?}" "${REPO:?}" "${REVISION:?}" "${HARDWARE:?}"
S=$OUT/shards
ENGINE=decision25_engine:Decision25Engine; [ -f "$PKG/d3_engine.py" ] && ENGINE=d3_engine:D3Engine
mkdir -p "$S"
[ -f "$S/shards.json" ] || python -m d25.vega.release.submission shard --kit "$KIT" --suite-dir "$SUITE" --n "$GPUS" --out "$S" \
  || exit 1
pids=()
for i in $(seq 0 $((GPUS - 1))); do
  n=$(printf %02d "$i")
  ( PYTHONPATH=$PKG:$KIT:${PYTHONPATH:-} python -m decision_index run --engine $ENGINE \
      --option model="$PKG" --option "device=cuda:$i" --rows "$S/shard-$n.jsonl.gz" --out "$S/run-$n" --compact \
      >> "$S/run-$n.log" 2>&1 ) &
  pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[ $failed = 0 ] || { echo "a runner failed; rerun to resume (see $S/run-*.log)"; exit 1; }
[ -f "$OUT/run/scores.json" ] || python -m d25.vega.release.submission merge --kit "$KIT" --suite-dir "$SUITE" --shards "$S" \
  --name "$NAME" --out "$OUT/run" || exit 1
[ -d "$OUT/stage" ] || python -m d25.vega.release.submission stage --kit "$KIT" --suite-dir "$SUITE" --run "$OUT/run" \
  --out "$OUT/stage" --manifest "$PKG/MODEL_MANIFEST.json" --repo "$REPO" --revision "$REVISION" --hardware "$HARDWARE" \
  ${SETTINGS:+--settings "$SETTINGS"} \
  || exit 1
echo ALL-DONE
