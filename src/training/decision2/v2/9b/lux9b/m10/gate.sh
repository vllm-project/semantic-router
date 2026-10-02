#!/usr/bin/env bash
# 9B M10 amendment 7 gate, node side (CPU): paired bootstraps of scored IX1 runs on this node minus M10-KIB4-a40-bf16,
# the current release's run, exactly as m10/ixchain.sh runs its bootstraps (v2.eval.ix1.paired_boot, 2,000 replicates,
# seed 20261002, cases resampled within benchmarks through the board weights; full panel and transfer-only). Each RUN
# (an M10 or arm-factory run) is read in place; the outputs go to ix1/m10/gate/RUN/ (private). No Index value is logged.
# The reference is node C's own run, or the copy that ix.sh kref placed in ix1/m10/refs.
#
# usage: gate.sh <mirror-dir> RUN...
set -uo pipefail
M=$1
shift
S=$M/src/training/decision2
R=/data/dev2/private/eval/index021/ix1
K=M10-KIB4-a40-bf16
kdir=$R/m10/refs/$K
[ -f "$R/runs/$K/merged/results.jsonl" ] && kdir=$R/runs/$K
[ -f "$kdir/merged/results.jsonl" ] || { echo "no $K run on this node (ix.sh kref)"; exit 3; }
umask 077
cd "$R/.." || exit 1
for RUN in "$@"; do
  run=$R/runs/$RUN out=$R/m10/gate/$RUN
  python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["scorers"]["pass"] else 1)' \
    "$run/merged/compare.json" 2> /dev/null || { echo "$(date -u +%FT%TZ) $RUN: not scored, or its scorer gate failed"; continue; }
  [ ! -f "$out/DONE" ] || { echo "$(date -u +%FT%TZ) $RUN: gate done before"; continue; }
  mkdir -p "$out"
  printf '{"reference_run": "%s", "run": "%s", "base_results_sha256": "%s", "new_results_sha256": "%s"}\n' "$kdir" "$run" \
    "$(sha256sum < "$kdir/merged/results.jsonl" | cut -c1-64)" "$(sha256sum < "$run/merged/results.jsonl" | cut -c1-64)" \
    > "$out/ref.json"
  echo "$(date -u +%FT%TZ) $RUN: gate bootstraps started (reference $kdir)"
  for v in full transfer; do
    x=()
    [ "$v" = transfer ] && x=(--exclude HoVer When2Call iSarcasmEval GSM8K BPoMP)
    [ -f "$out/paired-boot-$v-vs-kib4a40.json" ] && continue
    CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' PYTHONPATH=$S:$PWD/kit-19ad28ec nice -n 10 \
      venv/bin/python -m v2.eval.ix1.paired_boot --suite-dir suite-0.2 --base "$kdir/merged/results.jsonl" \
      --new "$run/merged/results.jsonl" --external "$R/../external/index021-frontier-gap-2026-10-01.json" \
      --replicates 2000 --seed 20261002 --workers 20 "${x[@]}" --out "$out/paired-boot-$v-vs-kib4a40.json" \
      >> "$out/boot.log" 2>&1 &
  done
  wait
  if [ -f "$out/paired-boot-full-vs-kib4a40.json" ] && [ -f "$out/paired-boot-transfer-vs-kib4a40.json" ]; then
    date -u +%FT%TZ > "$out/DONE"
    echo "$(date -u +%FT%TZ) $RUN: gate bootstraps done"
  else
    echo "$(date -u +%FT%TZ) $RUN: a gate bootstrap is MISSING (see $out/boot.log)"
  fi
done
