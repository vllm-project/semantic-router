#!/usr/bin/env bash
# One formal post-key same-panel run of an M7 finalist with the eval track's frozen runner, on the
# kernel image with a copy of the frozen Triton autotune cache (m6_formal.sh generalized to M7).
# Usage (node A): m7_formal.sh <gpu> <mirror-dir-name> <name> <package-dir> <revision|auto>
#   (a):  M7_ADAPTER=dev2-06b-causal-8k-sb m7_formal.sh 0 <sha>-src_training_decision2 m7a-mxcx-sb \
#           /data/dev2/runs/06b/m1/arms/m7a-mxcx-sb/full/best-export auto
#   (b):  m7_formal.sh 0 <sha>-src_training_decision2 m7-mxcx-soup \
#           /data/dev2/runs/06b/m1/arms/m7-mxcx-soup/full/best-export auto
# As m6_formal.sh: revision = the export manifest hash (`auto` reads best_export_manifest_sha256 from
# the package's ../COMPLETE.json; an explicit value must equal it and the manifest file's sha256);
# v3 typed FINAL + CSS15 + public 231, seal, report, then mlx-diag in <run>-mlx, each collection with
# a fresh writable copy of the frozen snapshot (M6-CACHE.json records it; m6_summary reads that name),
# paired compares (5,000 draws) against the M6 comparators plus m6-mxcx, the gate checks into
# <run>.gates/, M6-SUMMARY.json, and finally the M7 S3 Score gate (<run>.gates/score-s3.json, which
# also combines S1/S2 from M6-SUMMARY.json into the M7 successor verdict).
# Formal root /data/dev2/runs/06b/m7/formal. M7_ADAPTER selects the adapter spec (a name under
# records/adapters without .json, or a path; default dev2-06b-causal-8k); a package holding
# score_bias.json needs an adapter that passes --score-bias, and vice versa. M7_RELEASED_RUN,
# M7_CHECK_ONLY and M7_LEASE_ARGS behave as M6_RELEASED_RUN, M6_CHECK_ONLY and M6_LEASE_ARGS (which
# are still honoured when the M7_ name is unset).
set -euo pipefail
gpu="$1"
sha="$2"
name="$3"
pkg="${4%/}"
rev="$5"
S=/data/dev2/src/$sha/src/training/decision2
export PYTHONPATH=$S
[[ "$name" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "bad name: $name" >&2; exit 2; }
[[ -d "$pkg" ]] || { echo "package dir missing: $pkg" >&2; exit 1; }
manifest="$pkg.MANIFEST.json"
expected=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['best_export_manifest_sha256'])" "$(dirname "$pkg")/COMPLETE.json")
[[ "$rev" == auto ]] && rev="$expected"
[[ "$rev" == "$expected" ]] || { echo "revision $rev != COMPLETE.json export manifest $expected" >&2; exit 1; }
[[ "$(sha256sum "$manifest" | cut -d' ' -f1)" == "$rev" ]] || { echo "$manifest hash != $rev" >&2; exit 1; }
arm=$(basename "$(dirname "$(dirname "$pkg")")")
released=${M7_RELEASED_RUN:-${M6_RELEASED_RUN:-/data/dev2/runs/06b/m4/formal/m4-t-a7-soup}}
check_only=${M7_CHECK_ONLY:-${M6_CHECK_ONLY:-0}}
lease_spec=${M7_LEASE_ARGS:-${M6_LEASE_ARGS:-}}
snapshot=/data/dev2/runs/06b/m6/triton-cache-frozen
snapshot_manifest=$snapshot.MANIFEST.json
root=/data/dev2/runs/06b/m7/formal
run=$root/$name
gates=$run.gates
adapter=${M7_ADAPTER:-dev2-06b-causal-8k}
[[ "$adapter" == */* ]] || adapter=$S/v2/06b/records/adapters/$adapter.json
[[ -f "$adapter" ]] || { echo "adapter spec missing: $adapter" >&2; exit 1; }
if grep -q -- '"--score-bias"' "$adapter"; then
  [[ -f "$pkg/score_bias.json" ]] || { echo "adapter passes --score-bias but $pkg has no score_bias.json" >&2; exit 1; }
else
  [[ ! -e "$pkg/score_bias.json" ]] || { echo "$pkg has score_bias.json; use an adapter that passes --score-bias" >&2; exit 1; }
fi
pairs=(released="$released" kai1=/data/dev2/runs/eval/m1-adopt/kai1
  kai1-8k=/data/dev2/runs/06b/m2/formal/kai1-native-8k bosun06=/data/dev2/runs/eval/m1-adopt/bosun
  gliner25=/data/dev2/runs/eval/m1/p1-gliner25 m5-z=/data/dev2/runs/06b/m5/formal/m5-z-soup
  m5-x=/data/dev2/runs/06b/m5/formal/m5-x-soup lex=/data/dev2/runs/eval/m1/r3-lex
  causal-control=/data/dev2/runs/eval/m1-adopt/dev20-06b m6-mxcx=/data/dev2/runs/06b/m6/formal/m6-mxcx-soup)
for d in "$run" "$run-mlx"; do
  [[ ! -e "$d/GPU-TIME.json" ]] || { echo "$d already used" >&2; exit 1; }
  [[ ! -e "$d.triton-cache" ]] || { echo "$d.triton-cache exists" >&2; exit 1; }
done
[[ ! -e "$gates" ]] || { echo "$gates exists" >&2; exit 1; }
for pair in "${pairs[@]}"; do
  [[ -f "${pair#*=}/SEAL.json" && -f "${pair#*=}/REPORT.json" ]] || { echo "comparator not sealed: $pair" >&2; exit 1; }
done
python3 -m v2.06b.m6_cache tree "$snapshot" | python3 -c "import json,sys; d=json.load(sys.stdin); \
  m=json.load(open(sys.argv[1])); assert d['tree_sha256'] == m['tree_sha256'], 'snapshot changed'; \
  print('snapshot', d['tree_sha256'], d['files'], 'files')" "$snapshot_manifest"
if [[ "$check_only" == 1 ]]; then
  echo "check only: $name package=$pkg revision=$rev model_id=dev2-06b/$arm adapter=$adapter released=$released run=$run"
  exit 0
fi
end=$(date -u -d '+45 minutes' +%FT%TZ)
lease_args=()
[[ -z "$lease_spec" ]] || read -r -a lease_args <<<"$lease_spec"
collect() {
  local dir="$1" cache="$1.triton-cache"
  python3 -m v2.06b.m6_cache seed --snapshot "$snapshot" --manifest "$snapshot_manifest" --dest "$cache"
  set +e
  "$S/v2/eval/run_same_panel.sh" --gpu "$gpu" --track 06b-encoder --src "$sha" --run-dir "$dir" \
    --model-dir "$pkg" --purpose "0.6B M7 formal run: $name" --expected-end "$end" "${lease_args[@]}" \
    --env TRITON_CACHE_AUTOTUNING=1 --env TRITON_CACHE_DIR="$cache" --mount-rw "$cache" \
    -- --adapter-spec "$adapter" --model-path "$pkg" --revision "$rev" --extra "model_id=dev2-06b/$arm" "${@:2}"
  local code=$?
  set -e
  python3 -m v2.06b.m6_cache record --manifest "$snapshot_manifest" --cache "$cache" --output "$dir/M6-CACHE.json"
  return $code
}
collect "$run"
python3 -m v2.eval.same_panel seal --run-dir "$run"
python3 -m v2.eval.same_panel report --run-dir "$run" --label "DEV2.0-0.6B M7 $name (post-key same-panel)" \
  --tier 0.6B --family decision2 --count-safetensors "$pkg"
comparators=()
for pair in "${pairs[@]}"; do
  python3 -m v2.eval.same_panel compare --run-dir "$run" --comparator-run-dir "${pair#*=}" \
    --left-name "$name" --right-name "${pair%%=*}"
  comparators+=(--comparator "$pair")
done
collect "$run-mlx" --panels mlx-diag
python3 -m v2.eval.multilingual_panel score --panel /data/dev2/private/panels/mlx-diag-v1 \
  --predictions "$run-mlx/output/mlx-diag.predictions.jsonl" --output "$run-mlx/mlx-diag.score.json"
mkdir -p "$gates"
python3 -m v2.eval.gates paired --left "$run" --right "$released" --left-name "$name" --right-name released \
  --output "$gates/paired-vs-released.json"
python3 -m v2.eval.gates types --run "$run" --label "$name" --output "$gates/types.json"
extra=()
if [[ "$name" == m7-control* ]]; then
  python3 -m v2.06b.m6_answers_diff "$run" "$released" --json "$gates/answers-vs-released.json"
  extra=(--control --answers-diff "$gates/answers-vs-released.json")
fi
python3 -m v2.06b.m6_summary --run "$run" --mlx-run "$run-mlx" --gates "$gates" --name "$name" \
  --released released "${comparators[@]}" "${extra[@]}"
python3 -m v2.06b.m7_scorebias gate --run "$run" --gates "$gates" --label "$name" --output "$gates/score-s3.json"
echo "formal done: $name"
