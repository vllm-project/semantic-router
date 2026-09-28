#!/usr/bin/env bash
# One formal post-key same-panel run of an M6 finalist (or the released-package control) with the
# eval track's frozen runner, on the kernel image with a copy of the frozen Triton autotune cache.
# Usage (node A): m6_formal.sh <gpu> <mirror-dir-name> <name> <package-dir> <revision|auto>
#   finalist: m6_formal.sh 0 <sha>-src_training_decision2 m6-cx-soup \
#               /data/dev2/runs/06b/m1/arms/m6-cx-soup/full/best-export auto
#   control:  m6_formal.sh 0 <sha>-src_training_decision2 m6-control-released \
#               /data/dev2/runs/06b/m1/arms/m4-t-a7-soup/full/best-export auto
# As m5_formal.sh: revision = the export manifest hash (`auto` reads best_export_manifest_sha256 from
# the package's ../COMPLETE.json; an explicit value must equal it and the manifest file's sha256);
# v3 typed FINAL + CSS15 + public 231, seal, report, then mlx-diag in <run>-mlx. Each collection gets
# a fresh writable copy of the frozen snapshot (<run>.triton-cache) passed as TRITON_CACHE_DIR with
# TRITON_CACHE_AUTOTUNING=1; M6-CACHE.json records the snapshot tree before and the copy after.
# Then paired compares (5,000 draws), the gate checks (human transfer vs released, type collapse)
# into <run>.gates/, and M6-SUMMARY.json. Names starting with m6-control skip the successor verdict
# and diff their answers against the released run. M6_RELEASED_RUN overrides the released run
# (e.g. the control run, if the control's answers differ from the M4 run). M6_CHECK_ONLY=1 runs the
# identity, comparator and snapshot checks and exits before any GPU step. M6_LEASE_ARGS is passed to
# run_same_panel.sh as extra lease flags, e.g. "--lease-name owner.m6-formal --shared" for a recorded
# co-tenancy when another track's shared-lease job occupies the GPU (GPU-TIME.json records it).
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
released=${M6_RELEASED_RUN:-/data/dev2/runs/06b/m4/formal/m4-t-a7-soup}
snapshot=/data/dev2/runs/06b/m6/triton-cache-frozen
snapshot_manifest=$snapshot.MANIFEST.json
root=/data/dev2/runs/06b/m6/formal
run=$root/$name
gates=$run.gates
adapter=$S/v2/06b/records/adapters/dev2-06b-causal-8k.json
pairs=(released="$released" kai1=/data/dev2/runs/eval/m1-adopt/kai1
  kai1-8k=/data/dev2/runs/06b/m2/formal/kai1-native-8k bosun06=/data/dev2/runs/eval/m1-adopt/bosun
  gliner25=/data/dev2/runs/eval/m1/p1-gliner25 m5-z=/data/dev2/runs/06b/m5/formal/m5-z-soup
  m5-x=/data/dev2/runs/06b/m5/formal/m5-x-soup lex=/data/dev2/runs/eval/m1/r3-lex
  causal-control=/data/dev2/runs/eval/m1-adopt/dev20-06b)
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
if [[ "${M6_CHECK_ONLY:-0}" == 1 ]]; then
  echo "check only: $name package=$pkg revision=$rev model_id=dev2-06b/$arm released=$released run=$run"
  exit 0
fi
end=$(date -u -d '+45 minutes' +%FT%TZ)
lease_args=()
[[ -z "${M6_LEASE_ARGS:-}" ]] || read -r -a lease_args <<<"$M6_LEASE_ARGS"
collect() {
  local dir="$1" cache="$1.triton-cache"
  python3 -m v2.06b.m6_cache seed --snapshot "$snapshot" --manifest "$snapshot_manifest" --dest "$cache"
  set +e
  "$S/v2/eval/run_same_panel.sh" --gpu "$gpu" --track 06b-encoder --src "$sha" --run-dir "$dir" \
    --model-dir "$pkg" --purpose "0.6B M6 formal run: $name" --expected-end "$end" "${lease_args[@]}" \
    --env TRITON_CACHE_AUTOTUNING=1 --env TRITON_CACHE_DIR="$cache" --mount-rw "$cache" \
    -- --adapter-spec "$adapter" --model-path "$pkg" --revision "$rev" --extra "model_id=dev2-06b/$arm" "${@:2}"
  local code=$?
  set -e
  python3 -m v2.06b.m6_cache record --manifest "$snapshot_manifest" --cache "$cache" --output "$dir/M6-CACHE.json"
  return $code
}
collect "$run"
python3 -m v2.eval.same_panel seal --run-dir "$run"
python3 -m v2.eval.same_panel report --run-dir "$run" --label "DEV2.0-0.6B M6 $name (post-key same-panel)" \
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
if [[ "$name" == m6-control* ]]; then
  python3 -m v2.06b.m6_answers_diff "$run" "$released" --json "$gates/answers-vs-released.json"
  extra=(--control --answers-diff "$gates/answers-vs-released.json")
fi
python3 -m v2.06b.m6_summary --run "$run" --mlx-run "$run-mlx" --gates "$gates" --name "$name" \
  --released released "${comparators[@]}" "${extra[@]}"
echo "formal done: $name"
