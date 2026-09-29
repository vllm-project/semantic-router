#!/usr/bin/env bash
# One formal post-key same-panel run of an M8 finalist (m7_formal.sh generalized to M8, prereg
# sections 7-8) with the eval track's frozen runner, on the kernel image with a fresh copy of the
# frozen Triton autotune cache.
# Usage (node A): m8_formal.sh <gpu> <mirror-dir-name> <candidate> [<revision|auto>]
#   m8_formal.sh 0 <sha>-src_training_decision2 s5-b05
# Package /data/dev2/runs/06b/m1/arms/m8-<candidate>/full/best-export (must hold score_bias.json);
# revision = the export manifest hash (`auto`, the default, reads best_export_manifest_sha256 from
# ../COMPLETE.json; an explicit value must equal it and the manifest file's sha256). Adapter fixed
# to records/adapters/dev2-06b-causal-8k-sb.json. Run m8-<candidate> under
# /data/dev2/runs/06b/m8/formal: v3 typed FINAL + CSS15 + public 231, seal, report, then mlx-diag
# in <run>-mlx, each collection with a fresh writable copy of the frozen snapshot (M6-CACHE.json),
# paired compares (5,000 draws) against the M7 comparators, the gate checks into <run>.gates/
# (paired vs released and vs gliner25, types, public231 vs released = R7), M6-SUMMARY.json, the
# M8 Score report (score-report.json) and the R4 mlx-diag paired bootstrap (mlx-paired.json).
# M8_RELEASED_RUN overrides the released run (its mlx-diag run is <released>-mlx), M8_CHECK_ONLY=1
# stops after the identity checks, M8_LEASE_ARGS adds run_same_panel.sh lease arguments.
set -euo pipefail
gpu="$1"
sha="$2"
candidate="$3"
rev="${4:-auto}"
S=/data/dev2/src/$sha/src/training/decision2
export PYTHONPATH=$S
[[ "$gpu" =~ ^[01]$ ]] || { echo "gpu must be 0 or 1 (node A 0.6B allocation)" >&2; exit 2; }
[[ "$candidate" =~ ^s5h?-b(1|05|0)$ ]] || { echo "bad candidate: $candidate" >&2; exit 2; }
name=m8-$candidate
arm=m8-$candidate
pkg=/data/dev2/runs/06b/m1/arms/$arm/full/best-export
[[ -d "$pkg" ]] || { echo "package dir missing: $pkg" >&2; exit 1; }
[[ -f "$pkg/score_bias.json" ]] || { echo "$pkg has no score_bias.json" >&2; exit 1; }
manifest="$pkg.MANIFEST.json"
expected=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['best_export_manifest_sha256'])" "$(dirname "$pkg")/COMPLETE.json")
[[ "$rev" == auto ]] && rev="$expected"
[[ "$rev" == "$expected" ]] || { echo "revision $rev != COMPLETE.json export manifest $expected" >&2; exit 1; }
[[ "$(sha256sum "$manifest" | cut -d' ' -f1)" == "$rev" ]] || { echo "$manifest hash != $rev" >&2; exit 1; }
released=${M8_RELEASED_RUN:-/data/dev2/runs/06b/m4/formal/m4-t-a7-soup}
check_only=${M8_CHECK_ONLY:-0}
lease_spec=${M8_LEASE_ARGS:-}
snapshot=/data/dev2/runs/06b/m6/triton-cache-frozen
snapshot_manifest=$snapshot.MANIFEST.json
root=/data/dev2/runs/06b/m8/formal
run=$root/$name
gates=$run.gates
adapter=$S/v2/06b/records/adapters/dev2-06b-causal-8k-sb.json
[[ -f "$adapter" ]] || { echo "adapter spec missing: $adapter" >&2; exit 1; }
grep -q -- '"--score-bias"' "$adapter" || { echo "$adapter does not pass --score-bias" >&2; exit 1; }
gliner25=/data/dev2/runs/eval/m1/p1-gliner25
pairs=(released="$released" kai1=/data/dev2/runs/eval/m1-adopt/kai1
  kai1-8k=/data/dev2/runs/06b/m2/formal/kai1-native-8k bosun06=/data/dev2/runs/eval/m1-adopt/bosun
  gliner25="$gliner25" m5-z=/data/dev2/runs/06b/m5/formal/m5-z-soup
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
[[ -f "$released-mlx/mlx-diag.score.json" ]] || { echo "released mlx-diag run missing: $released-mlx" >&2; exit 1; }
python3 -m v2.06b.m6_cache tree "$snapshot" | python3 -c "import json,sys; d=json.load(sys.stdin); \
  m=json.load(open(sys.argv[1])); assert d['tree_sha256'] == m['tree_sha256'], 'snapshot changed'; \
  print('snapshot', d['tree_sha256'], d['files'], 'files')" "$snapshot_manifest"
if [[ "$check_only" == 1 ]]; then
  echo "check only: $name package=$pkg revision=$rev model_id=dev2-06b/$arm adapter=$adapter released=$released run=$run"
  exit 0
fi
cd /tmp
end=$(date -u -d '+45 minutes' +%FT%TZ)
lease_args=()
[[ -z "$lease_spec" ]] || read -r -a lease_args <<<"$lease_spec"
collect() {
  local dir="$1" cache="$1.triton-cache"
  python3 -m v2.06b.m6_cache seed --snapshot "$snapshot" --manifest "$snapshot_manifest" --dest "$cache"
  set +e
  "$S/v2/eval/run_same_panel.sh" --gpu "$gpu" --track 06b-encoder --src "$sha" --run-dir "$dir" \
    --model-dir "$pkg" --purpose "0.6B M8 formal run: $name" --expected-end "$end" "${lease_args[@]}" \
    --env TRITON_CACHE_AUTOTUNING=1 --env TRITON_CACHE_DIR="$cache" --mount-rw "$cache" \
    -- --adapter-spec "$adapter" --model-path "$pkg" --revision "$rev" --extra "model_id=dev2-06b/$arm" "${@:2}"
  local code=$?
  set -e
  python3 -m v2.06b.m6_cache record --manifest "$snapshot_manifest" --cache "$cache" --output "$dir/M6-CACHE.json"
  return $code
}
collect "$run"
python3 -m v2.eval.same_panel seal --run-dir "$run"
python3 -m v2.eval.same_panel report --run-dir "$run" --label "DEV2.0-0.6B M8 $name (post-key same-panel)" \
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
python3 -m v2.eval.gates paired --left "$run" --right "$gliner25" --left-name "$name" --right-name gliner25 \
  --output "$gates/paired-vs-gliner25.json"
python3 -m v2.eval.gates types --run "$run" --label "$name" --output "$gates/types.json"
python3 -m v2.eval.gates public231 --left "$run" --right "$released" --left-name "$name" --right-name released \
  --output "$gates/public231-vs-released.json"
python3 -m v2.06b.m6_summary --run "$run" --mlx-run "$run-mlx" --gates "$gates" --name "$name" \
  --released released "${comparators[@]}"
python3 -m v2.06b.m8_scorebias score-report --run "$run" --gates "$gates" --label "$name" \
  --output "$gates/score-report.json"
python3 -m v2.06b.m8_scorebias mlx-paired --candidate-run "$run-mlx" --released-run "$released-mlx" \
  --output "$gates/mlx-paired.json"
echo "formal done: $name"
