#!/usr/bin/env bash
# One formal post-key same-panel run of a causal 0.6B candidate with the eval track's frozen runner.
# Usage (node A): m4_formal.sh <gpu> <mirror-dir-name> <arm>
# The package is <arm>'s BEST (or soup) export at the 8,192-token native cap; its revision is the
# export manifest hash. Collects v3 typed FINAL + CSS15 + public 231, seals, reports, pairs it with
# the 0.6B comparators on this node, then collects and scores mlx-diag in a separate run directory.
set -euo pipefail
gpu="$1"
sha="$2"
arm="$3"
S=/data/dev2/src/$sha/src/training/decision2
export PYTHONPATH=$S
host=/data/dev2/runs/06b/m1/arms/$arm/full
pkg=$host/best-export
rev=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['best_export_manifest_sha256'])" "$host/COMPLETE.json")
run=/data/dev2/runs/06b/m4/formal/$arm
adapter=$S/v2/06b/records/adapters/dev2-06b-causal-8k.json
end=$(date -u -d '+45 minutes' +%FT%TZ)
collect() {
  "$S/v2/eval/run_same_panel.sh" --gpu "$gpu" --track 06b-encoder --src "$sha" --run-dir "$1" \
    --model-dir "$pkg" --purpose "0.6B M4 formal run: $arm" --expected-end "$end" \
    -- --adapter-spec "$adapter" --model-path "$pkg" --revision "$rev" --extra "model_id=dev2-06b/$arm" "${@:2}"
}
collect "$run"
python3 -m v2.eval.same_panel seal --run-dir "$run"
python3 -m v2.eval.same_panel report --run-dir "$run" --label "DEV2.0-0.6B candidate $arm (post-key same-panel)" \
  --tier 0.6B --family decision2 --count-safetensors "$pkg"
for pair in kai1=/data/dev2/runs/eval/m1-adopt/kai1 kai1-8k=/data/dev2/runs/06b/m2/formal/kai1-native-8k \
  lex=/data/dev2/runs/eval/m1/r3-lex bosun06=/data/dev2/runs/eval/m1-adopt/bosun \
  gliner25=/data/dev2/runs/eval/m1/p1-gliner25 causal-control=/data/dev2/runs/eval/m1-adopt/dev20-06b; do
  python3 -m v2.eval.same_panel compare --run-dir "$run" --comparator-run-dir "${pair#*=}" \
    --left-name "$arm" --right-name "${pair%%=*}"
done
collect "$run-mlx" --panels mlx-diag
python3 -m v2.eval.multilingual_panel score --panel /data/dev2/private/panels/mlx-diag-v1 \
  --predictions "$run-mlx/output/mlx-diag.predictions.jsonl" --output "$run-mlx/mlx-diag.score.json"
printf 'track=06b-encoder\nstatus=idle (M4 formal run done: %s)\nlast_job_end_utc=%s\n' "$arm" "$(date -u +%FT%TZ)" \
  > "/data/dev2/leases/gpu$gpu.lock/owner"
echo "formal done: $arm"
