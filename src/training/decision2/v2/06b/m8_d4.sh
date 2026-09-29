#!/usr/bin/env bash
# D4 of one M8 finalist (prereg section 6): a score5t-dev collection of its package with the
# -sb adapter at the formal settings, then the replay against the offline correction.
# Usage (node A): m8_d4.sh <gpu> <mirror-dir-name> <candidate>
#   m8_d4.sh 0 <sha>-src_training_decision2 s5-b05
# Package /data/dev2/runs/06b/m1/arms/m8-<candidate>/full/best-export (revision = its manifest's
# sha256, bound by ../COMPLETE.json), adapter records/adapters/dev2-06b-causal-8k-sb.json (8,192
# tokens), a fresh writable copy of the frozen Triton autotune cache (m6_cache seed / record), the
# exclusive owner lease via run_same_panel.sh. Run dir /data/dev2/runs/06b/m8/d4/<candidate>:
# gold-free seal SEAL-SCORE5T.json, eval readout READOUT-score5t.json, and D4.json from
# `m8_scorebias replay --check <dev>/CHECK.json` (same argmax on all 800, max |dp| <= 1e-4, online
# check-half flags equal to the offline correction's and to D1's, score_bias/model bindings).
# M8_CHECK_ONLY=1 stops after the identity checks.
set -euo pipefail
gpu="$1"
sha="$2"
candidate="$3"
S=/data/dev2/src/$sha/src/training/decision2
export PYTHONPATH=$S
[[ "$gpu" =~ ^[01]$ ]] || { echo "gpu must be 0 or 1 (node A 0.6B allocation)" >&2; exit 2; }
[[ "$candidate" =~ ^s5h?-b(1|05|0)$ ]] || { echo "bad candidate: $candidate" >&2; exit 2; }
home=/data/dev2/runs/06b/m1/arms/m8-$candidate/full
pkg=$home/best-export
dev=/data/dev2/runs/06b/m8/dev/$candidate
run=/data/dev2/runs/06b/m8/d4/$candidate
cache=$run.triton-cache
snapshot=/data/dev2/runs/06b/m6/triton-cache-frozen
snapshot_manifest=$snapshot.MANIFEST.json
adapter=$S/v2/06b/records/adapters/dev2-06b-causal-8k-sb.json
prompts=/data/dev2/private/panels/goldfree/score5t-dev.prompts.jsonl
[[ -d "$pkg" && -f "$pkg/score_bias.json" ]] || { echo "package or its score_bias.json missing: $pkg" >&2; exit 1; }
rev=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['best_export_manifest_sha256'])" "$home/COMPLETE.json")
[[ "$(sha256sum "$pkg.MANIFEST.json" | cut -d' ' -f1)" == "$rev" ]] || { echo "$pkg.MANIFEST.json hash != $rev" >&2; exit 1; }
cmp -s "$pkg/score_bias.json" "$dev/score_bias.json" || { echo "package score_bias.json differs from $dev" >&2; exit 1; }
[[ -f "$dev/CHECK.json" ]] || { echo "missing $dev/CHECK.json" >&2; exit 1; }
[[ ! -e "$run/GPU-TIME.json" && ! -e "$cache" ]] || { echo "$run already used" >&2; exit 1; }
python3 -m v2.06b.m6_cache tree "$snapshot" | python3 -c "import json,sys; d=json.load(sys.stdin); \
  m=json.load(open(sys.argv[1])); assert d['tree_sha256'] == m['tree_sha256'], 'snapshot changed'; \
  print('snapshot', d['tree_sha256'], d['files'], 'files')" "$snapshot_manifest"
if [[ "${M8_CHECK_ONLY:-0}" == 1 ]]; then
  echo "check only: $candidate package=$pkg revision=$rev adapter=$adapter run=$run"
  exit 0
fi
cd /tmp
python3 -m v2.06b.m6_cache seed --snapshot "$snapshot" --manifest "$snapshot_manifest" --dest "$cache"
set +e
"$S/v2/eval/run_same_panel.sh" --gpu "$gpu" --track 06b-encoder --src "$sha" --run-dir "$run" \
  --model-dir "$pkg" --purpose "0.6B M8 D4 replay: $candidate" \
  --expected-end "$(date -u -d '+15 minutes' +%FT%TZ)" \
  --env TRITON_CACHE_AUTOTUNING=1 --env TRITON_CACHE_DIR="$cache" --mount-rw "$cache" \
  -- --adapter-spec "$adapter" --model-path "$pkg" --revision "$rev" \
  --extra "model_id=dev2-06b/m8-$candidate" --panels score5t-dev
code=$?
set -e
python3 -m v2.06b.m6_cache record --manifest "$snapshot_manifest" --cache "$cache" --output "$run/M6-CACHE.json"
(( code == 0 )) || { echo "collection failed: exit $code" >&2; exit "$code"; }
predictions=$run/output/score5t-dev.predictions.jsonl
python3 -m v2.eval.htdev.score seal --prompts "$prompts" --predictions "$predictions" --output "$run/SEAL-SCORE5T.json"
python3 -m v2.eval.dev_readout --run-dir "$run" --label "m8-$candidate" --output "$run/READOUT-score5t.json"
python3 -m v2.06b.m8_scorebias replay --online "$predictions" --score-bias "$pkg/score_bias.json" \
  --check "$dev/CHECK.json" --output "$run/D4.json"
echo "d4 done: $candidate"
