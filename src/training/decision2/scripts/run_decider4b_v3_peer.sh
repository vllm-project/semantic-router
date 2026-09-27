#!/usr/bin/env bash
# Run the preregistered Decider 4B external peer on gold-free frozen panels.
# Mount only /work/code, /work/model, /work/panels, and /work/output.
set -euo pipefail

cd /work/code

check_sha() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "$path")"
  actual="${actual%% *}"
  if [[ "$actual" != "$expected" ]]; then
    echo "Frozen input hash mismatch: $path" >&2
    exit 2
  fi
}

check_sha b49054f1aef7a35c0a65b88dd5c1f1e5e252bb96dc210c7ef9e1d83942bb45ce /work/code/inference/run.py
check_sha ee8ce585b3cedd93206dd149b09b4bdd683174874211f85c77a36090b90c9fdd /work/model/model.safetensors
check_sha fc83293b6f707172e20176d603ea88dd1a3be2abd3596fdb1ccbec9404626baf /work/model/decider_config.json
check_sha 6359e5989fe922054c99446115a25d22acf1a943f0dea097409eaa49e2ef84f1 /work/model/decider/infer.py
check_sha e2a4a86bc978fc7497823e106533d8aa896a0307453d712f7bf99ee3174e87bd /work/panels/typed-final.prompts.jsonl
check_sha 7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6 /work/panels/css15.prompts.jsonl
check_sha 642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd /work/panels/public231.prompts.jsonl

export PYTHONPATH=/work/code
revision=eb5fbdfc9448473ec25e399882912863afbdb70e
for panel in public231 typed-final css15; do
  if [[ -e "/work/output/$panel.predictions.jsonl" ]]; then
    echo "Output already exists: $panel" >&2
    exit 2
  fi
done

for panel in public231 typed-final css15; do
  python3 -m inference.run \
    --backend decider \
    --model-path /work/model \
    --model-revision "$revision" \
    --input "/work/panels/$panel.prompts.jsonl" \
    --output "/work/output/$panel.predictions.jsonl"
done
