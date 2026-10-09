#!/usr/bin/env bash
# CPU worker: build the SYN1 plan once, then run generation shards against the vLLM Service until
# every shard of the range is sealed (unsealed shards resume from their request cache).
# Env: TAG (staged code), WORK (work dir name under /data/d25/vega/synth), SHARDS (first:last),
#      LLM_URL (default http://d25-vega-synth-llm:8000), GEN_ARGS (extra generate.py flags),
#      PLAN_SEEDS (default 18000), ATTEMPTS (default 6).
set -euo pipefail
: "${TAG:?}" "${WORK:?}" "${SHARDS:?}"
export PYTHONPATH=/data/d25/vega/src/$TAG
W=/data/d25/vega/synth
MODEL=/data/d25/shared/models/qwen3.5-397b-a17b-fp8
URL=${LLM_URL:-http://d25-vega-synth-llm:8000}
mkdir -p "$W/plan" "$W/logs" "$W/$WORK"
PLAN=$W/plan/syn1-plan.jsonl.gz
if [[ ! -f $PLAN ]]; then
  python3 -m d25.vega.data.synth.plan --seeds "${PLAN_SEEDS:-18000}" --out "$PLAN"
fi
first=${SHARDS%:*}; last=${SHARDS#*:}
total=$(python3 -c "import gzip; print(sum(1 for _ in gzip.open('$PLAN')))")
last=$(( last < (total + 99) / 100 ? last : (total + 99) / 100 ))
for attempt in $(seq 1 "${ATTEMPTS:-6}"); do
  until [[ "$(curl -s -o /dev/null -w '%{http_code}' "$URL/health")" == 200 ]]; do echo "waiting for $URL"; sleep 30; done
  # shellcheck disable=SC2086
  python3 -m d25.vega.data.synth.generate --plan "$PLAN" --work "$W/$WORK" --shards "$first:$last" \
    --gen-url "$URL" --gen-model qwen3.5-397b-a17b-fp8 --tokenizer "$MODEL" ${GEN_ARGS:-} \
    2>&1 | tee -a "$W/logs/generate-$WORK.log"
  open=0
  for ((i = first; i < last; i++)); do [[ -f $(printf "%s/%s/shards/%05d/DONE" "$W" "$WORK" "$i") ]] || open=$((open + 1)); done
  echo "attempt $attempt: $open unsealed shards in $first:$last"
  [[ $open == 0 ]] && exit 0
  sleep 60
done
exit 1
