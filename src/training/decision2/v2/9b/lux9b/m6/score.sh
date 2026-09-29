#!/usr/bin/env bash
# usage: score.sh SHA OUT_NAME [--runtime RSHA] ARM=RUN_PREFIX... -- COMPARE...
# CPU-only development readout (typed DEV + CSS pilot) with v2.dec.dev_readout from the runtime
# mirror RSHA (default the incumbent's 3277dec9d; container d2-9b-m6-OUT_NAME-score) into
# /data/dev2/runs/9b/m6/OUT_NAME. ARM=RUN_PREFIX reads RUN_PREFIX-dev/dev.predictions.jsonl and
# RUN_PREFIX-css-pilot/css-pilot.predictions.jsonl under /data/dev2/runs/9b (e.g.
# lux=m4/lux-16k, inc=m4/K-a13, k=m6/K5-a13); COMPARE is "a:b" (b minus a).
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
sha=$1; out_name=$2; shift 2
rt=$RUNTIME_DEFAULT
if [ "${1:-}" = "--runtime" ]; then rt=${2:?--runtime needs a SHA}; shift 2; fi
S=$(code_dir "$rt")
[ -d "$S" ] || { echo "runtime mirror $S missing" >&2; exit 2; }
out=$M6/$out_name
[ ! -e "$out" ] || { echo "$out exists" >&2; exit 66; }
mkdir -p "$out"
arms=(); compares=(); seen_sep=0
for a in "$@"; do
  if [ "$a" = "--" ]; then seen_sep=1; continue; fi
  if [ "$seen_sep" = 0 ]; then
    key=${a%%=*}; prefix=${a#*=}
    dev=/runs/$prefix-dev/dev.predictions.jsonl
    [ "$prefix" = m2/lux0c ] && dev=/runs/m2/lux0c-dev/dev.predictions.jsonl
    css=/runs/$prefix-css-pilot/css-pilot.predictions.jsonl
    [ "$prefix" = m2/lux0c ] && css=/runs/m2/lux0c-css/css-pilot.predictions.jsonl
    arms+=(--arm "$key=$dev,$css")
  else
    compares+=(--compare "$a")
  fi
done
printf '{"wrapper_commit": "%s", "code_commit": "%s"}\n' "$sha" "$rt" > "$out/mirrors.json"
dry docker run --rm --name "d2-9b-m6-$out_name-score" --network none -e PYTHONPATH=/code \
  -e PYTHONDONTWRITEBYTECODE=1 \
  --mount type=bind,src="$S",dst=/code,readonly \
  --mount type=bind,src="$DATA/dev2/private/panels/gold",dst=/gold,readonly \
  --mount type=bind,src="$RUNS",dst=/runs,readonly \
  --mount type=bind,src="$out",dst=/out -w /code "$IMAGE" \
  python3 -m v2.dec.dev_readout --typed-gold /gold/typed-dev.gold.jsonl --css-gold /gold/css-pilot.gold.jsonl \
    "${arms[@]}" "${compares[@]}" --output /out/readout.json > "$out/console.log" 2>&1
cat "$out/console.log"
