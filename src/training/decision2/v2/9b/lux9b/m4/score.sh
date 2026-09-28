#!/usr/bin/env bash
# usage: score.sh SHA OUT_NAME ARM=RUN_PREFIX... -- COMPARE...
# CPU-only development readout (typed DEV + CSS pilot) with v2.dec.dev_readout into
# /data/dev2/runs/9b/m4/OUT_NAME. ARM=RUN_PREFIX reads RUN_PREFIX-dev/dev.predictions.jsonl and
# RUN_PREFIX-css-pilot/css-pilot.predictions.jsonl under /data/dev2/runs/9b (e.g.
# lux=m4/lux-16k, k1=m4/K-s1, d=m3/D-soup); COMPARE is "a:b" (b minus a).
set -euo pipefail
sha=$1; out_name=$2; shift 2
S=/data/dev2/src/$sha-src_training_decision2/src/training/decision2
image=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
out=/data/dev2/runs/9b/m4/$out_name
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
docker run --rm --network none -e PYTHONPATH=/code -e PYTHONDONTWRITEBYTECODE=1 \
  --mount type=bind,src="$S",dst=/code,readonly \
  --mount type=bind,src=/data/dev2/private/panels/gold,dst=/gold,readonly \
  --mount type=bind,src=/data/dev2/runs/9b,dst=/runs,readonly \
  --mount type=bind,src="$out",dst=/out -w /code "$image" \
  python3 -m v2.dec.dev_readout --typed-gold /gold/typed-dev.gold.jsonl --css-gold /gold/css-pilot.gold.jsonl \
    "${arms[@]}" "${compares[@]}" --output /out/readout.json > "$out/console.log" 2>&1
cat "$out/console.log"
