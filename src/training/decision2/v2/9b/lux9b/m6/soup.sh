#!/usr/bin/env bash
# usage: soup.sh SHA GPU NAME [--runtime RSHA] MEMBER...
# CPU: uniform FP32 soup (v2.dec.soup, container d2-9b-m6-NAME-soup) into
# /data/dev2/runs/9b/m6/NAME-build/soup. A MEMBER is an M6 run name (its SELECT-chosen BEST
# full checkpoint) or an explicit full checkpoint path under /m6/, /m4/ or /m3/ (container views
# of /data/dev2/runs/9b/m{5,4,3}; e.g. the incumbent /m4/K-a13-build/soup). Members may repeat:
# k copies of a member out of N give it weight k/N. members.txt lists the resolved members in
# order; weights.json the effective weight of each distinct member. Then readout.sh (CAL698 and
# 16K typed DEV + CSS pilot). The soup and the readout run the runtime mirror RSHA (default the
# incumbent's 3277dec9d).
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
sha=$1; gpu=$2; name=$3; shift 3
rt=$RUNTIME_DEFAULT
if [ "${1:-}" = "--runtime" ]; then rt=${2:?--runtime needs a SHA}; shift 2; fi
L=$(code_dir "$sha")/v2/9b/lux9b/m6
RS=$(code_dir "$rt")
[ -d "$L" ] || { echo "mirror $L missing" >&2; exit 2; }
[ -d "$RS" ] || { echo "runtime mirror $RS missing" >&2; exit 2; }
[ $# -ge 2 ] || { echo "a soup needs at least two members" >&2; exit 2; }
build=$M6/$name-build
[ ! -e "$build" ] || { echo "$build exists" >&2; exit 66; }
members=(); paths=()
for run in "$@"; do
  case "$run" in
    /m6/*|/m4/*|/m3/*) host=$(host_path "$run"); path=$run ;;
    *)
      [ -f "$M6/$run/run/BEST.json" ] || { echo "no M6 run $run" >&2; exit 2; }
      best=$(best_of "$M6/$run/run/BEST.json")
      host=$M6/$run/run/$best; path=/m6/$run/run/$best ;;
  esac
  [ -f "$host/decision_config.json" ] || { echo "no checkpoint at $run" >&2; exit 2; }
  members+=(--member "$path"); paths+=("$path")
done
mkdir -p "$build"
printf '%s\n' "${paths[@]}" > "$build/members.txt"
printf '{"wrapper_commit": "%s", "code_commit": "%s"}\n' "$sha" "$rt" > "$build/mirrors.json"
python3 - "$build/members.txt" "$build/weights.json" <<'EOF'
import json, sys
from collections import Counter
from fractions import Fraction
paths = [line.strip() for line in open(sys.argv[1]) if line.strip()]
counts = Counter(paths)
weights = [
    {"member": p, "copies": k, "weight": str(Fraction(k, len(paths))), "weight_float": k / len(paths)}
    for p, k in sorted(counts.items(), key=lambda item: paths.index(item[0]))
]
json.dump({"members_total": len(paths), "distinct": weights}, open(sys.argv[2], "w"), indent=1)
EOF
date -u +%FT%TZ > "$build/start-utc.txt"
dry docker run --rm --name "d2-9b-m6-$name-soup" --network none --cpus 32 -e PYTHONPATH=/code \
  -e PYTHONDONTWRITEBYTECODE=1 \
  --mount type=bind,src="$RS",dst=/code,readonly --mount type=bind,src="$M6",dst=/m6,readonly \
  --mount type=bind,src="$M4",dst=/m4,readonly --mount type=bind,src="$M3",dst=/m3,readonly \
  --mount type=bind,src="$build",dst=/out -w /code "$IMAGE" \
  python3 -m v2.dec.soup "${members[@]}" --output /out/soup > "$build/console.log" 2>&1 || exit 1
if [ "${DRY_RUN:-0}" = 1 ]; then mkdir -p "$build/soup" && touch "$build/soup/decision_config.json"; fi
date -u +%FT%TZ > "$build/end-utc.txt"
"$L/readout.sh" "$sha" "$gpu" "$name" "/m6/$name-build/soup" --runtime "$rt" || exit 1
echo "done $name"
