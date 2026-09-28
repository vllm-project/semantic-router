#!/usr/bin/env bash
# usage: soup.sh SHA GPU NAME MEMBER...
# CPU: uniform FP32 soup (v2.dec.soup) into /data/dev2/runs/9b/m4/NAME-build/soup. A MEMBER is
# an M4 run name (its SELECT-chosen BEST full checkpoint) or an explicit full checkpoint path
# under /m4/ or /m3/ (container views of /data/dev2/runs/9b/m4 and /data/dev2/runs/9b/m3).
# Members may repeat: k copies of a member out of N give it weight k/N (rational
# interpolation). members.txt lists the resolved members in order; weights.json the effective
# weight of each distinct member. Then readout.sh: CAL698 temperatures (v2.dec.calibrate_ckpt)
# and gold-free typed DEV + CSS pilot predictions at 16,384 tokens on GPU.
set -uo pipefail
sha=$1; gpu=$2; name=$3; shift 3
S=/data/dev2/src/$sha-src_training_decision2/src/training/decision2
M3=/data/dev2/runs/9b/m3
M4=/data/dev2/runs/9b/m4
image=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
[ -d "$S" ] || { echo "mirror $S missing" >&2; exit 2; }
[ $# -ge 2 ] || { echo "a soup needs at least two members" >&2; exit 2; }
build=$M4/$name-build
[ ! -e "$build" ] || { echo "$build exists" >&2; exit 66; }
members=(); paths=()
for run in "$@"; do
  case "$run" in
    /m4/*) host=$M4/${run#/m4/}; path=$run ;;
    /m3/*) host=$M3/${run#/m3/}; path=$run ;;
    *)
      [ -f "$M4/$run/run/BEST.json" ] || { echo "no M4 run $run" >&2; exit 2; }
      best=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$M4/$run/run/BEST.json")
      host=$M4/$run/run/$best; path=/m4/$run/run/$best ;;
  esac
  [ -f "$host/decision_config.json" ] || { echo "no checkpoint at $run" >&2; exit 2; }
  members+=(--member "$path"); paths+=("$path")
done
mkdir -p "$build"
printf '%s\n' "${paths[@]}" > "$build/members.txt"
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
docker run --rm --network none --cpus 32 -e PYTHONPATH=/code -e PYTHONDONTWRITEBYTECODE=1 \
  --mount type=bind,src="$S",dst=/code,readonly --mount type=bind,src="$M4",dst=/m4,readonly \
  --mount type=bind,src="$M3",dst=/m3,readonly \
  --mount type=bind,src="$build",dst=/out -w /code "$image" \
  python3 -m v2.dec.soup "${members[@]}" --output /out/soup > "$build/console.log" 2>&1 || exit 1
date -u +%FT%TZ > "$build/end-utc.txt"
"$S/v2/9b/lux9b/m4/readout.sh" "$sha" "$gpu" "$name" "/m4/$name-build/soup" || exit 1
echo "done $name"
