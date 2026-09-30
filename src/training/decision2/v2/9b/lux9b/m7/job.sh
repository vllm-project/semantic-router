#!/usr/bin/env bash
# usage: job.sh SHA GPU NAME PURPOSE EXPECTED_MIN [--code CODE_SHA] -- python3-args...
# One Milestone 7 GPU job in the pinned trainer image (FLA overlay) through m7/run_gpu.sh (lease
# track 9b-m7, container d2-9b-m7-NAME-gGPU). The wrappers come from the mirror of SHA; the
# code mounted at /code is the exact subtree mirror of CODE_SHA (default SHA; readouts pass the
# runtime mirror). Run dir /data/dev2/runs/9b/m7/NAME (container /out, the only rw output
# mount). Read-only mounts: /model Lux 1.0, /d10 rights-clean data, /hfc the HF cache, /m7
# earlier Milestone 7 runs and data (/m7/data/DATA/build), /m6 Milestone 6 runs (K-s4, K-s5,
# K5-a12), /m4 Milestone 4 runs (K-s1..s3, incumbent K-a13), /m3 Milestone 3 runs (the Lux full
# checkpoint), /panels gold-free development prompts, /goldfree the gold-free formal-panel
# prompts (hs1-dev diagnostic).
# The M7 training autotune cache (/triton) is one copy of the M4 one, made once under flock
# (receipt m7/triton-cache.copy.json); its tree hash goes to m7/triton-cache.jsonl before and
# after every job. On GPU7 the job waits while the eval track's C1 lease entry is active.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
sha=$1; gpu=$2; name=$3; purpose=$4; expected=$5; shift 5
code=$sha
if [ "${1:-}" = "--code" ]; then code=${2:?--code needs a SHA}; shift 2; fi
[ "${1:-}" = "--" ] && shift
CM=$(mirror_dir "$code")
S=$(code_dir "$code")
L=$(code_dir "$sha")/v2/9b/lux9b/m7
TC=$M7/triton-cache
[ -d "$S" ] || { echo "mirror $S missing" >&2; exit 2; }
[ -d "$L" ] || { echo "wrapper mirror $L missing" >&2; exit 2; }
[ ! -e "$M7/$name/start-utc.txt" ] || { echo "$M7/$name was already used; pick a new run directory" >&2; exit 66; }
mkdir -p "$M7"
(
  flock 8
  if [ ! -d "$TC" ]; then
    [ -d "$M4/triton-cache" ] || { echo "M4 cache $M4/triton-cache missing" >&2; exit 2; }
    rm -rf "$TC.pending"
    cp -a "$M4/triton-cache" "$TC.pending"
    printf '{"source": "%s", "copied_utc": "%s", "files": %s, "source_tree_sha256": "%s", "tree_sha256": "%s"}\n' \
      "$M4/triton-cache" "$(date -u +%FT%TZ)" "$(find "$TC.pending" -type f | wc -l)" \
      "$(tree_sha "$M4/triton-cache")" "$(tree_sha "$TC.pending")" > "$M7/triton-cache.copy.json"
    mv "$TC.pending" "$TC"
  fi
) 8>"$M7/.triton-init.lock" || exit 2
cache_note() {
  printf '{"job": "%s", "phase": "%s", "utc": "%s", "files": %s, "tree_sha256": "%s"}\n' "$name" "$1" \
    "$(date -u +%FT%TZ)" "$(find "$TC" -type f | wc -l)" "$(tree_sha "$TC")" >> "$M7/triton-cache.jsonl"
}
tree=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tree"])' "$CM/.dev2-mirror.json")
yield_gpu7 "$gpu"
cache_note before
set +e
"$L/run_gpu.sh" "$gpu" "$M7/$name" "$purpose" "$expected" -- \
  -e PYTHONPATH=/code:/opt/decision-fla -e TRITON_CACHE_AUTOTUNING=1 -e TRITON_CACHE_DIR=/triton \
  -e DEC_SOURCE_COMMIT="$code" -e DEC_SOURCE_TREE="$tree" -e DEC_IMAGE_ID="$IMAGE" \
  --mount type=bind,src="$TC",dst=/triton \
  --mount type=bind,src="$S",dst=/code,readonly \
  --mount type=bind,src="$DATA/decision20-20260926/models/Decision-1.0-Lux-9B",dst=/model,readonly \
  --mount type=bind,src="$DATA/decision20-20260926/data",dst=/d10,readonly \
  --mount type=bind,src="$DATA/dev2/hf-cache",dst=/hfc,readonly \
  --mount type=bind,src="$M7",dst=/m7,readonly \
  --mount type=bind,src="$M6",dst=/m6,readonly \
  --mount type=bind,src="$M4",dst=/m4,readonly \
  --mount type=bind,src="$M3",dst=/m3,readonly \
  --mount type=bind,src="$DATA/dev2/runs/dec/panels",dst=/panels,readonly \
  --mount type=bind,src="$DATA/dev2/private/panels/goldfree",dst=/goldfree,readonly \
  --mount type=bind,src="$M7/$name",dst=/out \
  -w /code "$IMAGE" python3 "$@"
status=$?
set -e
[ ! -d "$M7/$name" ] || printf '{"wrapper_commit": "%s", "code_commit": "%s", "code_tree": "%s"}\n' \
  "$sha" "$code" "$tree" > "$M7/$name/mirrors.json"
cache_note after
exit $status
