#!/usr/bin/env bash
# MLX-DEV2 development-guard readout of one package (node side, private outputs).
#
#   mlx_dev2_run.sh --gpu G --package DIR --work DIR --panel DIR [--diag DIR] [--cache DIR]
#                   [--frozen-27b DIR --expect DIGEST] [--name NAME]
#
# Answers every MLX-DEV2 prompt (and, with --diag, the formal mlx-diag prompts) in the frozen
# image on one leased GPU through v2/eval/ix1/probe.sh (render node of GPU G only, --network
# none), then writes <panel>.predictions.jsonl for `mlx_dev2 compare`. --panel / --diag are
# panel directories with prompts.jsonl; the work directory must be under /data/dev2/private/.
# --cache copies a Triton cache; --frozen-27b copies a frozen 27B cache through triton_cache.
set -euo pipefail

S="$(cd "$(dirname "$0")/../.." && pwd)"
gpu="" pkg="" work="" panel="" diag="" cache="" frozen="" expect="" name="mlx-dev2"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu="$2"; shift 2 ;;
    --package) pkg="$2"; shift 2 ;;
    --work) work="$2"; shift 2 ;;
    --panel) panel="$2"; shift 2 ;;
    --diag) diag="$2"; shift 2 ;;
    --cache) cache="$2"; shift 2 ;;
    --frozen-27b) frozen="$2"; shift 2 ;;
    --expect) expect="$2"; shift 2 ;;
    --name) name="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -n "$gpu" && -d "$pkg" && -n "$work" && -f "$panel/prompts.jsonl" ]] || { sed -n '2,12p' "$0" >&2; exit 2; }
[[ -z "$diag" || -f "$diag/prompts.jsonl" ]] || { echo "no $diag/prompts.jsonl" >&2; exit 2; }
[[ ! -e "$work/answers.jsonl" ]] || { echo "$work/answers.jsonl exists" >&2; exit 2; }

umask 077
mkdir -p "$work"
: > "$work/empty.jsonl"
specs="--panel mlx-dev2:$panel/prompts.jsonl:$work/empty.jsonl:$(wc -l < "$panel/prompts.jsonl")"
mounts=(--mount "$S" --mount "$panel")
if [[ -n "$diag" ]]; then
  specs+=" --panel mlx-diag:$diag/prompts.jsonl:$work/empty.jsonl:$(wc -l < "$diag/prompts.jsonl")"
  mounts+=(--mount "$diag")
fi
extra=()
if [[ -n "$frozen" ]]; then
  (cd "$S" && python3 -m v2.27b.triton_cache copy --frozen "$frozen" --expect "$expect" --dest "$work/triton") > "$work/cache-copy.log" 2>&1
elif [[ -n "$cache" ]]; then
  extra=(--cache "$cache")
fi

bash "$S/v2/eval/ix1/probe.sh" --gpu "$gpu" --name "$name" --package "$pkg" --work "$work" "${mounts[@]}" "${extra[@]}" -- \
  "cd $S; date +%s > $work/start_epoch; PYTHONPATH=$S:/opt/decision-fla python3 v2/release/examples.py parity --package \$PKG --output $work/parity.json --device cuda:0 --threads 4 \${BASE:+--base-path \$BASE} --site /opt/decision-fla --require-kernels $specs --answers $work/answers.jsonl > $work/parity.log 2>&1; echo \$? > $work/exit_code; date +%s > $work/end_epoch"

[[ "$(cat "$work/exit_code")" == 0 ]] || { echo "readout failed; see $work/parity.log" >&2; exit 1; }
cd "$S"
python3 -m v2.eval.mlx_dev2 predictions --panel "$panel" --answers "$work/answers.jsonl" --output "$work/mlx-dev2.predictions.jsonl"
if [[ -n "$diag" ]]; then
  python3 -m v2.eval.mlx_dev2 predictions --panel "$diag" --answers "$work/answers.jsonl" --output "$work/mlx-diag.predictions.jsonl"
fi
