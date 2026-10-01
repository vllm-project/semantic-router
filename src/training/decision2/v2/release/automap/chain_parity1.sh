#!/usr/bin/env bash
# Run parity jobs one after another on one target (cpu or gpuN); each job is
# "<Repo-Name> <head-sha> <predictions-dir-or-"-"> <tag> [extra run_parity1.sh args]".
# Panels are the scored v3 (typed-final, css15) and public-231 prompts; with "-"
# instead of a predictions directory, <tag>'s reference predictions are used.
#
# Usage: chain_parity1.sh <cpu|gpuN> <work-root> <jobs-file>
set -uo pipefail
target="$1"; root="$2"; jobs="$3"
here="$(cd "$(dirname "$0")" && pwd)"
private=/data/dev2/private/dev1-automap
goldfree="$private/private/panels/goldfree"
while read -r repo head predictions tag extra; do
  [[ -z "$repo" || "$repo" == \#* ]] && continue
  [[ "$predictions" == "-" ]] && predictions="$root/reference-$tag"
  panels=()
  for panel in typed-final css15 public231; do
    panels+=(--panel "$panel:$goldfree/$panel.prompts.jsonl:$predictions/$panel.predictions.jsonl")
  done
  echo "$(date -u +%FT%TZ) start $tag on $target"
  # shellcheck disable=SC2086
  bash "$here/run_parity1.sh" "$repo" "$head" "$target" "$root/$tag" "${panels[@]}" $extra
  echo "$(date -u +%FT%TZ) end $tag rc=$?"
done < "$jobs"
