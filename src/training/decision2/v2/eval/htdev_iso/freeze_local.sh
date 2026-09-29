#!/usr/bin/env bash
# Copy node A's local training pools and derived files into a frozen snapshot.
#
# Usage: freeze_local.sh <iso-dir>
#
# Copies *.jsonl, *.jsonl.gz, *.json and *.parquet files older than 2 minutes (the
# indep-e2 rule) from each root below into <iso-dir>/training/local/<label>/, skipping
# triton caches. Code-tree mirrors inside the pools (m3b/tmp-code, m3b/gap/code) go to
# <iso-dir>/training/local-code/ instead. Writes <iso-dir>/training/local-roots.tsv.
set -euo pipefail
umask 077
iso="$1"
dst="$iso/training/local"; code="$iso/training/local-code"
mkdir -p "$dst" "$code"
roots=(
  m3a2=/data/dev2/runs/data/m3a2
  m3b=/data/dev2/runs/data/m3b
  m3a=/data/dev2/runs/data/m3a
  teachers-v2=/data/dev2/private/data/teachers-v2
  arms-v1=/data/dev2/private/data/arms-v1
  arms-v2=/data/dev2/private/data/arms-v2
  from-nodeB=/data/dev2/private/data/from-nodeB
  hf-uploads=/data/dev2/private/data
  a7-private=/data/dev2/private/a7/runs
  a7-runs=/data/dev2/runs/a7
  06b-m4=/data/dev2/runs/06b/m4/data
  06b-m5=/data/dev2/runs/06b/m5/data
  06b-m6=/data/dev2/runs/06b/m6/data
  9b-m2=/data/dev2/runs/9b/m2/data
  9b-m3=/data/dev2/runs/9b/m3/data
  dec-m2=/data/dev2/runs/dec/m2/data
)
: > "$iso/training/local-roots.tsv"
for item in "${roots[@]}"; do
  label=${item%%=*}; root=${item#*=}
  [[ -d "$root" ]] || { printf '%s\t%s\tmissing\n' "$label" "$root" >> "$iso/training/local-roots.tsv"; continue; }
  if [[ "$label" == hf-uploads ]]; then
    filter=(-path "$root/hf-upload-*")
  else
    filter=(-true)
  fi
  n=0
  while IFS= read -r -d '' f; do
    rel=${f#"$root"/}
    case "$label/$rel" in
      m3b/tmp-code/*|m3b/gap/code/*) target="$code/$label/$rel" ;;
      *) target="$dst/$label/$rel" ;;
    esac
    mkdir -p "$(dirname "$target")"; cp -p "$f" "$target"; n=$((n + 1))
  done < <(find "$root" -path '*triton-cache*' -prune -o -type f "${filter[@]}" \
      \( -name '*.jsonl' -o -name '*.jsonl.gz' -o -name '*.json' -o -name '*.parquet' \) -mmin +2 -print0)
  printf '%s\t%s\t%s\n' "$label" "$root" "$n" >> "$iso/training/local-roots.tsv"
done
cat "$iso/training/local-roots.tsv"
