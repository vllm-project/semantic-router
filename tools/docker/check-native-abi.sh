#!/usr/bin/env bash
# Inspect the actual shared libraries before linking a multi-provider router.
# Candle-only RISC-V builds pass a single argument and skip the ORT ABI check.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 CANDLE_LIBRARY [ORT_LIBRARY]" >&2
  exit 2
fi

abi_tmp=$(mktemp -d)
trap 'rm -rf "$abi_tmp"' EXIT

exports() {
  if [[ $(uname -s) == Darwin ]]; then
    nm -gU "$1" | awk '$(NF-1) == "T" {sub(/^_/, "", $NF); print $NF}'
  else
    nm -D --defined-only "$1" | awk '$(NF-1) == "T" {print $NF}'
  fi | LC_ALL=C sort -u
}

exports "$1" > "$abi_tmp/candle"
for task in sequence token embedding; do
  grep -qx "candle_instance_load_$task" "$abi_tmp/candle"
done
grep -qx "candle_instance_load_backbone" "$abi_tmp/candle"
for operation in clone close text_windows; do
  grep -qx "candle_instance_$operation" "$abi_tmp/candle"
done

if [[ $# == 1 ]]; then
  echo "Candle instance exports present (ORT skipped)"
  exit 0
fi

exports "$2" > "$abi_tmp/ort"
for task in sequence token embedding; do
  grep -qx "ort_instance_load_$task" "$abi_tmp/ort"
done
for operation in clone close text_windows; do
  grep -qx "ort_instance_$operation" "$abi_tmp/ort"
done

# Rust/C++ mangled implementation symbols are outside the public C ABI.
LC_ALL=C comm -12 "$abi_tmp/candle" "$abi_tmp/ort" | \
  grep -E '^[a-z][a-z0-9_]*$' > "$abi_tmp/collisions" || true
if [[ -s "$abi_tmp/collisions" ]]; then
  echo "Candle and ORT export conflicting C functions:" >&2
  cat "$abi_tmp/collisions" >&2
  exit 1
fi
echo "Candle and ORT instance exports coexist without C ABI collisions"
