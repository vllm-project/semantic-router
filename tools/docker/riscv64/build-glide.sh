#!/usr/bin/env bash
# Source fallback for the GLIDE FFI missing from the pinned Go module on RISC-V.
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
target=riscv64gc-unknown-linux-gnu
archive=
output=
while (($#)); do
  case "$1" in
    --archive|--output|--target)
      (($# >= 2)) || { echo "missing value for $1" >&2; exit 2; }
      case "$1" in
        --archive) archive=$2 ;;
        --output) output=$2 ;;
        --target) target=$2 ;;
      esac
      shift 2 ;;
    *) echo "usage: bash $0 --output DIRECTORY [--archive FILE] [--target RUST_TARGET]" >&2; exit 2 ;;
  esac
done
[[ -n "$output" ]] || { echo "--output is required" >&2; exit 2; }
[[ "$target" == riscv64gc-unknown-linux-gnu ]] || { echo "unsupported GLIDE target: $target" >&2; exit 2; }
version=$(awk '$1 == "github.com/valkey-io/valkey-glide/go/v2" {print $2}' "$root/src/semantic-router/go.mod")
[[ "$version" == v2.3.0 ]] || { echo "GLIDE source pin and Cargo.lock must match Go client $version" >&2; exit 1; }
if [[ -n "$archive" && ! -f "$archive" ]]; then
  echo "source archive not found: $archive" >&2
  exit 1
fi
scratch=$(mktemp -d "${TMPDIR:-/tmp}/vsr-glide.XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
if [[ -z "$archive" ]]; then
  archive="$scratch/source.tar.gz"
  curl --fail --location --retry 3 --connect-timeout 20 \
    "https://codeload.github.com/valkey-io/valkey-glide/tar.gz/refs/tags/$version" -o "$archive"
fi
checksum=$(sha256sum "$archive" | awk '{print $1}')
[[ "$checksum" == 861a5345b423206fa9090925767b3f9524e4bf135b851beaa6ccf41eb4153e53 ]] || {
  echo "GLIDE source checksum mismatch" >&2; exit 1;
}
mkdir -p "$scratch/source"
tar -xzf "$archive" -C "$scratch/source" --strip-components=1
cp "$root/tools/docker/riscv64/valkey-glide/Cargo.lock" "$scratch/source/ffi/Cargo.lock"
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$scratch/target}
mkdir -p "$CARGO_TARGET_DIR"
CARGO_TARGET_DIR=$(cd "$CARGO_TARGET_DIR" && pwd)
export CARGO_TARGET_DIR
if [[ "$(uname -m)" != riscv64 ]]; then
  export CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER=${CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER:-riscv64-linux-gnu-gcc}
  export CC_riscv64gc_unknown_linux_gnu=${CC_riscv64gc_unknown_linux_gnu:-riscv64-linux-gnu-gcc}
  export CXX_riscv64gc_unknown_linux_gnu=${CXX_riscv64gc_unknown_linux_gnu:-riscv64-linux-gnu-g++}
fi
(
  cd "$scratch/source/ffi"
  cargo build --locked --release --target "$target"
)
library="$CARGO_TARGET_DIR/$target/release/libglide_ffi.a"
machines=$(LC_ALL=C readelf -h "$library" | awk '/Machine:/ {sub(/^.*Machine:[[:space:]]*/, ""); print}' | sort -u)
[[ "$machines" == RISC-V ]] || { echo "GLIDE archive has unexpected ELF machines: $machines" >&2; exit 1; }
mkdir -p "$output"
install -m 0644 "$library" "$output/libglide_ffi.a"
sha256sum "$output/libglide_ffi.a"
