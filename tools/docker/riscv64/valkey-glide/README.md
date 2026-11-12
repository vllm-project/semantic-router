# GLIDE source dependency

The Router uses the GLIDE Go module's prebuilt FFI on amd64 and arm64. The
pinned v2.3.0 module has no RISC-V archive, so `../build-glide.sh` supplies a
source-built `libglide_ffi.a` for `riscv64gc-unknown-linux-gnu`.

The script checks the Go module version and release archive SHA-256, builds
with `cargo --locked`, and checks every archive object's ELF machine before
installing the library. The Docker build uses this entrypoint too.

The v2.3.0 source release does not include `ffi/Cargo.lock`. This directory
therefore carries that crate's resolved lock, generated with Rust/Cargo 1.90.
When upgrading the Go client, update the source pin, checksum and lock together
and rebuild the FFI. Source verification is not disabled for a local archive.
GLIDE's source is licensed under Apache-2.0.

## Build outside Docker

Install Rust 1.90 with the RISC-V GNU target standard library, GCC/G++, CMake,
Make, pkg-config, protobuf compiler, curl, tar, OpenSSL headers and GNU binutils.
Cross builds additionally need a RISC-V GNU C/C++ compiler; native builds use
the host compiler. Cargo's target linker and CC/CXX environment variables can
override the cross-compiler names.

```bash
bash tools/docker/riscv64/build-glide.sh --output "$PWD/bin/glide"
GLIDE_LIBRARY_DIR="$PWD/bin/glide"
(cd src/semantic-router && CGO_ENABLED=1 go build \
  -ldflags="-extldflags=-L$GLIDE_LIBRARY_DIR" -o ../../bin/router ./cmd)
```

Run both commands on RISC-V for a native Router build. For cross compilation,
also set `GOOS=linux GOARCH=riscv64 CC=riscv64-linux-gnu-gcc CGO_ENABLED=1`
on the Go command. `--archive /path/to/source.tar.gz` supplies an offline source
archive to the first command; `--output` selects the library directory.
Set `CARGO_TARGET_DIR` to retain compilation caches.
