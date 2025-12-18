# candle-binding

This directory contains Go bindings and tests for the `candle_semantic_router` native library.

## Prerequisites

- Go Version 1.24.1 or higher (matches the module requirements)
- Rust Version 1.90.0 or higher (for Candle bindings, supports 2024 edition)
- `cargo` (Rust's build tool)

## Build the Native Library

Before running the Go tests, you must build the native library using Rust:

```sh
cd candle-binding
cargo build --release
```

This will produce the library file (e.g., `libcandle_semantic_router.dylib` on macOS) in `candle-binding/target/release/`.

## Run the Go Tests

After building the native library, run the Go tests:

```sh
cd candle-binding
# If needed, set the library path (macOS):
export DYLD_LIBRARY_PATH=$(pwd)/target/release:$DYLD_LIBRARY_PATH

go test -v
```

- The `-v` flag enables verbose output.
- If you want to run a specific test, use:

  ```sh
  go test -v -run TestName
  ```

  Replace `TestName` with the name of the test function.

## Troubleshooting

- If you see an error like `library 'candle_semantic_router' not found`, make sure you have built the native library and that the library file exists in `target/release/`.
- Ensure your `DYLD_LIBRARY_PATH` (macOS) or `LD_LIBRARY_PATH` (Linux) includes the path to the built library.

## Notes

- The Go tests depend on the native library being present and correctly built.
- Some tests may download data from the internet (e.g., from norvig.com).
