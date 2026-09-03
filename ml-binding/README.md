# Traditional ML native binding

`ml-binding` loads model-selection artifacts and runs KNN, K-Means, or SVM
inference behind a Rust/CGo boundary. Training happens in
[`src/training/model_selection/ml_model_selection`](../src/training/model_selection/ml_model_selection/README.md);
this module only serves trained JSON artifacts.

| Selector | Runtime behavior |
| --- | --- |
| KNN | Chooses a model from nearby training samples. |
| K-Means | Maps the query to the nearest learned cluster. |
| SVM | Applies the serialized multi-class decision function. |

## Build and test

You need Go 1.24.1 or newer, Rust, Cargo, CGo, and a C compiler. Native Windows
is not supported; use Linux, macOS, or WSL.

```bash
cd ml-binding
cargo build --release
cargo test

export LD_LIBRARY_PATH="$PWD/target/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
go test ./...
```

On macOS, set `DYLD_LIBRARY_PATH` instead. From the repository root,
`make test-binding-minimal` runs the maintained cross-binding check.

## Use from Go

Load the artifact that matches the selector, call `Select`, and close the
selector when it is no longer needed:

```go
data, err := os.ReadFile("models/knn_model.json")
if err != nil {
    return err
}

selector, err := ml.KNNFromJSON(string(data))
if err != nil {
    return err
}
defer selector.Close()

model, err := selector.Select(features)
```

`KMeansFromJSON` and `SVMFromJSON` follow the same lifecycle. The feature vector
must have the same shape and ordering used during training; the binding does not
repair an incompatible artifact.

See [`ml_binding.go`](ml_binding.go) for the complete public Go API and
[`Cargo.toml`](Cargo.toml) for pinned Rust dependencies.
