# Keyword-classifier native binding

`nlp-binding` exposes BM25 and N-gram keyword classifiers to Go through a Rust
CGo library. The router uses them for keyword signals that need ranked or fuzzy
matching rather than literal regular expressions.

| Method | Use it for |
| --- | --- |
| BM25 | Ranking text against weighted keyword rules. |
| N-gram | Tolerating small spelling differences in short terms. |

## Build and test

The module requires Go 1.24.1 or newer, Rust, Cargo, CGo, and a C compiler.

```bash
cd nlp-binding
cargo build --release
cargo test

export LD_LIBRARY_PATH="$PWD/target/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
go test ./...
```

On macOS, set `DYLD_LIBRARY_PATH` instead. From the repository root,
`make test-binding-minimal` runs the maintained cross-binding check.

## Use from Go

```go
classifier := nlp.NewBM25Classifier()
defer classifier.Free()

classifier.AddRule(
    "urgent",
    "OR",
    []string{"urgent", "immediate", "emergency"},
    0.1,
    false,
)
result := classifier.Classify("This is an urgent request")
```

Use `NewNgramClassifier` and provide the N-gram size when fuzzy matching is
required. Rule methods, thresholds, and case handling are part of the router
configuration contract; see the
[keyword signal guide](../website/docs/tutorials/signal/heuristic/keyword.md) for
configuration examples.

[`nlp_binding.go`](nlp_binding.go) is the source of truth for the public Go API.
