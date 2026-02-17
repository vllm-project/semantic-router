# nlp-binding

Go bindings for BM25 and N-gram keyword classification, backed by Rust
implementations via C FFI.

## Rust Crates Used

| Crate | Version | Purpose |
|-------|---------|---------|
| [bm25](https://crates.io/crates/bm25) | 2.3 | BM25 (Okapi) keyword scoring and search |
| [ngrammatic](https://crates.io/crates/ngrammatic) | 0.7 | N-gram fuzzy string matching |

## Prerequisites

- Go 1.24.1+
- Rust 1.90.0+ with cargo

## Build the Native Library

```bash
cd nlp-binding
cargo build --release
```

## Run the Go Tests

```bash
# Linux:
export LD_LIBRARY_PATH=$(pwd)/target/release:$LD_LIBRARY_PATH

# macOS:
export DYLD_LIBRARY_PATH=$(pwd)/target/release:$DYLD_LIBRARY_PATH

go test -v
```

## Usage

### BM25 Classifier

```go
import nlp "github.com/vllm-project/semantic-router/nlp-binding"

c := nlp.NewBM25Classifier()
defer c.Free()

c.AddRule("urgent", "OR", []string{"urgent", "immediate", "emergency"}, 0.1, false)

result := c.Classify("This is an urgent request")
// result.Matched == true
// result.RuleName == "urgent"
// result.MatchedKeywords == ["urgent"]
// result.Scores == [1.20...]
```

### N-gram Classifier

```go
c := nlp.NewNgramClassifier()
defer c.Free()

c.AddRule("urgent", "OR", []string{"urgent", "immediate", "emergency"}, 0.4, false, 3)

// Exact match
result := c.Classify("This is urgent")

// Fuzzy match (typos!)
result = c.Classify("This is urgnet")  // matches "urgent" with ~0.56 similarity
```

## Architecture

```
YAML keyword_rules (method: "bm25" | "ngram")
    ↓
config.KeywordRule
    ↓
classification.KeywordClassifier (dispatches by method)
    ↓
nlp-binding Go API (BM25Classifier / NgramClassifier)
    ↓
C FFI (#cgo LDFLAGS → libnlp_binding.so)
    ↓
Rust crate (bm25 + ngrammatic)
```
