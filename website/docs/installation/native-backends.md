---
sidebar_position: 5
description: Choose a native inference backend and understand which Router features it supports.
---

# Native Backends

Semantic Router uses native bindings for learned classifiers, embeddings,
multimodal routing, and MLP model selection. The backend selected at build time
determines which of those features are available at runtime.

Use the capability table below when choosing an image or custom build. A
successful build does not imply that every native feature is available.

## Backend selection

| Build shape | Backend name | How it is selected |
|-------------|--------------|--------------------|
| Default CGo build | `candle` | Build without `onnx` and with CGo enabled. |
| ONNX build | `onnx` | Build with the `onnx` tag and CGo enabled. |
| Non-CGo or Windows build | `stub` | Build with `CGO_ENABLED=0` or on Windows. |

Examples:

```bash
# Default Candle-backed Router build.
make build-router

# ONNX-backed Router build (builds the binding and uses go.onnx.mod).
make build-router-onnx

# Non-CGo stub build for environments where native bindings are unavailable.
cd src/semantic-router
CGO_ENABLED=0 go build ./cmd
```

## Runtime capabilities

| Capability | `candle` | `onnx` | `stub` |
|------------|----------|--------|--------|
| Unified batch classification | Yes | No | No |
| LoRA batch classification | Yes | No | No |
| Batched embedding | Yes | Yes | No |
| Multimodal embedding | Yes | No | No |
| Modality routing | Yes | No | No |
| MLP selector | Yes | No | No |
| Local hallucination detection | Yes | No | No |
| Local hallucination NLI | Yes | No | No |
| Explicit reset | No | No | No |

The Router rejects a classifier configuration when the selected backend does
not advertise the required capability. In particular, an ONNX or non-CGo build
does not gain Candle-compatible classifier behavior merely because the Go
package compiles.

## Lifecycle expectations

Native model state is process-owned. Plan a process restart when changing the
backend or replacing model families that require a clean native state:

- prefer process restart for backend swaps or model-family changes that need a
  clean native state
- consult the runtime capability output before enabling backend-specific
  features in a control plane
- keep ONNX deployments to features that advertise support, primarily batched
  embedding
