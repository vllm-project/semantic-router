---
sidebar_position: 5
description: Native backend capability and lifecycle contract for Candle, ONNX, and non-CGO builds.
---

# Native Backends

Semantic Router uses native Rust and CGo bindings for learned classifiers,
embedding-backed signals, multimodal embedding, and MLP model selection. The
router exposes the selected backend as a runtime capability contract through
`CurrentNativeBackendCapabilities`; callers should use that contract instead of
inferring support from build tags or package names.

## Backend selection

| Build shape | Backend name | How it is selected |
|-------------|--------------|--------------------|
| Default CGo build | `candle` | Build without `onnx` and with CGo enabled. |
| ONNX build | `onnx` | Build with the `onnx` tag and CGo enabled. |
| Non-CGo or Windows build | `stub` | Build with `CGO_ENABLED=0` or on Windows. |

Examples:

```bash
# Default Candle-backed build from the Go router module.
cd src/semantic-router
go build ./cmd

# ONNX-backed build.
cd src/semantic-router
go build -tags=onnx ./cmd

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
| Explicit reset | No | No | No |

Unsupported unified or LoRA batch classification fails early through the router
classification layer when the selected backend does not advertise that
capability. Do not assume an ONNX or non-CGo build has Candle-compatible
classifier behavior just because the Go package compiles.

## Lifecycle expectations

No current backend advertises `explicit_reset`. Treat native model state as
process-owned for deployment and hot-reload planning:

- prefer process restart for backend swaps or model-family changes that need a
  clean native state
- use the runtime capability output when deciding whether to enable backend
  specific features in control planes
- keep ONNX deployments to features that advertise support, primarily batched
  embedding until ONNX classifier parity is implemented

Backend lifecycle reset support and deeper ONNX classifier parity are tracked as
open architecture debt in the native binding workstream.
