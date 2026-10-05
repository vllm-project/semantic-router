# The ROCm router image: PyTorch, causal-conv1d and a GPU smoke

The ROCm router image (`extproc-rocm`, `vllm-sr-rocm`) serves every GPU model
from the official PyTorch wheel for ROCm 7.2 and a causal-conv1d wheel it
builds itself. Its causal-conv1d computes what the released decoders' build
computes, instruction for instruction. One difference to the packages'
release image remains, in attention, and the exactness records name it per
model.

- **Date:** 2026-10-05.
- **Device:** AMD Instinct MI325X (gfx942).
- **Image:** `Dockerfile.extproc`, `ACCELERATOR=rocm`, target `extproc`, built
  from an exact checkout of `af71d5e82` (24.8 GB). The first build of the
  same packages, at `be7366c49`, lacked the MIOpen paths below.
- **Reference:** the packages' release image, where every ROCm golden answer
  and parity reference of Phase 1 was recorded: PyTorch 2.12.0 built from
  source on ROCm 7.2.3, Triton 3.7.1, FLA 0.5.2, causal-conv1d 1.7.0.

## The stack

| Package | Image | Release image |
| --- | --- | --- |
| Python | 3.12.15 | 3.12.13 |
| PyTorch | 2.12.0+rocm7.2, official wheel (HIP 7.2.53211, AOTriton 0.11.2) | 2.12.0, source build on ROCm 7.2.3 (HIP 7.2.53211, AOTriton 0.13.50) |
| Triton | triton-rocm 3.7.0 | 3.7.1 |
| FLA | fla-core 0.5.2 | 0.5.2 |
| causal-conv1d | 1.7.0, built in the image (below) | 1.7.0, local build |

- **Why the rocm7.2 wheel:** the rocm7.1 build of the same release crashes
  replaying the encoders' HIP graphs (`CUDAGraph.replay`, and
  `HSA_STATUS_ERROR_INVALID_PACKET_FORMAT` for the decoders at load). The
  rocm7.2 wheel served the router's five Vela 1.0 encoders in one process with
  16 callers and no device failure (`router-latency-rocm.md`).
- **Why not the release image's PyTorch:** it is a private source build that
  links the full ROCm 7.2.3 runtime; a public image can't reproduce it.
- **Triton 3.7.0 against 3.7.1** changes no answer: a Triton 3.7.1 venv gives the
  3.7.0 answers to the last digit for Decision 1.0, Vela 1.0 and Vela 2.0, and
  the decoders run their pinned FLA kernel choices on either.

## causal-conv1d

PyPI publishes no ROCm wheel. The `causal-conv1d-rocm` build stage compiles
the 1.7.0 sdist (sha256 `3202758494eaa7b5…`) against the image's own PyTorch,
with ROCm 7.2.3's compiler (AMD clang 22.0.0git, roc-7.2.3 26084) and the
release build's tools (setuptools 79.0.1, wheel 0.48.0, ninja 1.13.2), for
`CAUSAL_CONV1D_ARCHS` (gfx942 by default; 145 s on 16 vCPUs). The image
installs only the wheel (sha256 `58d292e607483d44…`); its extension is
`/usr/local/lib/python3.12/site-packages/causal_conv1d_cuda.cpython-312-x86_64-linux-gnu.so`.

**Check against the release image's wheel**, with the release image's ROCm
tools: both carry three gfx942 code objects (one per `.cu` file) of the same
sizes (18,586,992, 1,038,472 and 273,176 bytes). In each:

- `.text` (the machine code), `.rodata` (the kernel descriptors) and `.note`
  (the kernel metadata) are byte-identical, and so is the full disassembly with
  instruction encodings (2.7 million lines for the largest);
- only the symbol tables differ: one name, `__hip_cuid_<hash>`, which clang
  derives from the source path and command line (14–15 bytes of `.dynstr`,
  then the `.dynsym` and hash-table order).

That is the test the image passes: same machine code, descriptors and metadata
in every gfx942 code object. An independent build in
`rocm/dev-ubuntu-22.04:7.2.3-complete` passes it too, and its kernels give
byte-identical outputs to the release image's on 120 / 120 fixed inputs
(`causal_conv1d_fn` and `causal_conv1d_update`, FP32, FP16 and BF16, four
Qwen3.5 widths).

## Smoke as the charts run it

One runtime process in the image on one MI325X, as uid 65532 with a read-only
root filesystem, the model volume at `/app/models`, an offline model cache and
no network. It serves Vela 2.0 4B and 0.3B and the router's five Vela 1.0
encoders (Domain, PII, Guard, FactCheck, Feedback) on `rocm:0`, `exact`.

| Image | Ready | Golden | Requests (16 threads, every model) |
| --- | --- | --- | --- |
| `be7366c49` | 6 of 7; Vela 2.0 4B failed 5 / 5 loads (`miopenStatusUnknownError`) | `matched`, 6 / 6 | 240 / 240 |
| `af71d5e82` | 7 of 7, in 40 s | `matched`, 7 / 7 | 280 / 280 |

- **The `be7366c49` failure:** Vela 2.0's forest forward runs `F.conv1d`,
  which goes through MIOpen. MIOpen keeps its databases and lock files under
  `~/.config/miopen`, and the chart's uid has no home directory on a read-only
  root. `af71d5e82` sets `MIOPEN_USER_DB_PATH` and `MIOPEN_CUSTOM_CACHE_DIR` to
  `/app/models/miopen`, beside `TRITON_CACHE_DIR`.
- No request failed, no device fault was logged and nothing was retried.
  Golden `matched` is readiness (the 0.02 GPU tolerance); byte identity is the
  exactness check below.

## What still differs from the release image

Attention. A probe of 63 ops on fixed inputs (BF16 autocast and FP32 linears
at the decoders' widths, `bmm`, depthwise `F.conv1d`, norms, softmax and the
activations, reductions, `cumsum` and FLA's `chunk_gated_delta_rule`) finds 56
byte-identical between the two stacks. The 7 that differ are all
`scaled_dot_product_attention`. Both builds default to AOTriton's efficient
attention, and AOTriton 0.11.2 and 0.13.50 round differently; only the math
backend agrees, and neither build uses it. No official wheel carries AOTriton
0.13.50 (2.12.0 and 2.12.1 bundle 0.11.2, 2.13.0 bundles 0.12.0, 2.14.1
bundles 0.13.0).

So every model with attention misses the release image's answers by a small
amount on this stack, while each answers byte-identically across cold
processes on it. Each family's parity record gives the agreement with the
release image on its full panel, and each re-recorded golden names its
reason; the user's acceptance of the new stack (2026-10-05) set the
conditions.
