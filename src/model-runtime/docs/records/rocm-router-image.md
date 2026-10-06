# The ROCm router image: PyTorch, causal-conv1d and a GPU smoke

The ROCm router image (`extproc-rocm`, `vllm-sr-rocm`) serves every GPU model
with the PyTorch of vLLM's ROCm image (pinned by digest) and that image's ROCm
7.2.3 libraries, plus a causal-conv1d wheel it builds itself. Its
causal-conv1d computes what the released decoders' build computes,
instruction for instruction, and its attention is the release's. Every
decoder checked answers its full panel byte-identically to the packages'
release image.

- **Date:** 2026-10-05.
- **Device:** AMD Instinct MI325X (gfx942).
- **Image:** `Dockerfile.extproc`, `ACCELERATOR=rocm`, target `extproc`, built
  from an exact checkout of `a580be6b9`, then slimmed at `31d00387c` (below):
  14.4 GB of image content, against 25.5 GB at `a580be6b9` and 17.2 GB for
  `af71d5e82` (content summed from `docker history`; a containerd image store
  also counts the compressed layers, so `docker images` shows more). The
  images before `a580be6b9` took PyTorch from the official rocm7.2 wheel:
  `af71d5e82` and, without the MIOpen paths below, `be7366c49`.
- **Reference:** the packages' release image, where every ROCm golden answer
  and parity reference of Phase 1 was recorded: PyTorch 2.12.0 built from
  source on ROCm 7.2.3, Triton 3.7.1, FLA 0.5.2, causal-conv1d 1.7.0.

## The stack

| Package | Image | Release image |
| --- | --- | --- |
| Python | 3.12.15 | 3.12.13 |
| PyTorch | 2.12.0+git6bbd260 from `vllm/vllm-openai-rocm@sha256:1fd21abe…` (HIP 7.2.53211, AOTriton 0.13.50) | the same build (its base image) |
| ROCm user space | 7.2.3 `lib` and `share/miopen` from the same image | 7.2.3 |
| Triton | triton-rocm 3.7.0 | 3.7.1 |
| FLA | fla-core 0.5.2 | 0.5.2 |
| causal-conv1d | 1.7.0, built in the image (below) | 1.7.0, local build |

- **Why vLLM's ROCm PyTorch:** the release image is built on
  `vllm/vllm-openai-rocm` at that digest, a public image, and its PyTorch,
  AOTriton and ROCm libraries are byte-identical to the release image's. The
  official rocm7.2 wheel bundles AOTriton 0.11.2, whose attention rounds
  differently (below). The official wheel still installs torch's Python
  dependencies and Triton; the rocm7.1 build crashes replaying the encoders'
  HIP graphs.
- Torch needs GCC 16's C++ runtime (`GLIBCXX_3.4.32`), copied from the same
  image over Debian's; ROCm's libraries are found through `ldconfig`.
- **Triton 3.7.0 against 3.7.1** changes no answer: a Triton 3.7.1 venv gives the
  3.7.0 answers to the last digit for Decision 1.0, Vela 1.0 and Vela 2.0, and
  the decoders run their pinned FLA kernel choices on either.
- **No CPU LAPACK:** this PyTorch build has neither LAPACK nor MKL
  (`torch._C.has_lapack` is false). The Qwen3.5-based decoders' CPU path
  calls `torch.triangular_solve`, so such a model can't run on `cpu` in this
  image. The runtime refuses that placement before any weights load: the CPU
  accelerator reports `lapack` among its capabilities, these families
  require it on the CPU (`ModelSpec.requires`), and the model fails at once
  with the cause and the fix. Serve it on the GPU, or run CPU models from the
  CPU image.

## What the image leaves out

The `rocm-lib-slim` stage copies only what the serving stack can load from
the release image's ROCm tree, each file unchanged. Of `a580be6b9`'s files it
leaves out 3,353, about 11 GB:

| Left out | Size | Why nothing needs it |
| --- | --- | --- |
| Static libraries (`*.a`) outside LLVM, mostly composable_kernel's device-op archives | 4.9 GB | build-time only |
| ROCm's LLVM (`lib/llvm`, with its static libraries) | 2.4 GB | Triton links with its own `ld.lld`; comgr carries its compiler |
| rocFFT's kernel cache | 1.8 GB | rocFFT compiles what it misses, and no model runs an FFT |
| rocALUTION, hipTensor | 0.7 GB | no library in the image links them |
| rocBLAS, hipBLASLt and hipSPARSELt kernels, and MIOpen databases, of GPUs outside `ROCM_GPU_ARCHS` | 1.3 GB | the image serves gfx90a, gfx942 and gfx950, as `CAUSAL_CONV1D_ARCHS` |

Checked against `a580be6b9` on an MI325X (gfx942):

- **Files:** SHA-256 over every file under `/opt/rocm-7.2.3` and PyTorch's
  package: 17,607 kept, none changed or added.
- **Golden answers** recorded in the slim image equal the committed files in
  every value: Decision 1.0 (seven), Decision 2.0 (all six), Vela 1.0 (all 13 `task_heads` built-ins:
  the ten text models, Vela Embedding, Vela Reranker and Qwen3-Embedding), Vela 2.0 (all four sizes).
- **Panels**, the runtime at one commit, maximum drift 0.0: Vela 2.0 4B
  answers its 360 requests identically to `a580be6b9`, and 0.3B to the
  release image; Vela 1.0's AMD recipe gives its 4,376 answers identically
  to the release image (`vela1-parity.md`).
- **GPU smoke** (below): 7 of 7 ready in 36 s, 280 / 280 requests.

So the timing records stand as timed in `a580be6b9`.

## causal-conv1d

PyPI publishes no ROCm wheel. The `causal-conv1d-rocm` build stage compiles
the 1.7.0 sdist (sha256 `3202758494eaa7b5…`) against the official rocm7.2
PyTorch wheel (the stage's base; the image then serves with the release's),
with ROCm 7.2.3's compiler (AMD clang 22.0.0git, roc-7.2.3 26084) and the
release build's tools (setuptools 79.0.1, wheel 0.48.0, ninja 1.13.2), for
`CAUSAL_CONV1D_ARCHS`. The image installs only the wheel; its extension is
`/usr/local/lib/python3.12/site-packages/causal_conv1d_cuda.cpython-312-x86_64-linux-gnu.so`.

- The image the records and the smoke ran (`af71d5e82`) built gfx942 only: 145 s
  on 16 vCPUs, wheel sha256 `58d292e607483d44…`.
- The default is now gfx90a, gfx942 and gfx950 (Instinct MI200, MI300 and
  MI350), because a decoder fails on a GPU the wheel has no code for: 359 s on
  28 vCPUs, 21.5 MB. Its gfx942 code objects pass the same check below, so the
  records hold for it.

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
| `a580be6b9` | 7 of 7, in 35 s | `matched`, 7 / 7 | 280 / 280 |
| `31d00387c` (slim) | 7 of 7, in 36 s | `matched`, 7 / 7 | 280 / 280 |

- **The `be7366c49` failure:** Vela 2.0's forest forward runs `F.conv1d`,
  which goes through MIOpen. MIOpen keeps its databases and lock files under
  `~/.config/miopen`, and the chart's uid has no home directory on a read-only
  root. `af71d5e82` sets `MIOPEN_USER_DB_PATH` and `MIOPEN_CUSTOM_CACHE_DIR` to
  `/app/models/miopen`, beside `TRITON_CACHE_DIR`.
- No request failed, no device fault was logged and nothing was retried.
  Golden `matched` is readiness (the 0.02 GPU tolerance); byte identity is the
  exactness check below.

## Against the release image

The official wheel's images (`af71d5e82`) differed from the release image in
attention only. A probe of 63 ops on fixed inputs found 56 byte-identical; the
7 that differed were all `scaled_dot_product_attention`, because AOTriton
0.11.2 and 0.13.50 round differently. That moved 5.4% (4B) and 5.8% (9B) of
Vela 2.0's span sets, with no measurable quality change (`vela2-parity.md`),
and nearly every Decision 2.0 answer by a small amount.

`a580be6b9` serves with the release's PyTorch. The runtime at the same commit,
run in this image and in the release image (cold processes, cold MIOpen and
autotune caches):

- attention probe: 192 / 192 outputs identical;
- Vela 2.0 4B and 9B, 360-request panel, three processes each: 360 / 360
  identical, maximum drift 0.0;
- Vela 2.0 0.8B (added later), the same panel: 360 / 360 identical on the
  runtime and on the package's engine, maximum drift 0.0; its ROCm golden
  answers are recorded in this image;
- Decision 2.0, all six models, the four scored panels: 10,653 / 10,653
  identical (table below).
- Decision 1.0, all seven packages: the ROCm golden answers recorded in this
  image equal the stored release goldens in every value.

So the release image's golden answers and parity references hold byte for
byte on this image, and no ROCm golden is re-recorded.

Decision 2.0 in the release image against this image (2026-10-05; the
runtime at `a580be6b9` on both sides, one cold process per model on one
MI325X, `exact`, readiness `matched` on both sides). The panels are
`typed-final` (1,600 prompts), `css15` (6,547), `public231` (231) and
`mlx-diag` (2,275); a prompt is identical when every answer value is equal.

| Model | Prompts | Identical | Decision changes | Max diff |
| --- | --- | --- | --- | --- |
| Kai-0.6B | 10,653 | 10,653 | 0 | 0.0 |
| Eos-0.8B | 10,653 | 10,653 | 0 | 0.0 |
| Sol-2B | 10,653 | 10,653 | 0 | 0.0 |
| Nox-4B | 10,653 | 10,653 | 0 | 0.0 |
| Lux-9B | 10,653 | 10,653 | 0 | 0.0 |
| Vega-27B | 10,653 | 10,653 | 0 | 0.0 |

The answers stay with the program's exactness runs, outside the repository,
because the panels' prompts are not published.
