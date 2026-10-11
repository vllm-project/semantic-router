# Decision 3.0 parity

The `decision3` family answers exactly as the Decision 3.0 packages' own
server (`d3_server.py`, `POST /v1/systemone`) on CPU and on an AMD Instinct
MI325X, for text and image requests: every request of every model compared
below has byte-identical answers. Through Engine mode (the Router serving the
public System One API in front of a managed worker) the answers are the same
as well. Warm latency on the MI325X matches the package server for direct
requests. Engine mode adds Router time for large image bodies (last section).

- **Date:** 2026-10-11.
- **Packages:** the revisions `registry/tables/decision3.py` pins:
  `vllm-sr/d3` `dc6c41cb`, `vllm-sr/d3-flash` `581c9953`, `vllm-sr/d3-mini`
  `61dbd3a3`, `vllm-sr/d3-nano` `6601b4d1`, `vllm-sr/d3-lite` `b731454b`.
- **Devices:** one AMD Instinct MI325X (gfx942) per run, in the router's ROCm
  image (PyTorch 2.12.0 for ROCm 7.2, Triton 3.7.0, FLA 0.5.2) with this
  branch's runtime. CPU runs use PyTorch 2.10's CPU build on 16 cores of an
  AMD EPYC.
- **Reference:** the package's `d3_server.py` with its MI325X fast-path
  runtime (`d3_runtime.py`, `d3_fast.py`, `d3_kernels.py`), run with
  Transformers 5.17.0 and torchvision 0.27.0 on the same device and PyTorch.
  On CPU the package's runtime is called in the same process.
- **Runtime side:** on the MI325X, `vllm-srun serve` with the built-in model
  name, the exact profile and the default fast path (fused decoder layers,
  `fp64_naive` depthwise convolution); on CPU, the same runtime in process.
- **Requests:** 19 per model, sent in the same order to both sides:
  - 10 text requests with Choice, Noul and Score questions: 3, 6 or 10
    questions per request (10 takes two forwards), 2 to 12 options,
    object instructions and criteria, Unicode text, a JSON state, a
    one-character state and a long state.
  - 9 image requests: PNG, JPEG and WebP images of 96×96 to 1600×1200
    pixels, one to three images per request, and up to nine questions about
    one image. Images over 1.6 megapixels are resized to that size.

## Answers

A request is identical when both sides return the same answers: the same
choice, and every probability, Noul and Score value equal as JSON numbers.

| Model | CPU | MI325X, worker | MI325X, Engine mode |
| --- | --- | --- | --- |
| `vllm-sr/d3-lite` (0.8B) | 19 / 19 identical | 19 / 19 identical | 19 / 19 identical |
| `vllm-sr/d3-flash` (9B) | | 19 / 19 identical | |
| `vllm-sr/d3` (27B) | | 19 / 19 identical | |

The largest difference of any probability is 0.0 in every run, and no choice
changes. `tests/test_decision3_reference.py` checks the same exactness in CI
against Transformers' `Qwen3_5Model` on random-weight packages, for text and
for one and three images.

Exactness needs the released numerics:

- **Tokenizer.** The released runtime tokenizes with Transformers 5.17's
  `Qwen2Tokenizer`, which differs from `tokenizer.json`: it splits words with
  the Qwen2 pattern and adds every special token of `tokenizer_config.json`.
  The family builds the same tokenizer. It matched the Transformers tokenizer
  on 3,000 of 3,000 adversarial texts; `tokenizer.json` alone differed on 2,681.
- **Precision and attention.** BF16 parameters with no autocast on every
  device, an FP32 readout, and a full-attention mask that is always passed.
  Text uses plain positions even when it is left-padded; images use the
  padding-aware multimodal positions.
- **Images.** The Transformers 5.17 torchvision image processor's resize,
  normalization and patch layout. The vision tower runs once per batched
  question, and the patch embedding is a matrix product.
- **ROCm convolution.** The depthwise convolution of the Gated DeltaNet layers
  is MIOpen's naive kernel in the released runtime: FP64 accumulation,
  rounded to BF16. The `fp64_naive` variant computes the same values in
  Triton, both standalone and fused into the gated-delta preparation.

## Kernel choices

FLA's gated-delta kernels choose block sizes by timing them, and the choice
changes rounding. The release runs of each Decision 3.0 model shared one
autotune cache, so their answers are reproducible. `registry/kernel_choices.json`
pins the configurations of that cache for each model (`tools/kernel_choices.py`;
FLA 0.5.2, recorded on Triton 3.8.0). The runtime runs them instead of timing,
like the other built-in Qwen3.5 decoders (design §11).

In the runs above both sides read one shared Triton autotune cache. Run again
with the choices pinned, each side started from its own empty cache: the
runtime ran its pinned choices, and the package server ran the same choices
from FLA's config files (`FLA_CACHE_MODE=full`). Neither side timed a kernel.

| Model | Answers | Golden check |
| --- | --- | --- |
| `vllm-sr/d3-lite` | 19 / 19 identical | 5 / 5 matched |
| `vllm-sr/d3-flash` | 19 / 19 identical | 5 / 5 matched |

On the MI325X every built-in model's recorded choices install, and the
runtime resolves every recorded and unrecorded key as FLA resolves the same
entries (`tests/test_kernel_choices.py`, GPU cases).

## Latency

Warm medians in milliseconds, each request sent to both sides in turn after a
warm-up. Text is the 10 text requests above, cycled (200 requests). Image is a
1280×1280 PNG with one question (60 requests).

| Model | Text, package server | Text, `vllm-srun` | Image, package server | Image, `vllm-srun` |
| --- | --- | --- | --- | --- |
| `vllm-sr/d3-lite` | 15.8 | 16.6 | 73.1 | 70.8 |
| `vllm-sr/d3-flash` | 27.6 | 28.2 | 171.5 | 166.9 |
| `vllm-sr/d3` | 73.4 | 71.0 | 298.0 | 288.5 |

Through Engine mode, `vllm-sr/d3-lite` answered text in 16.5 ms (package server
16.1 ms in the same run). The image request's JSON body is about 6 MB, and
it took 202.8 ms against the package server's 80.0 ms. The Router decodes and
re-encodes the whole body several times before it reaches the worker. That
costs about 5 ms per 0.5 MB of image data, so a typical photo adds a few
milliseconds. Before images, the public System One API accepted bodies of
at most 2 MiB. It now accepts up to 32 MiB, and each image may have up to
8,000,000 bytes.
