# Decision 3.0 parity

The `decision3` family answers exactly as the Decision 3.0 packages' own
server (`d3_server.py`, `POST /v1/systemone`) on CPU and on an AMD Instinct
MI325X, for text, image and video requests: every request of every model
compared below has byte-identical answers. Through Engine mode (the Router
serving the public System One API in front of a managed worker) the answers
are the same as well. Warm latency on the MI325X matches the package server
for direct requests. Engine mode adds Router time for large request bodies
(last section).

- **Date:** 2026-10-11.
- **Packages:** the revisions `registry/tables/decision3.py` pins, each a
  model's latest release: `vllm-sr/d3` `a443f814` (v3.0.6), `vllm-sr/d3-flash`
  `abd00fe1` (v3.0.2), `vllm-sr/d3-mini` `94b1ea6e` (v3.0.3), `vllm-sr/d3-nano`
  `f1429487` (v3.0.4), `vllm-sr/d3-lite` `2227a85f` (v3.0.5) and
  `vllm-sr/d3-edge` `5d17a013` (v3.0.2, 15 layers), and the previous d3-edge,
  v3.0.1 (`84fa29a3`, 18 layers), served with `--revision`. An earlier round
  ran the d3, d3-flash and d3-lite revisions of 2026-10-11 morning (last two
  sections); d3-flash and d3-mini have not changed since.
- **Devices:** one AMD Instinct MI325X (gfx942) per run, in the router's ROCm
  image (PyTorch 2.12.0 for ROCm 7.2, Triton 3.7.0, FLA 0.5.2, OpenCV
  5.0.0.93) with this branch's runtime. CPU runs use PyTorch 2.10's CPU build
  on 16 cores of an AMD EPYC.
- **Reference:** the package's `d3_server.py` and runtime, run with
  Transformers 5.17.0 and torchvision 0.27.0 on the same device and PyTorch:
  each package's published runtime for text and images, and the v3.1.0 video
  runtime (the same files with video inputs) for videos. On CPU the package's
  runtime is called in the same process.
- **Runtime side:** on the MI325X, `vllm-srun serve` with the built-in model
  name, the exact profile and the default fast path (fused decoder layers,
  `fp64_naive` depthwise convolution); on CPU, the same runtime in process.
- **Kernel choices:** both sides run the model's pinned FLA kernel choices,
  each from its own empty Triton cache: the runtime from
  `registry/kernel_choices.json`, the package server from FLA's config files
  (`FLA_CACHE_MODE=full`). Neither side timed a kernel. d3-edge v3.0.1, which
  is not built in, ran on one Triton autotune cache shared by both sides.
- **Requests**, sent in the same order to both sides:
  - 10 text requests with Choice, Noul and Score questions: 3, 6 or 10
    questions per request (10 takes two forwards), 2 to 12 options,
    object instructions and criteria, Unicode text, a JSON state, a
    one-character state and a long state.
  - 9 image requests: PNG, JPEG and WebP images of 96×96 to 1600×1200
    pixels, one to three images per request, and up to nine questions about
    one image. Images over 1.6 megapixels are resized to that size.
  - 13 video requests: MPEG-4 clips in MP4, QuickTime and Matroska
    containers, 96×64 to 1920×1080 pixels, 8 to 30 frames per second and 1.5
    to 30 seconds long (4 to 32 frames read), one or two videos per request,
    a video with an image, up to nine questions about one video and a JSON
    state; 393 to 13,391 input tokens.

## Answers

A request is identical when both sides return the same answers and usage:
the same choice, and every probability, Noul and Score value equal as JSON
numbers.

| Model | Text and images | Videos | Golden check |
| --- | --- | --- | --- |
| `vllm-sr/d3` (27B) | 19 / 19 identical | 13 / 13 identical | 5 / 5 matched |
| `vllm-sr/d3-flash` (9B) | 19 / 19 identical | 13 / 13 identical | 5 / 5 matched |
| `vllm-sr/d3-mini` (4B) | 19 / 19 identical | 13 / 13 identical | 5 / 5 matched |
| `vllm-sr/d3-nano` (2B) | 19 / 19 identical | 13 / 13 identical | 5 / 5 matched |
| `vllm-sr/d3-lite` (0.8B) | 19 / 19 identical | 13 / 13 identical | 5 / 5 matched |
| `vllm-sr/d3-edge` v3.0.2 (0.66B) | 19 / 19 identical | 13 / 13 identical | 5 / 5 matched |
| `vllm-sr/d3-edge` v3.0.1 (0.72B) | 19 / 19 identical | | not built in |

The largest difference of any probability is 0.0 in every run, and no choice
changes. In the earlier round, `vllm-sr/d3-lite` was also 19 / 19 identical
on CPU and through Engine mode, with 3 to 6 MB image bodies.
`tests/test_decision3_reference.py` checks the same exactness in CI against
Transformers' `Qwen3_5Model` on random-weight packages, regular and pruned,
for text, one and three images, and videos with and without an image; it
also checks the video processing against Transformers' `Qwen3VLVideoProcessor`.

Exactness needs the released numerics:

- **Tokenizer.** The released runtime tokenizes with Transformers 5.17's
  `Qwen2Tokenizer`, which differs from `tokenizer.json`: it splits words with
  the Qwen2 pattern and adds every special token of `tokenizer_config.json`.
  The family builds the same tokenizer. It matched the Transformers tokenizer
  on 3,000 of 3,000 adversarial texts; `tokenizer.json` alone differed on 2,681.
- **Precision and attention.** BF16 parameters with no autocast on every
  device, an FP32 readout, and a full-attention mask that is always passed.
  Text uses plain positions even when it is left-padded; images and videos use
  the padding-aware multimodal positions, a video's frame pairs separated by
  their timestamps.
- **Layers.** The backbone follows the package's `layer_types`: d3-edge v3.0.2
  keeps 15 of d3-lite's layers in an irregular order that its
  `full_attention_interval` no longer describes.
- **Images.** The Transformers 5.17 torchvision image processor's resize,
  normalization and patch layout. In an image request the vision tower runs
  once per batched question, and the patch embedding is a matrix product.
- **Videos.** OpenCV 5.0.0 decodes from a file with its own FFmpeg, as in the
  release image, and the frames are sampled and capped as the d3 runtime
  configures the Qwen3-VL video processor. In a request with videos the tower
  reads the request's images and videos once and every question reuses them.
- **ROCm convolution.** The depthwise convolution of the Gated DeltaNet layers
  is MIOpen's naive kernel in the released runtime: FP64 accumulation,
  rounded to BF16. The `fp64_naive` variant computes the same values in
  Triton, both standalone and fused into the gated-delta preparation.

## Kernel choices

FLA's gated-delta kernels choose block sizes by timing them, and the choice
changes rounding. The release runs of each Decision 3.0 model shared one
autotune cache, so their answers are reproducible. `registry/kernel_choices.json`
pins the configurations of that cache for each model (`tools/kernel_choices.py`;
FLA 0.5.2, recorded on Triton 3.8.0): the fast-runtime release caches of d3,
d3-flash, d3-mini and d3-edge, and the cache of the runs that released d3-nano
v3.0.4 and d3-lite v3.0.5, which differs from the fast-runtime cache on 29
keys of the cumulative sum and L2 norm kernels. The runtime runs them instead
of timing, like the other built-in Qwen3.5 decoders (design §11). On the
MI325X every built-in model's recorded choices install, and the runtime
resolves every recorded and unrecorded key as FLA resolves the same entries
(`tests/test_kernel_choices.py`, GPU cases).

## Latency

Warm medians in milliseconds from the earlier round, each request sent to
both sides in turn after a warm-up. Text is the 10 text requests above,
cycled (200 requests). Image is a 1280×1280 PNG with one question (60
requests).

| Model | Text, package server | Text, `vllm-srun` | Image, package server | Image, `vllm-srun` |
| --- | --- | --- | --- | --- |
| `vllm-sr/d3-lite` | 15.8 | 16.6 | 73.1 | 70.8 |
| `vllm-sr/d3-flash` | 27.6 | 28.2 | 171.5 | 166.9 |
| `vllm-sr/d3` | 73.4 | 71.0 | 298.0 | 288.5 |

Through Engine mode (the Router on the configuration `vllm-sr serve
vllm-sr/d3-lite --engine --platform rocm` writes, in front of its managed
worker), the 13 video requests are 13 / 13 identical to the package server
as well, with request bodies of up to 15 MB. Warm medians in milliseconds for
d3-lite, each request sent to both sides in turn (20 requests), with the
Router replacing the served model name in the body without decoding it:

| Request | Body | Package server | Engine mode |
| --- | --- | --- | --- |
| Text, 1 question | 219 B | 15.2 | 16.0 |
| 1280×1280 PNG | 6.0 MB | 72.3 | 141.5 |
| 640×360 video, 5 s | 3.0 MB | 62.4 | 98.0 |
| 1920×1080 video, 2.6 s | 10.6 MB | 116.9 | 261.2 |

Before that change the 6 MB image took 202.8 ms through Engine mode (package
server 80.0 ms): the Router decoded and encoded the whole body again to set
the model name. It still reads and checks each body once, about 12 to 14 ms
per MB, so a typical photo or short clip adds a few to a few tens of
milliseconds. Before images, the public System One API accepted bodies of at
most 2 MiB. It now accepts up to 48 MiB, enough for one video of 32,000,000
bytes; each image may have up to 8,000,000 bytes.
