# Decision 1.0 parity

On the exact profile, the `decision1` family answers byte-identically to each
Decision 1.0 package's bundled runtime, for all seven packages. On ROCm this
holds on all four scored panels, on CPU on subsets of them. That is design
§17's bar for Decision 1.0: bit-identical on the same device class.

On the router's ROCm image (`a580be6b9`, which serves with vLLM's ROCm
PyTorch, the build the packages were released on), the runtime answers the
released references byte for byte: all seven models, all four panels, and
every router request, with the recorded golden answers reproduced in every
value. On the official-wheel stack that image replaced (below), the released
answers move by rounding: about 11.5% of prompts stay byte-identical on the
encoders, with no decision changed, and almost none on the decoders, where
0.3–0.6% of decisions change, each a near-tie.

- **Date:** 2026-10-04; the P1-4 runs, the `dc81682bb` column and the
  official-wheel stack on 2026-10-05; the adopted image on 2026-10-06.
- **ROCm:** one AMD Instinct MI325X (gfx942) per run, in the image the packages
  were released with: PyTorch 2.12 (ROCm), Transformers 5.17, Triton 3.7.1, FLA
  0.5.2, causal-conv1d 1.7.0.
- **CPU:** 16 cores (a cpuset), CPU PyTorch 2.10 with MKL, Transformers 5.17.
- **Reference:** the package's bundled runtime (Transformers remote code,
  `system_one`) on the same image, node and requests. On ROCm it runs with the
  same FLA kernel choices as the native side (the built-in table's).
- **Native side:** `tools/decision1_parity.py native`. It loads the package by
  repo id through `Decision1Family` (pinned revision, file digests), the native
  engine and the device's accelerator, with no package code. Requests go
  through `Runtime.call`, the API's request path, on the `exact` profile.
- **Comparison** (`tools/decision1_parity.py compare`): a request matches when
  its answers' canonical JSON is byte-identical. Otherwise it counts decision
  changes (a different choice or class), error mismatches and the largest
  probability difference. One normalization: the bundled runtime returns a
  structured Score legend description as an object, while the API contract
  returns every legend description as a string (the object's canonical JSON).
  The reference's legend is compared in the contract's form.
- **Raw results:** `decision1-parity.json`, one entry per model, device,
  profile and commit, with every panel's counts. From `c22bb15cd` an entry
  also carries what the engine ran: graph statistics per layer stack and the
  reduced copy. A P1-4 entry names the models served before it in the same
  process (`served_before`), and `concurrent` when they were asked at once.
  The stack entries name the `stack` and the `reference` they compare
  with (`released`, or the `stack`'s own one-model answers), and a
  `cross_process` entry holds the three cold processes.

## Exact profile, ROCm, four scored panels

Panels: typed-final (1,600 requests), css15 (6,547), public231 (231) and
mlx-diag (2,275), 10,653 in all. "Identical" means 0 decision changes, 0 error
mismatches and max |Δp| 0.0.

| Model | Runtime | Revision | `046f27883` | `11ab95952` | `d5b985e43` | `dc81682bb` |
| --- | --- | --- | --- | --- | --- | --- |
| Kai-0.6B | vela-encoder | `79263ba4` | identical | identical | identical | identical |
| Lex-0.6B | vela-encoder | `a5ba6895` | identical | identical | identical | identical |
| Route-0.6B | vela-encoder | `deed1f29` | identical | identical | identical | identical |
| Eos-0.8B | qwen3.5-decision | `2ca39a23` | identical | identical | identical | identical |
| Sol-2B | qwen3.5-decision | `5c698b1a` | identical | identical | identical | identical |
| Nox-4B | qwen3.5-decision | `7f65e1db` | identical | identical | identical | identical |
| Lux-9B | qwen3.5-decision | `2064c84d` | identical | identical | identical | identical |

- **`046f27883`:** the first full runs.
- **`11ab95952`:** the fused gfx942 layers run the BF16 residual stream
  (`8bc857d6f`), and the foundation and `vela1` are merged in.
- **`d5b985e43`:** the PR head merged in (`dcaf8a0b9`).
- **`dc81682bb`:** with the built-in kernel-choice table and per-model kernel
  choices (P1-4), the per-device GPU lock (P0-3) and IP3a merged in.
- **What runs on the GPU** (from the receipts): the decoders run every layer
  on the fused path (24 layers for Eos and Sol, 32 for Nox and Lux) and replay
  HIP graphs of exact shapes. Eos's causal convolution is the FP64-accumulating
  Triton kernel. The encoders run eagerly. Every load checked its ROCm golden
  answers (3 of 3 matched).

## Several decoders in one GPU process (review P1-4)

The default layout serves every model of a GPU in one runtime process, and
FLA's autotuners are process-wide. Before per-model kernel choices
(`accel/autotune.KernelChoices`, design §11), a decoder loaded after another
FLA model ran the first model's choices, because FLA read one config
directory per process. The released choices disagree on shared tuning keys
for 47 of 55 pairs of built-in GPU models, and the wrong ones change answers.

"B after A" serves Decision 1.0 A, then B, in one process on one MI325X
(`decision1_parity.py native --also A`, exact), and compares B's answers on
the four scored panels (10,653 requests) with B's bundled reference, which ran
B's own choices. Before: `ed4c500cc`, the runtime before the fix. After:
`8fc0bf02b`.

| Model | Served before it | Before: identical | Decision changes | Max abs diff | After: identical |
| --- | --- | --- | --- | --- | --- |
| Sol-2B | none (control) | 10,653 | 0 | 0 | |
| Sol-2B | Eos-0.8B | 75 | 79 | 0.193 | 10,653 |
| Eos-0.8B | Sol-2B | 75 | 74 | 0.100 | 10,653 |
| Lux-9B | Nox-4B | 199 | 38 | 0.148 | 10,653 |
| Nox-4B | Lux-9B | 199 | 65 | 0.070 | 10,653 |
| Lux-9B | Eos-0.8B | 10,636 | 0 | 0.016 | 10,653 |
| Eos-0.8B | Sol-2B, Nox-4B, Lux-9B | | | | 10,653 |
| Sol-2B | Eos-0.8B, Nox-4B, Lux-9B | | | | 10,653 |
| Nox-4B | Eos-0.8B, Sol-2B, Lux-9B | | | | 10,653 |
| Lux-9B | Eos-0.8B, Sol-2B, Nox-4B | | | | 10,653 |

- **Before, the loss was silent:** every one of those runs passed its golden
  check (3 of 3 within the GPU tolerance of 0.02). Lux after Eos differs least
  because the two share only `l2norm_fwd_kernel` keys; a model's other keys
  fell back to the first model's entries.
- **After, every case is byte-identical**, including each decoder in a process
  that serves all four, and no model ran unpinned. The last commit,
  `64d6046ad` (cheaper lookups, same routing), repeats Sol after Eos and Lux
  after the other three: identical.
- **The GPU tests** (`tests/test_kernel_choices.py`, at both commits): the
  resolver picks what FLA's own config-file lookup picks for every recorded
  key of all seven built-in models and for unrecorded neighbours (other
  numbers, flipped flags, another dtype), and two models' scopes alternate on
  FLA's real gated delta rule, each launch running its own configuration.

**Under concurrent requests.** The cases above ask one model at a time. With
`--concurrent 8`, every 8 prompts go to Sol and to Eos at once, so the two
models' batches and graph captures meet:

| Commit | Sol identical | Failed requests (Sol / Eos) | Health after | Sol's graphs |
| --- | --- | --- | --- | --- |
| `d7b06a6f9` (no device lock) | 1 / 10,653 | 10,652 / 10,652 | both degraded | 1 capture, failed |
| `79908fad8` (device lock) | 10,653 / 10,653 | 0 / 0 | both ready | 118 captures, 10,430 replays, 0 failed |
| `f25dae5dd` (and thread-local capture) | 10,653 / 10,653 | 0 / 0 | both ready | 118 captures, 10,430 replays, 0 failed |

- **Before the lock**, one model's graph capture (a shape's second use) ran
  while the other model launched work on the device; ROCm invalidated the
  capture (`hipErrorStreamCaptureInvalidated`), whatever the capture mode, and
  both models degraded. A runtime that may exit on a device error exits.
- **With the lock** (`GPUAccelerator.execute` serializes a GPU's device calls,
  design §9), Sol stays byte-identical on its own kernel choices while Eos
  runs its own, at the same time. From `f25dae5dd` the decoders capture in
  thread-local mode, so a device query from another thread (placement's, for a
  model loading next to served ones) can't invalidate a capture either.
- **Three encoders in one process** (`f25dae5dd`, Kai with Lex and Route, 8
  prompts at a time to all three): on `exact`, where encoders run eagerly,
  Kai's 10,653 answers stay identical. On `batching`, every model replays its
  own per-stack graphs (Kai's trunk: 10 captures, 341 replays, none failed),
  and Kai has 0 decision changes against its reference (max |Δp| 1.2e-5, the
  profile's approximation; 640 requests byte-identical, fewer than one at a
  time because concurrent requests coalesce). On both, no request failed and
  all three stayed ready.

## The router's ROCm image: vLLM's ROCm PyTorch

The router's ROCm image, `Dockerfile.extproc` at `a580be6b9` (`ACCELERATOR=rocm`), takes PyTorch 2.12.0+git6bbd260
(AOTriton 0.13.50) and the ROCm 7.2.3 libraries from vLLM's ROCm image,
pinned by digest, with Triton 3.7.0, FLA 0.5.2 and `causal-conv1d` 1.7.0. At
`1bd99cd37`, whose runtime code is the PR's (device lock and thread-local
graph capture included), exact, on node C, every result is byte-identical:

| Check | Kai | Lex | Route | Eos | Sol | Nox | Lux |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Four panels against the released references (10,653 prompts, 11,053 questions) | 10,653 | 10,653 | 10,653 | 10,653 | 10,653 | 10,653 | 10,653 |
| Router requests against the bundled runtime in the image (231 requests, 1,386 questions) | 231 | 231 | 231 | 231 | 231 | 231 | 231 |
| ROCm golden answers, two fresh processes, against the recorded file | equal | equal | equal | equal | equal | equal | equal |

- **Panels:** 0 decision changes and max \|Δp\| 0.0 on every model; every
  load matched its golden (3 of 3, Route 4 of 4). Three GPUs, one model at a
  time per GPU.
- **Router requests:** both sides in the image, with Transformers 5.17.0 (and
  `regex`) appended to `PYTHONPATH` for the bundled side.
- **Golden answers:** two processes, started together on two GPUs, recorded
  the ROCm answers of all seven packages with `tools/golden_answers.py`; each
  equals `registry/golden_answers_decision1.json` in every value.
- **CPU in this image:** its PyTorch is built without LAPACK and MKL
  (`torch._C.has_lapack` and `torch.backends.mkl.is_available()` are false).
  Kai answers on its CPU path, within 8.3e-7 of the CPU golden answers
  (matched). The Qwen3.5 decoders can't: their CPU gated-delta reference calls
  `torch.triangular_solve`, which raises without LAPACK, so their golden check
  fails at load. CPU deployments use the CPU image.

## The official-wheel ROCm stack: stack B plus `causal-conv1d`

Until `a580be6b9`, the router's ROCm image shipped the official PyTorch
2.12.0 wheel from the rocm7.2 index (HIP 7.2, Triton 3.7.0) with FLA 0.5.2
and `causal-conv1d` 1.7.0 built for gfx942. The references above, and the
recorded ROCm golden answers, come from the packages' release image: a
PyTorch 2.12 source build on ROCm 7.2 with Triton 3.7.1.

At `a1c7a284e`, exact, the four scored panels (10,653 prompts, 11,053
questions), against the released references. The official-wheel image
(`extproc-rocm72cc:be7366c49`) and `venv-rocm72-cc` (the image's pins with a
`causal-conv1d` wheel whose gfx942 device code equals the image's and the
release's) gave byte-identical answers on every prompt of all seven models:

| Model | Byte-identical prompts | Decisions changed | Max abs diff | Release golden |
| --- | --- | --- | --- | --- |
| Kai-0.6B | 1,241 / 10,653 (11.6%) | 0 / 11,053 | 1.0e-5 | matched 3 / 3 |
| Lex-0.6B | 1,260 / 10,653 (11.8%) | 0 / 11,053 | 9.8e-6 | matched 3 / 3 |
| Route-0.6B | 1,228 / 10,653 (11.5%) | 0 / 11,053 | 1.3e-5 | matched 4 / 4 |
| Eos-0.8B | 21 / 10,653 (0.2%) | 52 / 11,053 (0.47%) | 0.036 | matched 3 / 3 |
| Sol-2B | 14 / 10,653 (0.1%) | 67 / 11,053 (0.61%) | 0.17 | matched 3 / 3 |
| Nox-4B | 4 / 10,653 (0.04%) | 62 / 11,053 (0.56%) | 0.073 | matched 3 / 3 |
| Lux-9B | 4 / 10,653 (0.04%) | 31 / 11,053 (0.28%) | 0.084 | matched 3 / 3 |

- **Every changed decision is a near-tie** in the released answer: the median
  top-two margin is 0.003–0.005, the largest 0.017 (Eos), 0.036 (Lux), 0.058
  (Nox) and 0.062 (Sol). Of each decoder's changes, 4–15 are Score or Noul
  answers; the rest are Choice.
- **Why:** `causal-conv1d` closes part of the decoders' gap. Without it, Sol
  had 4 identical prompts, 95 changed decisions and a max diff of 0.46. The
  rest is the PyTorch build: the official wheel and the release image's source
  build compute some kernels differently. Kai runs no convolution and no FLA
  kernel, and it still differs by up to 1e-5 on every official wheel. Vela 1.0
  and Vela 2.0 show the same on this stack.
- **Readiness passes on either record:** every load on stack B matched the
  release goldens within the GPU tolerance (0.02).

**On the stack itself, exact holds.** Three cold processes started together,
one per GPU (node C GPU1, GPU2 and GPU5), sharing one empty autotune cache and
one empty Triton cache, each answered the four panels
(`tools/cross_process.py`), in the venv and again in the image. For every
model, each pair of processes is 10,653 / 10,653 identical. Every process
matched its golden and tuned nothing (0 autotune entries before and after):
the decoders run their pinned choices on Triton 3.7.0.

**Several models in one GPU process, on stack B** (in the venv and in the
image, alike), each against the stack's own one-model answers:

| Model | Served before it | Identical | Concurrent |
| --- | --- | --- | --- |
| Sol-2B | Eos-0.8B | 10,653 / 10,653 | no |
| Lux-9B | Eos-0.8B, Sol-2B, Nox-4B | 10,653 / 10,653 | no |
| Sol-2B | Eos-0.8B | 10,653 / 10,653 | 8 prompts at a time to both |

**Router requests.** The router asks a Decision 1.0 model its six router
signals about one prompt as one request: six questions of mixed types, which
no scored panel has. Exact against the bundled runtime on such requests
(public231's 231 prompts, each asking the six signals of Route's
`QUESTIONS.json` as explicit questions, as `tools/decision1_bench.py --router`
builds them) is byte-identical for all seven models on the official-wheel stack, both
sides in the image plus Transformers 5.17 (231 / 231 requests, 1,386
questions each). On CPU, Kai is 30 / 30 and Eos 10 / 10 (3 threads on both
sides).

## Exact profile, CPU

The first requests of each panel; public231 in full.

| Model | Requests (typed-final / css15 / public231 / mlx-diag) | `046f27883` | `51b6bd721` |
| --- | --- | --- | --- |
| Kai-0.6B | 1,431 (400 / 400 / 231 / 400) | identical | identical |
| Lex-0.6B | 1,431 (400 / 400 / 231 / 400) | identical | identical |
| Route-0.6B | 1,431 (400 / 400 / 231 / 400) | identical | identical |
| Eos-0.8B | 831 (200 / 200 / 231 / 200) | identical | identical |
| Sol-2B | 831 (200 / 200 / 231 / 200) | identical | identical |
| Nox-4B | 531 (100 / 100 / 231 / 100) | identical | identical |
| Lux-9B | 531 (100 / 100 / 231 / 100) | identical | identical |

`51b6bd721` has the same `decision1` runtime code as `d5b985e43`.

**At `89c4388aa`, `d3d1d7e68`, `69ae2d0c5`, `28d702760`, `2e0a87467` and
`84f263b46`** (per-stack graphs and copies, the scheduler that answers each job
when its own batches ran, the CPU-only changes to coalesced batches, backbone
weights copied into process memory instead of left as views of the
checkpoint's file mapping, oneDNN's larger primitive cache, and the runtime's
decisions surface rebuilt on the common surface path), a spot check on ROCm
and CPU: Kai and Eos on public231 and the first typed-final requests (400; Eos
on CPU 200), every answer byte-identical.

## What exactness takes

The family reproduces the bundled runtime's numerics instead of its code.

- **`vela-encoder` (Kai, Lex, Route):**
  - one ModernBERT embedding feeds three 22-layer stacks: Choice, Noul (the
    shared encoder) and Score. They load as branches of one backbone
    (`BackboneSpec.branches`), sharing the embedding and the rotary tables;
  - physical batches of 8 questions, stably sorted by type and padded to their
    longest row. Every type present in a batch runs its stack over the whole
    batch, as the bundled runtime does;
  - additive FP32 attention masks (`finfo.min`), through the kernel variant
    `sdpa:additive_masks`, with contiguous q/k/v on ROCm; FP32 throughout;
  - the typed heads (one `nn.TransformerEncoderLayer`-equivalent layer per type)
    and the MLP scorer over the mask-token markers, in FP32.
- **`qwen3.5-decision` (Eos, Sol, Nox, Lux):**
  - every backbone parameter in BF16 on a GPU, so the residual stream is BF16
    (`DtypePolicy.gpu_weights`); FP32 on CPU;
  - the fused gfx942 layers follow PyTorch's type rules on a BF16 stream: the
    residual sums, a BF16 norm weight's product and the BF16 RoPE products round
    where PyTorch rounds;
  - physical batches of 8 questions in request order, padded to a multiple of
    32 tokens; the Decision 2.0 candidate head and prompt layout;
  - the temperature softmax on the device, renormalized on the host;
  - Nox renders a null Choice description as its key. Eos's causal convolution
    on gfx942 accumulates in FP64 once batch × length reaches 2,048 (kernel
    variant `causal_conv1d:fp64_accumulate`).
- **Both runtimes, the System One rules:** request validation, presets
  (`{"preset": name}`) and one over-long question failing the whole request.

## Batch invariance

Neither device class computes a row the same way in every batch. Running each
question type's rows alone instead of inside the full mixed batch changes the
answers:

- **ROCm:** 0 of 640 rows equal, max |Δp| 0.08;
- **CPU (MKL):** 244 of 320 rows equal, max |Δp| 1.4e-3.

So the exact profile keeps the released shapes: full mixed batches, every
present type's stack over all of their rows. Running each type's stack over its
own rows only, packed without padding, is the `batching` profile's job (next
section).

## Approximate profiles

These profiles are opt-in. They change the shapes, so they are compared within
a tolerance rather than bit for bit.

- **`batching`, encoders, ROCm, four panels:** each type's stack runs over
  its own rows, packed without padding (`run_approximate`). From `c22bb15cd`
  every stack also replays its own bucket graphs; before, only the Noul stack
  did.

  | Model | Commit | Identical requests | Decision changes | Max \|Δp\| |
  | --- | --- | --- | --- | --- |
  | Kai-0.6B | `d5b985e43` | 10,253 / 10,653 | 0 | 9.6e-6 |
  |  | `c22bb15cd` | 6,002 / 10,653 | 0 | 1.4e-5 |
  | Lex-0.6B | `d5b985e43` | 10,253 / 10,653 | 0 | 5.7e-6 |
  |  | `c22bb15cd` | 6,007 / 10,653 | 0 | 2.6e-5 |
  | Route-0.6B | `d5b985e43` | 10,253 / 10,653 | 0 | 8.2e-5 |
  |  | `c22bb15cd` | 6,002 / 10,653 | 0 | 3.6e-5 |

  At `d5b985e43` every single-question request stayed byte-identical: one
  packed row computes exactly like one padded row, and the 400 requests that
  differed were typed-final's two-question requests. A bucket graph pads its
  rows and masks the padding, which moves the last bits of single-question
  Choice and Score requests too; no decision changes.
- **`max_speed`, encoders, CPU, `c22bb15cd`:** `batching` on the
  `float32-packed` copy of all three stacks (1.32 GB of pre-packed linears),
  on the CPU subsets above, against the bundled runtime.

  | Model | Identical requests | Decision changes | Max \|Δp\| |
  | --- | --- | --- | --- |
  | Kai-0.6B | 70 / 1,431 | 0 | 1.1e-5 |
  | Lex-0.6B | 56 / 1,431 | 0 | 4.5e-6 |
  | Route-0.6B | 61 / 1,431 | 0 | 3.5e-5 |

  The same counts at `d3d1d7e68`, where on CPUs the type heads read coalesced
  rows by length group and coalesced batches stay within 1,024 padded tokens.
  On GPUs `d3d1d7e68` runs approximate batches as `c22bb15cd` does; at
  `8d45cd2f2`, which also grouped there, `batching` on the four panels matched
  the `c22bb15cd` row.

- **`shared_context`, decoders, ROCm, `d5b985e43`:** the public many-question
  request (`tools/many_questions.py`) at 64 and 128 questions, 5 requests each
  (960 questions). The bundled runtime has no shared mode, so it answers every
  question from the full prompt. On the same requests, the exact profile is
  byte-identical for all four models.

  | Model | Max \|Δp\| | Decision changes (64 / 128 questions) |
  | --- | --- | --- |
  | Eos-0.8B | 0.046 | 0 / 0 |
  | Sol-2B | 0.031 | 10 of 320 / 15 of 640 |
  | Nox-4B | 0.025 | 0 / 0 |
  | Lux-9B | 0.032 | 5 of 320 / 0 |

  Every changed decision was a near-tie in the bundled runtime's own answer:
  a top-two margin of at most 0.0032 (Sol) and 0.034 (Lux).

## Golden answers

`tools/golden_answers.py` recorded CPU and ROCm answers for all seven packages
(`registry/golden_answers_decision1.json`). Every load checks them: a
deployment whose answers move further from the record than the check's
tolerance (1e-3 on CPU, 0.02 on a GPU) fails to load.

At `1bd99cd37` the file is reproduced in every value: on ROCm by two fresh
processes in the router's image (above), and on CPU in the CPU PyTorch
2.10 image at 16 threads. At 4 threads the CPU answers match within 3.5e-7,
since MKL's reductions depend on the thread count.

## Reproduce

```bash
python3 tools/decision1_parity.py reference --package PACKAGE_DIR --repo vllm-sr/Decision-1.0-Kai-0.6B \
  --device cuda:0|cpu [--threads 16] --panel typed-final:PROMPTS.jsonl:1600 --panel css15:PROMPTS.jsonl:6547 \
  --panel public231:PROMPTS.jsonl:231 --panel mlx-diag:PROMPTS.jsonl:2275 --answers reference.jsonl
python3 tools/decision1_parity.py native --model vllm-sr/Decision-1.0-Kai-0.6B --cache-dir HF_CACHE \
  --device rocm:0|cpu [--threads 16] [--profile batching] --panel ... --answers native.jsonl
python3 tools/decision1_parity.py compare reference.jsonl native.jsonl --output compare.json
# P1-4: Sol served after Eos in one process, against Sol's reference (add --concurrent 8 to ask both at once)
python3 tools/decision1_parity.py native --model vllm-sr/Decision-1.0-Sol-2B --also vllm-sr/Decision-1.0-Eos-0.8B \
  --cache-dir HF_CACHE --device rocm:0 --panel ... --answers sol-after-eos.jsonl
pytest tests/test_decision1_*.py tests/test_kernel_choices.py
pytest -m gpu tests/test_decision1_gpu.py tests/test_kernel_choices.py
```
