# Decision 1.0 parity

On the exact profile, the `decision1` family answers byte-identically to each
Decision 1.0 package's bundled runtime, for all seven packages. On ROCm this
holds on all four scored panels, on CPU on subsets of them. That is design
§17's bar for Decision 1.0: bit-identical on the same device class.

- **Date:** 2026-10-04.
- **ROCm:** one AMD Instinct MI325X (gfx942) per run, in the image the packages
  were released with: PyTorch 2.12 (ROCm), Transformers 5.17, Triton 3.7.1, FLA
  0.5.2, causal-conv1d 1.7.0.
- **CPU:** 16 cores (a cpuset), CPU PyTorch 2.10 with MKL, Transformers 5.17.
- **Reference:** the package's bundled runtime (Transformers remote code,
  `system_one`) on the same image, node and requests. On ROCm it runs with the
  same FLA kernel choices as the native side (the built-in table's, pinned
  before FLA is imported).
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
  reduced copy.

## Exact profile, ROCm, four scored panels

Panels: typed-final (1,600 requests), css15 (6,547), public231 (231) and
mlx-diag (2,275), 10,653 in all. "Identical" means 0 decision changes, 0 error
mismatches and max |Δp| 0.0.

| Model | Runtime | Revision | `046f27883` | `11ab95952` | `d5b985e43` |
| --- | --- | --- | --- | --- | --- |
| Kai-0.6B | vela-encoder | `79263ba4` | identical | identical | identical |
| Lex-0.6B | vela-encoder | `a5ba6895` | identical | identical | identical |
| Route-0.6B | vela-encoder | `deed1f29` | identical | identical | identical |
| Eos-0.8B | qwen3.5-decision | `2ca39a23` | identical | identical | identical |
| Sol-2B | qwen3.5-decision | `5c698b1a` | identical | identical | identical |
| Nox-4B | qwen3.5-decision | `7f65e1db` | identical | identical | identical |
| Lux-9B | qwen3.5-decision | `2064c84d` | identical | identical | identical |

- **`046f27883`:** the first full runs.
- **`11ab95952`:** the fused gfx942 layers run the BF16 residual stream
  (`8bc857d6f`), and the foundation and `vela1` are merged in.
- **`d5b985e43`:** the PR head merged in (`dcaf8a0b9`).
- **What runs on the GPU** (from the receipts): the decoders run every layer
  on the fused path (24 layers for Eos and Sol, 32 for Nox and Lux) and replay
  HIP graphs of exact shapes. Eos's causal convolution is the FP64-accumulating
  Triton kernel. The encoders run eagerly. Every load checked its ROCm golden
  answers (3 of 3 matched).

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

**At `89c4388aa`** (per-stack graphs and copies, and the scheduler that answers
each job when its own batches ran), a spot check on ROCm and CPU: Kai and Eos
on public231 and the first typed-final requests (400; Eos on CPU 200), every
answer byte-identical.

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

## Reproduce

```bash
python3 tools/decision1_parity.py reference --package PACKAGE_DIR --repo vllm-sr/Decision-1.0-Kai-0.6B \
  --device cuda:0|cpu [--threads 16] --panel typed-final:PROMPTS.jsonl:1600 --panel css15:PROMPTS.jsonl:6547 \
  --panel public231:PROMPTS.jsonl:231 --panel mlx-diag:PROMPTS.jsonl:2275 --answers reference.jsonl
python3 tools/decision1_parity.py native --model vllm-sr/Decision-1.0-Kai-0.6B --cache-dir HF_CACHE \
  --device rocm:0|cpu [--threads 16] [--profile batching] --panel ... --answers native.jsonl
python3 tools/decision1_parity.py compare reference.jsonl native.jsonl --output compare.json
pytest tests/test_decision1_*.py
pytest -m gpu tests/test_decision1_gpu.py
```
