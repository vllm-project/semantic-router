# BF16-resident runtime: runtime-only revisions of the six DEV2.0 repositories (2026-10-01)

Coordinator note 2026-10-01 10:30 UTC+8 (b5f60b33): roll the shipped runtime's BF16-resident Linear weights out to
all six released private repositories as runtime-only revisions, each only with 0 answer changes on every scored
prompt and on mlx-diag, and measure latency and memory old vs new. Release worker, worktree `vllm-sr-dev2-release`
(branch `xunzhuo/decision-2-training-release`); state [`dev2-bf16-resident-state.md`](dev2-bf16-resident-state.md).
Node receipts are copied under [`dev2-bf16-resident-2026-10-01/`](dev2-bf16-resident-2026-10-01/); node paths stay
in records only. Times are UTC. Everything stays private.

## Result

RESULT_TABLE

## 1. Runtime change (`5dc962b00`, shared module, separate commit)

`v2/release/runtime/qwen.py` used to load every Qwen package with `model.float()` and run the backbone under BF16
autocast, so every Linear weight was held in FP32 and cast to BF16 before every matmul. On a GPU the runtime now
calls `keep_linear_bf16(model.backbone)` before moving the model to the device:

- every `nn.Linear` weight (and bias) of the backbone whose value BF16 represents exactly is stored in BF16 — the
  exact tensor autocast would have produced, so every product is unchanged;
- everything else stays FP32: the embedding table, all norms, gated-delta `A_log` / `dt_bias`, conv filters, the
  decision head (outside the backbone), any weight shared with a non-Linear module (tied embeddings), and any
  Linear weight BF16 cannot hold exactly — the 27B adapter's FP32-trained LoRA factors, which stay unmerged PEFT
  modules (autocast still casts them per call, as before);
- autocast is unchanged (`torch.autocast("cuda", torch.bfloat16)` around the backbone, head in FP32 with autocast
  off); CPU loading is unchanged (FP32, no autocast);
- `Decision2.from_pretrained(..., bf16_resident=False)` keeps the FP32 copies (`examples.py run --fp32-master`);
  example and parity receipts now record the residency counts and the loaded elements by dtype.

Converting on the host before `.to(device)` also means the GPU never holds the FP32 copies: memory after loading
falls by the size of the Linear weights in FP32 minus BF16.

## 2. Tests

- `v2.release.tests.test_bf16_resident` (image, CPU): only exact Linear tensors move (tied / inexact stay FP32,
  values unchanged); repeat calls are stable; tiny Qwen3 dense, Qwen3.5 hybrid (gated-delta tensors stay FP32) and
  unmerged-LoRA backbones give **bitwise-equal outputs** under BF16 autocast before and after the conversion. 5 / 5.
- `v2.release.tests.gpu_bf16_resident` (image, node A GPU0): tiny real `qwen-full` Qwen3 and Qwen3.5 packages and a
  `qwen-adapter` LoRA package, built exactly as releases are, answer the release examples in fresh isolated
  processes BF16-resident vs FP32-master: **byte-identical answers** for all three; the resident runs hold every
  backbone Linear weight in BF16 except the LoRA factors (Qwen3 14 / 0, Qwen3.5 15 / 0, adapter 14 / 28); the
  FP32-master and CPU runs hold every parameter in FP32. 3 / 3.
- The release suite (146 stdlib tests) passes; `check_no_private.sh` before every commit.
- Image limits found on the way: Transformers 5.17 binds the image's CUDA-only `causal_conv1d` even for CPU tensors,
  and the image's PyTorch has no CPU LAPACK for the chunked gated-delta reference, so the Qwen3.5 CPU unit test uses
  the torch references (recurrent form) and the GPU fixture runs its CPU leg for the Qwen3 packages only (the
  Qwen3.5-family cards already say CPU is not verified).

## 3. Latency and memory, old vs new runtime (same GPU)

`v2/release/runtime_bench.py`: one isolated container process per runtime (old = the verified download of the
released revision with its own vendored runtime; new = the preview build of the runtime-only spec), the scored
image and kernels, a fresh copy of the frozen autotune cache each, the first 400 typed-final prompts (one question
each) as single in-process `system_one` calls with `torch.cuda.synchronize()` around each. The 400 prompts run once
untimed (first use of each input shape), then the same 400 are timed. Memory is `torch.cuda` allocated bytes: after
loading, and the peak during the timed requests (weights + activations). One run at a time per node (node A GPU1;
27B node B GPU2).

| Model | p50 ms | p95 ms | mean ms | items/s | after load GiB | peak GiB | Linear BF16 / FP32 | identical answers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DEV2.0-0.6B | 19.2 → 16.6 | 19.4 → 16.9 | 19.0 → 16.5 | 52.5 → 60.6 | 2.22 → 1.40 | 2.32 → 1.50 | 196 / 0 | 400 / 400 |
| DEV2.0-0.8B | 22.7 → 21.6 | 22.9 → 22.7 | 25.3 → 24.7 | 39.5 → 40.5 | 2.81 → 1.90 | 2.91 → 2.01 | 186 / 0 | 400 / 400 |
| DEV2.0-2B | 24.1 → 23.5 | 24.8 → 23.7 | 24.2 → 23.4 | 41.4 → 42.7 | 7.02 → 4.46 | 7.14 → 4.57 | 186 / 0 | 400 / 400 |
| DEV2.0-4B | 28.5 → 28.1 | 29.0 → 28.9 | 32.8 → 28.2 | 30.5 → 35.5 | 15.68 → 9.12 | 15.83 → 9.25 | 248 / 0 | 400 / 400 |
| DEV2.0-9B | 33.8 → 27.2 | 53.3 → 27.6 | 36.6 → 27.8 | 27.3 → 36.0 | 29.58 → 16.70 | 29.79 → 16.83 | 248 / 0 | 400 / 400 |
| DEV2.0-27B | 122.9 → 93.3 | 127.5 → 96.8 | 124.3 → 94.0 | 8.0 → 10.6 | 97.26 → 51.90 | 97.56 → 52.07 | 496 / 992 | 400 / 400 |

- Every tier answered all 400 bench prompts **bit-identically** (0 changes, drift 0.0) old vs new.
- Peak memory falls 31–47%: the Linear weights are held once in BF16 instead of in FP32 (and the per-call BF16
  copies disappear). The gain in latency grows with size: −0.4 to −2.6 ms up to 4B, −6.7 ms at 9B, −29.6 ms (−24%)
  at 27B (the coordinator's estimate was ~10 ms at 9B and ~30 ms at 27B). The old 9B runtime also had a slow tail
  (p95 53 ms vs p50 34 ms) that the new one does not have.
- Superseded bench runs (kept on the nodes, counted in GPU-hours): a first pass with 20 warm-up requests (p95
  dominated by first-use shape spikes) and a second pass whose node-A runs overlapped on GPU0 and GPU1 (a model load
  on one GPU disturbed timings on the other). Their answers were also 400 / 400 identical.

## 4. Releases

RELEASE_SECTION

## 5. GPU-hours

GPU_SECTION

## 6. Commits

COMMITS_SECTION
