# Decision 3.0 graphs

Decision 3.0 backbones are noncausal, and the native engine used to skip its
host masks and graphs for noncausal backbones, so every d3 forward ran
eagerly and short text requests were bound by host launch time. With
`fast.NoncausalMasks` the d3 text forward is captured per exact padded shape
like the causal decision models.

- **Date:** 2026-10-12.
- **Device:** one AMD Instinct MI300X (gfx942) in the router's ROCm image
  `vllm-sr-rocm:latest` built 2026-10-11 (PyTorch 2.12.0, Triton 3.7.0,
  FLA 0.5.2), exact profile, pinned kernel choices.
- **Model:** `vllm-sr/d3-flash` at the built-in revision `521525ef`.
- **Code:** main as of 2026-10-12 against this branch.

## Answers

Every request below returned the same probabilities with and without graphs,
equal as JSON numbers: six in-process shapes (1, 2 and 8 questions, two
780-token rows, an image with 1 and 2 questions) and, over HTTP, the four
fixed requests of the latency table. `tests/test_gpu_fast_path.py` checks the
same exactness on a random noncausal Qwen3.5 with left-padded rows.

## Latency

`vllm-srun serve` over HTTP, warm medians of 40 requests after 5, one at a
time:

| Request | Main | This branch |
| --- | --- | --- |
| Text, 1 question (102 tokens) | 38.7 ms | 13.7 ms |
| Text, 2 questions | 39.6 ms | 18.0 ms |
| Text, 8 questions | 42.2 ms | 39.8 ms |
| Image and 2 questions | 146.9 ms | 149.9 ms |

A shape is captured on its second use. 120 two-question requests of new
lengths (5 to 250 words), sent three times in the same order, took a mean of
46.3, 40.9 and 41.2 ms on main and 46.8, 51.7 and 26.9 ms on this branch:
the second pass pays the captures, the third replays them.

In process, one capture costs 61 to 71 ms on d3-flash at 2 rows of 145 to
295 tokens, against 40 to 49 ms for the eager forward before it: a shape captured after an
eager use has already compiled and tuned its kernels, so only the capture
stream's first capture runs a warm-up forward. Replay time grows with length
(17 ms at 2 × 120 tokens, 29 ms at 2 × 295) while eager stays near 40 ms, so
graphs help most at short lengths. Larger forwards are
GPU-bound.

## Memory

40 captured shapes added 389 MiB of allocated memory, 157 MiB of it graph
outputs. With two warm-up forwards per capture, each on a new pooled stream
that took its own BLAS workspace, the same 40 shapes held 2,669 MiB. Without
the warm-ups a new stream per capture takes no workspace of its own (389 MiB
either way), so the saving at that size comes from the single capture stream
together with the skipped warm-ups.

## Not covered

Image and video requests take `forward_images` and still run eagerly; they
are GPU-bound (image and 2 questions: 129 ms of kernel time in 144 ms).
