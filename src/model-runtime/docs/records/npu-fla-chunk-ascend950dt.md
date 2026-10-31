# The fla-npu Ascend C chunk on Ascend 950DT

Issue: #5422 (accepted, Router Models & Inference Runtime). Precedents:
`npu-fused-ascend950dt.md` (the exact fused op), `cuda-fused-approximate.md`
(the approximate consent channel).

## What was measured

The `chunk_gated_delta_rule` slot of the d3 decision models on Ascend 950DT
(CANN 9.1, torch_npu 2.10.0.post4), d3-mini at the served shape (T=1343,
v_heads=32, GDN 128-wide k16/v32, chunk 64):

| arm | isolated chunk call | served d3-mini forward (c1 p50) |
| --- | --- | --- |
| reference (`kernels.chunk_gated_delta_rule_ref`) | 107.1 ms | 2,914 ms |
| fla-npu Ascend C (`flash_gated_delta_rule`) | **0.42 ms** | **77.5 / 79.8 ms** |

Parity arm: the CUDA production path (H20, serving the FLA chunk) on the same
corpus and shape measures **107.3 ms** — the NPU path is 28% faster at equal
footing. The slot is the model's dominant term: the e2e difference
(2,914.7 − 79.8 = 2,834.9 ms) is entirely the chunk replacement (97% of the
baseline forward).

fla-npu (https://github.com/flashserve/flash-linear-attention-npu) ships the
Ascend C GDN ops vllm-ascend binds for Qwen3.5 — the official integration
guide builds exactly these kernels through fla-npu and ships them in
vllm-ascend v0.20+.

## Numerics (approximate slot)

Two probes (dual-run bypass and injection, 3,072 calls each across the
60-answer office corpus):

- **Per-layer injection is sparse and bounded**: unique max values sit on the
  bf16 ulp grid (2^-14..2^-8); mean |diff| ~3e-7 — most elements are
  bit-identical. Independent audit: both implementations sit at max 1 ulp from
  the FP64 reference — two equally-correct rounding paths, not a wrong kernel.
- **No compounding**: the injection curve does not grow across the 24 GDN
  layers (growth 0.96x), and single-call divergence does not grow with
  sequence length (T=200..500 all <= 3.9e-3 = 2^-8 — the reported rounding
  floor of BF16 GDN kernels, independently measured in arXiv:2609.04098).
- **Label impact is concentrated on borderline decisions**: 4 of 60 requests
  flip, one leaf each (`fact_check.choice` on two requests, `jailbreak.choice`
  on two — 2 prompts x 2 runs, deterministic; row agreement 56/60 = 93.3%).
  Every flipped leaf has a two-option probability gap of <= 0.02 (margins
  0.008, 0.019); the two flip directions are opposite, consistent with a
  random walk rather than a systematic bias. The two fused implementations
  (triton-FLA and Ascend C) agree with each other on all 60; the NPU reference
  agrees with the CUDA production recordings on all 60.
- Ecosystem precedents for this phenomenon (GDN, ulp-level divergence between
  implementations of the same math): IsoExec (vLLM blog, 2026-08) measures
  mean 1.7e-2 / max 0.25 between FLA's chunkwise and vLLM's recurrent kernels
  on H100 — three orders of magnitude larger; vllm-ascend's release notes
  carry multiple "GDN accuracy regression" fixes; generative ecosystems accept
  the class via KL tolerance and benchmark scores.

**Consent verdict under the current family policy**: 93.3% row agreement <
the "no decision changed, >=99% per answer type" bar → the released BF16 d3
packages keep exact kernels; this slot stays dormant without consent (the
consent-off serve measures 2,914 ms). A quantized d3 package — validated as
its own model — is the long-term path.

## Registration

- Approximate slot (`exact=False`), SoC-gated to `Ascend950DT` (the
  `MEASURED` set), bf16-only (other dtypes fall back to the reference), the
  wrapper loaded from the installed `fla_npu` package or `FLA_NPU_EXAMPLES`.
- The released BF16 packages are unchanged (receipt shows the reference
  kernels); a consenting package gets the full win.
- Unit tests (`tests/test_npu_fla.py`) run the registry and consent gate
  without a device.

## Reproduction

All artifacts on the 950DT host under `/root/sr-fused-logs/`:
`d3-plain-answers.json` / `d3-benchfused-answers.json` / `demo-*-answers.json`
(the label-diff arms), `layercurve.json` / `curve_inj_*.json` (the probes),
`req28-{request,response}.json` (a full request/response sample), and the
`serve-*.log` receipts.
