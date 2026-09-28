# 0.6B independent-option prefix cache: technical v2 preregistration

**Status: preregistered, no v2 GPU result.** This is a distinct attempt after
the [v1 device-map failure](small06-option-cache-technical-preflight-result-2026-09-28.md).
The only planned change is the render-device mapping. No model weights,
renderer, cache implementation, synthetic requests, parity thresholds, or
resource thresholds change. This is still a hidden-state feasibility screen,
not training, benchmark evaluation, or a Decision 2.0 model result.

## Immutable inputs and gates

- Use official `Qwen/Qwen3-0.6B-Base` revision
  `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, with config SHA-256
  `504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59`
  and weight SHA-256
  `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba`.
  Check all source-file hashes before load.
- Use unchanged source code from commit
  `35fa92868c807de86e652b2680c1d2aee9093a02`. The renderer, cache core
  and probe SHA-256 values respectively remain
  `f2646a0c18e99f6c71b6ab5c93c52665a24c59bfabc957fc00d07106a5cba675`,
  `c7510d552ce66b6442f3b17ec0b0329329704083396b3c4a2fedefac0cd6505d`,
  and `198ed4fafad82413603ce0feb4aa11c2c30f0ff18f9bc28b028fa1f4b0bef95f`.
- Use the pinned container image digest
  `sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`.
  A read-only host map must show the selected card's bus address matching an
  AMD (`0x1002`) render node before starting the container. In v2 the mapped
  node is the first AMD render device corresponding to the selected idle
  accelerator, rather than the virtual display node used by v1. Mount exactly
  that render node and `/dev/kfd`, and set `ROCR_VISIBLE_DEVICES=0` and
  `HIP_VISIBLE_DEVICES=0`. The GPU-visible preflight and probe must run in the
  *same* container with identical image, devices, environment, and read-only
  code/model mounts; output alone is writable.
- Before model load, verify exact code/source hashes, tokenizer loading, all
  seven deterministic synthetic request shapes and native 8,192-token cap,
  one visible BF16 GPU, and the selected AMD device-map identity. Any
  mismatch or exception is a v2 HOLD without a retry or changed flags.
- If and only if preflight passes, run the frozen probe once, within a
  15-minute container wall cap. Compare independent full-prefix versus
  shared-prefix-cached final hidden vectors for Choice 2/3/10/255, Noul 2,
  Score 3/10. For 255 options, evaluate all cached branches and eight frozen
  reference branches. Chunk size is eight, BF16 autocast applies to both,
  and neither path truncates.
- Every checked vector must be finite, maximum absolute drift <= 0.01 and
  minimum cosine >= 0.999. Reversing the Choice-10 option list must merely
  permute aligned branch vectors. The complete cached Choice-255 path must
  remain below 40 GiB peak allocated memory and 120 seconds. Record every
  case's latency, peak memory, parity, and the actual container/GPU time.

The frozen probe must write one mode-0600 private receipt at a new v2 output
path. Do not replace a missing or failed receipt with a hand-computed score.
Retain the failure state if the cache API, ROCm visibility, model load, parity,
or resource gate fails. No teacher outputs, private labels, model scoring, or
training occur in this arm.
