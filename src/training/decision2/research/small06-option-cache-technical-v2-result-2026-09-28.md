# 0.6B independent-option prefix cache: v2 technical HOLD

The [separately frozen v2 attempt](small06-option-cache-technical-v2-prereg-2026-09-28.md)
mapped an AMD render device correctly and passed its exact-code, official
source, tokenizer, seven-request, BF16 and one-GPU preflight. One bounded run
of the unchanged hidden-state probe completed. Its result is
**HOLD_TECHNICAL** because all seven cases exceeded the preregistered 0.01
maximum absolute drift limit. This is not a trained Decision model or an
accuracy, calibration, or product throughput result.

| Request | Cached seconds | Cached peak GiB | Maximum absolute drift | Minimum cosine | Parity gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Choice, 2 options | 2.652 | 3.068 | 0.218 | 0.999962 | FAIL |
| Choice, 3 options | 0.041 | 3.370 | 0.379 | 0.999972 | FAIL |
| Choice, 10 options | 0.081 | 4.742 | 0.351 | 0.999969 | FAIL |
| Choice, 255 options | 0.641 | 4.742 | 0.224 | 0.999966 | FAIL |
| Noul, 2 outcomes | 0.039 | 3.071 | 0.463 | 0.999982 | FAIL |
| Score, 3 levels | 0.040 | 3.372 | 0.311 | 0.999972 | FAIL |
| Score, 10 levels | 0.060 | 4.742 | 0.480 | 0.999973 | FAIL |

The Choice-10 reverse-order branch comparison also exceeded the absolute
drift limit (0.323; cosine 0.999975). The Choice-255 cached path covered all
255 branches and stayed below the 40-GiB and 120-second technical ceilings;
its independent reference used the eight frozen branches. The first measured
case includes cold-path work, so the displayed latencies are individual probe
measurements, not a throughput benchmark. High cosine similarity does not
override the absolute-drift gate, and no alternate threshold was used.

The immutable official source revision was
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, the frozen probe source
commit was `35fa92868c807de86e652b2680c1d2aee9093a02`, and the pinned
container image digest was
`sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`.
The private mode-0600 preflight and model receipts have SHA-256 values
`ff9805f6371d44f94f8b5d88cab98d95fc8bd50c3b21fa7f364bbcb87449a6ee`
and `57bec5bd3b8d7a184aedad31a11af7ddcd2b4ae871a31f338b37d5190b1c87a5`.
The container window was 92.450 seconds (0.02568 reserved-slot GPU-hours
upper bound); the sole model process lasted 10.683 seconds (0.00297 model
GPU-hours upper bound). The container was stopped and removed and GPU memory
returned to its preflight baseline.

No teacher outputs, private answer labels, decision head or training were
involved. This architecture remains ineligible for model selection or release
on this evidence. Two explanations remain **hypotheses**, not results:
the ROCm/BF16 attention or matrix kernels may vary with batch shape and
padding, or the repeated prefix cache may use different position/mask
semantics from the independent full-prefix path. The reverse-order cached
drift is consistent with batch-shape sensitivity, but it does not isolate the
cause.

One prospective *diagnostic*, requiring its own frozen protocol before any
new GPU use, is to hold the same source and candidate text fixed while
crossing cache on/off with branch batch size one/eight, and separately
recording BF16 versus FP32 drift and explicit position IDs. If unbatched
cached and uncached paths disagree, cache position/mask semantics remain a
candidate cause; if the difference appears only with batched BF16 branches,
kernel or padding numerics become more plausible. This would diagnose the
failed feasibility gate, not establish an accuracy gain or authorize training.
The v2 receipt remains unchanged and no further GPU run occurred.
