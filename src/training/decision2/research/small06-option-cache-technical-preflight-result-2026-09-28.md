# 0.6B independent-option prefix cache: preflight HOLD

The frozen technical screen in
[`small06-option-cache-technical-prereg-2026-09-28.md`](small06-option-cache-technical-prereg-2026-09-28.md)
did **not** reach model loading or the seven-case hidden-state comparison.
This is no Decision 2.0 model, accuracy result, throughput result, or release
evidence.

| Item | Observed result |
| --- | --- |
| Preregistered source commit | `35fa92868c807de86e652b2680c1d2aee9093a02` |
| Official source revision | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |
| Container image digest | `sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54` |
| Source/config/weight/code hashes | Matched every frozen value before model loading. Ten source-file hashes are in the private receipt. |
| CPU preflight | Tokenizer loaded; all seven synthetic request shapes passed the native 8,192-token cap; exact code bytes matched. |
| GPU visibility | **Failed:** PyTorch reported `device_count=0`, `is_available=False` in the pinned container, despite the selected render device and KFD device appearing inside it. |
| Hidden-state parity and 255-option cost | **Not measured.** No trained head or model prediction was run. |
| Resource | Container window 51.904 seconds; reserved-slot upper bound 0.01442 GPU-hours; actual model GPU-hours 0. |
| Private evidence | Mode-0600 failure receipt SHA-256 `2fdbfb466d4feab7ec16dca6ccecbf75fc621eca862c7d675fec63a85264e17d`. |

The container was stopped and removed. GPU memory returned to its preflight
baseline. There was no retry or change to the frozen probe code. This arm
remains **HOLD**. A different GPU mapping or container runtime would require
a separately versioned preregistration and a fresh preflight; it must not be
silently treated as the result of this attempt.
