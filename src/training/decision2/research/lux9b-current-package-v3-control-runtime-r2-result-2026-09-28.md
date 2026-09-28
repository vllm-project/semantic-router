# Lux 1.0 r2 runtime repair: gold-free technical result

The separately locked r2 intervention **passed its two technical gates**.
No typed FINAL, CSS15, public JevBench or protected answer key was used in r2;
it is not a benchmark score. The r1 infrastructure failure and zero-item
receipt remain intact.

| Fixed comparison | r1 | r2 |
| --- | --- | --- |
| ROCm image, package and physical device | Same pinned image and current Lux 1.0 bundle | Same |
| Visibility flags | Both `HIP_VISIBLE_DEVICES=7` and `ROCR_VISIBLE_DEVICES=7` | Only `ROCR_VISIBLE_DEVICES=7` |
| Device discovery | Native initialization raised `No CUDA GPUs are available` before model load | No-model probe reported exactly one available `gfx942` device |
| Native current-package check | Not reached | Two fixed requests/five Choice, Noul and Score answers; runtime qualified and exact package revision; category changes **0**, maximum drift from the previously sealed current-bundle run **0.0** versus `1e-6` gate |

This single-flag comparison supports double visibility filtering as the cause
of r1's inability to discover the GPU. It does not determine how the two ROCm
variables are applied internally and is not evidence about model quality.

The no-model probe used **1.916 GPU-wall seconds**; the full native five-answer
check used **45.320 seconds**. Combined r2 technical use was **47.236 seconds
or 0.013121 GPU-hour**. The task container exited and device memory returned
to baseline. Private probe, native-gate and stop receipt SHA-256 values are
`d628c7ebcbcf317ef8592179dcfccc03edfccb3413944c77940082c4bff86062`,
`997eac396637de370e884eb182bcc85c1b8ead78b1e4f7f1b63fe0fde01f629b`,
and `1fea5d5243882dd21005a1a842a923b43693de670ba96bf871b1c00aa3436324`.

The r2 session ended after gold-free parity. A distinct prospective protocol
is required for a full current-package v3/public-231 comparator; r1's failed
attempt cannot be called a formal result and r2's five examples cannot be
extrapolated to 8,147 formal originals. Raw inputs, predictions, logs and
machine details remain private.
