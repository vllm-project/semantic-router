# Qwen3.5 9B ROCm synthetic backward probe: FAIL

The single prospectively frozen [probe](qwen35-9b-rocm-synthetic-probe-prereg-2026-09-27.md) ran on a reserved, otherwise idle accelerator in the pinned training image. The script SHA-256 was `7f26d37fba07c987737b85c08a4554b2776d4605b6414022704575189d757a0a`. The official-source update-64 checkpoint's metadata, trainer state, adapter and head hashes matched the preregistration before and after the run. Source, checkpoint and code were read-only; no optimizer, scheduler, parameter update, real task row, or evaluation label was involved.

| Evidence | Observed |
| --- | --- |
| Container UTC interval | 2026-09-27 12:32:35–12:33:41 |
| Exit / OOM | 139 / false |
| Completed iterations | 14/20, through the 4,096-token case; next scheduled length was 6,144 |
| Failure | Native Python segmentation fault during `torch.autograd._engine_run_backward` |
| Peak allocated GPU memory before failure | 38.093 GiB |
| Charged GPU time | 0.0184 GPU-hour |
| New matching kernel event | None observed in the bounded post-run check |

This is a **FAIL** under the frozen criterion. The later fault's exact native library site was not identified; the earlier interrupted 9B training also exited 139 with a host-kernel record in the ROCm runtime. Because the no-optimizer synthetic path fails, the 9B training arm remains **HOLD**. Do not resume training, automatically retry the probe, change the original GPU-hour budget, or infer model quality from this runtime failure. A separately preregistered diagnostic on a different host may test whether the failure is host-specific; it would not itself authorize training continuation. The probe reservation was released after the container exited; the 4B continuation on another accelerator remains untouched.
