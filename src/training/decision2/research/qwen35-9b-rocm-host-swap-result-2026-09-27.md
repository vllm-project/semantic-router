# Qwen3.5 9B ROCm host-swap synthetic probe: FAIL

After the signed [setup amendment and CPU-only smoke](qwen35-9b-rocm-host-swap-amendment-2026-09-27.md), exactly one corrected GPU diagnostic ran on the second authorized node. The pre-launch and post-exit 848-file source/checkpoint/code/script manifest checks both passed; manifest SHA-256 was `af84d7cd5f80a23d2424176f4314ce3dedfaea97614d90f40dde294675157b46`. The image ID was `sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`, and the frozen probe script SHA-256 was `7f26d37fba07c987737b85c08a4554b2776d4605b6414022704575189d757a0a`. No optimizer, scheduler, parameter update, real task row, evaluation label or teacher output was used.

| Evidence | Observed |
| --- | --- |
| UTC interval | 2026-09-27 12:54:42–12:58:12 |
| Exit / OOM | 139 / false |
| Completed iterations | 14/20, through the 4,096-token case; next scheduled length was 6,144 |
| Failure | Native Python segmentation fault during autograd backward |
| Peak allocated GPU memory | 38.093 GiB of approximately 256 GiB available |
| Container wall and charged diagnostic time | 209.6 s / 0.0582 GPU-hour |
| Private raw-log SHA-256 | `41936ebbdc46b86501467ae0ec4639ef2096d2e7ed081f35c82ef4466e003209` |
| New matching kernel event | None observed; the bounded kernel tail was unchanged |
| Accelerator after exit | Baseline memory and zero reported use |

This is a **FAIL** under the frozen exit-139 criterion. The first-node synthetic probe also stopped after iteration 14 with a native autograd segmentation fault before its next 6,144-token case. Reproduction with byte-verified inputs and the same image on a separate node makes a fault confined to one host unlikely. It does **not** identify the exact failing library or distinguish a common ROCm/driver issue from a model-code interaction; both nodes had the same kernel and ROCm runtime versions. The 38 GiB peak and Docker's negative OOM flag argue against simple GPU-capacity exhaustion.

The official-source 9B optimizer arm remains **HOLD**. Do not resume or automatically repeat it, alter the original 4.0 GPU-hour training budget, or treat this fault as a model-quality result. Raw logs and container metadata are retained privately. The probe reservation was released; existing router/vLLM service containers on the second node and the separate 4B continuation were not changed. Read-only process ownership found no active Decision 2.0 GPU container on the occupied accelerators.
