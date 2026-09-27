# Qwen3.5 9B host-swap probe: one-time setup amendment

The [first host-swap attempt](qwen35-9b-rocm-host-swap-setup-2026-09-27.md) stopped before import. This prospective amendment changes **only container setup** so the [frozen 20-iteration probe](qwen35-9b-rocm-host-swap-probe-prereg-2026-09-27.md) can actually run. It does not change the script, image, source revision, update-64 checkpoint, seed, sequence schedule, timeout, GPU-hour cap, or the 9B training HOLD.

The complete reference-versus-failed-attempt diff across image, command, working directory, user, mount destinations/modes, device bindings, group, IPC, shared memory, ulimit, network, security, resource limits and environment was:

| Field | First-node reference | Failed host-swap setup | Corrected trial |
| --- | --- | --- | --- |
| `ROCR_VISIBLE_DEVICES` | `6` | `7` | `7`, the newly reserved idle accelerator |
| `PYTHONPATH` | `/work/src/training/decision2` | Inherited image default `/opt/decision-fla` | Explicitly `/work/src/training/decision2` |
| `PYTHONUNBUFFERED` | `1` | Absent | Explicitly `1` |
| Security option | `seccomp=unconfined`, `label=disable` | `label=disable` only | Both reference options |

No other examined field differed. `PYTHONDONTWRITEBYTECODE=1`, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` already matched but will be set explicitly. The corrected GPU trial will retain both original GPU device mounts; the CPU-only smoke intentionally omitted those mounts. Source, checkpoint, code and probe script remain mounted read-only.

## CPU-only import smoke before GPU retry decision

The exact pinned image and read-only source/checkpoint/code/script mounts were used, with the corrected environment, security options and working directory. **No GPU devices were passed to the container**; the smoke asserted `/dev/kfd` was absent, imported `torch`, `DecisionModel` and `per_example_loss`, and exited successfully.

| Evidence | Observed |
| --- | --- |
| UTC interval | 2026-09-27 12:52:16–12:52:18 |
| Exit / OOM | 0 / false |
| Image | `sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54` |
| GPU device mounts | None |
| Output | `no_gpu_import_smoke=PASS` |

This smoke establishes that the missing module path has been fixed. It does **not** test model loading or ROCm backward. One corrected, no-optimizer, 600-second GPU probe may run only after separate review, a fresh exclusive accelerator check, and another 848-file hash check. An exit-139 GPU fault remains FAIL; successful 20/20 iterations would be PASS for this synthetic path only. Neither result automatically authorizes optimizer training.
