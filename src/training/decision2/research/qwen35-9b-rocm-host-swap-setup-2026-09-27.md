# Qwen3.5 9B host-swap diagnostic: setup failure

The first attempted container on the second authorized node exited **before model import, load, or any GPU forward/backward**. This is **INDETERMINATE** for the prospectively frozen [host-swap probe](qwen35-9b-rocm-host-swap-probe-prereg-2026-09-27.md), not a completed 20-iteration trial or evidence about a host-specific ROCm fault. The 9B optimizer arm remains HOLD.

| Evidence | Observed |
| --- | --- |
| UTC interval | 2026-09-27 12:48:49–12:48:50 |
| Exit / OOM | 1 / false |
| Error | `ModuleNotFoundError: No module named 'training'` at probe import |
| Iterations / GPU work | 0 / none |
| Conservative container-wall charge | 0.00036 GPU-hour |
| Source, checkpoint, code and script | 848/848 file hashes matched the first-node manifest before and after; manifest SHA-256 `af84d7cd5f80a23d2424176f4314ce3dedfaea97614d90f40dde294675157b46` |
| Exact image | `sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54` |
| Accelerator after exit | Baseline memory and zero reported use; no new matching kernel fault observed |

The first-node reference container set `PYTHONPATH=/work/src/training/decision2`. The host-swap command unintentionally inherited the image default `PYTHONPATH=/opt/decision-fla`, so its import failed immediately. A complete reference-versus-attempt configuration diff and a CPU-only import smoke are required before seeking approval for one corrected launch. No automatic retry is authorized.

Read-only process ownership on the second node found that the seven memory-occupied accelerators were used by existing router/vLLM service containers unrelated to this task. The reserved accelerator had no active task process. No service was stopped or modified, and no stale active Decision 2.0 GPU container was identified.
