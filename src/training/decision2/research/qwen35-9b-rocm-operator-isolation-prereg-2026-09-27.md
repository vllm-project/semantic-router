# Qwen3.5 9B: prospective backward operator isolation

The official-source 9B optimizer arm remains **HOLD**. This is one diagnostic
for its ROCm autograd exit 139, not an authorized optimizer resume, checkpoint
search or release evaluation. The [first and second node probes](qwen35-9b-rocm-host-swap-result-2026-09-27.md)
both failed after 14/20 synthetic iterations at their third 6,144-token
exposure. [Disabling gradient checkpointing](qwen35-9b-rocm-no-checkpoint-result-2026-09-27.md)
still failed at its first 6,144-token exposure. None was OOM-killed. The raw
Python frames stop in autograd backward and do not name a native operator.

## Source-based hypothesis

The official 9B text config has 24 linear-attention layers and 8 full-attention
layers. The pinned Transformers fallback for linear attention performs two
FP32 `torch.linalg.solve_triangular` calls per chunked layer, followed by a
serial recurrent-state scan. The full-attention layers call SDPA. In prior
9B logs the gated-delta wrapper used the **PyTorch reference**, not the
optimized FLA implementation. The next question is whether either backward
path can fail at the official layer dimensions *without loading the model*.
The reference gated-delta path is the first hypothesis because of its long
FP32 triangular solves; this is an ordering choice, **not a proven cause**.

## Frozen single-container diagnostic

- Runner `training/model/rocm_9b_operator_isolation.py` SHA-256
  `c00c8ada89122acea19b035f25383e9bc3f9e2dc8504a80651c111918a5c7134`.
  Its only input is the official `Qwen/Qwen3.5-9B` config at revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, SHA-256
  `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05`.
  It refuses unexpected layer counts or head dimensions. No model weights or
  update-64 checkpoint are loaded or modified.
- Use the same pinned image ID
  `sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`,
  PyTorch `2.12.0+git6bbd260`, HIP `7.2.53211`, Transformers `5.17.0`.
  The modeling file SHA-256 must be
  `762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b`;
  the unwrapped reference function source SHA-256 must be
  `e4116a769851c6a455d1a277f0b5ad777090d73c9da491a84e4e9773e17c45d6`.
- First run the same script's **CPU-only 64-token import/autograd smoke**.
  Only if it passes, take a fresh `amd-yes --list`, resolve ownership and use
  one exclusively idle accelerator on either authorized node. Do not stop
  another service. The exact runner and config are read-only mounts; verify
  their SHA-256 immediately before and after. One container, no network, no
  optimizer, no full model, no task data, no labels, no checkpoint writes.
- Seed `20260927`. In one process, test the reference gated-delta function
  first, then SDPA **only if gated-delta completes**. Each path receives one
  4,096-token control and three 6,144-token trials, all at the official
  attention-head dimensions with BF16 Q/K/V and FP32 gated-delta decay/beta.
  Test finite forward loss, backward gradients and GPU synchronization. Print
  a flushed phase event before backward, so an exit 139 records its case and
  length. At most 8 iterations, 300 seconds wall, **0.0834 GPU-hour**; no
  automatic retry, sequence change, alternate seed or second GPU cell.
- Preserve private stdout/stderr, container exit and OOM flags, event count,
  wall time, peak GPU allocation and runner/config hashes. A setup, SHA or CPU
  smoke failure is **INDETERMINATE**; no GPU launch. Exit 139 during the first
  phase implicates the standalone reference path; exit 139 during the second
  implicates standalone SDPA under these synthetic inputs. Timeout/OOM is
  **INDETERMINATE**. Eight finite iterations with exit 0 show both isolated
  paths pass this test, but **do not exonerate** the complete model, combined
  graph, checkpointing, repeated layers or production training.

The diagnostic has no model-quality result and cannot lift the 9B training or
release HOLD by itself. A positive isolated fault would motivate a separately
registered kernel/runtime mitigation; all-pass would motivate combined-graph
instrumentation. No further GPU work is authorized by this note.
