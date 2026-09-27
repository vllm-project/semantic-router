# Qwen3.5 9B: prospective backward isolation after two matching faults

The official-source 9B training arm remains **HOLD**. This note freezes one
diagnostic before any new GPU work. It does not change the arm's 466 updates,
SELECT rule, budget or release eligibility.

## What the two raw logs establish

The [first probe](qwen35-9b-rocm-synthetic-probe-result-2026-09-27.md) and the
[byte-verified second-host replay](qwen35-9b-rocm-host-swap-result-2026-09-27.md)
used the same 20-iteration no-optimizer script (`7f26d37fba07c987737b85c08a4554b2776d4605b6414022704575189d757a0a`),
image (`sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`),
official source and update-64 checkpoint. Both completed iterations 1–14, then
exited 139 in native autograd backward before completing iteration 15. The
length cycle is 512, 1,024, 2,048, 4,096, 6,144 tokens. **Both previous
6,144-token iterations succeeded**, so a simple length threshold is not
established. Docker did not report OOM and the observed peak allocation was
38.093 GiB. The prior training failures did have a `libhsa-runtime64` kernel
record, but no native library/site was established for these two probes.

The raw logs on both hosts print that Qwen3.5's gated-delta function **fell
back to the Transformers PyTorch reference**. The image contains the FLA
package, but its optimized implementation was not selected for those GPU
calls. Do not attribute these two faults to an FLA Triton kernel. The reference
uses FP32 batched `torch.linalg.solve_triangular` and a serial chunk scan;
the model also has SDPA full-attention layers. The crashing native operation
within backward is still unknown.

## Working hypothesis and discriminating cell

A plausible mechanism is interaction between non-reentrant gradient
checkpointing/recomputation and the long Qwen3.5 reference-gated-delta or
SDPA autograd graph on this pinned ROCm/PyTorch image. Repeated mixed lengths,
not length alone, may matter. This is a hypothesis, not a root-cause finding.
The only new cell disables gradient checkpointing while retaining all other
synthetic inputs and backward work. The already completed, twice-failed
checkpointed runs are its prospective control; do not spend another GPU-hour
repeating them.

- Runner: `training/model/rocm_9b_no_checkpoint_probe.py`, SHA-256
  `bf99ed3ada1ed10bd607dc726f0dc98813962c812fa63332a3d87b03637ad857`.
  It is the original probe with only checkpoint enabling removed, an explicit
  disabled-state assertion and a start-event field. The runner imports the
  **same** immutable model/loss code as the previous probes.
- Use the same pinned image, source revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, exact update-64 checkpoint,
  source/code hashes, three endpoints, synthetic motif, seed `20260927`, BF16
  autocast/FP32 head, labels, 20 iterations and five-length cycle. Mount
  inputs read-only. Verify the prior 848-file manifest and the new runner hash
  before launch and verify source/checkpoint hashes again after exit.
- One exclusive idle accelerator after a fresh ownership check; no running
  service may be stopped. One container, 600-second wall timeout, at most
  0.167 GPU-hour. No optimizer, scheduler, parameter update, real task row,
  teacher output, evaluation label or checkpoint write. Compare trainable
  tensors bitwise before/after successful completion. Preserve private raw
  logs, iteration count, exit/OOM flags, memory peak and bounded kernel events.
- A CPU-only import/precondition smoke may run before GPU admission. The GPU
  cell requires separate review and approval. A setup/hash mismatch, occupied
  accelerator or import failure is **INDETERMINATE**; do not improvise a fix and
  rerun. A timeout or OOM is also **INDETERMINATE** for the mechanism. Exit 139
  is **FAIL** for the no-checkpointing intervention; 20/20 finite iterations,
  unchanged weights and no new native fault is **PASS for this isolated cell**.
  Stop after this one GPU cell regardless of outcome.

If the intervention passes, checkpointing or its memory/recomputation pattern
is implicated, but the precise native operation is unproven. If it fails,
checkpointing is not sufficient to explain the fault; a later, separately
registered operator-level test could distinguish reference triangular solve
from SDPA. Neither result authorizes resuming the frozen 9B optimizer arm,
selecting its checkpoint, or running development/formal evaluation.
