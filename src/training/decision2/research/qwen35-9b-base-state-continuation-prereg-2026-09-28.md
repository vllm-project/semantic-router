# Qwen3.5-9B Base: bounded exact-state fault discriminator

**Prospective technical cell; no release or benchmark evaluation.** The
official Base 458-update arm remains HOLD. The [step-107 no-update replay](qwen35-9b-base-runtime-probe-2026-09-28.md)
passed 34 real TRAIN backwards from the saved step-64 adapter/head, so it did
not test the original optimizer and random-number trajectory. The original
process completed 106 updates and crashed in the next backward.

## One falsifiable question

Does the fault recur when the exact saved step-64 optimizer, model, Python,
PyTorch and accelerator RNG state follows the original 42 complete TRAIN
updates and enters the step-107 backward in a fresh process? A repeat at the
same update and row hash supports a deterministic dependency on this saved
training trajectory. A clean, numerically matched update 107 falsifies a
deterministic fault from this checkpoint and schedule under a fresh process;
the original event could still depend on unsaved process/runtime state or
nondeterminism. Neither outcome alone proves an operator bug or authorizes a
new full arm.

The original immutable source is
`Qwen/Qwen3.5-9B-Base@68c46c4b3498877f3ef123c856ecfde50c39f404`.
The original TRAIN SHA-256 is
`fe9c419a3e751e4a4173b5c90c49e0a83bba7e1c35c683441bb036138acb971c`,
trainer provenance SHA-256 is
`21a544b82a3b98a5bcb5a123ae37d5802ecf5fa301e2ce0e03572ea8fdc6d94c`,
step-64 optimizer/RNG state SHA-256 is
`cbe3b4be2f9fd42db9fd7c11e631608eb10a3b4feef829ffaf8f8f509f5c6786`,
original console SHA-256 is
`59e2aecf8e2c6d684598d0cbd994c305cd38c9ed83834ba2673f5bd559f13ded`,
and prior private row/sequence manifest has canonical digest
`d1554946856a3c9ee29a22376a61951d6934ad8a2b77d54c9d11900bb28eaa08`.
The original container image digest is
`sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`.
The trainer was PyTorch `2.12.0+git6bbd260`, HIP `7.2.53211`, PEFT
`0.21.0`, BF16 backbone autocast, FP32 LoRA/head/AdamW state and reference
gated-delta fallback. No gradient checkpointing or `torch.compile` was used.
The new bounded-continuation script SHA-256 is
`5930c3b4b15ce259571f1d12ccd3321c5a93dec9730f9afe04cdec36f8079bca`;
its frozen schedule/identity helper SHA-256 is
`4263bf8c9ca2b5605280144a847cd3f46ccf4c0dff80f07b131cbd20c7a41956`.

## Admission and exact bound

1. Rerun `amd-yes --list` and live per-card utilization, bus/render-node and
   reservation checks. Use one truly free card on an authorized node with the
   original immutable artifacts. If source, runtime, saved state, TRAIN,
   trainer hashes or previously sealed row order differ, **stop before GPU**.
2. In a network-free pinned container, load the complete original step-64
   checkpoint and restore AdamW plus Python/CPU/GPU RNG. The diagnostic code
   uses the original trainer's model, loss, schedule and learning-rate helper;
   no SELECT/CAL/FINAL labels are mounted. Recreate only updates 65–107, at
   most 43 optimizer updates, and write no weight or optimizer checkpoint.
3. Compare every completed update 65–106 to the original full-arm log:
   tokens exactly, loss within 1% relative and pre-clip gradient norm within
   5% relative. Any mismatch stops **before** entering update 107, marked
   INDETERMINATE. Those thresholds check trajectory fidelity, not quality.
4. For the 16 target TRAIN microbatches, flush private row-hash and phase
   events around forward/backward and attach output-gradient markers to
   backbone layers, attention and MLP components. On a crash, the last marker
   narrows the active module interval; it is **not** automatically the exact
   faulting ATen or ROCm operator. Record the container exit, kernel tail,
   OOM status, peak allocated memory, and event-file hash. Do not log raw
   examples or private row IDs in public artifacts.
5. One container and one launch only; hard wall cap **840 seconds** (<0.25
   one-GPU-hour), no retry or changed thresholds. A source/length/runtime
   mismatch, OOM or timeout is INDETERMINATE; exit 139 is a native fault; a
   complete numerically matched update 107 is a narrow negative finding.

Any operator-path repair requires its own prospective test and native output
parity. The Base initialization cell stays HOLD until a root cause and repair
are demonstrated; this probe never selects a model or runs JevArena/JevBench.
