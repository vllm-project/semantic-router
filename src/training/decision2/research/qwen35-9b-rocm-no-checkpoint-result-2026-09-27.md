# Qwen3.5 9B: no-checkpointing backward diagnostic FAIL

The one GPU cell frozen in the [prospective plan](qwen35-9b-rocm-rootcause-next-prereg-2026-09-27.md)
ran once, without optimizer, scheduler, real task data or evaluation labels.
The official-source 9B training arm remains **HOLD**. No optimizer resume or
public/development evaluation follows from this diagnostic.

## Source and preflight

- The prior failing probe SHA-256 was
  `7f26d37fba07c987737b85c08a4554b2776d4605b6414022704575189d757a0a`.
  A verified unified diff against the new runner
  `bf99ed3ada1ed10bd607dc726f0dc98813962c812fa63332a3d87b03637ad857`
  contains only the explanatory docstring, replacement of
  `gradient_checkpointing_enable(use_reentrant=False)` with a disabled-state
  assertion, and an explicit `gradient_checkpointing: false` start-log field.
  The model forward, loss, labels, length cycle, backward, synchronization,
  unchanged-weight check and timeout contract are otherwise identical.
- The pinned image ID was
  `sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`.
  The existing 848-file source/checkpoint/code/script manifest SHA-256 was
  `af84d7cd5f80a23d2424176f4314ce3dedfaea97614d90f40dde294675157b46`;
  every file passed SHA-256 validation immediately before and after the GPU
  cell. The new runner hash matched its local signed source and remote mirror.
- A first CPU-only smoke encountered a missing writable temporary directory
  under a read-only container root. A corrected CPU-only smoke with a small
  temporary filesystem imported the model and loss modules, parsed the runner,
  confirmed no checkpoint-enabling call and asserted no GPU device was passed.
  It passed. Local syntax, formatting, security and training-contract checks
  also passed before launch.
- The selected accelerator had 0% reported VRAM allocation and activity
  immediately before launch. Existing service containers were left unchanged.

## Observed run

| Field | Result |
| --- | --- |
| UTC interval | 2026-09-27 13:13:15–13:14:21 |
| Container wall / GPU-hours | 65.609 s / 0.01823 GPU-hour |
| Exit / Docker OOM flag | 139 / false |
| Completed iterations | 4/20: 512, 1,024, 2,048, 4,096 tokens |
| Failed iteration | Fifth, first 6,144-token exposure, during native autograd backward |
| Peak allocated before fault | 107.987 GiB on an approximately 256 GiB accelerator |
| Private raw-log SHA-256 | `4722c7ea327f38804b99a978525146455a1ab6d3d8ada09da9e9c7d97909b133` |
| Post-exit accelerator | 0% reported VRAM allocation and activity |

The run is **FAIL** under the preregistered exit-139 rule. Both original
checkpointed probes completed two 6,144-token iterations and faulted on the
third; disabling checkpointing faulted on the **first** such iteration and
raised peak allocation substantially. Thus gradient checkpointing is **not a
necessary condition** for this native fault. The different allocation and
recomputation patterns prevent interpreting the earlier fault as having the
same immediate trigger. The raw log shows a Python autograd backward frame,
but no trustworthy native operator or library frame. No matching bounded
kernel-log event was observed in this run. Image, PyTorch/ROCm, reference
gated-delta, SDPA and their interaction remain candidates; an FLA optimized
kernel was not selected in the original or current 9B runs.

Stop this diagnostic here. An operator-level backward isolation, if later
worth its GPU cost, needs a new prospective control and stop rule. It must not
be used as an unregistered retry or a route to resume the frozen 9B training
arm.
