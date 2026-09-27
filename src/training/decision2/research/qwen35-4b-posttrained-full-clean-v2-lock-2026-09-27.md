# Official Qwen3.5 4B Posttrained: prospective full clean-v2 source arm

**Status: CPU-only lock for independent review. No full training or release
authorization.** This is one source-initialization ablation against the
completed [official Base BEST466 control](qwen35-4b-official-base-full-development-result-2026-09-27.md).
The source and native-input audit is in the earlier
[prospective source protocol](qwen35-4b-posttrained-vs-base-prereg-2026-09-27.md).
The corrected [bounded admission](qwen35-4b-posttrained-admission-v2-prereg-2026-09-27.md)
passed, including two fresh SELECT zero starts, a one-TRAIN-update smoke and
separate reload. That one-update state is **not** the initialization for this
full arm. The Base control is **not** rerun. No protected or public benchmark
labels are admitted to the training lock.

## Causal contrast and frozen inputs

The sole intended difference is official Qwen initialization: Posttrained
`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` versus
completed Base
`Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b`.
Both share the exact model configuration. The full 8,855-row native-input
CPU audit found zero token-ID differences across TRAIN, SELECT and CAL.
The Posttrained full-weight loader, one-update path and SELECT zero/reload
parity passed in the pinned Base-compatible image. They establish feasibility,
not a development gain.

| Fixed component | Value |
| --- | --- |
| TRAIN rights-clean v2 | 7,455 rows; SHA-256 `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755` |
| SELECT | 700 rows; SHA-256 `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6` |
| CAL | 700 rows; SHA-256 `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`; split audit only |
| Native TRAIN input | 4,194,465 unpadded tokens; maximum 6,596 < cap 8,192 |
| Data budget | One epoch, microbatch 1, accumulation 16 including final partial window, **466 updates**; `max_steps=None` |
| Objective and optimizer | CE + 0.5 Brier; rank-16 LoRA, alpha 32, dropout .05, trainable 256-d head; LoRA peak LR `1e-4`, head peak LR `2e-4`, AdamW decay .01, gradient clip 1.0, warmup .05 and cosine decay |
| Seed and precision | `20260926`; FP32 weights/head/loss and BF16 backbone autocast; gradient checkpointing on |
| Runtime | Image `sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`, `/usr/bin/python`, Transformers `5.17.0`, Torch `2.12.0+git6bbd260`, PEFT `0.21.0` |
| Base evidence | Completed Base provenance SHA-256 `49b58711f9645af6806374d89dfce3db453460cf2921ae8777742290356ec7c8`; completion SHA-256 `a8543f3b8173cecfe7028444a6e3b015a98c1a6cdd80757d362cc299dd4490c9` |
| Posttrained admission | Private v2 lock SHA-256 `cb2690675dc1d6d39370acbf4445ea667b0d08ee54c369077ee8c9f55fcfb86a`; PASS receipt SHA-256 `4b8a21348a8e0d957cf7f5f520d1b2ee7a81f8816453b25d8f50ee4a5aa21054` |

The independent CPU-only
[`prepare_qwen35_4b_posttrained_full.py`](../scripts/prepare_qwen35_4b_posttrained_full.py)
must verify all archived evidence hashes, source shards, exact trainer file
hashes against the completed Base provenance, each current split's bytes and
row counts, native token-budget audit, identical update planner, image and
runtime. It creates a mode-`0600` one-shot lock within a mode-`0700` private
directory. A fresh no-device process must reopen and verify the lock. A lock
is a proposed execution contract, never launch authority.

## Training command and CAL boundary

The private lock contains the **exact argument vector** and private paths.
Its public form is:

```text
/usr/bin/python -m training.model.train
  --model-path <PINNED_OFFICIAL_POSTTRAINED_SOURCE>
  --init-kind posttrained
  --base-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
  --train <FROZEN_TRAIN> --select <FROZEN_SELECT> --cal <FROZEN_CAL>
  --output <FRESH_PRIVATE_ARM_OUTPUT>
  --epochs 1 --microbatch 1 --accumulation 16 --eval-batch 2
  --max-length 8192 --head-dim 256
  --backbone-lr 2e-5 --head-lr 2e-4 --weight-decay 0.01
  --warmup-ratio 0.05 --save-every 64 --seed 20260926
  --gradient-checkpointing --objective ce_brier --brier-weight 0.5
  --train-mode lora --lora-rank 16 --lora-alpha 32
  --lora-dropout 0.05 --lora-lr 1e-4
```

Omitting `--max-steps` is deliberate: the completed Base contract has
`max_steps=None`. No replay, teacher, sample weight, zero-step-only flag,
resume, smoke state, chat template or alternate checkpoint selector is used.
The trainer currently requires `--cal`, but source inspection shows it
**parses and hashes CAL solely for TRAIN/SELECT/CAL split-isolation**. It
encodes only TRAIN and SELECT, and evaluates/checkpoint-selects only SELECT.
CAL is never tokenized, passed to a model forward, scored, used to fit a
temperature, used for checkpoint selection, or used for a release claim in
this arm. The sealed provenance must say `cal_examples_audited_only=700`,
and a complete receipt must say `calibration_status=untouched`. Any other
CAL use invalidates this lock.

## Bounded execution and checkpoint rule

On one **currently reverified idle** GPU, cap cumulative container training
walltime at **3.0 one-GPU hours**, including SELECT inference and saves.
The operator must use an external watchdog that stops and removes the exact
container when the cumulative cap is reached; a mere client timeout is not
enough. Do not launch without separately reviewing the private lock, exact
container invocation, live device/process telemetry and watchdog. Stop on
source/data/code/runtime drift, missing type, overlength input, OOM,
nonfinite loss or gradient, failed checkpoint, interrupted cursor or budget
expiry. Preserve the failure receipt. There is **no automatic retry**; an
exact optimizer/data-cursor resume requires a separate reviewed decision
under the same cumulative cap and unchanged schedule.

Save and evaluate SELECT after updates **64, 128, 192, 256, 320, 384,
448 and 466**. Choose the single BEST by SELECT family-macro accuracy
descending, normalized Brier ascending, then earliest update. Verify a
32-row native reload with zero categorical change and probability drift
p99 ≤ `.005`, maximum ≤ `.02`. Do not pick a different checkpoint after
seeing typed DEV, CSS pilot, JevArena, JevBench or any public score.

## Decision after development, and limit of this test

Only after a complete 466-update receipt and BEST parity should a separate
SELECT-selected package be run once on the existing typed DEV 1,600 and CSS
pilot 1,430, using the previously frozen development screen: composite proxy
at least completed Base `65.33730 + 2.0`, valid/within-budget coverage at
least 99%, Noul ≥ `230/400`, Score ≥ `355/400`, exception-family count at
least Base, CSS pilot task-median macro-F1 ≥ `.520209`. Record failures as
well as any pass. No DEV/CSS/published/FORMAL score is read during this lock
or used to change the budget, step, seed, rule or checkpoint. A development
pass alone does not establish release quality; the project has already
accessed formal labels, so subsequent same-panel formal results are post-key
comparisons and require fresh independent corroboration for an independent
validation claim.

The completed Base 4B scored v3 `53.218`, below own Nox 1.0 `56.470` and
third-party Decider `61.88`; its principal deficits included typed Noul,
Score, exception and CSS transfer. The official-source Posttrained arm is
therefore a narrow falsifiable initialization test, not a comprehensive data
repair. The Decider 4B author reports roughly 742M first-stage tokens and a
later 29,325-row hard/replay stage, versus this arm's 4.194M native input
tokens, a nominal ~177× count gap across non-matched tokenization, sources,
objectives and schedules. The gap is a limitation and hypothesis, not a
causal explanation or an expected Posttrained gain.

## Sealed CPU lock and launch audit

The signed preparer source is revision `f9b405dd5cf92aaf2e333133d44d2e12eeb37c19`,
SHA-256 `f6ec4578dfcc8a59bc4b23e349fbead4c6d9c430347fd27beebe2a94752710d3`.
The first **no-device** invocation stopped before creating a lock because the
operator omitted a container bind mount; its private log SHA-256 is
`36d93d2300d8f0067bda06e7d85db492471df5ca8c0c35887c8d3cc293882990`.
It did not consume a GPU or change the archived admission. A corrected fresh
private directory produced full-arm lock SHA-256
`7a3fe34eef8a4845edb71bead0aa12660993d0b81ad3a99184714f69e37809cd`.
A second fresh no-device process independently verified the lock, source,
TRAIN/SELECT/CAL hashes and exact trainer code. Its private log SHA-256 is
`d43773097880cba1501e1baf0f38df84e1911ff28950a9db8bfc14f5830736b9`.
The local and remote exact Python mirror has 57 files and canonical digest
`8d7809920cd48f38b9f2980f2e888006ae3f07753714ec666a8cec6fff009118`.

The separately signed single-shot
[`launch_qwen35_4b_posttrained_full.py`](../scripts/launch_qwen35_4b_posttrained_full.py)
defaults to dry-run. Its source SHA-256 is
`77d1a8fdd0bc18be559cfa38bdd4d9eea2a39efe65afb9b0bd67b31a939a076c`.
It rechecks the sealed lock, trainer mirror, source and data bytes, pinned
image and selected render-device node before creating one named container.
Its exact-container-ID watchdog is armed before starting training and stops
that container at 10,800 seconds; no other container or process is a cleanup
target. The complete training receipt, logs, checkpoints and SELECT predictions
remain private. The repository training-contract suite completed successfully
before launch. The lock and scripts do not authorize CAL model use, DEV/CSS
evaluation, formal evaluation, upload or release.
