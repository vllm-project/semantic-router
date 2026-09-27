# Official Qwen3.5 4B Posttrained: prospective GPU admission lock

**Status: HOLD after the first bounded GPU admission phase failed before any
model forward pass. No retry under this lock.** This is the next bounded stage of the
[official-source matched ablation](qwen35-4b-posttrained-vs-base-prereg-2026-09-27.md).
It is an engineering source/numerics check, not a model result. The completed
official Base BEST466 control and its v3 HOLD stay immutable. No typed FINAL,
CSS15, public JevBench, or CAL label is read, and the one-update smoke does
not initialize the later full 466-update arm.

## Immutable bindings

| Bound component | Frozen value |
| --- | --- |
| Direct source | Official [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Full weight shards | SHA-256 `26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61`, `cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188` |
| Rights-clean v2 | TRAIN 7,455 SHA-256 `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`; SELECT 700 `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`; CAL 700 `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a` |
| CPU full-shard/native-input audit | SHA-256 `0c45417d4cc8f3e4c32edc55aec3c230e0922476500acc1fabc3b5c2151ac82b`; 738 tensors, 4,659,865,088 stored parameters, 8,855/8,855 Base/Posttrained token sequences equal, TRAIN 4,194,465 tokens, max 6,596 |
| Runtime image | Exact image ID `sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`, the same image ID used in the completed Base control |
| Admission runner | [`admit_qwen35_4b_posttrained.py`](../scripts/admit_qwen35_4b_posttrained.py), SHA-256 `9e7e715ef5bf6404c007077e633534f2cfb4354ac0b17d03e0896239d56dce90` |
| Private immutable lock | SHA-256 `24dc3a714e0a2cd9fc8c7568f0aaada5fa13430a3c9f35c3e98c48e1c66cfb6d`; `LOCKED_NO_GPU` |

The source is a **clean official snapshot** downloaded by the HF CLI at the
pinned revision. A prior download directory mixed official files with CPU
audit JSON, which would have changed the trainer's `source_fingerprint`.
That directory is not a training source. The clean snapshot's ten relevant
root files are hash-bound in the private lock; Hugging Face cache metadata
for all downloaded files attests the pinned revision. The lock also binds the
exact trainer, renderer, data validator, source-fingerprint code, LoRA,
objective and scheduler hashes and the private input/output paths. The
lock's parent is mode 0700 and the lock is mode 0600. A separate no-device,
network-disabled container reopened it with the exact image ID and returned
`CPU_LOCK_VERIFIED` with the same lock SHA-256. No GPU was exposed or used.

## Ordered GPU stages, only after review

The [`runner`](../scripts/admit_qwen35_4b_posttrained.py) accepts a private
lock and refuses changed source/data/code or a repeated output directory.
The operator must run the container by **image ID**, mount only the frozen
source and data read-only, expose exactly one BF16 GPU, and keep all logs and
SELECT predictions private. Check the specific device reservation before
launch. All stages use the same seed `20260926`, 8,192-token no-truncation
native Choice/Noul/Score renderer, fresh 256-dimensional head, rank-16
LoRA/alpha32/dropout .05, CE+0.5 Brier, microbatch1/accumulation16, LoRA
LR `1e-4`, head LR `2e-4`, AdamW decay .01 and FP32 head/loss with BF16
backbone computation.

1. **`zero-a`:** fresh source, no optimizer update, all SELECT700 rows,
   Choice320/Noul290/Score90. Require exact row IDs, type/order, source
   fingerprint, finite distributions and complete private receipt. Hard cap
   **540 seconds** including preflight. Stop on source/code/data mismatch,
   unsupported type, invalid probability, OOM or timeout.
2. **`zero-b`:** a second independent fresh process with identical source,
   head seed, data, code and runtime. Same cap and validity. Neither start
   may be substituted by a checkpoint or the Base source.
3. **`compare-zero`:** CPU-only, row-wise pair comparison of the two sealed
   starts. Require zero categorical changes, per-row maximum option
   probability drift p99 ≤`.005` and absolute maximum ≤`.02`, identical
   prompt/token hashes and exactly 320/290/90 typed records. A failed gate
   stops the arm; no threshold or backend substitution.
4. **`one`:** only after a PASS zero gate, a third *fresh* official source
   start with exactly **one** optimizer update. Require finite loss and
   gradient norm, nonzero token count, durable checkpoint, 700-row SELECT
   before/after, exact source fingerprint and COMPLETE step=planned=1.
   Cap **360 seconds** including preflight and save. The smoke uses
   `--max-steps 1`, so its scheduler horizon differs from the prospective
   466-step training arm; it tests numerical feasibility and reload, not a
   matched one-step learning effect.
5. **`reload`:** a separate fresh process loads the one-update LoRA/head
   checkpoint from the frozen source and runs the first fixed 32 SELECT
   rows at batch size 2. Require zero category changes, p99 drift ≤`.005`
   and max ≤`.02` against same-batch step-one in-memory predictions.
   Cap **180 seconds** including preflight and source load. Any miss is HOLD.

`zero-a` and `zero-b` cap combined at 0.30 GPU-hour, and `one` plus
`reload` at 0.15 GPU-hour; the full admission cap is **0.45 GPU-hour**.
The runner writes private immutable phase receipts and retains failures.
Subprocess failures receive a failure receipt; a postcondition or reload
failure leaves its private outputs without a PASS receipt and must be
recorded separately before any new protocol.
No failed phase is retried under this lock. The first draft lock made while
hardening the runner is retained privately but **superseded** by the SHA
above; it has no GPU runs and cannot admit any phase with the final code.

Passing admission permits a separately reviewed and signed 466-update
development arm at the already fixed data/token budget and SELECT-only
selector. It does **not** authorize running the full arm automatically,
reading formal labels, retuning calibration, or publishing 4B. The
[Base v3 transfer failure](qwen35-4b-official-best466-v3-formal-hold-2026-09-27.md)
and [2B Posttrained Score collapse](qwen35-2b-official-dual-source-full-result-2026-09-27.md)
remain substantive risks.

## Bounded execution result: HOLD

The approved `zero-a` phase exited nonzero after **13.205 seconds** inside the
runner. The private failure receipt and complete console log have SHA-256
`837723a9c502167182c7894abb982e0f45b79ea95bb53a1ffd42aaae6571b023`
and `aaac74af32b618857ba1de8a008361e22236790b70b4980c321997f2e37cceb6`.
The private HOLD decision, bound to the unchanged lock, has SHA-256
`a299ebe2974694f2f554b47313f1aa0478319c3f7eb0bd8f7a7d8e4377d38241`.
The reserved-card accounting proxy is **0.003668 GPU-hour** for the runner's
measured phase; container startup is excluded, and the complete reservation was
below 0.01 GPU-hour. There were no model forward passes, predictions, optimizer
updates, or trained weights. `zero-b`, `compare-zero`, `one`, and `reload` were
not started. No formal or public benchmark label was accessed.

The stack trace ends at `DecisionModel.from_base` calling
`AutoConfig.from_pretrained`, which rejects `model_type=qwen3_5`. The official
Base and Posttrained configuration files have the **same SHA-256**
`ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`,
the same `model_type=qwen3_5`, and the same
`Qwen3_5ForConditionalGeneration` architecture. The archived successful Base
zero-step, smoke, and initial full-run containers used this exact immutable
image ID and the same `decision_model.py` SHA-256
`ee3db820db73011d60e99b98c3067e33d85c8f4cbae08b53ea0604869ffba0ca`.
Their command invoked `/usr/bin/python`, whose Transformers version in the
image is **5.17.0**. The admission runner invoked its container entrypoint
`/opt/vllm-sr/venvs/vela/bin/python` and then reused that `sys.executable`
for training; that interpreter has Transformers **4.57.6** and does not support
Qwen3.5. Thus the frozen runtime invocation, rather than a Posttrained weight
or architecture difference, caused this admission failure. The source arm has
not yet been tested.

The current lock cannot admit any further phase. A future corrected attempt
would require a revised runner, a CPU-only pinned-interpreter source-load
preflight, a new lock and independent review before GPU use. It must not
relabel this failed run or reuse its zero-step output.
