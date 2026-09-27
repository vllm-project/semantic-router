# Gemma 4 text LoRA: one-step TRAIN-only numeric and reload gate

This is a **prospective preflight**, not a Decision benchmark or training
campaign. It does not authorize a GPU launch on its own. The prior official
source-versus-fresh-LoRA 32-prompt zero-step comparison passed with hidden
drift 0.0. The next question is whether the frozen text-only q/o LoRA and
shared Decision head support one finite gradient update and exact package
reload on the actual training input format.

## Frozen source and data

- Source: official `google/gemma-4-26B-A4B-it` revision
  `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`; configuration SHA-256
  `ed0c1eb3633de771906e9ba004a44cc5635bcc06ee2062077c3d2e88a50707d3`,
  tokenizer SHA-256
  `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`,
  two shard SHA-256s
  `1127684971bbca40465435a5cad69d67ad603bf5e61c6dfd5561fae4a3bcfdb3`
  and `aab47033e1e8a492ef8e581efae1cf36478d0433567e7729b3c1728bc8970db7`.
  Loaded text parameters: 25,233,141,760. Source keys, shapes, tied head and
  stored buffers must again pass the strict loader check.
- Data: one row of the already authorized rights-clean v2 **TRAIN** split,
  full file SHA-256
  `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`.
  The private lock binds its exact row/group ID, raw line hash, prompt hash,
  token-ID hash, label index and metadata. It is a three-level, Chinese Score
  row of 268 source tokens. The row ID and text remain in the private lock
  and private TRAIN file. No SELECT, CAL, DEV, JevArena, JevBench or other
  evaluation row or label is loaded.

The source uses only the Gemma text decoder. The 60 exact q/o LoRA targets
use rank 8, alpha 16 and dropout .05; all official weights, MoE router,
experts and vision modules are frozen. The random, shared 2,816→256 Decision
head is trainable. Its three logits are computed from the Score row's option
endpoints and final query using the same segmented-option protocol as the
other Decision training arms. This one row cannot establish Choice/Noul
optimizer behavior or any task quality.

## One-update cell and acceptance rules

Run on one freshly free isolated ROCm GPU with BF16 source, FP32 decision
head, PyTorch AdamW, zero weight decay, LoRA learning rate `1e-4`, head
learning rate `1e-3`, seed `20260927`, no accumulation and exactly **one**
optimizer update. Max input 4,096; the frozen row must admit 268 tokens with
no truncation. Cross entropy uses only that TRAIN label. Clip global gradient
norm to 1.0 after recording the unclipped norm. No scheduler, replay,
calibration, checkpoint search or metric-based selection is permitted.

The cell passes only if the lock, source and row hashes match; there are
exactly 120 trainable LoRA A/B tensors plus the expected head, with no frozen
parameter in the optimizer; the forward yields finite `(1, 3)` logits and
loss in `(0, 1e6)`; every gradient is finite, the unclipped norm is in
`(0, 1e6)`, all 60 LoRA B tensors receive nonzero gradients and change after
the step, and at least one head tensor changes. Save only the adapter and
head to a new private package. Then unload the first full model, independently
reload the same official source and saved adapter/head, require exact adapter
and head tensor equality, finite native logits and maximum absolute logit
drift ≤`1e-3` against the pre-save post-update logits. Missing weights,
unsupported backward ops, OOM, nonfinite values, hash mismatch, numerical
drift or timeout mean **STOP**; do not vary a threshold or select another
checkpoint. The output is a private technical receipt, not a publishable
Decision weight.

The external wall-time cap is 20 minutes, at most **0.3334 conservative
GPU-hour**. Estimated use is 6–12 minutes or 0.10–0.20 GPU-hour; this is an
estimate, not an observed result. Check GPU occupancy and exact code/data/
model/runner hashes immediately before launch. The runner records exit code,
time, conservative GPU-hours and hashed private logs even on failure, removes
its task container and releases the device. No retry is included in this
preregistration.

The final private lock SHA-256 is
`88faa7837b6256f8f455c1b6668b8de55d644aac64d1932c3a4a9f0a0e03f798`.
The exact one-step probe and private runner SHA-256s are
`f933a82d17030bb6267435e4183deccfa277ccdd1963ab559fde782cb4668bfc`
and `c5c45adc185cd62952232fa56838989b6e17d02bcfbfd41a4640115ae1925f9b`.
The no-device Docker admission and full private-lock check passed on the
268-token TRAIN row; the private admission receipt SHA-256 is
`343a99c82f138fb202ee017b76d25eace02ca435f4eed812149a78f2ad2b91ce`.
Three focused CPU tests pass. A pass admits only a future
matched-data development ablation after a separate preregistration and
review; it does not justify formal JevArena/JevBench access or publication.

## Approved one-step result

After separate approval, the single locked cell passed with exit status 0.
It finished in **94 seconds = 0.02611 conservative GPU-hour**, below the
20-minute cap. The one pinned TRAIN row produced finite `(1, 3)` logits and
loss. The unclipped gradient norm was 16.054; all 60 LoRA B tensors had
nonzero gradients and changed after the one optimizer step, and 10 head
tensors changed. The private adapter/head package reloaded onto a second
independent official source load with exact tensor equality and maximum
same-input native logit drift **0.0**. Peak allocated GPU memory was
54,173,890,048 bytes; the selected device returned to its pre-run baseline
and the task container exited.

The private result, timing and log SHA-256s are respectively
`bd934a35ff8c4893661f1d59debccc3ce07bea57c847c3c74f06baa9e6044351`,
`26fceb8bccc3a5fed309487ee93871764b10de2910ee272c99489bf63e33c197`,
and `7220d9a3e45a5061bbb38224f7f01e71a97df918e1a16bb2704d0497333b45a7`.
The loss change on the **same TRAIN row** is a numerical smoke observation,
not transfer or quality evidence. SELECT, CAL and all evaluation labels
remained unread. This result admits the native Gemma training mechanics for
a separately preregistered, matched development arm only. It is neither a
Decision 2.0 score nor a publication candidate.
