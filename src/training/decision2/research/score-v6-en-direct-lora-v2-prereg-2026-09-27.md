# Score v6 English matched continuation v2: direct LoRA prospective protocol

**Status: preregistration only. No v2 code, model inference, optimizer update or
selector scoring has run.** V1 stopped on its separately sealed native BF16
parity failure; it must not be relabeled a pass. The v1 public-safe negative
receipt SHA-256 is
`00a870e8c14eee007910b0e273c6632e07a72110f3ea413b0a8cbe5e77e95fb5`.
V2 changes only the initialization path: continue the exact frozen PEFT LoRA
adapter itself with a **fresh optimizer**, without merging into the base or
stacking a second adapter. This is a new source/loader protocol, not a silent
v1 amendment. Neither v1 nor v2 addresses missing Chinese review or licenses
a release claim.

## Immutable source and data

The start is the completed 27B clean-v2 `BEST368` PEFT checkpoint at native
inference fingerprint
`d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2`,
loaded against the same immutable Qwen3.8-27B revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` and its pinned source-file
fingerprint. The adapter is rank 8, alpha 16, dropout .05, with its selected
FP32 Decision head. No original optimizer state, alternate step or merged
checkpoint is admissible. The v1 materialized full model is retained only as
negative diagnostic evidence and is excluded from v2 initialization.

The v1 CPU preparer froze exactly the same arm data for this prospective
contrast: private arm-manifest SHA-256
`b10eae10192096c2dcf52af60ad6805c5180abd00f7ff2a5af52a4f5dadc736c`;
Arm A TRAIN SHA-256
`6a6ef7d3f2eac2a63cdd61cd806275e67c0aa772e78b45bf0953200f2a776235`;
Arm B TRAIN SHA-256
`1c705c9a8271ce2e526b6bc91affe18d463a52bb86ce99b6b1bcb5007d467b41`.
Each has 729 treatment/control plus the same 2,048 parent Choice/Noul replay
rows, 2,777 total; the treatment/control sections each have 220,932 native
tokens, padding differs by only 0.02865%, and the parent-only control has 203
Score rows. The same parent English SELECT588 and CAL files are SHA-256
`e41774cfc6f1dcea58fa0baa3c8a0cef3940af36fcd464f2b0d5ca01598a95ce`
and `ff6d80d04a07427939ac1ec4fe807e57fdaa921ff90e0abb50a033094cc485a8`.
The r2 gold-free English prompt file is SHA-256
`3ad89c37170de17ccda63f149aaf14a7d4112e75db01741a3ef1b4b2c161b726`;
its 192 English rows/64 groups and 48 held Chinese rows remain unscored.
The answer key is never a loader or training input. No new rows, replay
substitution, translation or changed overlap rule enter v2.

## Loader implementation and zero-step source gate

Before training, add an explicit `decision2-lora` fresh initialization mode to
the trainer; the current `decision2` mode refuses a source path and cannot
load a LoRA as a fresh source. The new mode must require the immutable base
`--source-path`, verify the full original checkpoint plus base fingerprint,
and call `DecisionModel.from_checkpoint(..., source_path=...,
trainable_adapter=True)`. It must reuse the existing LoRA tensors and selected
head, skip `attach_lora`, and initialize a new AdamW optimizer and schedule
without reading the prior optimizer state. Reject a full checkpoint, nested
adapter, mismatched PEFT topology/version, incomplete source, unexpected
trainable non-LoRA backbone tensor, or a source whose fingerprint differs.
Record the original adapter-plus-base fingerprint separately from the
base-source fingerprint in provenance and preserve the base contract for
exact resume. The same signed loader code/container is mandatory for both
arms. Tests must cover source mutation, trainability, initial tensor/head
equality, no second adapter, and exact resume before any GPU optimizer run.

After exact package transfer and hashing on the experiment host, generate
read-only native BF16 predictions from that original source and from each
arm's zero-step in-memory loader on the previously frozen 32-row gold-free
roster, SHA-256
`193404fb2ed3905cbb9e34379400a2f33a40d86aaae971940454c6fe71163bc5`.
Use one physical GPU type, container, tokenizer, prompt, batch, code,
temperature 1, no truncation and `model.eval()`. The two arm starts must have
identical byte-pinned adapter and head tensor digests. Each zero-step result
must match the same-host original source on **32/32 argmax** with maximum
absolute option-probability drift **at most `1e-4`**. Bind all input/token,
prediction, adapter, head, source-file, container/software and GPU hashes in
private receipts before the first update. This is a new direct-LoRA start
gate; the failed v1 merged-model comparison remains failed. If either start
gate fails, stop v2 and do not adjust tolerance or source silently.

## Frozen two-arm optimizer and blind comparison

Only after both zero-step receipts pass, run one matched pair in isolated
directories. Each arm receives one epoch of its 2,777 fixed rows, microbatch
1, accumulation 16, **174 updates**, max length 1,024 with no truncation,
BF16 backbone autocast and FP32 head, rank-8/alpha-16/dropout-.05 existing
LoRA, AdamW, LoRA LR `2e-5`, head LR `1e-5`, weight decay .01, warmup .05,
cross entropy, seed `20260927`, and gradient checkpointing. Save only fixed
final step 174. Parent English SELECT588 provides a baseline/final retention
receipt, never an early checkpoint search; CAL is lineage-only. Record exact
data, model, source, optimizer, step, raw/padded-token, code, precision,
hardware and runtime hashes for both arms. A mismatch or resource failure
stops the pair; do not silently reduce context, steps or data.

After **both final checkpoints and all gold-free native r2 English prediction
files are sealed**, a separate scorer may open the r2 English key once. Score
all 192 rows, 64 groups and four operations, with invalid/missing answers
counted as failures, and keep the original BEST368 as a third reference.
Preserve the v1 advance rule unchanged: Arm A must gain at least 12/192 over
B, have positive lower bound for its paired group-bootstrap 95% interval,
improve at least two of four operations with no operation losing more than
2/48, and keep parent English SELECT Choice/Noul accuracy losses at most two
points each, normalized Brier worsening at most .02 each, and no increase in
invalid responses. All 174 steps, equal admitted raw tokens, padding within
5%, zero truncation and all source hashes must also pass. If any criterion
fails, seal a negative result without searching another step, seed or mix.
Passing allows at most one independent typed DEV1600 and CSS pilot1430
assessment; it is not a release victory or Chinese-transfer claim.

## Release and rollback boundaries

The present publication builder accepts only merged full weights, whose
BF16 parity failed. A future passing v2 adapter requires a new
adapter-preserving bundle/runtime with the immutable base revision and all
source-file hashes, explicit native loader instructions, final package versus
scored-model output parity, and parameter accounting that includes the full
base. Do not package the lossy merged model or transfer the source LoRA's
scores to it. Independent authored release data, full JevArena same-panel
evaluation, calibration, robustness, multilingual review and rights gates
remain separate. V1's frozen artifacts and BEST368 are the rollback points;
neither is overwritten by v2.
