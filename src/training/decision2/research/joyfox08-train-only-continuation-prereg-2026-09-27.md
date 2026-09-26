# Joyfox 0.8B source-preserving continuation — preregistration

Status: **design and preflight only**. No optimizer step or release claim is
implied by this document. This arm addresses the failed Qwen3.5 0.8B Base and
posttrained clean-v2 controls while retaining the strongest open native 0.8B
comparison as initialization.

## Pinned source and rights boundary

- [Open model](https://huggingface.co/joyfox/Qwen3.5-0.8B-JEV) revision
  `ae7b7040aeff7802f6f2bcfdd27f08a72d5cd969`; 752,655,424 parameters.
  Its model card declares Apache-2.0. The exact backbone, head, tokenizer and
  configuration hashes are checked by `inference.joyfox.verify_release`.
- [Published native inference source](https://github.com/joyfoxai/jev-inference)
  commit `2677b5a3714489847668175de793e2d92fe183f0`, Apache-2.0.
  Use the published 128-dimensional query/key head, one state-plus-question
  row, BF16 and the released 1,024-token no-truncation contract.
- The source card says the checkpoint learned from archived Jev teacher
  distributions; its reported teacher agreement is **not** independent
  accuracy. This inherited lineage must be disclosed if a derivative is ever
  released. Our continuation uses no Jev API or mirror outputs, archived
  teacher distributions, public benchmark answers, or sealed labels as
  supervision. New supervision comes only from the rights-audited private
  clean-v2 TRAIN partition. This arm cannot establish that the source model's
  earlier training corpus was free of overlap with an evaluation panel.

The fixed clean-v2 TRAIN/SELECT/CAL row counts are 7,455/700/700, with SHA-256
`61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`,
`32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`,
`3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`;
rights manifest SHA-256 is
`61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8`.
Partition IDs, lineage groups and canonical inputs must be disjoint. The
existing rights manifest records upstream source licenses and redistribution
limits; neither samples nor labels are committed or placed in the public gist.

## Frozen pilot design

1. Use the source's own `validate_record` and `encode` on every TRAIN row.
   Discard only inputs rejected by its 1,024-token cap or native schema; never
   truncate or rewrite targets. Convert structured state and instructions to
   canonical UTF-8 JSON. Preserve Choice candidate order and keys; sort Score
   levels numerically; use the published default false/true Noul wording.
2. Choose exactly **512** eligible TRAIN rows by SHA-256 rank of the fixed
   seed plus row ID: Choice 128, Noul 192, Score 192. A shortfall in any type
   blocks the arm, rather than silently changing the data mix. Record the
   selected file SHA, source-family/language mix, token lengths, and every
   exclusion count before training. No duplicate group may enter SELECT/CAL.
3. Freeze the released query/key head and source backbone weights. Attach
   rank-8, alpha-16, dropout-.05 LoRA to every full-attention,
   linear-attention and MLP projection identified by the existing exact
   Qwen3.5 `select_target_modules` contract; verify an actual
   finite native forward/backward gradient through at least one LoRA tensor
   before the first optimizer step. Use BF16, native 1,024 max tokens,
   microbatch 1, accumulation 8, **64 optimizer updates**, AdamW
   learning rate `1e-5`, weight decay `.01`, epsilon `1e-8`, eight warmup
   steps then cosine to `1e-6`, gradient norm cap 1, seed `20260927`.
   The loss is hard-label categorical CE plus `.25` times the summed Brier
   squared error over the offered candidates. No replay teacher probabilities.
4. Save steps 32 and 64. Score source step 0 and both candidates on the same
   700 SELECT rows with native probability semantics; invalid and overflow
   answers are failures. Predeclared selector: highest family-macro accuracy,
   then lowest family-macro Brier, then earliest step. A continuation advances
   only if it exceeds source family-macro accuracy by at least `.015`, loses
   no more than `.01` accuracy in any type and has no increase in invalidity.
   A checkpoint at step 0, a failed gradient parity preflight, non-finite loss,
   or a source-identity mismatch blocks downstream scoring.
5. If SELECT passes, freeze checkpoint, package hash and any calibration plan
   **before** evaluating independent typed DEV1,600 and human CSS pilot1,430
   once. The source comparator is the same released 1,024-token native adapter,
   not its 4,096-token ablation. The continuation needs at least +32/1,600
   DEV correct, no more than eight losses in any of four 400-item families,
   nonnegative Noul and Score correct deltas, at least +28/1,430 CSS correct,
   and CSS median task macro-F1 at least `.02` above source. Use group-paired
   confidence intervals, task-level CSS detail, and the existing paired
   option-order/label-renaming robustness measures; paired-simultaneous
   correct rate may not fall more than `.02` versus source. Any failed gate keeps the
   checkpoint private as a negative research result.
6. Only a candidate passing those gates gets one public JevBench-subset231
   diagnostic. It must beat Decision 1.0 Eos's 142/231 and report all 41
   source-native long-input failures; this public score cannot select a
   checkpoint or become an official hidden-set rank. A 4,096-token adapter
   remains a distinct experiment. Full JevArena release and 0.8B publication
   require independent sealed evaluation, broader transfer, calibration,
   robustness, long-input and package parity gates.

The source comparison receipts already exist: native DEV 993/1,600 (Choice
616/800, Noul 205/400, Score 172/400), CSS pilot 510/1,430 with median
macro-F1 `.30119`, public subset 136/231. The 4,096-token no-training
ablation scored 146/231 and is deliberately excluded from the released-source
contract. The prior 0.8B Base and posttrained clean-v2 controls reached only
392/1,600 and 509/1,600 DEV, respectively, and failed broad transfer.

This is a causal *source initialization/low-rank continuation* screen, not a
same-data architecture comparison against those two controls: its train sample
and source head differ. A positive result would justify a later matched-data
ablation before claims about which backbone or head is best.

[Sun et al., 2026](https://arxiv.org/abs/2609.26758v2) found that semantic
option-name swaps can override explicit rubric definitions in Jev and two
other open decision families; **Joyfox was not one of the tested open models**.
Its native encoding includes the option ID in each candidate span, so this is
a concrete transfer risk to measure, not evidence that Joyfox exhibits the
paper's measured effect. The fixed pilot does not add label-swap training; a
separate preregistered augmentation would be required if the robustness gate
fails.
