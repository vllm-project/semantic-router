# Gemma 4 ~26B: prospective full Decision development arm

**Status at freeze: CPU lock and audit PASS; GPU execution was pending separate
review.** The subsequent one-cell execution and HOLD decision are recorded in
the [signed result](gemma4-full-development-arm-result-2026-09-28.md).
This implements the staged arm proposed in the
[Gemma/Qwen development plan](gemma4-qwen27-development-arm-prereg-2026-09-27.md).
The preceding single-GPU [three-type long-input gate](gemma4-long-train-gate-prereg-2026-09-27.md)
passed at 4,090/4,076/4,044 tokens with exact package reload. Those three
TRAIN updates measured feasibility, not decision quality. This prospective
document did not authorize the full run or a formal evaluation.

## Frozen source, task path and cohort

- Official direct weight start: `google/gemma-4-26B-A4B-it` revision
  `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`. The private lock binds
  the exact configuration, tokenizer and both model-shard hashes. The loader
  requires a complete indexed state, tied language head and actual loaded
  parameter accounting. **25,233,141,760 loaded text parameters**;
  25,805,933,872 when the untouched vision component is included.
  Header accounting separates 22,837,985,280 packed expert parameters from
  2,395,156,480 other text parameters. With the pinned top-8-of-128 expert
  routing, the nominal per-token **active text parameters** are
  **3,822,530,560** (other text + 8/128 of packed experts). This is an
  architectural active-count estimate, distinct from loaded/storage size;
  it is not the Pareto size axis or a measured runtime FLOP count.
- Text-only Decision adapter: official text decoder with 60 q/o attention
  LoRA targets, rank 8/alpha 16/dropout .05, and a fresh shared 2,816→256
  dynamic-option head. Exactly **6,540,800** adapter/head parameters train;
  vision, source embeddings, MoE router and experts remain frozen. Source
  versus fresh adapter identity has a separate 32-input zero-step PASS
  receipt. No third-party Decision model initializes this arm.
- Native task path: `state` plus typed `instructions` and dynamic `options`,
  covering Choice, Noul and Score with calibrated-probability-capable native
  logits. The official Gemma BOS is prepended to the shared segmented
  option/query prompt; inputs are never truncated. This trains the Decision
  head and adapter, not a chat answer generator.
- Data: rights-clean v2 TRAIN SHA-256
  `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`,
  SELECT SHA-256
  `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`.
  TRAIN/SELECT source, group and exact-input isolation is recomputed. CAL is
  not mounted or opened; its previously completed isolation audit remains a
  separate control. No DEV, JevArena, JevBench, Decision Index or public
  benchmark file is an accepted argument.

The no-GPU admission independently recounted the exact common no-truncation
cohort under pinned Gemma and official Qwen tokenizers: **7,287 TRAIN rows**
(Choice 3,804, Noul 2,982, Score 501), **3,620,578 Gemma unpadded tokens**,
and the previously signed ordered-row SHA-256
`3d6f96168e38761b45630e9a8a61c60c85b19811df3049e715696acae0edd30b`.
The schedule is one seeded epoch, microbatch 1, accumulation 16, no replay,
exactly **456 optimizer updates**. SELECT has 700 rows (Choice 320, Noul 290,
Score 90); all Gemma inputs fit and its maximum is 228 tokens. The SELECT
readout therefore controls checkpoint choice but cannot itself establish
long-context transfer. Each update uses the precise sampled TRAIN rows,
native option count, valid-candidate CE + 0.5 Brier, AdamW with LoRA LR
`2e-5`, head LR `1e-4`, weight decay `.01`, cosine decay and 5% warmup.
Non-reentrant activation checkpointing and BF16 backbone/FP32 head are fixed.

## Immutable development gates and choice

Save new, private adapter/head/optimizer checkpoints only at updates
**16, 64, 128, 256 and 456**. At update 16, inspect numeric health and the
observed throughput projection against a **24 wall-hour / 24 conservative
GPU-hour total ceiling**. If the projected remaining token processing plus
10% margin exceeds that ceiling, stop at the partial checkpoint. No silent
budget extension, row truncation, model switch or new seed is allowed.

Only at 64, 128, 256 and 456, read the exact SELECT 700 under native Gemma
option logits. All rows contribute; absent/invalid predicted choices fail,
while nonfinite or non-simplex probabilities stop the arm. Record per-family
accuracy/Brier, type accuracy, and Score predicted-level distribution. At
256, stop for futility if the **best** milestone SELECT family macro accuracy
remains below `.70`; also stop at 256 or later for Score collapse when fewer
than three levels are predicted or one level occupies more than 95% of the
Score subset. Any nonfinite TRAIN loss/gradient, absent trainable gradient,
gradient norm outside `(0, 1e6)`, OOM, hash change or wall timeout stops
without automatic retry.

Among completed predeclared milestones, choose the highest SELECT family
macro accuracy, then lower normalized Brier, then the earlier step. A selected
macro of at least `.794` permits consideration for a **separately authorized**
calibration and independent diagnosis. This is a compute-allocation rule, not
a release requirement or a result. A completed or stopped arm never gains
permission to read formal labels, publish a model or reuse partial weights
without another decision.

## Honest comparison and resource envelope

The completed official Qwen3.8-27B reference used 7,324 TRAIN rows,
3,579,176 Qwen tokens, 458 updates and 63,627,776 trainables. This Gemma arm
uses the same upstream rights-clean v2 pool and near-identical optimizer
settings but **different admitted rows, tokenizer, exact token budget,
initialization and trainable capacity**. Within the same 7,287-row cohort,
Gemma tokenization has 5.47% more tokens than Qwen. This is a comparative
development screen, not a strictly matched causal ablation or a rerun of the
Qwen control. The eventual external ~27B peers must be scored on the same
JevArena/JevBench panel before claiming a relative result.

The prospective site is one previously used 8-GPU authorized node with the
official source snapshot, frozen private data and pinned offline image
already present. A current read-only check found all GPUs idle, about 1.4 TB
free on the local-output filesystem, and 33 TB free on the data filesystem;
the specific candidate device is physical ordinal 3. These are **observed
capacity**, not a standing reservation. Recheck fleet, GPU process/VRAM,
storage, source/shard hashes and image immediately before any separately
approved launch. Only task-owned containers/files may be touched. The private
launcher has a hard timeout below 24 hours, creates a new output directory,
captures exit status and GPU-hours even on failure, and never retries.

## CPU preparation receipts

An offline, network-disabled container with **no GPU devices** generated the
owner-only lock and independently recomputed admission/schedule against it.
Both `prepare-lock` and `cpu-audit` passed. The private lock binds source,
both tokenizers, TRAIN/SELECT checksums, 32-input identity and 4K gate
receipts, exact code, private launcher, image, cohort/order, optimization,
SELECT steps, stop/selection policy and device. It contains no benchmark
label or prompt. Raw data, exact private paths and model predictions remain
private. The public receipt below supports later audit without distributing
restricted rows.

| Item | SHA-256 or identity |
| --- | --- |
| Private prospective lock | `20b7524b0d6dceeb57237e312fc8d341b31b962bca5b177688ffb7be3b1798c3` |
| Private offline launcher | `fcd5952cfc031cd644026ae919cb7402503dea9a7eb78ddaf33c01be5a6731b7` |
| Tracked Gemma development runner | `eb593ecee5082f56895832af7069b8dbb4a99da28835bf7a90df0dfa4e34f1b0` |
| Tracked pure schedule/stop policy | `8a34c34149b3f6dcf28b6be9fc567d9cab45a2a614eee4b50ef6c59499970c4f` |
| Pinned image ID | `sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54` |
| Ordered training-step index digest | `19cade6feda817c1cca89d2e92a5417e889291ea7cc5a3b6dc866629be6767c7` |
| Per-step token-count digest | `568f8f152c562e55c1632de57741c2778587f88cd03ed7897d726701561c0f57` |

The model, source and data hashes must match before a future GPU run. The
lock's exact code digests are retained privately; changing even a comment
invalidates it and requires a new signed, reviewed lock. No full development
optimizer update had been executed under this lock when it was frozen.
