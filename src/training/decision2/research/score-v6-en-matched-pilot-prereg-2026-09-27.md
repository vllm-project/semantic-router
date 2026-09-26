# Score v6 English-only matched continuation: prospective diagnostic

**Status: protocol only. No arm data, materialized source, optimizer update, r2
model response or held-out score has been produced under this protocol.** This
is a one-pair exploratory screen, not approval of the complete v6 TRAIN set,
Chinese-transfer evidence or a release candidate. The v6 Chinese 240 rows and
r2 Chinese 48 rows remain held for qualified independent review. R1 SELECT
remains blocked. A result here cannot be promoted to a JevArena FINAL result.

## Frozen antecedents and hypothesis

The source is the completed Qwen3.8-27B posttrained-source clean-v2 run's
`checkpoint-0000368`, selected before this protocol by its original SELECT.
Its native inference fingerprint is
`d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2`;
the upstream Qwen revision is
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Direct inspection of the
frozen checkpoint confirms `peft-lora/1`, rank 8, alpha 16, dropout .05, a
256-dimensional Decision head and the expected prompt version. The current
trainer cannot start a fresh `decision2` LoRA continuation directly from this
LoRA checkpoint: its fresh initializer does not accept the required immutable
source path. There is no verified merged full copy. The two-arm optimizer is
therefore **blocked** until the separate materialization and parity gate below.

The parent TRAIN, SELECT and CAL file SHA-256 values are, respectively,
`61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`,
`32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`,
and `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`.
V6 candidate SHA-256
`6aee966cc5499a87d2a77241676586c9c9f801b3c662a078daf025f001169f54`
contains 729 English rows in 243 complete 0/1/2 groups plus 240 held Chinese
rows. The source-disjoint r2 gold-free packet SHA-256
`091e3023b84d64131a72b23b90b3eacf837027ed23d58045de801e92a331f683`
contains 192 English rows in 64 groups plus 48 held Chinese rows. Its private
oracle key SHA-256 is
`8e9d1232c1d0db75d7fc1e583d07753e5db6716c85aca39040744a2ec834eda1`;
the key is **not** an input to arm preparation or training. The v6 and r2
review notes bind their prior mechanical and seal-first editorial checks.

Hypothesis: the four new Score mechanisms improve unseen three-level decisions
relative to the same source continued for the same exposure budget on parent
TRAIN alone, without damaging parent Choice/Noul retention. The English-only
test does not cure v6's missing Chinese editorial gate or its disclosed
abstract similarity to one blocked authored diagnostic.

## Materialization and source-identity gate, before optimizer freeze

1. Verify the exact original LoRA package, its source file fingerprint and
   original `BEST.json`/`COMPLETE.json`; recompute the native inference
   fingerprint. Any mismatch stops this version. Use the existing audited
   `training.model.materialize` path once on CPU to merge that LoRA into a
   *new* full checkpoint; leave original source and adapter untouched. Record
   the full-model fingerprint, source/merged file hashes, code hash and PEFT,
   Transformers and PyTorch versions in a private receipt. Available memory
   and disk were checked read-only; this is a capacity observation, not a
   materialization or parity result.
2. On a fixed, gold-free 32-row English parent SELECT input roster, compare
   original LoRA and merged full native predictions using identical tokenizer,
   prompt, max length, dtype, batching and no temperature. Require identical
   input/token hashes, 32/32 same option argmax, and max absolute option
   probability drift at most `1e-4`. Freeze both prediction files and a signed
   parity receipt **before** forming either optimizer arm. A failed parity
   blocks training; repair requires a new source version or protocol, never a
   silent different initialization.
3. Both arms must load the **same byte-pinned merged full checkpoint** and
   attach the same new LoRA topology. The original LoRA is an untouched third
   reference; neither arm resumes its optimizer or alters original weights.

## Data and matched exposure, before the first optimizer step

The CPU-only preparer admits **all 729 English v6 rows**, keeping each of 243
triplets intact. It excludes every v6/r2 Chinese row. It chooses 2,048 common
parent English replay rows, 1,024 Choice and 1,024 Noul, each at most 192
native tokens. The parent-only control receives 729 *distinct* additional
English parent TRAIN rows outside that replay, each 200–512 tokens with at
least 200 parent Score rows. The control's total additional native tokens
must equal the v6 English total **exactly: 220,932**. A bounded deterministic
integer feasibility solver may choose the control IDs once; the chosen IDs,
type histogram, lengths, seed, solver version and both arm file hashes are
frozen before any training. The control is parent-only; no translated,
synthetic Score v6 or SELECT row enters it. Arm A has v6 Score plus identical
parent replay; arm B has matched parent-only rows plus identical replay.

The 729-row control is feasible under those count/token/type constraints in a
CPU-only preliminary calculation. A stricter 200–400-token control cannot
match the 220,932-token budget (even its maximum is below target), so the
preparer uses the prospectively stated 512-token cap. Exact *semantic* token
counts and optimizer batch counts match; dynamic padded sequence lengths and
task-type proportions can differ and must be reported. If padding workload
differs by more than 5% or control Score rows fall below 200, block the run
rather than change the cap after seeing outcomes. This is an exposure control,
not a claim that parent examples are content matched to v6.

Before writing arm files, validate the pinned SHA-256 values, train/select/cal
schema, 0/1/2 v6 group completeness, English-only filters and native 1,024
token cap. Recompute exact ID, group and input separation against parent
SELECT/CAL, and exact plus bounded near-state/full-prompt overlap against the
gold-free r2 packet. The original v6/r2 protected-inventory receipts remain
mandatory; this smaller recheck does not establish semantic independence.
Write private immutable arm/selector manifests with every input, tokenizer,
prompt, source, output and code hash. Keep restricted raw text off public Git,
gist, HF public datasets and model cards.

## Fixed optimizer and blinded selection

After source parity and both arm manifests pass, launch exactly one pair in
isolated directories: one epoch, 2,777 rows per arm, microbatch 1,
accumulation 16, **174 optimizer updates**, no early checkpoint search,
1,024-token cap, BF16 backbone autocast with FP32 head, LoRA rank 8/alpha 16/
dropout .05, AdamW, LoRA LR `2e-5`, head LR `1e-5`, weight decay .01, warmup
.05, cross entropy, seed `20260927`, gradient checkpointing on. Save only
the fixed final step 174, with parent English SELECT588 used by the trainer
for a baseline/final retention receipt. The original parent CAL is parsed
only for lineage isolation; do not fit a new temperature here. Same code,
source, seed, optimizer schedule, precision, batch count and admitted tokens
are mandatory. Record actual tokens, padded tokens, updates, loss, runtime and
hardware in private receipts; unmatched quantities invalidate the causal
interpretation. If a resource error prevents either arm, stop both and amend
the protocol rather than silently reduce context or steps.

Only after both final arm checkpoints and **all gold-free native r2 English
prediction files are sealed** may a separate scorer open the r2 English key.
Score all 192 rows, all 64 independent groups and each of four operations;
invalid/missing answers fail. Use one fixed final checkpoint per arm, so r2
cannot be used for repeated step selection. Keep the original source as the
third scored reference. On parent English SELECT588, separately report Choice
277, Noul271 and Score40, with raw accuracy, normalized Brier and invalid
rate. There is no r2 Chinese inference or scoring in this pilot.

## Frozen advance, stop and rollback rules

Arm A advances to **one** independent typed DEV1600 and CSS pilot1430
assessment only if all of these hold against B on the frozen EN selector:

- r2 English correct count improves by at least 12/192, its paired
  group-bootstrap 95% interval has lower bound above zero, and at least two
  of four operations improve with no operation losing more than 2/48;
- parent English SELECT Choice and Noul each lose at most 2 percentage points
  in accuracy, each normalized Brier worsens by at most .02, and invalid
  answers do not increase;
- 174/174 updates complete in each arm with exact equal raw admitted tokens,
  padding workload within 5%, no truncation, and source/checkpoint/adapter
  hashes matching the pre-step freeze.

If any gate fails, freeze the two arms and the negative report; do not inspect
DEV/CSS/public results to choose an alternate step, seed or mixture. Never
open 15-task formal transfer or authored FINAL gold for this experiment.
A passing diagnostic still cannot justify publishing `dev-2.0-27b`: the
complete v6/Chinese editorial gate, independent authored release panel, full
JevArena same-panel comparison, calibration/robustness, rights and package
parity remain outstanding. Preserve BEST368 and all previous results. The r2
English selector is consumed by this single paired attempt; a new optimizer
search needs a separately authored, source-disjoint selector.
