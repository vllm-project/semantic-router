# 0.6B Choice/Score diagnosis and next source gate

**Decision: HOLD new GPU training.** The next experiment should change
source-quality and typed training exposure while holding the official Qwen
initializer and shared readout fixed. No new source has passed that admission
gate, so there is no defensible frozen replacement roster or executable
training arm yet. This note is an internal research decision, not a product
model card or a model result.

## What the current comparison establishes

The official-Qwen 0.6B full466 package scores 38.520 on the post-key same-panel
JevArena v3 against own Kai1's 35.938 and Bosun v3.1's 38.524. The paired
interval for its difference from Kai1 is [-2.032, +7.846]. Its human-transfer
median improves (.4801 versus .3569), but typed Choice is **109/800 versus
277/800** and Score **80/400 versus 98/400**; Noul improves 458/800 versus
404/800. On typed DEV all 400 Score decisions selected level zero. The
post-key v3 and open DEV observations diagnose a real weakness; they do not
establish that any proposed remedy works. See the [sealed v3 result](qwen3-06b-official-full466-v3-postkey-result-2026-09-27.md)
and [development result](qwen3-06b-official-base-full466-development-result-2026-09-27.md).

The current model is a causal official-Qwen backbone with candidate-endpoint
states and a final global-query state fed to a shared bilinear/MLP head.
Its native renderer scores the complete typed `state`/`instructions`/options
input with no truncation, returning a distribution over runtime options,
not chat text. This must retain the System One Choice, Noul and ordered Score
semantics; the [API reference](https://docs.typesafe.ai/api) describes Choice
probabilities, Noul's yes probability and Score's probability-weighted level.

## Frozen TRAIN exposure: rows conceal the largest imbalance

The [aggregate-only CPU auditor](../training/data/audit_small06_train_balance.py)
recomputed the **exact native rendered token count** of rights-clean v2 TRAIN
using the pinned official Qwen3-0.6B tokenizer. It read TRAIN only, no SELECT,
CAL, typed DEV/FINAL, human-transfer or public benchmark rows. It verifies
TRAIN SHA-256 `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`,
tokenizer JSON SHA-256 `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`
and the production renderer's parsed-function hash before counting. The
aggregate receipt is SHA-256 `4bc3e9f2a5bdea8bcfea64670dddac9fbdbb902d89c390c10c461ea725c3021c`.
It contains no row text or identifiers.

| TRAIN slice | Rows | Native tokens | Token share |
| --- | ---: | ---: | ---: |
| All, 5,392 independent groups | 7,455 | 4,094,489 | 100% |
| Choice | 3,908 | 2,482,423 | 60.6% |
| Noul | 3,031 | 1,322,873 | 32.3% |
| Score | **516** | **289,193** | **7.1%** |
| Three-level Score | **102** | **72,973** | **1.78%** |
| Stage-4 composition source, all types | 1,987 | 3,124,797 | **76.3%** |
| Original human GoEmotions source | 2,800 | 286,140 | 7.0% |

Three-level Score appears in only three programmatic families: dense table
17 rows/51,655 tokens, ordinal 53/12,238 and prior logic replay 32/9,080.
The level mix is 102 three-level, 59 four-level, 211 five-level, 58 six-level,
47 seven-level and 39 eight-level rows. Thus even the 516 Score row count
overstates exposure to the three-level rubric used in typed DEV/FINAL.
Stage-4 composition alone supplies 1,842,610 Choice, 1,033,284 Noul and
248,903 Score tokens. This is a token/source distribution fact, not evidence
that any particular stage-4 item is invalid.

Declared English is 6,085 rows/2,449,936 native tokens (59.8% of tokens),
Chinese 1,370 rows/1,644,553 tokens (40.2%). Choice/Score token shares by
language are respectively English 1,532,326/142,730 and Chinese
950,097/146,463. There are no other declared languages. The Chinese token
share is substantially larger than its row share, and neither number proves
cross-source multilingual transfer. Audit output includes each type/language
cell so future replacement exposure can be compared under the same renderer.

## Kai and competing 0.6B training are not matched controls

The pinned [Kai1 benchmark document](https://huggingface.co/llm-semantic-router/Decision-1.0-Kai-0.6B/blob/7185f514f54b8f93c55998b1e8f9c5cc67f0d029/BENCHMARK.md)
reports 21,367 all-type training rows: 16,529 shared TRAIN and 4,838
historical Noul replay rows, spanning 18 languages. Its
[source attribution](https://huggingface.co/llm-semantic-router/Decision-1.0-Kai-0.6B/blob/7185f514f54b8f93c55998b1e8f9c5cc67f0d029/TRAINING_ATTRIBUTION.md)
identifies original NLI, BoolQ, QA, intent, multilingual, human ordinal and
programmatic families, but does **not** publish exact row IDs, per-source
token exposure or overlap against today's protected transfer panel. The final
Kai release combines its updated Choice/Score paths with the restored old
Noul path after all-type training harmed BoolQ; see its pinned
[model card](https://huggingface.co/llm-semantic-router/Decision-1.0-Kai-0.6B/blob/7185f514f54b8f93c55998b1e8f9c5cc67f0d029/README.md).
Old Kai training cannot simply be replayed as an assumed clean matched arm.

Bosun v3.1 reports an official general Qwen3 start, rank-16 LoRA, learned
decision-token slots and roughly 130,000 training rows (80,000 base and
50,000 additional), but its exact training rows are not public; see the
[author's account](https://huggingface.co/blog/Hanno-Labs/decisionbench-bosun-v3-1)
and [0.6B model card](https://huggingface.co/Hanno-Labs/bosun-v3.1-0.6b).
The approximately 17-fold row difference from our clean-v2 roster and the
late-token readout both vary. Its performance cannot isolate either cause.
The [Kev primary repository](https://github.com/jaredpalmer/kev/blob/main/README.md)
similarly motivates multi-source and targeted incremental data, not importing
its train rows without source and protected-panel review.

In our own matched-data tests, type-separated heads, candidate interaction,
official posttrained initialization, Score RPS, external soft-teacher and
projected gradients failed their frozen development gates. Those results
reject the tested implementations, not all possible late-token or encoder
topologies. The QuALITY and RACE wholesale Choice source pilots failed
answer-blind evidence-necessity review (7/24 and 6/24); the Score-v6
replacement failed a blind shortcut review. Repeating these sources or
putting another head on the same scarce Score exposure has low information
value. See [release status](release-status-2026-09-28.md) and the individual
arm receipts linked there.

## One prospective experiment, conditional on a CPU source gate

The **only proposed next 0.6B arm** is a *data-only*, matched-budget Choice
plus three-level Score replacement against the archived official-Qwen Base
shared-head control. The initializer, native renderer, head, objective,
optimizer, seed, full-input cap and 466-update budget stay fixed. All
human-source and Noul rows stay byte-identical to protect the
observed transfer gain. The exact prospective removal pool is 256 stage-4
Choice singleton groups and 384 non-three-level programmatic Score rows:
264 stage-4 singletons plus 60 of 75 complete targeted-source pairs. This
preserves the original 102 three-level Score rows and the remaining 30
targeted-source Score rows. The 640 removed rows would be replaced with
admitted source-disjoint groups, keeping 7,455 row
exposures and 4,094,489 native tokens within ±0.5%. No group is split across
TRAIN/SELECT/CAL or protected roles. This contrast tests source/type exposure
separately from readout topology. It does **not** prescribe use of Kai,
Bosun, Kev or any third-party decision weights as initialization.

Before freezing any row roster or allocating GPU, the CPU gate must pass:

1. Recover original source ID, version, terms, native source group and
   answer rubric for every candidate. First try to recover Kai1's exact
   training roster for provenance and overlap diagnosis; do not assume its
   public attribution document is sufficient to admit its rows. A new source
   is acceptable if Kai records cannot be recovered.
2. Require at least **256 independent Choice groups** across two mechanisms
   and **128 independent Score groups** across two mechanisms. For Score,
   three answer levels must be supported by independent evidence changes;
   three related variants of one group count as one group. Build a whole-group
   candidate of exactly 256 Choice and 384 Score row exposures, with balanced
   correct-option position and Score levels. Original human transfer examples
   and Noul exposures remain untouched.
3. For each source/mechanism, hash-freeze a stratified **24-group**
   answer-blind packet before review. Require at least **18/24** cases where
   the decisive evidence is necessary (the answer must change or become
   indeterminate when it is removed), at least **22/24** independently
   reproducible labels, zero unresolved ambiguity and no systematic position,
   template, timestamp or eligibility shortcut. A failed source contributes
   zero rows; do not filter its packet post hoc to rescue the frozen arm.
4. Audit source rights and redistribution scope, exact native lengths,
   original IDs, group boundaries and all TRAIN/SELECT/CAL/protected/public
   prompt roles for exact, normalized, near and suspected semantic overlap.
   Any protected match or unresolved provenance is HOLD. The complete-input
   cap is 8,192 native tokens; no truncation is allowed. Keep original text,
   label keys and review packets private.
5. Only then deterministically select whole replacement groups to reach
   640 rows, native total within ±0.5%, unchanged Noul/human rows and
   declared Chinese native-token share within five percentage points of the
   archived 40.2% whole-TRAIN share. Freeze final row IDs, order,
   tokenizer/prompt/code/model hashes, training steps, stop limits and
   SELECT advancement thresholds **before** the first optimizer update.

If admitted, a one-step zero-source/parity, length, finite-gradient and
save/reload gate precedes one 466-update run. The frozen SELECT gate should
require all 700 valid decisions, total at least the archived **562/700**,
targeted Score at least **40/90** (archived 32/90), GoEmotions Choice at least
**151/200** (archived 161/200) and GoEmotions Noul at least **165/200**
(archived 175/200). These last two limits precommit the tolerated 5-point
transfer tradeoff; they are not release metrics. If any gate fails, stop
without CAL, typed DEV/FINAL, CSS or JevBench evaluation. If SELECT passes,
run the already open typed DEV/CSS pilot **once** as a development diagnostic,
then freeze an independent source-disjoint confirmation before any new
release claim. Do not reselect checkpoints on keyed v3.

**Current feasibility:** no source passes the CPU gate, no replacement IDs or
source manifest are frozen, and no GPU arm should start. The next useful work
is source/provenance recovery and the answer-blind gate, not another training
run. Source quality failure remains a recorded result. A later readout-only
experiment can be considered after the data-only question is answered.

## Separate later architecture hypothesis

The current causal endpoint of an early option can see preceding options but
not later ones; the final global query sees all. On the open typed DEV panel,
option-order paired joint correctness was only 69/400. A distinct topology
test would encode each candidate against the same prefix with **no other
candidate present**, then use a shared set-equivariant readout over all
candidate vectors. This is not the failed ordinary candidate-interaction
head, which retained the order-sensitive endpoint stream. It could use
prefix KV reuse, but inference cost is still one prefix plus one branch per
candidate, rather than one serial stream; without reuse the full backbone
cost could approach the option count times the current cost. Before freezing
such an arm, measure exact 2/4/8/32-option native latency and memory at
short and long context, require permutation equivariance by construction,
and compare under the *same admitted data, token/update and initializer
budget* as its endpoint-head control. It remains a hypothesis, not a GPU
authorization or a claim of higher accuracy.
