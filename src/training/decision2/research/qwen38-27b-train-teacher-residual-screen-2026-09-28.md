# ~27B first release: TRAIN-only teacher residual screen

**Status: prospective design, CPU admission HOLD.** This is one bounded A/B
development experiment, not a trained result, JevArena score, or publication
approval. No formal key, GPU optimizer, model upload, or new dataset was used
to prepare this note. The existing six-mechanism authored Score arm remains a
separate HOLD and is neither replaced nor silently recycled here.

## Evidence and causal question

Our eligible start is the official `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` plus our own clean-v2
`BEST368` LoRA/head fingerprint
`d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2`.
It achieved typed **development** Score `162/400` and three-human-task pilot
median macro-F1 `.62052`; the same-panel native AutoJev peer scored
`400/400` and `.66605`. The full native teacher pass already yielded valid
private TRAIN distributions for `6,939/6,939` Choice/Noul and `516/516`
Score inputs, pinned respectively by SHA-256
`8cf211e5af88920de556dc84aa0fcb8b16677fdfd5209abb04db0ffe8e8b195c`
and `072cd519657caaa883eea1f5077789e5bacbf85f8ee20ab44cd562acc317701b`.
But teacher Score uniquely matches only `324/516` TRAIN labels; only `75/102`
three-level rows agree, and the parent has just 102 three-level Score rows.
The teacher's own training overlap is unknown. An official-Qwen 0.6B arm with
gold-agreeing, confident teacher KL on 3,690 Choice/Noul rows fell
`562→510/700` on SELECT and worsened Brier `.142634→.240292`. Thus a full
TRAIN KL replay, a larger KL weight, or a teacher-rank claim is unsupported.

The single question here is whether **source-limited soft information**, in
addition to hard labels on the same rows, can improve the existing 27B
student's three-level Score decisions *and* independently sourced human-task
transfer, without taking a third-party weight start. It is a risky
hypothesis; a negative result is a useful answer and ends this version.

## One fixed, equal-exposure contrast

Both arms resume exactly the eligible official-Qwen/own-`BEST368` weights,
with fresh identical optimizer, native System One typed head, tokenizer,
prompt and option ordering. The teacher is only a row-local probability
target. Use the rights-clean v2 TRAIN bytes SHA-256
`61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`.
First freeze a **2,560-exposure, 160-update** one-epoch schedule: include all
parent Score rows admitted by the unchanged Qwen tokenizer at 4,096 tokens
without truncation (`S` rows); fill `2,560−S` with Choice and Noul rows from
the same admitted parent TRAIN, as equally as integer counts allow. Select
eligible whole groups in ascending hash order within each source/type bucket,
using parent source proportions rounded by a fixed largest-remainder rule;
a group straddling a quota is skipped. If the exact quotas cannot be filled,
stop rather than split a group or change the rule. Freeze selected IDs,
order, source counts, raw/padded token exposures and SHA-256 before any
gradient. Require `S≥460`; otherwise hold rather than alter the budget.
The **same row IDs, order, tokens, 160 updates and fixed final checkpoint**
must be used for A and B. No row is drawn from SELECT/CAL, typed DEV/FINAL,
CSS, JevBench, rejected v6–v8 Score corpora or any new authored pool.

**CPU admission amendment, before model outcomes:** The first CPU-only gate
returned `WHOLE_GROUP_QUOTA` (private receipt SHA-256
`b74f04e60f3ccdf45c266ad1d35f7bc03c463af7a3e1d20457b03c432e8f10b2`).
Several source/type buckets contain multi-row groups, so the greedy skip rule
above can miss a feasible exact quota. The versioned admission implementation
now uses deterministic exact subset selection over the same hash-ordered whole
groups. This changes only the subset solver: **2,560 exposures, 160 updates,
all admitted Score rows, Choice/Noul type targets and largest-remainder
source quotas remain frozen**. An infeasible exact group subset still returns
HOLD. The first receipt remains preserved; this amendment precedes any 27B
weight load, gradient, teacher-mask result, SELECT/DEV result or formal/public
score. No sampler search after model outcomes is allowed.

The auxiliary mask is determined once from TRAIN provenance, TRAIN gold and
the pinned teacher vectors, never from model or SELECT outcomes. A row is
eligible only with a unique teacher top level equal to TRAIN gold and gold
probability at least `.50` for Score, or `.70` for Choice/Noul from the
original human-labelled GoEmotions, CosmosQA, SQuAD2, SNLI and FLUTE source
buckets. Programmatic Choice/Noul and every teacher-wrong/tied row remain
hard-label-only in both arms. Require at least **160** masked Score rows,
including **40** three-level Score rows, and **400** masked genuine-source
Choice/Noul rows; otherwise stop before training. Report the actual counts by
language, source, class and number of offered levels. The mask can highlight
easy teacher-agreeing TRAIN cases, so it does not itself prove useful signal.

Both arms use native hard-label CE on *every* exposure. On exactly the same
masked rows, B adds `.02 × CE(one_hot_gold, student)` while A adds
`.02 × CE(teacher_distribution, student)`; this is teacher-to-student KL up
to a constant independent of student parameters. Unmasked rows receive only
base CE. No teacher temperature, threshold, weight, subset, checkpoint or
CAL search is allowed. This matched auxiliary weight is important: A−B
isolates **soft versus hard auxiliary targets**, not extra row frequency.
Use the prior direct-LoRA rank-8/alpha-16/dropout-.05, BF16 backbone/FP32
head and loss, AdamW, LoRA LR `2e-5`, head LR `1e-5`, weight decay `.01`,
warmup `.05`, microbatch 1, accumulation 16, maximum 4,096 tokens and seed
`20260928`. Stop at exactly update 160. The contrast does not identify which
masked source caused any gain; report Score and human-source slices
separately and do not infer cross-source transfer from TRAIN agreement.

## Admission and execution stops

1. Recheck the two private teacher artifact digests and full row/input/option
   identity against the parent TRAIN. A fresh source ledger must confirm every
   admitted row's use terms and redistributable scope. Group-level exact,
   bounded-near and reviewed semantic overlap checks cover TRAIN,
   SELECT/CAL, gold-free typed/CSS inventories and the public subset. The
   parent manifest's zero exact/near match is evidence, not a complete
   semantic or third-party-pretraining guarantee. Quarantine suspect groups
   and stop if the fixed quotas cannot still be met.
2. Implement a **new versioned 27B teacher attachment and direct-LoRA start
   receipt**. The current `--external-teacher` gate is byte-locked to the
   failed 0.6B full466 experiment and explicitly excludes Score; the existing
   direct-LoRA receipt is bound to earlier Score arms. Do not bypass or
   overwrite either. Unit-test wrong teacher/base/row hash, option reorder,
   invalid mass, wrong/tied mask, missing type, stale receipt and control
   weight parity. The generic loss can score a masked distribution, but its
   current wrapper does not authorize this arm.
3. On the same runtime, each new arm must reproduce `BEST368` on a fixed
   32-item gold-free typed roster at zero step: **zero category changes**,
   maximum probability drift `≤1e-4`. A capped one-step backward/reload
   checks finite loss, gradients, actual Score and human-mask exposure and
   32/32 restored answers. On 16 TRAIN-only fixed Score and 16 genuine-source
   examples, measure the adapter/head CE-versus-auxiliary gradient cosine;
   require finite values and nonnegative median separately in both strata.
   This preflight directly probes the 0.6B conflict risk; failure stops both
   arms without tuning a new weight.
4. Cap technical preflight at **0.25 GPU-hour**. If it passes, each arm has
   a **2.5 GPU-hour** one-run ceiling (total A/B plus preflight ≤5.25
   GPU-hours). At update 16, project remaining time with a 10% margin and
   stop if the cap is unreachable. Match accelerator class/image/precision;
   two separate devices may run in parallel only if the exact zero-step
   parity passes on both. No partial checkpoint is selected. Nonfinite
   gradients, OOM, tokenizer overflow, missing vector or failed package
   reload ends this version. These are ceilings, not an observed cost; the
   previous 4,096-token run does not give a reliable time for this schedule.

## Fixed development decision, then separate release work

Seal A/B final weight, native predictions and source/score hashes before
reading SELECT700 labels. SELECT is a safety gate, **not** evidence of
three-level Score transfer: its 90 Score items are five-level quantized
median. A may reach a single, predeclared typed DEV1,600 and CSS pilot1,430
diagnostic only if its SELECT family macro accuracy is at least B−`.02`,
Choice and Noul each lose at most two percentage points, normalized Brier
worsens at most `.02`, and invalid answers do not increase. Report Score90
regardless of direction; do not select a milestone or CAL temperature.

On the already open development panels, A is worth a separately frozen
formal release preparation only if it beats B by **≥20/400 typed Score**,
scores **≥192/400 Score absolutely**, and improves the CSS pilot median
task macro-F1 by **≥.015 versus both B and BEST368**, with at least two of
three tasks improving and no Choice/Noul typed loss above two percentage
points. Report all Score levels, source families, both languages, Brier,
invalids, paired group/task uncertainty and long-input failures. Those
thresholds are future prospective gates, not retroactive changes to the
existing 27B first-release policy. A failure ends this arm; no alternate
KL weight, seed, checkpoint or same-key rerun follows. A pass would still
be **development evidence only**: CSS pilot and typed DEV have been used
before, the current Score TRAIN is synthetic and mechanism-mismatched, and
teacher pretraining overlap is unknown. Before a private product release,
separately establish exact package/download/native parity and matched
JevArena v3 and public JevBench results against eligible peers. A claim of
independent validation additionally needs source-disjoint confirmation;
without it, disclose the post-key comparison. Do not call post-key v3 a new
blind validation or claim a 27B frontier position from this screen.
