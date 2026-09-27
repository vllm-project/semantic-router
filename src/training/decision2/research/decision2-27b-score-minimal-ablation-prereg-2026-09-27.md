# 27B Score: minimum prospective data and objective screen

**State: pretraining design, formal HOLD.** This records a proposed experiment
before any new v7p Score row, new selector label, optimizer step or v3 FINAL
prediction. The exact row manifests/hashes and new parity receipt do not yet
exist, so this note is **not** an authorization to launch. The earlier v6
English direct-LoRA pair remains `DO_NOT_ADVANCE`: its 11/192 gain missed the
frozen 12/192 gate, and its r2 selector is consumed. Neither that pair nor the
published AutoJev DEV advantage can be rescued by changing thresholds.

## Question and minimum arms

BEST368 started from an official general Qwen weight and was trained on 516
Score rows, of which only 102 are three-level. On matched typed DEV it solves
162/400 Score, versus 400/400 for the same-size open peer. The v6 treatment
raised middle-level accuracy but harmed high-level accuracy. The next useful
test separates **Score evidence data** from a **proper-loss target**:

| Arm | Frozen data concept | Objective | Comparison |
| --- | --- | --- | --- |
| B, parent control | Eligible existing parent Score groups + exactly the common parent Choice/Noul replay | categorical CE | Matched exposure control |
| A, data change | New evidence-state Score groups + exactly the same replay | categorical CE | A vs B identifies the data intervention under matched source/budget |
| C, target change, conditional | Byte-identical A data and replay | CE + 0.5 Brier, both native option distributions | C vs A isolates the supported target change; this is *not* a new ordinal-specific loss |

The three arms begin from the byte-pinned **original BEST368 LoRA and head**
plus official `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, with new AdamW state and
no inherited optimizer. No third-party Decision weight initializes an arm.
Arm C is run only if A passes the fresh Score selector and parent-retention
gate below; this sequential decision is declared now. If A fails, C is not
run and there is no objective conclusion.

## New data and selector, before GPU use

This is a **small English-only v7 pilot (`v7p`)**, distinct from the proposed
full 480-group multilingual v7 curriculum. Limit new TRAIN to **120
independent case groups × 3 related 0/1/2 levels = 360 rows**; limit the
fresh r3 selector to **80 separate case groups × 3 = 240 rows**. The selector
author does not see v7p TRAIN groups, prior r2 key, item predictions or
failures. Do not reuse v6 or r2 rows, even with paraphrases. These sizes are
ceilings, not quotas: unqualified groups are dropped and insufficient quality
blocks the version.

Each triplet requires two independently necessary current/archived evidence
channels and a scoped disqualifier. Under a fixed rule, level 0 proves an
applicable disqualifier; level 1 lacks a disqualifier but leaves a requirement
unresolved; level 2 positively verifies all requirements. Only the operative
evidence state changes within a triplet. Require two independently coded
oracles, a source-necessity counterfactual, evidence-removal witnesses and
group-held-out one-field/one-source/length/count/position/date/lexical
shortcut probes. No single-source or shallow feature may perfectly decode
all levels. An independent blind review of complete groups must resolve
answerability, ambiguity and realism before admitting labels. This experiment
makes **no multilingual claim**; any Chinese expansion has its own native
editorial and independent selector gate.

For A and B choose **360 Score rows** and the same **2,048 parent English
Choice/Noul replay rows** (1,024 of each), hence 2,408 training rows per arm.
Control Score rows come only from rights-audited parent TRAIN and are
group-disjoint from replay, SELECT, CAL and all protected panels. Before an
optimizer: freeze immutable row IDs, source licenses, tokenizer/prompt/code
revisions, arm hashes, raw native-token sums, dynamic padded-token sums,
Score-level distribution, longest input and option count. Match A/B raw
native-token exposure within **1%**, padded exposure within **5%**, equal
2,408 example count and no native input longer than **1,024 tokens**. If this
cannot be met with eligible parent control rows, stop and version a new
design; do not duplicate or shorten a benchmark input to make the budget fit.
Arm C uses A's exact bytes. Exact/group, bounded near-duplicate and semantic
overlap review covers TRAIN, SELECT, CAL, DEV/pilot, v3 gold-free typed/CSS
prompt inventories, public JevBench, earlier Score curricula and the fresh
selector. The earlier v6 source/QA receipts do not substitute for this audit.

## Frozen start, optimizer, panels and stop rules

The existing trainer supports `ce` and `ce_brier`, but its direct-LoRA parity
gate currently hardcodes the **old v6 A/B TRAIN hashes**. This is an explicit
implementation blocker: **do not pass a forged old receipt or edit the old
hashes in place**. First add a separately versioned `direct-lora-start/2`
receipt that binds the new A/B/C TRAIN hashes, exact BEST368/base identities,
gold-free 32-item roster and all three zero-step starts; retain the old v1
gate for reproducibility. Unit-test wrong source, stale TRAIN, missing arm and
receipt mutation. On the same physical GPU/runtime, each new zero-step start
must match the pinned source on **32/32 categories** with max option drift
≤`1e-4`; otherwise no optimizer.

Freeze a one-epoch, 2,408-row, microbatch-1, accumulation-16 run:
**151 optimizer updates**, BF16 backbone/FP32 head and loss, existing rank-8,
alpha-16, dropout-.05 LoRA, AdamW, LoRA LR `2e-5`, head LR `1e-5`, weight
decay `.01`, warmup `.05`, max length `1024`, seed `20260927`, gradient
checkpointing, one fixed final checkpoint at step 151 and no early checkpoint
search. A/B/C must use the same code/image, hardware class, precision,
learning-rate schedule, admitted tokens, CAL lineage, native prompt and
fixed output budget. `SELECT` is for the trainer's baseline/final receipt
only, not for changing the frozen endpoint. No temperature is fit on r3.
Both A/B final checkpoints and all gold-free r3 predictions are sealed before
the r3 key is read. Invalid/missing/over-budget predictions are wrong.

The first contrast A−B uses 240 r3 Score rows, **80 independent triplet
groups** for a 10,000-replicate paired bootstrap with seed `20260927`. To
advance to one same-panel DEV/CSS pilot comparison, A must improve by at
least **15/240** (6.25 percentage points), have paired 95% interval lower
bound above zero, and lose no more than **4/80** at either the 0 or 2 endpoint
level. Parent English SELECT Choice and Noul may each lose at most two
percentage points; normalized Brier may worsen at most `.02`; invalid counts
must not rise. The latter are safeguards against learning only Score. All
source/exposure/parity/update checks are conjunctive. If A fails, preserve the
negative result and stop. It does not matter whether one exploratory slice
looks favorable.

If A passes, C uses the **same r3 packet once**, but scores C−A against the
predeclared objective question: Score normalized Brier must improve by at
least `.01`, Score correct count may lose at most **2/240**, and parent
Choice/Noul and invalid guards must still pass. Publish each level's accuracy
and confusion even if an aggregate passes. This objective contrast is
exploratory because A eligibility depends on the same selector; it does not
justify a standalone independent claim. Choose A or C by these fixed rules
before a single diagnostic DEV/CSS recheck. No second seed, altered weight,
different checkpoint, new teacher or r2 reuse follows a failure.

The matched DEV/CSS pilot check reports the same two panels as the [27B peer
screen](decision2-27b-same-panel-dev-peer-2026-09-27.md), but is another
development comparison. To justify spending formal v3 GPU-hours, the chosen
arm should close at least **half of the existing 10.62-point DEV/CSS proxy
gap** to the qualified same-size peer (≥`73.84` on the fixed proxy), without
new native invalids or package/source parity failure. This prospective
screening threshold is a resource decision, **not** the release gate. Formal
v3 remains the separate disclosed same-panel, non-virgin comparison and
requires a fresh package, independent overlap audit, sealed predictions and
public231 run. An untouched external or source-separated confirmation is needed
for a claim of independent validation.

## Exact command shape and current execution boundary

The following command is the **actual trainer CLI** once v2 direct-start
attestation, arm files and their signed hashes exist. It is intentionally
fail-closed now: required variables must be set to audited immutable paths,
and the current v1-only parity validator will reject new hashes. The first
operational command today is CPU-only code inspection, not a 27B optimizer:

```bash
cd /ABS/vllm-sr/src/training/decision2
python3 -m training.model.train --help
python3 -m pytest training/model/tests/test_direct_lora_start.py -q
```

After `direct-lora-start/2` is implemented, tested, and a PASS receipt is
sealed, set these **private** environment variables from the preregistration
manifest: `QWEN_SOURCE`, `BEST368`, `ARM_A`, `ARM_B`, `PARENT_SELECT`,
`PARENT_CAL`, `START_RECEIPT`, `START_SHA256`, `RUN_ROOT`. Do not put raw
private paths or tokens in public logs. The launch shape for each arm is:

```bash
: "${QWEN_SOURCE:?}" "${BEST368:?}" "${ARM_A:?}" "${ARM_B:?}"
: "${PARENT_SELECT:?}" "${PARENT_CAL:?}" "${START_RECEIPT:?}"
: "${START_SHA256:?}" "${RUN_ROOT:?}"
python3 -m training.model.train \
  --model-path "$BEST368" --source-path "$QWEN_SOURCE" \
  --init-kind decision2-lora --initial-model-sha256 \
  d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2 \
  --direct-lora-parity-receipt "$START_RECEIPT" \
  --direct-lora-parity-sha256 "$START_SHA256" --direct-lora-arm A \
  --train "$ARM_A" --select "$PARENT_SELECT" --cal "$PARENT_CAL" \
  --output "$RUN_ROOT/data-A" --train-mode lora \
  --objective ce --epochs 1 --microbatch 1 --accumulation 16 \
  --max-length 1024 --max-steps 151 --save-every 151 \
  --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05 \
  --lora-lr 2e-5 --head-lr 1e-5 --weight-decay 0.01 \
  --warmup-ratio 0.05 --seed 20260927
```

For B substitute `--train "$ARM_B"`, `--direct-lora-arm B` and output
`$RUN_ROOT/control-B`. Conditional C requires a v2 receipt binding its own
zero-step start and exact A bytes, output `$RUN_ROOT/target-C`, and changes
only `--objective ce_brier --brier-weight 0.5`. Do not launch A/B/C merely
because these commands parse: data QA, direct-start v2 contract, memory and
first two-update finite-loss smoke, whole-run GPU-hour ceiling, and current
GPU reservation must pass and be recorded before any long run. No v3 gold is
an input to any of these commands. The 27B full-training duration remains
unmeasured for this new schedule; measure the smoke before fixing the
GPU-hour ceiling rather than inventing one.
