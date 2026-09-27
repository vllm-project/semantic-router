# Decision 2.0 ~27B: next official-source Score experiment

**Status: prospective, CPU-data HOLD.** This is a single matched continuation
experiment, not a trained model, a JevArena result, or a release authorization.
The data, fresh selector, source parity receipt and immutable arm manifests do
not yet exist. No GPU optimizer, CAL, formal-set, public-set or HF action is
authorized by this note. Once any candidate sees the selector answers, a failed
gate ends this version; the thresholds and endpoint cannot be revised.

## Why this experiment, rather than another backbone run

The completed official `Qwen/Qwen3.8-27B` clean-v2 model, BEST368, is the
stronger existing ~27B Decision development start. Its package-native typed
DEV result is Choice 791/800, Noul 258/400 and **Score 162/400**; the same-size
native AutoJev peer scored Score 400/400 and led the DEV/CSS-pilot proxy
79.15 to 68.53. These are opened development panels, not JevArena v3. The
official `google/gemma-4-26B-A4B-it` full 456-update arm instead reached
SELECT family macro **0.72491**, below its predeclared **0.794** diagnostic
gate, so it remains HOLD without a DEV or release run. Repeating either full
control would consume compute without isolating the failure.

The earlier Qwen Score v6 matched pair gained 11/192, below its frozen 12/192
advance floor and with a level-2 loss. The v7p synthetic A/B corpus failed
independent realism review; v8.3 had contradictions and repeated templates;
v8.4 was quarantined for shortcut/copy defects; v8.5 failed its fixed long-row
quota. **None is admitted to this experiment.** The unresolved causal question
is whether a source-grounded, multi-mechanism three-level Score curriculum can
recover levels 1 and 2 without sacrificing Choice/Noul or level 0.

## Fixed contrast and starting weights

Use the official posttrained `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` and our own completed
BEST368 LoRA/head fingerprint
`d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2`.
Both arms start from those exact weights with a fresh optimizer. No third-party
Decision weights initialize either arm. Freeze the source snapshot, BEST368
checkpoint, tokenizer, native System One adapter, scoring code and runtime
image in the future private arm lock. The task remains structured `state`,
typed `instructions` and options, returning a scored Choice/Noul/Score
decision. A generative chat prompt is not a substitute.

| Arm | Exactly 2,288 TRAIN rows | Objective | Endpoint |
| --- | --- | --- | --- |
| A, new evidence | 1,024 parent Choice + 1,024 parent Noul replay + 240 new Score | Native option CE | One checkpoint at update 143 |
| B, exposure control | The **same** 2,048 replay rows + 240 eligible parent Score | Native option CE | One checkpoint at update 143 |

Choose replay as whole eligible English projections of parent TRAIN groups,
disjoint from both Score pools. Do not borrow SELECT, CAL, DEV, FINAL or public
rows. The prior v7p control files are not silently reused: create new
versioned arm manifests and independently attest their exact rows and tokens.

## Data admission before any device reservation

Author 80 **new TRAIN** cases and 80 independently sourced **fresh selector**
cases. Each case has exactly three related 0/1/2 Score variants, yielding
240 rows in each role. Allocate cases across six distinct mechanisms with
counts 14, 14, 13, 13, 13, 13 in each role: dependency readiness, timed
feasibility, stock uncertainty, scoped policy exception, multi-source
attestation and state reconciliation. A case varies the operative evidence
between levels; variants are one independent group, not three independent
cases. The selector author must not read TRAIN cases or the used v6/r2 and
rejected v7/v8 item keys. English is the language of this narrow pilot; it
makes no multilingual or long-context claim.

For every case, create structured facts and the deterministic oracle first,
then render distinct source documents. A separate parser of the rendered
documents must recover all three labels. Require evidence-removal witnesses,
counterfactual source necessity, correct scope/recency, decoy controls and a
source-grounded rather than repeated-filler presentation. Audit single-field,
lexical, option-position, date, length and document-order shortcuts. Run
exact/group, bounded near-duplicate and reviewed semantic-overlap checks
against parent TRAIN/SELECT/CAL, all available gold-free protected prompts,
public supplement prompts and previous Score corpora. Freeze separate labeled
rows, keys and answer-free blind packets. Independent reviewers must solve
**all 160 groups** and seal their answers and realism/ambiguity findings
before opening any key. Any unresolved mismatch, systematic template cue,
source redundancy, license issue or source overlap HOLDS the entire fixed
version; no favorable subset or new seed is substituted.

Each native input must fit **1,024 tokens without truncation**. Require 2,288
distinct rows per arm, exactly one epoch of 143 updates, A/B admitted raw
token totals within **1%** and dynamic padded-token totals within **5%**.
Target the already feasible roughly **438,000 native tokens per arm**; the
future CPU manifest must fix an exact common exposure between **430,000 and
450,000** before any optimizer. If realistic documents cannot fit that
matched budget with eligible parent controls, stop and version a different
experiment; shortening evidence or padding useless text is prohibited. The
data and selector content hashes, source/rights ledger, removal list, group
IDs and scored-prompt hashes belong in private artifacts. Only digests and
aggregate counts belong in public notes.

## Native preflight, training budget and one-use SELECT gate

First implement a separately versioned direct-LoRA start receipt for the new
A/B manifests; retain the old v6 receipt unchanged. In two fresh processes on
the **same physical GPU and runtime**, each arm's 32 gold-free Choice/Noul/Score
zero-step answers must reproduce BEST368 with **0 categorical changes** and
maximum option-probability drift at most `1e-4`. A separately capped one-step
finite backward, save and reload must pass 32/32 answer identity and the same
drift ceiling before full training. The technical preflight has a **0.25
GPU-hour** cap. No experiment launches if source/data/code hashes, native
context, zero-step, gradient or reload checks fail.

If the preflight passes, A and B each run once on comparable isolated hardware:
microbatch 1, accumulation 16, BF16 backbone and FP32 head/loss, gradient
checkpointing, rank-8/alpha-16/dropout-.05 LoRA, AdamW, LoRA LR `2e-5`, head
LR `1e-5`, weight decay `.01`, warmup `.05`, maximum length 1,024 and fixed
seed `20260928`. Preserve the prior native prompt and option-scoring path.
Each arm has **143 updates**, one final checkpoint and no intermediate
checkpoint selection or automatic retry. Cap each arm at **1.5 GPU-hours**;
at update 16, project remaining time with 10% margin and stop if the cap will
be exceeded. Total technical plus two-arm optimizer reservation is at most
**3.25 GPU-hours**; native selector inference gets a separately logged
**0.25 GPU-hour** ceiling. Nonfinite loss/gradient, OOM, stale identity or
partial checkpoint stops the arm. Preserve the failure receipt.

The first and only answer-key read occurs after A and B final weights and all
240 fresh selector predictions are sealed. Missing, invalid or over-budget
answers are wrong. Use 10,000 paired bootstrap replicates over the **80
independent groups**, seed `20260928`. A advances only if it wins at least
**15/240** Score answers over B, the paired 95% lower bound is above zero,
neither endpoint level loses more than **4/80**, Choice and Noul on the
unchanged parent SELECT each lose at most two percentage points, normalized
Brier worsens by at most `.02`, and invalid rate does not increase. Report all
three Score levels and six mechanisms even on a failed aggregate. No CAL fit,
DEV/CSS recheck, formal v3, public231 or HF upload is part of this arm. A
passing selector would merely permit a separately frozen single DEV/CSS-pilot
diagnostic before considering the already documented v3 release protocol.

## Current resource and evidence boundary

The read-only audit found the Qwen458-update completion and BEST368 selection
receipts still present. The Gemma signed completion/SELECT record is preserved
in the repository research notes. There is no active ~27B training container
from this task on the two authorized nodes; one node has unrelated model
services, which remain untouched. No GPU was allocated to this audit and no
new model score was obtained. Next action is the **CPU-only 160-group source
and blind-review admission**, followed by exact-token matching and the
versioned zero-step gate. Until then, the Qwen and Gemma candidates remain
HOLD for ~27B first release.
