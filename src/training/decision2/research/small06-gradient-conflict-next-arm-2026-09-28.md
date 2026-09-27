# 0.6B: test task-gradient interference before another full run

**Status: opt-in preflight implemented; no real-source gradient receipt yet.**
No GPU diagnostic, optimizer update, new model score, or release decision
follows from this note. Freeze the implementation and all thresholds before
executing it. The [official-base v3 comparison](qwen3-06b-official-full466-v3-postkey-result-2026-09-27.md)
is a post-key same-panel result, not a fresh blind test.

## Evidence and hypothesis

The eligible Qwen3-0.6B-Base shared-head model scores 38.520 on JevArena v3
versus Kai1's 35.938, with paired 95% difference interval [-2.032, +7.846].
Typed Choice is 109/800 versus 277/800 and Score is 80/400 versus 98/400;
Noul rises from 404/800 to 458/800. On opened typed DEV, the Qwen model
chose level 0 on all 400 Score rows. The rights-clean v2 TRAIN has 3,908
Choice, 3,031 Noul, and only 516 Score rows, spanning different Score level
counts. These observations suggest either insufficient mechanism coverage or
destructive sharing during training; neither is established causally.

Changing only the readout to three type-specific heads missed the frozen
SELECT gate (553/700 versus 562/700 control). Candidate interaction missed it
more widely (333/700); an added Score ranked-probability loss got 276/700;
official general posttrained initialization got 548/700; and same-TRAIN
AutoJev teacher KL got 510/700. The authored Score-replacement candidate is
on quality HOLD after shortcut review. Repeating those arms or selecting on
the already keyed v3 labels would add little evidence.

**Single next ablation:** keep the eligible official Qwen3-0.6B-Base source,
shared head, all 7,455 TRAIN rows and exact tokenization/order, loss, 466
updates, optimizer hyperparameters, checkpoints, SELECT/CAL, and native
System One Choice/Noul/Score inference fixed. Change only the *shared-backbone
gradient aggregation*: project conflicting Choice, Noul and Score task
gradients within each existing accumulation window, while leaving the head
gradient as the original ordinary sum. This asks whether cross-type backbone
interference contributes to the Choice/Score loss. It does not add data or
claim that gradient surgery remedies missing Score mechanisms. The motivating
method is [Yu et al., Gradient Surgery for Multi-Task Learning](https://papers.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html);
this is a fixed norm-matched variant, not a claim to reproduce that paper.

## Paired protocol and stop gates

| Contract | Frozen value |
| --- | --- |
| Start | `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`; same random-head seed `20260926` as the archived hard-label control, with zero-step native logits checked |
| Data | Identical rights-clean v2 TRAIN SHA-256 `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`, 7,455 rows / 4,094,489 Qwen tokens; SELECT SHA-256 `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`; no added teacher, replay, or source weights |
| Budget | One epoch, microbatch 1, accumulation 16, exactly 466 planned AdamW steps; unchanged CE + 0.5 Brier, 8,192-token no-truncation input, LR schedule, BF16/FP32 settings and gradient clip |
| Control | Reuse the completed identical-start standard-gradient arm (BEST466, SELECT 562/700, family macro .77259); do not retrain it |
| Selector | Same eight fixed checkpoints and family-macro, Brier, earliest-step ordering as the control; no formal or public results used to choose a checkpoint |

For each accumulation window, calculate each type's FP32 backbone gradient
`g_t` with every example retaining its original `1/window_size` contribution.
Thus `sum(g_t)` must reconstruct the standard window gradient. For each
nonempty type in fixed Choice, Noul, Score order, project a negative component
against each **original** other-type gradient in that order:
`g'_t <- g'_t - min(0, dot(g'_t,g_u))/(||g_u||^2+1e-12) * g_u`.
Sum projected vectors and rescale **per backbone optimizer group** to the
norm of the original summed vector before the original global clip; if either
norm is below `1e-12`, use the ordinary sum for that group. Preserve the
historical, unprojected shared-head gradient. Missing types are skipped.
The rescaling makes the tested change primarily gradient direction rather
than a hidden loss-weight increase. Record task gradient norms, pairwise
cosines, projection frequency, and final gradient norms without raw prompts.

Before a full run, use the archived deterministic shuffled order to identify
the first eight windows containing all three types, without consulting
evaluation labels. At the untouched source weights, measure their gradients
without an optimizer update. **Stop** if fewer than three of these eight
windows have a Choice/Score-involving pair with cosine at most `-0.05`: this
sample does not support spending the full budget on this mechanism. Also stop
if a projection-disabled one-update reconstruction fails native zero-step
identity or diverges from a *same-schedule, one-update technical replay* of
the unmodified trainer beyond `1e-5` maximum probability difference on 32
fixed SELECT rows. Do not compare against the older `--max-steps=1` smoke:
its learning-rate schedule need not match the 466-step arm. If nonfinite
gradients, save/reload drift, source/data mismatch, changed token count, or
the 2 GPU-hour cap occurs, retain a failure receipt. These are technical and
mechanism gates, not favorable-checkpoint search.

If the preflight passes, run only this one treatment and use its original
checkpoint selector. Advance beyond SELECT only if the chosen checkpoint has
at least 569/700 correct, family macro at least .78259, quantized Score at
least 37/90 (control 32), and human GoEmotions Choice no worse than 156/200
(control 161). Then inspect the already opened typed DEV and CSS pilot once:
require Choice at least 206/800 (control 186), Score at least 105/400
(control 85), and CSS pilot task-median macro-F1 no more than .01 below the
control's .290959. Report all other type/family scores, Brier, invalids,
paired robustness, actual parameters, and GPU-hours regardless of outcome.
These are development gates, not publication evidence. If they pass, freeze a
downloadable package and seek a new source-disjoint confirmation before
claiming a major 0.6B gain. The keyed v3 and public231 panels remain
post-key/diagnostic for any new candidate.

If the conflict or performance gate fails, do **not** sweep projection orders
or thresholds on SELECT. Prioritize an independently admitted, genuinely
source-disjoint Choice/three-level Score corpus, since the current label
coverage and mechanism mismatch remain the stronger unresolved explanation.

## Implemented read-only preflight boundary

[`gradient_conflict_preflight.py`](../training/model/gradient_conflict_preflight.py)
recreates the exact epoch-0 row order with the control scheduler and takes the
first eight accumulation windows containing all three types. Its default CPU
mode only hashes the private TRAIN, verifies pinned official source files,
tokenizes every row, checks the 4,094,489-token total, and emits an aggregate
roster receipt. Explicit `--measure --device cuda` additionally reloads the
same official source and random head, uses BF16 backbone autocast, measures
each window's ordinary and per-type backbone gradients, and applies **zero**
optimizer steps. It never accepts SELECT, CAL or benchmark keys. The receipt
has an allowlisted schema: window index, type counts, native token count,
roster hash, gradient norms/cosines and technical status, with no IDs, text or
labels. CPU synthetic tests independently compare the grouped gradient norm
against ordinary `.backward()` and verify unchanged parameters. CPU test
success is implementation evidence, **not** a measured Qwen conflict result.

The future real-source measurement requires one BF16 GPU, the pinned official
source and private TRAIN, and enough HBM for four FP32 backbone-gradient
accumulators (roughly 9.6 GB before model, activations and temporary grads).
The loop stops after six minutes (0.10 GPU-hour); reserve at most 0.20
GPU-hour for source loading, tokenization and the measurement. A conflict
receipt passing three of eight windows is only permission to implement the
separate same-schedule one-update parity gate, not permission for the full
466-update arm or a publication claim. If it fails, record HOLD without
changing the window roster or threshold.
