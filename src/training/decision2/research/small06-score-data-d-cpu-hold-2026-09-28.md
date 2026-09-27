# 0.6B Score data contrast D: CPU construction and review HOLD

**Disposition: HOLD after independent gold-blind AI editorial review.** This is
a private TRAIN candidate, not a model result or permission to run a GPU arm.
The previously completed shared-head 0.6B control and failed type-separated
head arm remain unchanged. No SELECT, CAL, typed DEV/FINAL, CSS, JevBench or HF
operation was used to select this corpus.

## Fixed source and construction

The [D contrast](small06-type-separated-head-prereg-2026-09-27.md) fixes the
official Qwen3-0.6B initializer, shared head and 7,455-row rights-clean v2
control, replacing exactly 384 of its 516 Score rows with 128 complete
three-level source groups. The parent TRAIN and rights manifest SHA-256 values
are `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`
and `61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8`.
The new rows are a deterministic subset of the already mechanically screened,
internally authored Score v6 private TRAIN candidate, SHA-256
`6aee966cc5499a87d2a77241676586c9c9f801b3c662a078daf025f001169f54`;
its manifest SHA-256 is
`1217eb9a9003615562715ba11f9164c0e3cf09277433169d8b92ef3d741fe736`.
That upstream candidate explicitly remained **HOLD for independent editorial
review**; subsetting it does not clear the HOLD or turn it into a new source.

The fixed [builder](../training/data/build_score_replacement_d06.py) and
[oracle/matcher tests](../training/data/tests/test_build_score_replacement_d06.py)
are signed on branch `xunzhuo/decision2-06b-score-data-d` at commit `8b8fcf705`.
Its builder SHA-256 is
`e5835aeee959de2e02dce54471caac841f31d8c1b2d78b4585173f4f34b70389`.
The pinned CPU image is
`sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1`.
The first CPU attempt stopped on a wrong assumption that the parent source
rights ledger was a mapping; the frozen parent manifest actually stores a
list. A second attempt stopped because some gold-free protected prompts were
not mounted in the isolated CPU container. Both defects were corrected locally
or in the isolated mount before the successful output; neither attempt wrote a
partial TRAIN file. Earlier successful audit outputs were preserved under
their own immutable names as the code gained the rights detail and grouped
blind-review packet.

## Measured private-candidate audit

| Check | Final CPU result |
| --- | ---: |
| Selected complete source groups / new Score rows | 128 / 384 |
| Mechanisms | evidence intersection 32; latest core obligation 32; directed route 32; calendar streak 32 |
| Language groups | English 96; Chinese 32 |
| New Score labels 0 / 1 / 2 | 128 / 128 / 128 |
| Original Score rows replaced | 384, in 328 complete old groups |
| Original Score rows retained | 132 |
| Final Choice / Noul / Score rows | 3,908 / 3,031 / 516 |
| Final three-level Score rows | 402 / 516 |
| Qwen3-0.6B parent / candidate / removed / final native tokens | 4,094,489 / 112,442 / 112,442 / 4,094,489 |
| Token-budget deviation | **0.000%**; the prospective ±0.5% constraint passes |
| Maximum new-row native tokens / complete-input cap | 342 / 8,192 |
| Protected gold-free prompt roles | 23, including typed FINAL; exact/context/complete-prompt near matches 0 |
| Oracle disagreement / incomplete triplets | 0 / 0 |
| Blind individual / grouped packets | 384 / 128, with a separate sealed answer key |

The 384 old Score slots were replaced **in place**, preserving the order and
bytes of every Choice and Noul row. Old Score groups were removed whole; 56
two-row groups and 272 singleton groups are absent from the treatment. The
independent CPU oracle recomputed all 384 labels from displayed facts without
calling the v6 authoring oracle. The original v6 QA had separately checked
source necessity and shallow count shortcuts; this 128-group subset retains
that provenance but has not received a new independent naturalness review.
The source-rights ledger and per-source retained counts are kept in the
private candidate manifest. Added text is internally generated; the parent
upstream conditions still apply to retained rows.

The private TRAIN candidate SHA-256 is
`3bb43b2b101b5121b45e1d457a81ad513102895de9cae27e8a9ab084bfadb6ed`;
the 384 new rows SHA-256 is
`b68122f85976dd3ebe95578baf4cefe996399cf039952b41d680985a81f809b3`.
The private manifest SHA-256 is
`8ad43f17a24d62d378d5d72ad38f61b0cb6e0af4664ded237898382c4d8541a4`.
The answer-free individual and grouped packets have SHA-256 values
`c71babc4dd94054d2b0594f1d95081de19911fa8672e3fe489b2662586837749`
and `d4a0a743ac816d1a1bc8f485fc7882b3bd1c506a9024d2d5f8c0f94d0b9bd899`.
Their key is separate and not for reviewers.

## Subsequent blind review

A separate AI reviewer inspected all 128 answer-free grouped triplets before
the key comparison. Its blind receipt SHA-256 is
`98f33b6118b423cf9e280b1343557e6f4123a87ccabb99b25bd7f5fc03ed1478`;
the later key-comparison receipt SHA-256 is
`479d4b1d2c8599f94f47e5216de62e639eab9e70b00abaad4004d2fd03eeff07`.
The reviewer reconstructed all 384 labels correctly and found no ambiguous
answer in these packets. It also found material shortcuts: the eligible
filter never changes the answer in the 96 evidence rows, and the two core
controls share the latest timestamp in all 96 relevant rows. Some Chinese
renderings were stiff. These are data-quality failures for the intended
generalization contrast, even though the answer oracle is internally
consistent. This was AI review, not human annotation or a new model score.
The candidate remains HOLD; no GPU arm should train on it as this contrast.

## Interpretation and next gate

This treatment isolates the *training-data composition* hypothesis from the
failed type-separated-head treatment; it does not show that 0.6B Score,
Choice, Noul, transfer, calibration or composite score improved. These four
programmatic mechanisms are short and structured, so they cannot establish
real-world or long-context transfer. The blind review above found shortcuts;
it does not clear the candidate for training.
Near-overlap algorithms cannot rule out every semantic paraphrase. Any
material review failure requires a new prospectively specified treatment,
not deletion of unfavorable rows from this frozen roster.

If a separate review clears this corpus, freeze the final source manifest,
original Score replacement IDs, trainer and SELECT-only advancement gate
before any one-update GPU preflight. For 0.8B and 2B, the 128-group metadata and
blind review may be reused as a source pool, but their different Qwen tokenizer
and non-Score replacement designs require separate exact budgets and locks.
