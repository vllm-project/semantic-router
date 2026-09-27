# 27B prospective pooled-quota schedule: capacity, not TRAIN admission

**Status: schedule candidate only; GPU training HOLD.** The earlier frozen
strict source/type quota experiment failed twice at `WHOLE_GROUP_QUOTA` and
remains failed. This is a distinct, input-only CPU run of the minimum-L1
pooled-quota rule proposed after those failures. It reads no target labels,
teacher probabilities, SELECT/CAL, development or formal answers. No model,
optimizer, checkpoint, prediction or HF package was loaded or changed.

The signed code is `training/data/plan_27b_pooled_schedule.py` at source commit
`9f213fd64`; its exact mirrored file has SHA-256
`b4d66226ecb179556f1e4bef533218429dcb846bfdc1020e2ab890c3220e80a0`.
The pinned rights-clean v2 TRAIN SHA-256 is
`61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`;
the official Qwen3.8-27B tokenizer JSON SHA-256 is
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`.
The CPU container image ID is
`sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1`.
The private 0600 receipt SHA-256 is
`d76eb49a12a6dd04861239a7d7e6697f6f96c0f80015246b4060362c5bc156d7`;
it contains row IDs and is not a public dataset. Only aggregates appear here.

| Input-only property | Result |
| --- | ---: |
| Scheduled rows / updates | 2,560 / 160 |
| Choice / Noul / Score | 1,026 / 1,027 / 507 |
| English / Chinese rows | 2,027 / 533 |
| Source-quota minimum L1 shift | 2 rows, entirely within Noul |
| Raw / 8-token-padded exposure | 1,244,036 / 1,253,048 tokens |
| Source groups partially selected across task types | **600** |
| Training or inference GPU-hours | **0** |

The selected-row/input/token schedule hash is
`6ea5e54d207535bd1ca06e9f2ad4ff19fbdae0d4c6143fce0b20be8fcbe52181`.
Whole-group selection is exact **within each source and task type**; the
original corpus has groups spanning types. The 600 partially selected
cross-type groups need an explicit training-design decision and protected-role
source audit. This receipt does not show that their variants are independent,
nor that the candidate can safely train without them. Unlike the old exact
quota run, the L1=2 result is a measured schedule rather than only a capacity
bound, but it is still not a data-admission pass.

Before any GPU reservation, verify selected IDs against full original rows,
source rights, teacher option identity and frozen mask minima; run exact,
near and reviewed semantic overlap over the **complete** native input surface
against TRAIN-disjoint SELECT/CAL and all protected roles. The previous
state-only checker is insufficient. Freeze the selected set and cross-type
group treatment before using labels or model outcomes. Then perform the
predeclared zero-step, one-step and gradient checks. A failure keeps this
new version HOLD; do not reinterpret either earlier failed schedule as passed.
