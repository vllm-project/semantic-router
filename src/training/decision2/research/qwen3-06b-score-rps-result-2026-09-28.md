# Official Qwen3 0.6B Score RPS ablation: SELECT HOLD

This is the completed result of the prospective [Score RPS
arm](qwen3-06b-score-rps-prereg-2026-09-28.md), not a release score. The direct
start is official Qwen3-0.6B-Base. Its data, seed, full-backbone/shared-head
schedule, eight fixed selection times and native scoring are the archived
rights-clean v2 control's. The only training-objective change is a normalized
ranked-probability term with weight 1.0 on the 516 Score TRAIN rows. No
additional training data, teacher, model-source change or chat generation was
used.

The zero-step 700-item SELECT output exactly matched the control: 232 correct,
family-macro accuracy 0.27804843. The one-step technical run had finite loss
and gradient, saved and reloaded with zero category or probability drift on
the first 32 SELECT items. A separate native Score backward had a nonzero,
finite head gradient. Its preflight receipt SHA-256 is
`73bee316d0c544df2aa2410ec2b052033dca0564a6571aa1c7f987b20e270c76`.
An added CPU regression test confirms that enabling the ordinal branch leaves
**non-Score** per-example losses and logit gradients byte-equal to the legacy
CE+Brier branch. This post-run test was diagnostic; it did not change the
experiment's training code or data.

The independent full run completed exactly 466/466 updates and exited zero.
Its `COMPLETE.json`, `BEST.json` and source-provenance SHA-256 values are
`5b80f97ff2735e6792393b67d62cfc7b2eef6af5c14bf095a5a69c102bf8ce2a`,
`0d1c5055fb324d60b237652e90c6187136170ed4d9a6673ee89666826fa89420`
and `9077eca1312c41073e37a7e32c6ee1316783baf30dfc1743a87f6094658606fd`.
The fixed selector chose step **448**; no later checkpoint was substituted.

| SELECT700 checkpoint | Correct | Six-family macro |
| --- | ---: | ---: |
| Zero-step | 232 | 0.27805 |
| 64 | 245 | 0.31711 |
| 128 | 272 | 0.36772 |
| 192 | 243 | 0.29623 |
| 256 | 251 | 0.30437 |
| 320 | 258 | 0.32793 |
| 384 | 269 | 0.33691 |
| 448, fixed BEST | **276** | **0.38075** |
| 466 | 270 | 0.36127 |
| Same-start CE+Brier control BEST466 | **562** | **0.77259** |

At the selected step, the targeted Score family got **38/90** versus the
control's **32/90**, but GoEmotions Choice fell to **53/200** versus **161/200**,
Noul to **96/200** versus **175/200**, and narrative reading to **64/130**
versus **130/130**. The apparently favorable Score change is narrow and does
not establish ordinal transfer: on those 90 five-level SELECT cases it emits
only levels 0 or 2, with no predictions at levels 1, 3 or 4. The selected
checkpoint misses both frozen
SELECT gates, 562/700 and 0.77259, by a large margin. Its family-macro Brier
is 0.34361; the selected metrics SHA-256 is
`45ebc2d782a54584ec9571333c3f801cfb8ea347f84c7586b0ab5a5188c7f2b0`.

The three recorded GPU-container durations were 24.15 seconds zero-step,
41.06 seconds one-step and 1,106.01 seconds full training, totaling about
**0.325 GPU-hour**; a separate brief native reload/backward probe keeps this
arm well below the frozen one-GPU-hour cap. GPU5 is released. CAL,
typed DEV, CSS pilot, typed FINAL, CSS15, public JevBench and HF were **not**
used on this arm. Its failure is preserved, not promoted. The large same-data
degradation makes simply adding this RPS weight to the shared head an
unpromising route; the exact optimization mechanism is not isolated by one
run. A separately sourced, independently reviewed Score curriculum and
cross-task retention contrast is the next substantive experiment.
