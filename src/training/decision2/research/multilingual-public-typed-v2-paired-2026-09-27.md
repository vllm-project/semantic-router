# Public Chinese/Russian typed DEV v2: paired 4B diagnostic

**Status: completed exposed DEV diagnostic; not a release score.** This note
supersedes no other panel. The [v1 serialization correction](multilingual-public-typed-v1-fingerprint-correction-2026-09-27.md)
quarantined both earlier prediction files. All numbers here come from fresh,
complete native inference on the corrected v2 panel.

The panel has 4,052 typed questions from 21 task slices and 3,729 observable
group IDs: 3,768 Russian and 284 Chinese. Its manifest, prompt, and target
SHA-256s are `fea7c433419d76e150e606cdca4c5029830a966b7febc674a5478fb729f94ede`,
`4147c42006a4dfb39331936ce99e7c964235e5b3de7ac4cc8c19b9c87dbdb75a`,
and `d93a64f96e5ee69cd67cef56274f37251e1e311e37c5f4ecc9226fead92369cb`.
The exact upstream source revisions, rights, translation/source-ID checks,
and ten excluded ambiguous Russian MASSIVE rows are in the panel manifest and
[`public_typed_dev.py`](../multilingual/public_typed_dev.py). The exposed labels
make this panel suitable for development diagnosis only. It is not JevArena
FINAL, unseen-task transfer, or an independently sealed multilingual test.

Both models answered all 4,052 identical serialized prompts with zero native
invalids. The candidate was the **unpublished** Eikos clean-v2 standalone 4B
checkpoint-0232 under its `decision2-eikos-semif-native-v1` adapter and
package-manifest SHA-256
`7e005ef609553d973c3a6232840436d1384657a19f5765e42752ebcc04906e39`.
The baseline was published Decision 1.0 Nox 4B at exact revision
`0bb833504965c0eabdb9630b7bbd385cb2fe5cd4`, under
`native-published-v2`. No calibration or task-specific prompt was fitted on
these panel labels. Candidate/baseline prediction SHA-256s were
`8b261b4cade7f0b641613f7227016eed1f139e80b47071b6bb2a67551e7daf00`
and `3b276d3ceb821ce82468dcccf51908068b447eeed7d934cd59093b38e1ab6a84`;
native score-report SHA-256s were
`3699d81f641b2ea05052934fccef8a5731f2d3b1f91d94be0c48693b792e296a`
and `8ab12bfddb1c209a94afa8473731d28e77d7a4afd2437080e68f305b9a7c1852`.

| Slice | Questions | 2.0 correct | 1.0 correct | Difference |
| --- | ---: | ---: | ---: | ---: |
| All | 4,052 | 3,285 (81.07%) | 3,184 (78.58%) | +2.49 pp |
| Russian Choice | 2,089 | 1,666 | 1,591 | +3.59 pp |
| Russian Noul | 1,553 | 1,272 | 1,256 | +1.03 pp |
| Russian Score | 126 | 106 | 105 | +0.79 pp |
| Chinese Choice | 204 | 195 | 187 | +3.92 pp |
| Chinese Noul | 55 | 33 | 29 | +7.27 pp |
| Chinese Score | 25 | 13 | 16 | **−12.00 pp** |

There were 279 candidate-only and 178 baseline-only correct answers. The
reproducible [paired script](../multilingual/public_typed_compare.py) uses
`random.Random(20260927)`, 3,000 with-replacement resamples of observable
group IDs within each slice, a question-weighted difference, and percentile
limits. Its source and report SHA-256s are
`fa6e274ffdd6a11041200c42c0f0dd1d6e82e865843732a8ed27c7a4fd77fbb0`
and `04bda61836f5609a29f98d78c204c7498c273de3fdab0f31488bc987999e8e31`.
The all-row paired difference is +2.493 pp with an **exploratory** 95% group
bootstrap interval of +1.475 to +3.542 pp. Russian is +2.442 pp
(+1.397 to +3.515); Chinese is +3.169 pp (−0.685 to +6.909).
Group IDs are only observable proxies: some upstream source identities are
unavailable, so independence is **not certified**. These intervals must not
be used as release confidence bounds.

Task regressions remain: Russian RCB 99 versus 106/220, injection 214 versus
216/300, support 224 versus 226/300, and Chinese e-commerce urgency/Score
13 versus 16/25. The Chinese Score slice is too small to establish a stable
effect, but it is a concrete non-regression target. This aggregate gain does
not satisfy the family release gate: independent multilingual and 15-task
transfer, Score calibration, long input, package parity, and full JevArena
FINAL still need separate evidence.
