# AutoJev 27B: same-panel v3 and public231 external peer

**Result: external comparator measured; Decision 2.0 27B remains HOLD.** This
run evaluates pinned AutoJev on the same JevArena v3 questions, native task
types and scorers intended for our 27B candidate. There is no corresponding
qualified Decision 2.0 27B v3 result yet, so these values give a target and
cannot establish a 2.0 win, release gate or Pareto position. The v3 labels had
been opened earlier in the project; this run is a disclosed same-panel
comparison, not a fresh blind independent test.

## Pinned model and pre-score audit

- Open model [`denis-pplx/autojev-27b`](https://huggingface.co/denis-pplx/autojev-27b/tree/6f5b557e037f5edb25c7dc92dbc6553e5a19c015), revision `6f5b557e037f5edb25c7dc92dbc6553e5a19c015`, 26,086,635,760 loaded parameters; published native [runtime](https://github.com/denis-pplx/autojev-27b/tree/ee63c1515980491a742f0bd0685c8dc5ca1f00c3) revision `ee63c1515980491a742f0bd0685c8dc5ca1f00c3`, temperature `2.207568021892729`.
- Full package/source reattestation matched native fingerprint `d0b1e161c17d60889744b6ccb5fa588bf80f9856f8535e6a04e083ffcf667ca2`; adapter SHA-256 `2075676129c663a93cac4c903ea33cb3116115d32fb7afd2fdce04a41ba3a3ec`. Fixed reference-runtime smoke: 32 prompts, zero category changes and maximum option-probability drift `0.0140843`, below the previously fixed `0.02` comparator bound. The accelerated runtime previously failed and was not used.
- The signed [preregistration](autojev27-v3-public-peer-prereg-2026-09-28.md) at `248bb75d2` precedes prediction. Typed FINAL prompts `e2a4a86bc978fc7497823e106533d8aa896a0307453d712f7bf99ee3174e87bd`, CSS15 prompts `7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6`, and public231 prompts `642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd` all match the preregistered bytes. The public panel manifest SHA-256 is `e0e7c67701cf05f996d3b4eac09abf1bb3e6bb363cfe007089acd3644a38ed35`.
- One primary full inference was retained per panel. Before scoring, prediction SHA-256s were typed `36abd1fc29d8a755e5e2a3bb30b2068c8f41165a68df75f749e1c5804eb8b769`, CSS `d64f330530f627e5f8814e1f162e05e88bccef9c0dfe1b924b0140edc1fe704d`, public `00bf4fd34ea5cc678b6c98b6ac91d0587f9df32bc1a6d6ffc5ff7ac881c53ce4`. The gold-free seal SHA-256 is `afba189f24f2cff2609e184cab5db8365092ab442f212b094ab8d12dcc2cce4e`, completed before any label read. It checked every input digest, ID, question key, model/runtime hash and denominator. Independent read-only review confirmed this chronology and the unchanged predictions.
- **Audit erratum:** an older 27B roster-plan JSON and the first seal implementation had a public prompt hash missing one `4e`. The signed peer preregistration and original public panel manifest contained the correct full hash. The initial seal failed without reading targets (private failure receipt SHA-256 `4ad1d80b1a25ee17b813cf398b396e900a8a7fd1cc8b1ccf2b20da32e47e2ffd`). Signed fix `83fa194bb` corrected only the sealer hash and bound it to the earlier peer preregistration in a regression test. The same three prediction files, not a rerun, were then sealed. The stale roster cannot be used as a passing 27B candidate preflight.

## JevArena v3: 8,147 original questions, 8,547 scored answers

`T` is the mean accuracy of the four typed FINAL families across 1,600 original
questions and 2,000 scored answers. `H` is the median macro-F1 of 15
human-label transfer tasks across 6,547 questions. The predeclared aggregate
is `100 × sqrt(T × H)`.

| Metric | AutoJev 27B |
| --- | ---: |
| Typed `T` | 0.886875 |
| Human transfer `H` | 0.589571 |
| JevArena v3 aggregate | **72.3101** |
| Typed correct / scored | 1,819 / 2,000 |
| CSS valid / items; micro correct | 6,534 / 6,547; 61.22% |
| Typed Brier / ECE10 | 0.06392 / 0.08223 |

| Typed FINAL type | Correct / total | Accuracy |
| --- | ---: | ---: |
| Choice | 800 / 800 | 100.00% |
| Noul | 737 / 800 | 92.13% |
| Score | 282 / 400 | 70.50% |

The typed families scored constraint competition `1.0000`, exception stack
`0.8425`, evidence join `1.0000` and resource ledger `0.7050`. All 2,000 typed
answers were valid. CSS had 13 native invalid answers, which
remain in its full denominator. The transfer weakness is material: among the
15 task macro-F1s, `tropes` was `0.1868`, `talklife` `0.2381`,
`conv_go_awry` `0.4091`, whereas `flute` reached `0.8843` and `mrf` `0.7731`.
The full per-task and per-item reports remain in private experiment storage;
no restricted source records are copied here. The previous typed DEV result
of 400/400 Score is not a v3 result; its drop to 282/400 FINAL warns against
treating a familiar development template as cross-source skill.

## Separate JevBench public subset: 231 questions

| Difficulty | Correct / total | Accuracy |
| --- | ---: | ---: |
| Easy | 48 / 48 | 100.00% |
| Standard | 71 / 72 | 98.61% |
| Hard | 81 / 111 | 72.97% |
| **All public** | **200 / 231** | **86.58%** |

All 231 public answers passed the scorer's validity rule. This is an
independent rerun of the **public subset**, not JevBench's unavailable sealed
official result or a complete ranking. Public subset report SHA-256:
`df3ebc7fd125f3c32bb66b3b0f3d501ac2ac4a24c68d6040394e13fa6808c786`.
Typed/CSS score-report SHA-256s are respectively
`9449e3cd003d75ec033326189cc6b6487f0f9c7ba6ffab90e92df7fcc8c1d02f`
and `213094121fa8190a59eada4b28ee4c312f7a1dd0da21186b5d2c18a14a3442ad`.
The v3 validator recomputed `T`/`H` and confirmed every scorer's prediction
hash matched the pre-score seal.

## Cost and decision

The three complete one-GPU process windows took **193 + 574 + 100 = 867
seconds, 0.2408 observed GPU-hours**, from process launch to exit. This is
reserved-wall time, including model load and excluding the two additional
32-question smoke processes, downloads and CPU hashing/scoring; the latter
smoke start times were not instrumented, so no precise all-in GPU-hour claim
is made. Accounting receipt SHA-256:
`c307ed48f88f950e0be93520ff1d08d79af6c7f9349d8a77dce212b2e49f7244`.

AutoJev is now a qualified strong near-identical-size peer for future frozen
27B same-panel comparisons. This result does not alter our official-Qwen 27B
candidate's development HOLD, does not turn its previous development score
into a formal score, and does not authorize an HF model publication. Before
any Decision 2.0 27B rank, the candidate needs its own qualified complete v3
and public231 predictions plus same-panel evidence and truthful label-exposure
disclosure. External peer training/evaluation overlap cannot be independently
excluded from the released model's public documentation.
