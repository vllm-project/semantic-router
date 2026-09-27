# Decision 2.0 27B: same-panel development peer screen

**Decision: formal v3 HOLD.** This is a gold-free-prediction, subsequently
scored **development** comparison on typed DEV 1,600 and CSS pilot 1,430. It
contains no typed FINAL, CSS 15-task, JevArena v3, or JevBench peer result.
The known v3 label exposure elsewhere also prevents calling a later v3 run a
virgin blind test. This screen qualifies a strong same-size open opponent and
identifies the next capability gap; it does not authorize 8,147-item inference
or publication.

## Fixed identities and comparable inference

| Item | Frozen identity |
| --- | --- |
| Our weight start | Official `Qwen/Qwen3.8-27B` commit `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; clean-v2 BEST checkpoint `0000368`, native fingerprint `d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2` |
| Our proposed public identity | `llm-semantic-router/DEV2.0-27B`, currently a private experiment package; manifest SHA-256 `47f1e934062ec5d8f75dc6b74812c4fbfaf2d567f1a099f7f9f5cb2d41ffe782`; 25,688,227,840 loaded parameters |
| Our source/package identity | Full matched-GPU DEV+CSS native parity receipt SHA-256 `d53236dabc9c4e09479f6eef72aff1bb367426ecd9ee6f7c2f1603a4b95c4426`: zero categorical changes and zero p99/maximum option-probability drift. The two CSS over-budget answers remain invalid. Existing scored source files therefore describe this byte-matched package on these development panels only. |
| Open same-size peer | [`denis-pplx/autojev-27b`](https://huggingface.co/denis-pplx/autojev-27b/tree/6f5b557e037f5edb25c7dc92dbc6553e5a19c015) commit `6f5b557e037f5edb25c7dc92dbc6553e5a19c015`; published native [runtime](https://github.com/denis-pplx/autojev-27b/tree/ee63c1515980491a742f0bd0685c8dc5ca1f00c3) commit `ee63c1515980491a742f0bd0685c8dc5ca1f00c3`; 26,086,635,760 loaded parameters. |
| Peer file and runtime audit | Full-file/package receipt SHA-256 `11335b3a8d222b9106cac3eb86546bdb82c9775c02d60254472b01c45eaad935`; attestation SHA-256 `5d154ee21792b99709110bc7cc1cd97af616608e614ae7ce5e7ebc0873cf3f3c`; native model fingerprint `d0b1e161c17d60889744b6ccb5fa588bf80f9856f8535e6a04e083ffcf667ca2`. |
| Peer calibration and panel | Published temperature `2.207568021892729`; fixed native adapter and no truncation. The peer's gold-free DEV and CSS prompt SHA-256 values are `a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a` and `598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda`. The scored panel gold digests match our reports exactly: DEV `c7a8b86bda0d0d6120e572b94dfc756bf10264108554af76307141ae02fbf5dc`, CSS pilot `9a7274760dc4ced5ce5219b300974a1cf54c7d5e7e0c7de05d78bb33f2959391`. Native adapters can render the same questions differently. |

The external peer's native adapter was qualified *before* these scored
panels. The fixed 32-item gold-free smoke prompt SHA-256 was
`3376ed4093c7efb591519912b88605960923cf33fd19d8adfa5018eb02270a55`.
Two independent no-FLA PyTorch-reference processes had **0/32 categorical
changes, zero maximum/p99 option-probability drift, and zero invalid answers**;
repeat receipt SHA-256
`0814d2c0cb6c21dd29966760a5bdb62089f6e7e4e1d033696e467b83fb6ea14e`.
Its first complete scored run is primary regardless of score. The original
peer decision config and calibration were not refit on these panels.

The accelerated flash-linear-attention (FLA) path is a **negative runtime
result**. Two fresh 32-item processes kept all categories, but maximum
option-probability drift was `0.02794454698995147`, above the previously
fixed external-peer tolerance `0.02`; failed receipt SHA-256
`2b4d947338d603c0c2a8c2b88ccc238665923bfc93e85fce8cb37b7e2e88b44c`.
It was not used for DEV or CSS and must not be relabeled qualified. The first
GPU visibility attempt failed before model output; the corrected reference
runtime uses one physical GPU. These failures are retained in private logs.

## Development scores, same questions and scoring programs

| Model | Loaded B | DEV overall | Choice | Noul | Score | Invalid DEV | CSS pilot overall | CSS task-median macro-F1 | Invalid CSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Our BEST368, package parity passed | 25.688 | 1,211/1,600 = 75.69% | 791/800 | 258/400 | 162/400 | 0 | 845/1,430 = 59.09% | 0.62052 | 2 |
| AutoJev native reference | 26.087 | 1,505/1,600 = 94.06% | 800/800 | 305/400 | 400/400 | 0 | 928/1,430 = 64.90% | 0.66605 | 0 |

CSS per-task macro-F1 on the frozen pilot is, in task order discourse,
implicit-hate and SemEval stance: our model `0.62052 / 0.47488 / 0.68757`,
peer `0.66605 / 0.52952 / 0.77209`. Peer DEV coverage-adjusted Brier is
`0.03949` and ECE10 `0.02960`; ours are `0.19625` and `0.13790`. The
peer's scored DEV prediction/report SHA-256 values are
`5b2878c8f9d06f797dc623f1fff928090622991f4b1cbfe89f68d81adfaada39`
and `8e4e922f8b8507751835467a9a3249ba426bfdc791c0aed1105be42032253133`.
Its CSS pilot prediction/report values are
`b93a32df1d11454dcebb83c8504d68a86c242c3f57fb3eb1d601fdccbc2ef012`
and `6a656811d97b530da269bf6bbee117a42e304b83887b27e081ee279ec8f3aaba`.
Our existing same-panel source DEV/CSS score report digests are
`eb903d873d7f0bb6f1180b1608c45e62a83a3854c9317185f28e749f1c034aa6`
and `f633d86d68d064bd2400c27c3cd9e2bd0e2c508de4ba5529aeb12eeed15ad5a6`.

The purely diagnostic `100 × sqrt(DEV family-macro accuracy × CSS pilot
task-median macro-F1)` is **79.15** for AutoJev and **68.53** for our
BEST368, a **10.62-point** gap. It is not JevArena v3: neither axis is the
v3 FINAL or 15-task transfer panel. AutoJev's 100% on several generated DEV
families can reflect template familiarity; the three human pilot tasks give
the separate transfer view. No Decision Index historical score is mixed into
this comparison.

## Resource accounting and next decision

All new peer inference used one GPU at a time. Seven successful reference or
FLA inference processes (two reference smoke, three accelerated attempts,
DEV and CSS) account for **526.8 seconds = 0.1463 observed GPU-hours**, measured
from each first model-runtime log timestamp to its prediction-file completion
timestamp. This is an approximate active window, not billed allocation; it
excludes HF download, CPU file hashing, shell startup and the failed
pre-model GPU-visibility attempt. Summed per-item inference latencies were
about 0.109 GPU-hours, a lower bound on actual process windows. A conservative
task reservation ledger may charge **<0.3 GPU-hours** for this peer screen.
No new 27B training or formal 8,147-item evaluation was run here.

The current candidate's principal shortfall is ordinal Score **162/400 vs
400/400** on DEV; CSS human transfer also trails the peer on all three tasks.
This development result alone does not establish release failure on v3, but
it makes a full v3 run poor information value now. The next decision is the
separately preregistered, matched-budget Score data/objective experiment in
[`decision2-27b-score-minimal-ablation-prereg-2026-09-27.md`](decision2-27b-score-minimal-ablation-prereg-2026-09-27.md).
Keep BEST368 and the rejected v6 pilot immutable. Only a new qualified
candidate, same-panel development recheck, package parity and then a frozen
v3 roster could justify lifting HOLD. That v3 comparison must disclose prior
label exposure and needs truly untouched or external confirmation for an
independent-validation claim.
