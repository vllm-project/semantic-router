# Official Qwen3.5 2B matched-source full arm: development HOLD

**Decision: HOLD for both arms.** The two separately admitted, official-source
arms completed the frozen 466-update budget and native reload test. Neither
cleared the preregistered `+2.0` development-proxy promotion threshold over
our own Decision 1.0 Sol 2B. No protected JevArena v3, public JevBench,
calibration-based rescue, Hugging Face upload, or alternative checkpoint
selection followed this result. The results below are **development** evidence.

This is the completed readout of the [matched full-arm protocol](qwen35-2b-official-dual-source-full-prereg-2026-09-27.md)
and its [source preflight](qwen35-2b-official-dual-source-preflight-2026-09-27.md).
The two arms share all training and inference settings except the official
initial weights/tokenizer at the pinned revisions. Both initialize a new
dynamic-option head; neither starts from a third-party Decision model.

## Frozen identity and selection

| Item | Base arm | Posttrained arm |
| --- | --- | --- |
| Direct source | `Qwen/Qwen3.5-2B-Base@b1485b2fa6dfa1287294f269f5fb618e03d52d7c` | `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Source weight SHA-256 | `928acbf11878c32185bbd863514d191769285065ab9ea14fbfe431303f5fdf2d` | `aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1` |
| Source tokenizer SHA-256 | `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927` | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| Completed updates / exit | 466/466; exit 0 | 466/466; exit 0 |
| Final frozen SELECT BEST | step 448 | step 466 |
| BEST SELECT correct / 700 | 587 | 583 |
| BEST SELECT family-macro accuracy / Brier | `.797407` / `.132004` | `.805185` / `.128074` |
| Inference model fingerprint | `2a3edab8839586f5df783fa14086d2bd110ad0d4e2bb3b5bb58e94076376895c` | `1e19dbccc42cc36193b9ecbe8bdf0510334edfcdf502737052b86ec1eabedc3d` |
| Provenance SHA-256 | `c67f27832f52f9a7a6d26f748b9da8051b8ee385ba103edf198cfccc5ab3ea95` | `2201f16619fe47005423a9d0090f5ae88d6170443314ade8cbb3084cf99a2a27` |
| BEST / COMPLETE SHA-256 | `0d1c5055fb324d60b237652e90c6187136170ed4d9a6673ee89666826fa89420` / `574923845588f4a2dae1b5eb344c6aa963e17880557d5894ba916cb34f6b8c81` | `e96277b9e0b07b7ef8c8df4b2fcbb71b7bdb9e8b4b8ed58816d8381dc221e3b6` / `09e1066dd7c2a345d6d3e03fad3fd0e06749d58dc499bde1e46a84d90dfe328e` |

The common rights-clean v2 TRAIN 7,455, SELECT 700 and untouched CAL 700
SHA-256 values are respectively `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`,
`32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`
and `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`.
Both consumed 4,194,465 full native TRAIN tokens with no source/data/hash
contract difference. The shared training code SHA-256 values in each
`provenance.json` matched the exact local code and remote read-only mirror.
Both used the same runtime image ID
`sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`.
CAL was neither used for checkpoint selection nor fit after the failed screen.
The local and remote mirror SHA-256 values matched for native inference
`b25382a2667f4a82167d52e35a54dadea6b0838eef407127b23f5dc45b18914b`,
typed scoring `d02a3b2bbaa08ec45928fc354532b3c3b5aef80e0a5d8e9ed6348ad6d30e2bcc`,
CSS scoring `cfe199a1826bb89b27c9eb746f808d74f16b46ca6585ac7b0ff7e440d44eeaca`,
and the 32-row reload checker
`19160f9d7aeb0a1726c0a4efb7887457b97a61af9ff0c6b9d65eb0f174a373f1`.

The prescribed first 32 SELECT rows were reloaded with batch size 2 at each
frozen BEST. Both returned 32/32 with **zero changed categories, zero p99
probability drift and zero maximum drift**. Each comparison receipt SHA-256 is
`3a8e165b2a3a1ce188072a477216a19357025ee6b8e33d6ada3240b1f2a5ee77`.
There was no checkpoint replacement after this test.

## Same-panel development result

Every candidate answered all 1,600 typed DEV and 1,430 human CSS pilot
requests; both panels had zero invalid, missing, overflow or truncated
answers. The same-panel own Sol 1.0 control was the already frozen complete
native run, not a newly trained baseline. Its typed family macro accuracy
`T=.590625` and pilot task-median macro-F1 `H=.3151728272` give proxy
`100*sqrt(T*H)=43.14498`.

| Development metric | Own Sol 1.0 | Official Base BEST448 | Official Posttrained BEST466 |
| --- | ---: | ---: | ---: |
| Typed DEV four-family `T` | `.590625` | `.575000` | `.490625` |
| Typed correct / 1,600 | 945 | 920 | 785 |
| Choice correct / 800 | 416 | 487 | 471 |
| Noul correct / 400 | 218 | 192 | 219 |
| Score correct / 400 | 311 | 241 | 95 |
| Typed normalized Brier ↓ | `.278928` | `.278036` | `.335474` |
| CSS pilot task-median `H` | `.315173` | `.289243` | `.358025` |
| CSS pilot micro correct / 1,430 | 554 | 538 | 587 |
| CSS pilot median Brier sum ↓ | `.808894` | `.839039` | `.789799` |
| Frozen proxy `100*sqrt(T*H)` | **43.14498** | **40.78168** | **41.91133** |
| Proxy difference from own Sol 1.0 | — | **−2.36330** | **−1.23365** |

The Base arm improved typed Choice by 71 questions but lost 26 Noul and 70
Score questions. Its pilot discourse / implicit-hate / stance macro-F1 values
were `.246248/.289243/.557198`, versus own Sol
`.306738/.315173/.481115`; one strong task gain did not repair the median.
Base Score predicted levels 0/1/2 on `199/70/131` of 400 questions, so it
did not collapse to a single level, but its Score accuracy `.6025` remained
below own Sol `.7775`; Score Brier was `.228445` and expected-value MAE
`.537975`.

The Posttrained arm improved all three human pilot task macro-F1 values to
`.343496/.358025/.486224`, but typed Score fell to `.2375`. It predicted
levels 0/1/2 on `390/0/10` of 400 Score questions; Score Brier `.542086`,
ECE10 `.577018` and expected-value MAE `1.044871` expose severe ordinal
bias and overconfidence. The SELECT gains did not transport to typed Score.
Neither arm reaches even the unchanged own Sol proxy, so the frozen `+2.0`
promotion rule fails without relying on a confidence interval or public
benchmark. No formal roster was prepared because neither arm qualifies.

The raw gold-free typed DEV and CSS pilot input SHA-256 values are
`a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a`
and `598319a429de16c659b59ede0eac3c269939356b059f4e08d1983e44def3dda`.
All four predictions were sealed before development scoring at
`2026-09-27T12:55:14Z`; the private seal SHA-256 is
`4f137318c00c8a4168cf8b534a6de6e69d6ec105437e51aaf40c55b9c8ae36ef`.

| Arm | Typed prediction / score SHA-256 | CSS prediction / score SHA-256 |
| --- | --- | --- |
| Base | `4f5785df653b39c649c4553560d20ff220540ebc5d665a22980047c666813987` / `2faa60677c20e02a7b1b0854a4ee7b5821be03fc5f0ee606ee4607f5a5494d37` | `4d8cd5cba40739fa0c13af72991797778e13e2c7a0e6446307a1bfcc3975db4f` / `5978cf62a510cf0f02cfa7ba7e0db3431a60e12065e66d0554f6c3fb1f45819a` |
| Posttrained | `eb8725c4ee2059fc5d76c4ad8b106d71fd7ff42cf18d3a7129056f4b210b2508` / `830a0b7c9cc6dada4294f072d8627d27f72cf832dc70acaabb59bc0e1a0f707f` | `dab051f028eb89d1471acc71af6d7ef51af8d3fe979ce5bd644e1579aff57f2b` / `b5460801f7dc43813b7b85028ac60f33e36c89a212e7a04c27876a212884207f` |

## Resource use and next discriminating experiment

Docker start/exit intervals on one reserved GPU per arm give `.95332` and
`.98519` GPU-hour for the complete Base and Posttrained training runs.
Their six reload/DEV/CSS inference containers add `.12222` GPU-hour combined:
**2.06072 GPU-hours** for full training plus development readout. The earlier
separately frozen zero/one-step admission consumed approximately `.0957`
GPU-hour combined. CPU scoring is excluded. All eight task-owned containers
exited zero; their evidence and selected checkpoints remain private.

**Next experiment, not launched:** keep these two selected packages as fixed
controls. Construct one audited, source-disjoint training mixture that
replaces a fixed token share of the English short-form rows with (a) real
human-label transfer tasks from sources absent from the 15-task evaluation
and (b) independently checked, balanced three-level Score questions. Require
the Score set to pass the outstanding quality review before training. Apply
that *same frozen mixture*, full-input token budget, optimizer and SELECT
rule to two new arms initialized separately from the same pinned official
Base and Posttrained revisions. This matched source-by-data comparison tests
whether the Base CSS deficit and the Posttrained ordinal collapse respond to
data composition. Freeze exact rows, token counts, group overlap audit,
control contrasts and stop threshold first; do not look at JevArena v3 or
public231 to set the mix. If either new arm advances on open development
panels, obtain a new gold-free post-key formal lock and independent
corroboration before a broad release claim.
