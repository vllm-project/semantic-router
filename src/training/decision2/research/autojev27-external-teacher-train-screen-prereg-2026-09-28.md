# AutoJev 27B external teacher: bounded TRAIN-only screen

**Status:** prospective source screen, registered before loading this teacher on
the frozen TRAIN roster. The published
[AutoJev-27B](https://huggingface.co/denis-pplx/autojev-27b) is an external
distillation **teacher** and model peer, never a Decision 2.0 weight
initializer. This screen cannot establish student improvement, JevArena rank,
or publishable performance.

| Component | Frozen value |
| --- | --- |
| Weights | `denis-pplx/autojev-27b@6f5b557e037f5edb25c7dc92dbc6553e5a19c015`; published package manifest tree SHA-256 `d0b1e161c17d60889744b6ccb5fa588bf80f9856f8535e6a04e083ffcf667ca2` |
| Native runtime | `denis-pplx/autojev-27b` source commit `ee63c1515980491a742f0bd0685c8dc5ca1f00c3`, clean local Git checkout, `inference.autojev27` release verification and unmodified `autojev.model.DecisionModel`/`answer` |
| TRAIN | rights-clean v2, exactly 7,455 rows, SHA-256 `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755` |
| Roster | Same deterministic gold-free 96-group selection as the separate Eikos teacher screen, 32 independent groups per Choice/Noul/Score; SHA-256 `18dce35a5ca58864f1b92399344fab679ec98fb7ff4ddd05ee71cfeecebb1722` |
| Interface | Original TRAIN state, instructions, option descriptions and ordered Score levels rendered to one native typed System One question; no gold label in the request, no chat API, no prompt search |
| Output | 32-row/type denominators; native valid/invalid/tie, hard-label agreement, gold probability, half Brier sum; private aggregate only, mode 0600, no row text/predictions/labels |
| Runtime | `decision20-train-fast:host2` image SHA-256 `f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`; one newly checked exclusive GPU on the first authorized node, 15-minute wall cap including model load, no retry or parameter variation |
| Script | [`autojev_teacher_train_pilot.py`](autojev_teacher_train_pilot.py), SHA-256 `eabf3c9f9d9a7857519b026fa64c5b950fe6374d7f9af03ecb7e32db25713296` |

Admission order: confirm weight/source revision and untouched manifests; mirror
the signed local script; CPU-only strict TRAIN hash and roster dry run; verify
the image and live GPU ownership; then run at most one full 96-row GPU source
screen. `verify_release` recomputes the full package and source inventory.
Unsupported native context or candidate limits count as invalid without
truncation. Any unexpected runtime/format error stops the screen and leaves a
HOLD, not a substituted chat or projection score. Do not alter thresholds,
calibration, roster or revision after seeing a result.

All 96 outputs must be structurally valid for AutoJev to be considered for a
later, separately registered distillation arm. Agreement and probability
statistics are descriptive only on 32 TRAIN groups/type. Compare with the
Eikos TRAIN-only screen solely when both used this exact roster and native
semantics. A future student arm must predeclare allowed official/own weight
origin, teacher coverage, KL weight, equal student token and step budget,
source overlap screen, SELECT gate and independent confirmation. No weights or
probability vectors leave the teacher screen.

**No AutoJev TRAIN result exists at preregistration.**

## CPU and runtime lock before GPU

The signed source commit is `028d9cb86de9c19f561a6275e455ef3eb8f6fd47`.
Its exact script mirror has the SHA-256 above. The CPU-only dry run selected
96 rows from 96 independent groups and reproduced the fixed roster SHA.
The separate full package/source verifier recomputed the frozen package SHA,
26,086,635,760 parameters, model config SHA-256
`bacbcbb281a53af5ef5cc6c9028601097d155bf981129f18a727219517921dcd`
and clean runtime source tree SHA-256
`550ccd857350c6771a1de03e4bcba9fb4412247b58bcdc9e8ac03de2ab9641a5`.
No native teacher inference or gold scoring occurred in these checks. GPU 4
on the first authorized node was idle at the precheck; it must be rechecked
immediately before the single bounded run.

## Frozen source screen result

The one permitted native run finished in 74 wall seconds, approximately
0.0206 GPU-hours. The private aggregate (SHA-256
`a5b105f59435a62ae91fe4d5debb7455051d2f2c1b2f30d40bc073be8015d58b`)
was written with mode 0600 and passed row/roster invariants. Its model and
runtime hashes match the CPU lock. No student optimization or evaluation-set
inference was performed. The task-owned container exited and GPU memory
returned to its prior idle level.

| Native TRAIN type | AutoJev correct / 32 | AutoJev valid / 32 | Ties | Mean gold probability | Half Brier / row | Eikos correct / 32 on exact roster | Eikos half Brier / row |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Choice | 26 | 32 | 1 | 0.6936 | 0.1277 | 21 | 0.2193 |
| Noul | 26 | 32 | 0 | 0.7778 | 0.1294 | 26 | 0.1275 |
| Score | 25 | 32 | 0 | 0.4850 | 0.2037 | 10 | 0.3843 |

AutoJev has 77/96 unique-max hard-label matches versus Eikos 57/96 on this
one TRAIN roster, with equal Noul hard-label matches. This is a descriptive
teacher-source check, not a model rank: there are just 32 groups/type,
third-party training overlap is unknown, the models have very different sizes,
and no student has been trained. The Score contrast supports considering an
AutoJev soft-teacher arm only after separately predeclaring source checks,
matched student budget and independent SELECT/transfer gates. Do not use this
TRAIN result as a JevArena/JevBench or release claim.
