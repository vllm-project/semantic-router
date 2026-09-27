# Official Qwen3-0.6B posttrained source contrast: SELECT HOLD

This is the completed one-arm source-only contrast preregistered in
[`small06-official-posttrained-source-prereg-2026-09-28.md`](small06-official-posttrained-source-prereg-2026-09-28.md).
The treatment starts from the official general posttrained
`Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`, not a
third-party Decision model. The comparison is the already completed official
`Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`
shared-head full466 control. Source initialization is the only intended
variable. This is an internal development result, **not** a formal evaluation
or release claim.

## Controls and execution

The signed preregistration and code lock are commit `39209c0c3` and SHA-256
`c268cce8e08a847788fd15442c863183262283afd109959a8a2badf3a76d3c40`.
Official sources, Apache-2.0 LICENSE bytes, dataset partitions, all 8,855
native rendered input/option identities, and equal TRAIN token exposure
(4,094,489) passed the CPU audit. CPU receipt SHA-256:
`269427d9be15e8bf47876e0c8bfd62bdfa730e1135e6df4b6ef8b6077`.

The one-GPU zero/one-step technical gate passed: exactly matched fresh shared
head, 597,103,104 loaded parameters, stable normalized Choice/Noul/Score
probabilities, finite nonzero gradients for all three types, and 32/32 native
save/reload parity with zero probability drift. Technical result SHA-256:
`d897c1139e5d803553c3b2c1f48939c9b2c50cf9bc6358482a05edc376b43a8d`.
It consumed 18.951 single-GPU seconds = **0.00526 GPU-hours** of measured
model execution (27.311 seconds of conservative container wall time).

One fixed full466 arm ran to completion on one reserved GPU: one epoch,
7,455 TRAIN items, 466 optimizer updates, eight prescribed SELECT milestones,
no resume, no search and no changed threshold. The launch lock SHA-256 is
`dc601311c381e618aeaf9e09cfd92fb4faa243fef230251d2077cd889168ed94`.
The complete training wall time was **1,052.916 single-GPU seconds = 0.29248
GPU-hours**, below the frozen 1.5-hour cap. All 700 native responses at each
SELECT milestone were valid, distinct and finite. The private decision receipt
SHA-256 is
`547da02cd2478744e0787977fc7e8a0b6383427fa2e023633e53d1112cdb2ab2`.

## Eight fixed SELECT milestones

| Update | Base correct/700 | Posttrained correct/700 | Base six-family macro | Posttrained six-family macro | Posttrained Score-targeted correct/90 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 329 | 431 | .38482 | .59796 | 7 |
| 128 | 441 | 520 | .51337 | .70731 | 35 |
| 192 | 460 | 525 | .57700 | .71843 | 32 |
| 256 | 501 | 542 | .67042 | .73926 | 32 |
| 320 | 542 | 534 | .73565 | .72926 | 32 |
| 384 | 552 | 534 | .75963 | .73259 | 32 |
| 448 | 557 | 546 | .75972 | .74796 | 34 |
| 466 | **562** | **548** | **.77259** | **.75065** | 35 |

The frozen selector chose update **466**. Its 548/700 and .750648 macro miss
the preregistered advancement limits 562/700 and .77259 by 14 answers and
.02194 macro respectively. **Decision: HOLD**. The initial posttrained
advantage peaked early and did not persist to the control's selected
checkpoint. At update 466, compared with the Base control on the same SELECT
rows, GoEmotions Choice was 150 versus 161/200, Noul 173 versus 175/200,
narrative 130 versus 130/130, open-world abstention 40 versus 40/40, string
composition 20 versus 24/40, and targeted Score 35 versus 32/90. The small
Score-targeted +3/90 is neither a general Score repair nor independent
evidence; its across-milestone values were 7, 35, 32, 32, 32, 32, 34, 35.

No CAL scores, typed DEV, CSS pilot, JevArena formal, JevBench, other public
benchmarks or HF product evaluation were obtained for this arm. Therefore we
cannot conclude that official posttrained initialization repairs the opened
0.6B typed Score regression. The source-only hypothesis did not meet its
advancement criterion. A future Score-specific data or ordinal-loss contrast
would be a **new**, separately preregistered arm, with its own source-matched
control and a fresh independent transfer check before any formal claim.
