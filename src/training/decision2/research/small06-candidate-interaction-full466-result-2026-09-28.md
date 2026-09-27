# 0.6B candidate-interaction head: fixed 466-update result

**Decision: HOLD.** The prospective architecture-only contrast completed its
single fixed-budget arm and failed both frozen SELECT thresholds. This is a
development result, not JevArena or JevBench evidence and not a release
candidate. No typed DEV, CSS pilot, CAL scoring, formal, public or Hugging Face
evaluation followed the failure.

## Identity and technical gates

- Implementation and [preregistration](small06-candidate-interaction-prereg-2026-09-28.md)
  are signed in Git commit `241d79b30706ce9aaa611635987cd28e1d70e990`.
  The completed shared-head control has the same official
  `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`
  source, rights-clean v2 TRAIN/SELECT/CAL groups, 4,094,489 tokenizer-measured
  TRAIN tokens, seed, 466 optimizer updates, objective, schedule, complete
  8,192-token cap and native System One renderer. Only the head changes.
- CPU source/data/partition audit, 16 focused contracts and repository
  `make check` passed. Gate lock SHA-256:
  `a824569bb817d3f0530f16c94cbcb5938553e20e10aec6e46c5a44b46abc662f`.
- The real-source zero-step treatment and control logits matched exactly.
  After the fixed first 16 TRAIN examples, the treatment saved and reloaded
  with zero category changes and zero probability drift on 32 SELECT inputs.
  The technical receipt SHA-256 is
  `b2eaf82a060976c71beb9aae26afb4bd66f734cc84a1f46c55aaa23d352b0960`.
- The existing 400-group paired-order typed DEV input roster was frozen
  before optimizer work, without labels. It comprises 200 Choice, 100 Noul and
  100 Score groups; SHA-256
  `e834300bc9c578b0400f58b3e5eee0e14aa4c3c8bd3795cfc5118d6a5db65fb5`.
  Its post-SELECT test was never run because SELECT failed.

## Complete fixed SELECT trajectory

| Update | Shared-head control correct/700 | Interaction correct/700 | Shared-head family macro | Interaction family macro |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 329 | 246 | .38482 | .31844 |
| 128 | 441 | 236 | .51337 | .33231 |
| 192 | 460 | 254 | .57700 | .32132 |
| 256 | 501 | 246 | .67042 | .32885 |
| 320 | 542 | 268 | .73565 | .33438 |
| 384 | 552 | 283 | .75963 | .36975 |
| 448 | 557 | 325 | .75972 | .39820 |
| 466 | 562 | **333** | .77259 | **.43174** |

The precommitted gate required at least **562/700** correct *and* family macro
accuracy **.77259**, with 700 valid answers. The sole run selected update 466
by its fixed family-macro, normalized-Brier, earliest-step rule; even this
checkpoint misses the gate by 229 correct and .34085 macro points. All eight
prescribed reports have 700 rows. The training process exited successfully at
update 466, and CAL remained untouched for scoring. The private HOLD receipt
SHA-256 is
`c73f19bae1f11c30d6f4ad2b5099265e5ac9c89dfb8347de17eb0f83a1670709`.

At the final SELECT checkpoint, family correct counts were: human GoEmotions
Choice 81/200 versus control 161/200; human GoEmotions Noul 107/200 versus
175/200; narrative 86/130 versus 130/130; open-world abstention 11/40 versus
40/40; string composition 13/40 versus 24/40; and targeted quantized median
35/90 versus 32/90. The three extra correct Score examples on this one opened
SELECT family do not support a general ordinal improvement. Noul's direct
head path is unchanged, yet its fall suggests the jointly trained backbone or
shared head was adversely affected; the current contrast does not isolate
which mechanism caused it.

The full arm used one GPU for **1,076.324 wall seconds = 0.29898 conservative
GPU-hours**, below the 2.0-hour limit; zero/one-step preflight used another
17.661 GPU seconds = 0.00491 GPU-hours. No additional model evaluation or
checkpoint search was performed.

## Next source-only contrast (proposal, not launched)

The official general posttrained `Qwen/Qwen3-0.6B` current immutable Hub
revision is `c1899de289a04d12100db370d81485cdf75e47ca` (checked with the
Hugging Face CLI). A useful next arm would compare that official source with
the completed official Base shared-head control while keeping the **shared**
head, native renderer, same TRAIN/SELECT groups and matched optimizer budget.
First pin the downloaded weight-file hashes and actual loaded parameter count;
then audit tokenizer-measured exposure. If its tokenization differs, prospectively
define a matched-token design before training. Require source-matched zero-step,
length, numerical and save/reload preflight and a new independent transfer
check before any formal post-key interpretation. This source contrast has no
new GPU/data result and needs its own signed preregistration.
