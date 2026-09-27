# 0.8B fixed-checkpoint temperature transport result

This is a **development diagnostic, not a release model or new calibration
selection**. The [protocol was signed](eos08-temperature-transport-prereg-2026-09-27.md)
as `ad0f286dd` before scoring. The exact source and frozen evidence are signed
as `5e68eee31`; local focused tests and `make check` passed. The two authorized
remote execution sites received the exact tracked Decision 2.0 source archive
SHA-256 `f44947e22d7dcb9400cdbe19394a736bdc6e8f9f2bb2572917dc72a614b29002`.
The diagnostic/evidence/temperature-transform/typed-scorer/CSS-scorer source
SHA-256 values are respectively `9cb0bd8d15767ce9365f28427e38b50c76ed444f09383ab290e03a8103fea216`,
`d5c63bcfe971ff692dd06bb6d35e6c8f43bc65331133601b0ad9f92f2ecd2bd5`,
`4808b1a497074cf663d1f443a7677170e2f9e49208ec96b19fa21fdbc482a910`,
`d02a3b2bbaa08ec45928fc354532b3c3b5aef80e0a5d8e9ed6348ad6d30e2bcc`,
and `cfe199a1826bb89b27c9eb746f808d74f16b46ca6585ac7b0ff7e440d44eeaca`.

All original CAL/prediction/sidecar/scorer/gold fingerprints matched the
preregistered evidence. The original calibrated score recomputed under the
current frozen scorer. Each of the six prediction panels contained every
expected row and no invalid, truncated, missing, zero/one or non-finite
probability. Positive-temperature inversion produced six private T=1
prediction files. Every per-item categorical answer, valid count and point
accuracy stayed identical to its calibrated source. Each private result
summary is sealed by SHA-256: Eos continuation
`fa5313398da158f6b1c893d22110ea5a3e10908394da471c10a83f8bbf101aa1`,
Qwen3.5 Base `2ae358521b80de0c3129ea072a5b6cc80d4b53b298b273e190586ca35dbdbaae`,
and Qwen3.5 Posttrained
`9e89468d315fb1d3994e0ee889437ac84d1c63c1227f77f6242a3121caa137e3`.
Row-level outputs and human gold remain private.

## Matched checkpoint: calibrated temperature versus T=1

Lower Brier/NLL/ECE is better. Brier uses the typed scorer's half-scaled
multiclass convention for DEV; CSS uses full-sum multiclass Brier, so their
absolute values should not be compared with each other. CAL is the already
used fit partition; DEV and CSS pilot are separate development panels.

| Checkpoint and panel | Calibrated Brier | T=1 Brier | Calibrated NLL | T=1 NLL | Calibrated ECE | T=1 ECE | Correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Eos continuation CAL900 | .293903 | .304432 | 1.130804 | 1.170140 | .083050 | .117457 | 399/900 |
| Eos continuation typed DEV1,600 | .399506 | .392865 | 2.037176 | 1.541786 | .270937 | .293565 | 795/1,600 |
| Qwen3.5 Base CAL700 | .140556 | .141096 | .510943 | .524243 | .032850 | .040096 | 548/700 |
| Qwen3.5 Base typed DEV1,600 | .346224 | .336884 | 1.136343 | 1.108561 | .196901 | .178500 | 392/1,600 |
| Qwen3.5 Posttrained CAL700 | .172195 | .173290 | .606198 | .610653 | .050923 | .039585 | 500/700 |
| Qwen3.5 Posttrained typed DEV1,600 | .365836 | .352019 | 1.191291 | 1.150892 | .177429 | .154263 | 509/1,600 |

For Eos, the overall DEV Brier gain from T=1 is `.006641`, whereas CAL-fitted
temperatures are better by `.010530` on the very CAL partition where they were
fit. The DEV difference is concentrated in Score: calibrated/T=1 Brier
`.776907/.724290` and NLL `5.456240/3.360901` on 400 Score items. Eos's
Choice and Noul DEV Brier instead favor their CAL temperatures:
`.244926/.248045` and `.331264/.351078`. The previous observation that the
Eos continuation has worse typed DEV Brier than the published 1.0 Eos
(`.399506` versus `.36231`) therefore cannot be reduced to temperature alone:
even its T=1 `.392865` remains worse. Its typed DEV correct count rises only
three questions over Eos 1.0 (795 versus 792).

Both Qwen3.5 controls show the same Score direction on independent typed DEV:
Base calibrated/T=1 Brier `.402870/.360640`, Posttrained
`.455064/.402934`, each on 400 Score items with 85 correct. Their CAL700 has
only 90 Score rows, on which the chosen temperatures optimize NLL. This
CAL-to-DEV reversal is evidence of temperature transport failure for this
particular Score distribution; it does not establish that T=1 will calibrate
another unseen distribution. On typed DEV, Posttrained improves point accuracy
over matched Base by 117/1,600, entirely through Choice (232 versus 115 of
800); Noul and Score correct are tied at 192 and 85. Temperature scaling leaves
all these point decisions unchanged.

The CSS three-task pilot is Choice-only. Its calibrated/T=1 median task
full-sum Brier and median task ECE15 are:

| Checkpoint | CSS correct / 1,430 | Median Brier calibrated / T=1 | Median ECE15 calibrated / T=1 |
| --- | ---: | ---: | ---: |
| Eos continuation | 483 | .821650 / .885647 | .103678 / .212902 |
| Qwen3.5 Base | 481 | .811886 / .816781 | .078850 / .107451 |
| Qwen3.5 Posttrained | 439 | .821684 / .832394 | .085290 / .114108 |

Task detail matters: Eos's CAL temperature improves discourse Brier
`.885647→.821650` and implicit-hate `.914274→.822711`, while T=1 improves
SemEval stance `.564067→.538400`. The Base/Posttrained CSS point gap (481
versus 439 correct) is therefore an initialization/training transfer issue,
not a post-hoc temperature artifact. The original comparison on the exposed
public 231-item subset also favored Base 129 versus Posttrained 118; it was
not used in this calibration diagnostic or for model selection.

## Decision and next independent experiment

**Keep all three checkpoints private as development controls.** No
`dev-2.0-0.8b` qualifies. Do not replace the published/native temperatures
with T=1 based on already inspected DEV or CSS pilot labels. The two panels
pull in opposite directions, and Eos's overall ECE worsens under T=1 despite
its Brier improvement. A future new candidate needs an independently drawn,
source-disjoint CAL with enough Score examples and a frozen domain/level mix,
then a predeclared transport gate on untouched typed and human transfer tasks.
Temperature cannot repair the observed categorical transfer reversal. Any new
training arm should first test data/objective/backbone changes against that
same-panel transfer and Score gate; repeating the completed Base/Posttrained
recipe or tuning on exposed public items would add no independent evidence.
