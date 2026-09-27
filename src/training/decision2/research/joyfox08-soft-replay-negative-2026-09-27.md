# Joyfox 0.8B source-probability replay: fixed-budget negative arm

**Decision: HOLD.** This is an open development diagnostic, not a JevArena v3
result or a Decision 2.0 release candidate. No sealed typed FINAL, CSS 15-task
labels, or JevBench public subset were opened for this arm. The matched
hard-label control was reused, not retrained.

## Frozen intervention and provenance

The preregistration is
[`joyfox08-soft-replay-source-disjoint-design-2026-09-27.md`](joyfox08-soft-replay-source-disjoint-design-2026-09-27.md).
The pinned source is `joyfox/Qwen3.5-0.8B-JEV@ae7b7040aeff7802f6f2bcfdd27f08a72d5cd969`
with native code `joyfoxai/jev-inference@2677b5a3714489847668175de793e2d92fe183f0`.
Source `pilot.py` and `train.py` SHA-256 values remained the same as the saved
hard-label control (`c5d76e83…` and `99cc3445…`). The TRAIN sample was exactly
512 rows, 91,956 native encoded tokens, SHA-256 `ecb50a755c351c903a72a73a285d54622b141fce3e09809bf79d609ae8d2e532`.
The 512-row source-logit cache SHA-256 was
`2e4cd2c94167469ab7de1a4a50d838100671742bdf18293c19c9dcf7b13917c9`.
Two independent BF16 source processes agreed on all 32 frozen smoke rows:
zero category changes and maximum probability drift `0`, below `1e-6`.
Zero-initialized LoRA also reproduced source logits on all 32 with maximum
absolute drift `0`; the replay loss and LoRA gradient were finite.

The only optimizer change was
`0.2 × 2² × KL(source(T=2) || student(T=2))` added to the existing
hard-label CE + 0.25 Brier objective. The source head stayed frozen. The
rank-8 LoRA, sample order, BF16 native 1,024-token contract, seed, 64 updates,
learning-rate schedule, accumulation, and checkpoint selection were fixed.
Only step 64 was evaluated. Its adapter SHA-256 was
`7f03009c8d2a7651cea4bd6fdecc4a9dc088922c4c0d3591c1c0c03abc1d9acc`;
training receipt SHA-256 was
`27f19543ea36468841435bbb879b666be291ad94c67dfd09d2b042d1530292f7`.

## Open development results

| Panel | Source | Saved hard-label step 64 | Soft-replay step 64 |
| --- | ---: | ---: | ---: |
| SELECT family-macro accuracy, 700 items | .668426 | .670926 | .668426 |
| SELECT Score correct, 90 items | 32 | 32 | 32 |
| Typed DEV correct, 1,600 items | 993 | 995 | 995 |
| Typed DEV Choice / Noul / Score correct | 616 / 205 / 172 | 618 / 205 / 172 | 618 / 205 / 172 |
| Typed DEV Brier | .236526 | .236243 | .236057 |
| CSS pilot correct, 1,430 items | 510 | 509 | 510 |
| CSS pilot median task macro-F1, 3 tasks | .301191 | .299479 | .303202 |
| CSS pilot median task Brier | .781927 | .781430 | .781554 |
| CSS pilot valid items | 1,407 | 1,407 | 1,407 |

The soft model gained only **2/1,600** DEV correct over source, with **0/400**
Score gain. It missed the preregistered +32 DEV and +8 Score thresholds. Its
CSS median F1 gain was `.002011`, below the required `.02`. The CSS task-level
F1 changes versus source were discourse `-.000284`, implicit-hate `+.002011`,
and stance `-.002378`; no task had an acute `.01` drop. Brier shifted slightly
in favorable directions, but these small changes do not override the failed
capability and transfer gates. Soft replay beat the saved hard-label run on
CSS median F1 by `.003723`, while their typed DEV classifications were equal.

Paired bootstrap, 5,000 replicates: typed DEV family-macro accuracy
**soft minus source** point `+.00125`, 95% interval `[-.001875, +.004375]`
over the 400 independent four-variant groups. CSS pilot three-task median
macro-F1 **soft minus source** point `+.002011`, item-paired 95% interval
`[-.006049, +.010865]`. These intervals include zero and reflect the fixed
open task set; they do not establish new-task transfer.

The prior source DEV/CSS predictions were reused after matching prompt SHA,
model revision, native source revision and recorded Torch/HIP/Transformers
versions. Their original container image digest was not recorded; this is an
additional comparability limit for the open diagnostic. The new treatment and
saved hard-label control used the same pinned image and collector path.

The source model's original teacher-distillation lineage remains a limitation.
This experiment made no Jev API calls and used no archived Jev outputs for new
training. It tests one fixed, English-heavy, two-mechanism-dominated sample;
it does not show that replay is generally ineffective. The next 0.8B arm
should change Score mechanism coverage and real-label transfer data under a
separate preregistered, token-matched control, rather than increasing KL weight
after seeing this result.

## Cost and stop

All eight task-owned GPU containers exited successfully. Their wall-clock GPU
occupation totaled **0.169942 GPU-hours**, including source cache, repeat,
training, smoke, and hard/soft DEV/CSS pilot inference. The fixed 64-step
training container occupied `0.062163 GPU-hours`; its measured training loop
was `0.055967 GPU-hours`. The earlier hard-label optimizer control is excluded
from this new cost. The arm is stopped before CAL fitting, public231, sealed
JevArena v3, or publication. The private all-artifact receipt SHA-256 is
`1be2b42c2160e7f94316db59974040ced543f3006aa45b98229037b05731ec00`;
the source and hard/soft prediction and scorer hashes are bound there.
