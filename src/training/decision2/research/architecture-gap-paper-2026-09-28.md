# Decision 1.0 paper comparison: what Decision 2.0 has and has not changed

The primary source is the project's [Decision 1.0 paper](https://vllm-sr.ai/decision-paper), specifically Sections 3–5. This note compares architectures and training evidence, not historical benchmark scores with JevArena v3.

## Observed architecture and data differences

| Component | Decision 1.0 paper | Current eligible 0.6B Decision 2.0 candidate |
| --- | --- | --- |
| Direct weights | Kai: Vela/mmBERT encoder | Official Qwen3-0.6B-Base causal decoder |
| Candidate representation | Joint bidirectional attention to typed question, candidates and state | Causal candidate endpoints plus suffix-final query |
| Type specialization | Separate 22-layer Choice, Noul and Score paths, interaction layers and scorers | One shared candidate head across the three types |
| Training described | Kai: approximately 21,000 complete-input Choice/Score examples, retaining an earlier Noul path | Rights-clean v2: 7,455 TRAIN rows, including 516 Score rows, from a fresh official initialization |

The current 0.6B candidate differs substantially from Kai, but its shared causal readout follows the existing Decision 1.0 decoder branch's basic candidate-scoring design. It is not evidence by itself of a new superior architecture. The paper also says its families were released systems with different recipes, not a controlled comparison of attention direction or head sharing.

At 0.8B, 2B, 4B and 9B, the completed eligible 2.0 experiments either continue our corresponding 1.0 decision weights or adapt official Qwen3.5 weights with the same basic causal candidate-head family. They test initialization, data and optimization more than a new topology. The 27B work adds an official Qwen3.8 text backbone and a new size tier, but still uses a candidate head; it has no qualified formal result. Across the family, architectural innovation remains a research goal, not a demonstrated release property.

## Same-panel evidence and limits

On post-key JevArena v3, the private 0.6B package scores **38.520** versus Kai's **35.938**. Its paired gain interval is **[-2.032, +7.846]**, so the composite gain is not statistically established. Choice falls from Kai's **277/800** to **109/800**, and Score from **98/400** to **80/400**; Noul rises from **404/800** to **458/800**. The separate 231-item public JevBench result is **143** versus **114**. Neither result establishes a size frontier, and the type regressions are material for System One use.

The same-start, same-data 0.6B type-separated-head contrast failed its frozen SELECT gates (**553/700**, macro **0.74583** versus shared-head control **562/700**, **0.77259**). An official Qwen3-0.6B Posttrained source contrast also missed them (**548/700**, **0.75065**). Adding a Score-only ranked-probability loss reached **276/700**, macro **0.38075**, while only improving its targeted Score slice from **32/90** to **38/90**. These negatives do not show that specialized heads or ordinal objectives are generally ineffective; they rule out the particular frozen implementations and budgets tested here.

A previously completed same-source candidate-interaction head also missed the SELECT gate after all 466 steps: its fixed best was **333/700**, macro **0.43174**. It adds leave-self-out candidate attention to Choice and Score, starts with the shared head's exact zero-step logits, and was stopped before any transfer or formal panel. Its [frozen protocol](small06-candidate-interaction-prereg-2026-09-28.md) and [result receipt](small06-candidate-interaction-full466-result-2026-09-28.md) rule out repeating that specific run as a new architecture experiment.

The same-TRAIN, source-aware AutoJev soft-target contrast also failed its frozen SELECT gate: **510/700** versus the hard-label control's **562/700**. Its teacher was an external research signal, never the student's direct weight initialization. The failure rules out this teacher loss and mixture at this budget; it does not rule out all distillation. A later read-only diagnostic found pairwise negative task gradients in **6/8** fixed mixed-type TRAIN windows at the untouched official base. The projection-disabled, same-schedule one-update path matched the ordinary control on all 32 fixed SELECT prompts with zero categorical and probability drift. The projected 466-update treatment then completed but failed its frozen SELECT gate at **547/700**, below both the **562/700** ordinary control and **569/700** advancement threshold; Score remained **32/90**. No formal evaluation or release gain followed.

## Next discriminating sequence

1. Preserve the failed projection result and do not rerun that fixed optimization arm. Independently prioritize source-disjoint, naturally written Choice and three-level Score data with explicit oracle/label custody, answer-position and shortcut screens, and transfer checks. The paper's larger training mixtures make a coverage gap plausible, but current comparisons cannot isolate it from architecture and initialization.
2. Only after a high-quality data arm passes CPU and small-model controls, compare Kai continuation and an official-base causal candidate under the same data/token budget and native System One protocol. Report the Choice/Score tradeoff even if the frozen composite rises.

No model should be described as architecture SOTA or Pareto-optimal until a same-panel size-matched comparison supports that claim. The published 1.0 paper's 3,766-item, 54-slice suite and post-hoc weighted overall are context, not numbers to merge into v3.

## Prospective option-isolation hypothesis, not a model result

The current causal endpoint for option *i* can attend to options before *i*.
The final query sees all options, so replacing only the head with a set module
does not make its input representations permutation-equivariant. The existing
0.6B arm has only **69/400** jointly correct original/reordered Choice pairs
on typed DEV, though this diagnostic alone cannot apportion the error between
training data, causal position and the head. The failed candidate-interaction
arm did not remove the causal information asymmetry.

A distinct architecture experiment would encode each candidate against the
same state and question, without other candidates or their keys, then apply a
shared scalar readout followed by a softmax. This construction is equivariant
to option reordering if the render, encoder and scorer contain no option index
or key signal. A symmetric set-interaction layer could subsequently restore
comparative information; it must not silently reintroduce order embeddings.
The cost is up to one backbone pass per option without verified shared-prefix
cache reuse, which could be unacceptable for 255-option System One Choice.
It may also lose tasks whose answer is defined only relative to the complete
candidate set. Both are release-relevant limits, not implementation details.

Before allocating training GPU-hours, freeze a small native CPU shape and
permutation test, a realistic 2/3/10/255-option memory and latency envelope,
and a data/token-matched training contrast from the same official 0.6B start.
Count complete input token exposure, not just optimizer steps; report paired
Choice/Score, human transfer and probability quality on independent groups.
Any gain on the existing post-key v3 panel remains post-key corroboration.
This idea is motivated by the measured position weakness and by independent
research on [option-permutation vulnerabilities](https://proceedings.mlr.press/v235/zong24b.html)
and [permutation-aware optimization](https://aclanthology.org/2026.acl-long.1621/).
Those papers do not demonstrate that this architecture improves System One
decisions; reduced order sensitivity and improved accuracy are separate claims.
