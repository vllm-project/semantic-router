# Decision 2.0: System One architecture audit and 0.6B repair hypotheses

**Scope:** CPU and source audit only; no new weights, inference, labels, scores,
publication, or release-gate change. This is an internal research note, not
model-card copy. Evidence is the [Decision 1.0 paper](https://vllm-sr.ai/decision-paper.pdf),
the [System One API reference](https://docs.typesafe.ai/api), and the pinned
same-panel receipts linked below. The current formal panel has already been
keyed in this project; future work must not call another use of it untouched
blind validation.

## What changed from Decision 1.0

The [paper](https://vllm-sr.ai/decision-paper.pdf) defines a common decision
request as state plus runtime-supplied typed questions and candidates. Choice
returns a selected candidate and a distribution, Noul a probability of true,
and Score a distribution over ordered levels plus its expected **zero-based
level index**. Kai-0.6B is a bidirectional encoder with candidate markers,
independent Choice/Noul/Score transformer paths and a 1,024-token complete
input cap. The paper's Eos/Sol/Nox/Lux causal family uses candidate endpoints,
a suffix query, and one shared FP32 bilinear/MLP candidate head, with 16,384
tokens as the public cap. These are **decision** paths, not chat generation.

The released DEV2.0-0.6B starts from official
[`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
at pinned revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, adds a
randomly initialized shared dynamic-candidate head, and trains the full text
backbone/head for 466 updates on rights-clean v2. It loads **597,103,104**
parameters and admits complete inputs up to **8,192** tokens without silent
truncation. Its [`CandidateHead`](../training/model/decision_model.py) and
segmented option/suffix packing implement the same *kind* of causal endpoint
plus global-query bilinear/MLP readout described for the 1.0 causal models.
The backbone changes from Kai's bidirectional Vela/mmBERT encoder to an
official Qwen3 causal text model, and the context cap increases. This is a
meaningful **family-architecture change at 0.6B**, but the readout is not a
new, independently validated 2.0 architectural invention. Nor does this
Kai-versus-Qwen comparison isolate architecture: initialization, tokenizer,
training sources and token exposure also differ. See the
[frozen training plan](qwen3-06b-official-base-full-clean-v2-prereg-2026-09-27.md)
and [development result](qwen3-06b-official-base-full466-development-result-2026-09-27.md).

## What the 0.6B measurements actually show

The [same-panel v3 result](qwen3-06b-official-full466-v3-postkey-result-2026-09-27.md)
is a post-key comparison on 1,600 typed original items / 2,000 answer slots
and 6,547 human-labeled transfer items. Its composite is
`100 * sqrt(T * H)`, where T is four typed-family macro accuracy and H is
15-task median macro-F1. The separate JevBench result is the **231 public
items**, not an official closed-set rank.

| Measure | DEV2.0-0.6B | own Kai 1.0 | Bosun 0.6B peer |
| --- | ---: | ---: | ---: |
| v3 composite | **38.5200** | 35.9383 | 38.5243 |
| T, four-family macro | 0.309063 | 0.361875 | 0.433750 |
| H, transfer task median | **0.480094** | 0.356907 | 0.342161 |
| typed Choice | 109/800 | 277/800 | 365/800 |
| typed Noul | 458/800 | 404/800 | 541/800 |
| typed Score | 80/400 | 98/400 | 83/400 |
| JevBench public 231 | **143** | 114 | 133 |

The +2.5817 v3 point estimate over Kai is primarily a transfer gain;
its paired 95% interval is **[-2.0317,+7.8464]** and includes zero. Typed
Choice falls 21 percentage points, Score 4.5 points, while Noul gains 6.75
points. The 0.6B development result found all **400/400** Score predictions
at level zero and only 85 correct; that is a *development* diagnosis, not
proof that every formal Score output is zero. Typed Brier 0.3535, ECE10
0.1785, and Score expected-value MAE 1.1828 are material for probabilistic
decision use. The data has only 516 Score rows across varying ordinal scales;
this suggests a hypothesis, not a demonstrated cause of the Score loss.

The own-Kai continuation on the same clean-v2 corpus also failed v3, scoring
35.0897 versus Kai1's 35.9383. Its Score improved (0.245 to 0.320), while
Choice and Noul fell. Its 1K cap left 404/6,547 human-transfer and 44/231
public answers invalid. The [Kai continuation result](kai06-v3-postkey-result-2026-09-27.md)
supports capacity and retention as separate factors; it does not show that
causality lies solely in the backbone.

## System One contract: present support and gaps

The [API reference](https://docs.typesafe.ai/api) defines one state and a
map of independently answered, named questions. State may be text, an object,
or an array. Instructions may also be text/object/array; Choice descriptions
may be text/object/array/**null**; Score levels are an ordered list of
text/object/array descriptions. Choice permits up to 255 options, Score 2–10
levels. Choice responses contain the key, all probabilities and confidence;
Noul returns a probability of yes; Score returns the probability-weighted
numeric level, legend, probabilities and confidence. TypeSafe describes
confidence as derived from the distribution but does not publish a unique
formula in the [confidence guide](https://docs.typesafe.ai/confidence).

The 2.0 native [`system_one`](../publication/full_runtime_api.py) method
does accept a structured JSON state and multiple named questions, evaluates
each independently through candidate logits, returns Noul `p(true)`, and
computes Score as `sum(level * p(level))`. It does not call a chat completion
or sample label text. However its current
[`question_to_row` and `normalized_answer`](../training/model/infer.py)
accept only **string** instructions and criteria descriptions; they reject
the documented structured instructions/criteria and Choice `null` option.
Choice/Score responses omit documented `confidence`, and Score omits
`legend`; a tied Choice currently emits `choice: null` although the documented
Choice answer key is a string. Therefore the current package supports a
System One-shaped *subset*, not the full documented API. These are runtime
contract gaps, not evidence of model-quality degradation. Fix and verify them
separately from the frozen v3 scorer; do not reinterpret already scored
predictions or invent equivalence with TypeSafe's unpublished confidence
formula. Keep explicit unknown/abstain as a supplied Choice candidate; do not
silently introduce an extra output class.

## Priority experiments before a stronger 0.6B claim

1. **Decision-contract gate, no optimizer:** Add an independent runtime matrix
   for text/object/array state; string/object/array instructions; structured
   Choice descriptions and `null`; two and 255 Choice options; two and ten
   ordered Score levels; optional Noul criteria; several named questions in
   one request; and no silent truncation. Verify normalized probabilities,
   Score expectation/legend and a documented confidence function against the
   emitted distribution. Keep invalid/overlength responses explicit. Audit
   latency versus state length, question count and candidate count rather
   than comparing it to a chat API.
2. **Matched causal-head ablation:** From the *same* official Qwen3-0.6B
   revision, tokenizer, TRAIN groups, token/update budget, optimizer and
   native input cap, compare the current endpoint/query head with a small
   permutation-equivariant candidate interaction head. Candidate endpoint
   states already encode earlier options, so a set head alone may *reduce*
   but cannot guarantee order invariance. Freeze a fresh development selector
   before running it; use paired option permutations/renamed keys and an
   independent Choice/transfer panel. Record extra parameters, peak memory,
   and K-scaled inference cost. Promote only if capability gain survives
   the counterfactual tests and transfer does not collapse.
3. **Ordinal Score ablation:** With the same Qwen source and exposure, compare
   existing K-way categorical CE+Brier with adjacent-level soft targets and
   an ordered-threshold or distributional ordinal head. Preserve user-defined
   level descriptions and the exact expected-index output. Evaluate per-level
   confusion, expected-value MAE, proper Brier/NLL and calibration on an
   independent three-level development set *and* variable 2–10 level rubrics.
   A higher argmax accuracy alone is insufficient for Score's API semantics.
4. **Data versus architecture control:** Retain completed Kai clean-v2 and
   Qwen clean-v2 controls. On new source-disjoint development groups, report
   a common <=1,024-token subset to compare decision quality at shared
   capacity, and a separate long-input panel to measure the larger Qwen cap.
   Do not relabel the existing unmatched model results as a causal backbone
   study. Test targeted Choice constraints/evidence and Score-level-balanced
   data as separate arms; a soft-distribution replay arm can test whether
   the transfer gain persists without the old typed regression. The formal
   keyed v3 labels remain reporting-only, never checkpoint selection.

**Decision:** retain the 0.6B result as a real same-panel tradeoff. A
decision-model SOTA or Pareto claim needs same-protocol, same-size peers and
fresh independent evidence, especially for Choice, Score and probability
quality. Any future private first-release card should be concise product
copy with the decision API, supported tasks and a matched rank/matrix;
this note stays out of model repositories.
