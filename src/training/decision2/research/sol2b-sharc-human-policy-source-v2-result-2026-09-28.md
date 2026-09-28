# Sol 2B ShARC source v2: metadata feasibility passes, training HOLD

The prospective [v2 diagnostic](sol2b-sharc-human-policy-source-v2-prereg-2026-09-28.md)
ran once on the same publisher TRAIN archive as v1, using a byte-identical
local-code mirror on the authorized remote CPU. It printed only aggregate
counts. **This is not an admitted training source or a model result.** No
protected evaluation answers, model weights or GPU were used.

| TRAIN-only source inventory | V2 result |
| --- | ---: |
| Unique original utterances | 21,890 |
| Original tree IDs | 628 |
| Exact snippet and URL uniform within tree | 628 each |
| Exact or normalized question uniform within tree | 0 |
| Opposite Yes/No pairs within a normalized same-question subgroup, one per tree | **589** |
| Distinct rule snippets represented by those pairs | **581** |
| Distinct source URLs represented by those pairs | **180** |
| Frozen feasibility floors | 100 pairs / 100 snippets / 50 URLs |

The v2 feasibility count passes all three floors. Its difference from v1 is
principally **grouping within each original tree by a matching question**, not
just typography normalization: every tree contains multiple question strings
while its exact rule snippet and source URL are stable. Thus v1's whole-tree
exact-tuple filter stays failed; v2 neither alters nor recasts that receipt.
The publisher's TRAIN labels count exact Yes 6,773, exact No 7,057 and 8,060
other free-form answers. No raw other-answer text was written to v2 output.

The publisher archive SHA-256 is
`72dca3f4f3ba73b1d796b40e952a80d53cd2011ef90b2168b8bcaa818f5edd1e`;
the TRAIN member SHA-256 is
`d37d349758a69644fbf49827b1a0893f6099752b0ccc9c7e29bd315e5228429c`;
v2 audit code SHA-256 is
`9811536cc3e5238c53c58415d6a0a610d73e13e10b3e1b30e2a24c17a1964e9b`;
private aggregate receipt SHA-256 is
`cd7270222940608469366d9e995bf41d84685fa751f01298ff2ddc8d20d3af6e`.
The exact source revision is the signed code commit `ff7d4f8b3`.

**Next admission boundary:** construct a deterministic one-pair-per-tree
answer-blind packet without `answer` or `evidence`, confirm the candidate
states really differ on a policy-relevant condition and the publisher Yes/No
labels are semantically defensible, and examine negative-question sampling.
Then assess source URL/snippet disjointness against TRAIN/SELECT/CAL and
protected prompts, native token exposure, state-removed shortcut, rights and
matched-budget Sol1 zero-step parity. Until those gates pass, all **589 pairs
remain only source candidates**, with zero data rows admitted and zero
GPU-hours spent.
