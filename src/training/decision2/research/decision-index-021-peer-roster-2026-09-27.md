# Decision Index 0.2.1 peer selection for Decision 2.0

This is a **comparator roster**, not a Decision 2.0 result. The public
[Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)
`data/index.json` snapshot was generated `2026-09-27T03:42:13Z`, downloaded
with the Hugging Face CLI, and has SHA-256
`0e48d92d43f661391d1ff56392d9e84610d2f6737c2af2177334e494f355f73c`.
Its edition is `release-v2.1`, labelled `Decision Index 0.2.1`, with 67 model
entries. The index `balanced_skill` values below are only for choosing strong
models to **rerun** on our JevArena v3 and public JevBench 231 panels. They
cannot be placed in either panel's ranking or compared numerically with our
results. Index `served_params` metadata is a rough roster aid; actual loaded
parameters must be measured in each native rerun.

| Decision 2.0 tier | Strong same-size or adjacent public models to qualify | Index `balanced_skill` | Current action |
| --- | --- | ---: | --- |
| 0.6B | Bosun v3.1 0.6B; GLiNER2.5-Decide | 14.32; 11.21 | Bosun is already rerun on the frozen v3/public231 panel. Further peers need native adapters and matching panels. |
| 0.8B | JPT-0.8B; own Eos 1.0 | 19.22; 18.41 | Use JPT as first open peer after the own-origin candidate passes development and package gates. |
| 2B | Decider 2B; this-that 1.2 | 28.97; 28.14 | Decider is the primary same-size peer; an own-origin candidate must first pass development and package parity. |
| 4B | Decider 4B; Hopper | 40.70; 39.67 | Start with Decider and/or Hopper once an own-Nox/Qwen-origin candidate qualifies; historical Index values are not JevArena scores. |
| 9B | JPT-9B; adjacent larger Winnow-12B | 46.89; 50.02 | JPT native DEV/CSS pilot/public231 rerun is complete; Winnow is an optional stronger, larger stress test. |
| ~27B | Surogate Rune 26B-A4B v3; Decider chat Gemma-4-31B; AutoJev-27B | 57.44; 57.33; 56.40 | AutoJev is the closest same-backbone peer and already has a native DEV/CSS pilot rerun. Rune and Decider are additional first-tier targets if their exact downloadable native artifacts can be qualified. |

For initial release, require meaningful combined JevArena v3 improvement over
the same-size Decision 1.0 where it exists, with individual tradeoffs disclosed,
and a result approaching a qualified strong peer. The ~27B tier has no direct
1.0 reference and is judged against its first-tier peers. After initial
release, continue improving toward or beyond the best qualified same-size
models. Every published rank or model-by-task matrix must use models actually
run on the same frozen panel and scorer. The first-release model card includes
JevArena and public JevBench rank charts and a task matrix, with no Pareto
chart. Source model versions, native inference paths and actual loaded
parameter counts belong in each peer's individual qualification receipt.
