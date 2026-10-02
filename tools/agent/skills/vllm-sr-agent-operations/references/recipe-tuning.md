# Tune a recipe

Start from the requested quality, cost, latency or reliability objective and its
acceptable tradeoffs. Discover the active recipe and relevant schema; use
[sr-bench](https://vllm-sr.ai/install/agent/vllm-sr/references/sr-bench.md) for measured single-model/MoM comparisons.

## Locate the policy change

Follow entrypoint → recipe → signals/projections → decision → algorithm/model →
plugins. Change the component responsible for the observed failure. Prefer a
small set of meaningful actions over a decision per keyword or classifier label.
At least two eligible, reachable models are needed to demonstrate model-selection
tradeoffs; one model can only establish delivery.

Use protocol facts for explicit requirements, semantic evidence for intent and
complexity evidence for effort. Topic alone is not difficulty, FactCheck does not
verify truth, and PII detection does not redact content. Evaluate counterexamples
alongside intended matches, including negation, quoted instructions, missing
history and unknown signals. Feedback needs a prior assistant reply.

## Inspect details only when relevant

- **Embedding:** raw scores and emitted matches differ. Check thresholds, soft
  matching, `top_k` and prototype compression before changing examples. Cosine
  similarity is not a probability; use per-rule evidence for a predicate.
- **Complexity:** the local margin is hard score minus easy score. Independent
  `hard_above` and `easy_below` boundaries leave equality in the medium class.
  Compare all three distributions and false positives; more escalation is not
  itself better discrimination.
- **Unknowns:** review `rules.on_unknown` with negative predicates. Inference
  failure must not silently become an ordinary false under `NOT`. Signal families
  may run concurrently, so early conditions do not necessarily avoid inference.
- **Retrieval:** verify real retrieval/reranking and grounded answer quality.
  Preview shows plugin selection, not its execution. Keep document identities and
  measure added latency/context; a relevance score is not truth confidence.
- **Multi-model workflows:** inspect actual planner, worker and final calls.
  Fallback plans, duplicate workers, reasoning-only output and truncation do not
  establish completed work. Stage budgets share the end-to-end deadline.

Use schema surfaces and recipe definitions for detailed settings and examples;
do not load every signal or plugin contract for a small change.

## Learning and continuity

Inspect `global.router.learning.protection` and decision `adaptations` when the
policy uses session state. Built-in initialization defaults to protection on and
online adaptation off, preserving explicit base settings. Verify the actual
config and send stable session/conversation identities.

Learning preview observes a read-only snapshot; it does not advance a conversation
or guarantee a later live selection. Keep Learning enabled when it is the policy
under test. Check the actual model on continuation, tool completion, correction,
model failure and conversation reset. An observed recommendation is not an
applied hold; a hold cannot retain a model outside the eligible pool.

## Iterate from evidence

Freeze a baseline and relevant dev cases before observing candidate outputs.
Make one coherent change through the [configuration workflow](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md),
check the active revision, preview affected routes and run actual delivery.
Compare paired quality, routing outcomes, tokens, cost and latency. Keep cases
used for tuning separate from independent validation, and keep related
paraphrases together. Offline policy recomposition is diagnostic, not live proof.

Keep or revert the candidate based on the stated objective and hard constraints.
Preserve failed evidence and a reproducible recipe artifact. Route correctness,
completed delivery and answer quality are separate outcomes; qualify missing
measurements and untested paths instead of inferring success.
