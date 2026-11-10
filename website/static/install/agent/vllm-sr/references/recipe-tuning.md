# Tune a recipe

Start from the requested quality, cost, latency or reliability objective and
its acceptable trade-offs. Discover the active recipe and the schema you need;
use [sr-bench](https://vllm-sr.ai/install/agent/vllm-sr/references/sr-bench.md) for measured single-model and MoM comparisons and
[Optimization](https://vllm-sr.ai/install/agent/vllm-sr/references/optimization.md) for the decision model, caches and cost.

## Locate the policy change

Follow entrypoint → recipe → signals and projections → decision → algorithm
and model → plugins. Change the component responsible for the observed
failure. Prefer a small set of meaningful actions over a decision per keyword
or label. At least two eligible, reachable models are needed to demonstrate
model-selection trade-offs; one model can only establish delivery.

Use heuristics (`keywords`, `context`, `conversation`, `input_modality`) for
explicit facts, application signals for the questions they define, and a
`systemone` question for anything else. Topic alone is not difficulty,
`fact_check` predicts a need for checking rather than verifying a claim, and
`pii` detection does not redact (masking does). Evaluate counterexamples
alongside intended matches: negation, quoted instructions, missing history and
unknown answers. `user_feedbacks` needs a prior assistant reply.

## Questions and predicates

Every model-judged signal is a System One question, and a rule matches when the
answer meets its `predicate` for its `labels`:

- **Read the answer, not just the match.** A traced preview
  (`vllm-sr route preview … --trace --json`) shows each condition's `state`
  and `confidence`; compare the distributions of intended matches and
  counterexamples before moving a threshold.
- **Thresholds.** A rule without a `predicate` uses its model's calibrated
  threshold (vela3-fast and Vela 2.0 only) or 0.5 with `uncalibrated_preset`.
  Write a `predicate` only from measured cases, and keep the old value for
  comparison.
- **Your own questions.** A `systemone` rule asks one thing in plain words:
  `noul` for yes or no, `choice` with a `description` per option, `score`
  with ordered levels. Narrow the wording before adding rules; models answer
  every question zero-shot, and a trained ID only exists for the built-in
  presets.
- **Complexity** is the preset with `easy`, `medium` and `hard`; route on the
  label (`{type: complexity, name: difficulty, label: hard}`). More escalation
  is not better discrimination.
- **Spans.** `pii`, `hallucination` and `span` questions need a span-capable
  deployment (vela3), and `ignore_labels` narrows what counts.

## Unknown answers

A late or failed answer leaves a rule unknown, and the decision's
`rules.on_unknown` (`no_match` by default, `match` or `fail_request`) resolves
it. Check every decision that negates a model-judged rule: a failed model must
not turn into evidence for a cheaper route under `NOT`. Signal families may run
concurrently, so an early condition does not avoid inference. A traced preview
lists unanswered rules in `signal_errors`.

## Selection and execution

- **Selectors.** `static` takes the first candidate, `multi_factor` weighs
  quality, latency, cost and load, and `systemone` asks the decision model to
  choose from the candidates' descriptions; describe each candidate by what it
  is good at, not by its name. A failed selector falls back to the first
  candidate.
- **Executors.** `escalate` calls candidates in order and stops at the first
  accepted answer (`accept.confidence`, `accept.systemone` with
  `answer_check`, `accept.refusal`); inspect the actual calls, since fallback
  plans, duplicate workers, reasoning-only output and truncation do not
  establish completed work. Stage budgets share the end-to-end deadline.
- **Retrieval.** A `context` plugin's RAG source recalls candidates and its
  `relevance` judge keeps the best. Preview shows plugin selection, not its
  execution; verify real retrieval, grounded answers and the added latency.

Use schema surfaces (`vllm-sr config schema --surface algorithm:escalate`) for
detailed settings; do not load every signal or plugin contract for a small
change.

## Learning and continuity

Inspect `global.router.learning.protection` and decision `adaptations` when the
policy uses session state. Built-in initialization defaults to protection on
and online adaptation off, preserving explicit base settings. Verify the
actual config and send stable session and conversation identities
(`x-vsr-session-id`, `x-conversation-id`).

Learning preview observes a read-only snapshot; it does not advance a
conversation or guarantee a later live selection. Keep Learning enabled when
it is the policy under test. Check the actual model on continuation, tool
completion, correction, model failure and conversation reset. An observed
recommendation is not an applied hold; a hold cannot retain a model outside
the eligible pool.

## Iterate from evidence

Freeze a baseline and relevant dev cases before observing candidate outputs.
Make one coherent change through the [configuration workflow](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md#change-a-running-stack),
check the active version, preview affected routes and run actual delivery.
Compare paired quality, routing outcomes, tokens, cost and latency. Keep cases
used for tuning separate from independent validation, and keep related
paraphrases together.

Keep or revert the candidate based on the stated objective and hard
constraints. Preserve failed evidence and a reproducible recipe artifact.
Route correctness, completed delivery and answer quality are separate
outcomes; qualify missing measurements and untested paths instead of inferring
success.
