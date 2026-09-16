---
title: Tune and Verify a Recipe
description: Improve routing quality, latency, cost, and agent continuity with real requests.
---

# Tune and Verify a Recipe

A useful recipe sends requests to the right handling path and delivers better
results within your latency, cost, and safety requirements. This guide shows
how to improve one using real Preview requests, routed responses, and session
traces.

Start with a running deployment and reachable model endpoints. Follow the
[agent installation guide](../installation/agent) to set up a new deployment.
Use the management origin for configuration and Preview, and the inference
origin for real model requests.

## Choose an objective and a baseline

Pick an outcome you can measure: fewer unnecessary reasoning calls, better
answers to consequential questions, faster tool turns, or fewer model switches
during an agent run. Set a quality floor and acceptable latency or cost before
adjusting thresholds.

Export the built-in bundle or save your current configuration. Record the active
recipe, runtime image, model revisions, and assignments. Keep the original test
cases so you can run the same requests before and after a change.

Build a small representative dataset for every decision and fallback. Include
normal requests, near-boundary cases, multilingual paraphrases, long inputs,
quoted instructions, and multi-turn conversations. Add negative examples: a
medical definition need not receive the same route as personalized treatment
advice, and a quoted attack should not automatically be treated as an instruction.

Pair some requests with identical background information but different tasks.
Vary concise and detailed answer styles separately. This tests whether effort
signals recognize the required work instead of the topic or response length.
Treat related paraphrases and tool variants as one family when reporting coverage.

## Give each part a clear job

| Part | Use it to |
| --- | --- |
| Signals | Observe meaning, risk, effort, and explicit request constraints. |
| Projections | Combine evidence into scores or categories that decisions can use. |
| Decisions | Choose a small set of distinct handling paths. |
| Algorithms | Select or coordinate models within each path. |
| Plugins | Apply retrieval, tools, caching, and data-handling behavior. |
| Models | Execute the request within declared capabilities and context limits. |

Keep decision names short, such as `simple`, `medium`, and `reasoning`. Add a
new decision when the action changes, rather than for every topic or language.

Use heuristics for explicit facts such as required tools and structured output.
Use learned signals where meaning matters: Embedding for intent, Complexity for
effort, Domain with FactCheck for consequential questions, and Feedback with
Reask for recovery. Combine clues through projections when one signal is too
broad. Topic alone does not establish difficulty, and FactCheck predicts a need
for checking rather than verifying a claim.

Review how unknown signals affect every decision, especially conditions using
`NOT`. A failed classifier should not turn into evidence for a cheaper route.
Compare easy and hard signal scores before changing thresholds. Strong overlap
calls for better task examples or a better model; sending every uncertain request
to reasoning increases usage without establishing better discrimination.

Adding a keyword condition also does not guarantee less inference: used signal
families can run concurrently before decisions are evaluated. Measure the actual
request cost.

For recovery, include the earlier assistant reply: Feedback routing skips
requests without one. Test direct repair instructions and self-contained error
reports separately from Feedback matches, with harmless tone or formatting edits
as controls. For collaboration, distinguish multiple-worker authorization from
a discussion of agents or a request for one independent worker. The workflow
handles its internal stages; users should not have to name them to request
collaboration.

Check that multilingual examples survive prototype compression. A rule-local
`prototype_scoring` override stays with the Recipe, while an omitted override
inherits global settings. Use `enabled: false` to retain all deduplicated
candidates, and specify `best_weight` and `top_m` for their combined score.
Measure the effect before replacing the baseline.

## Validate and preview the candidate

Discover the running contract before editing configuration:

```bash
vllm-sr config schema --endpoint "$ROUTER_ORIGIN" \
  --surface algorithm:multi_factor
vllm-sr config validate --config candidate.yaml \
  --endpoint "$ROUTER_ORIGIN"
vllm-sr config plan --config candidate.yaml \
  --endpoint "$ROUTER_ORIGIN"
```

Apply a hot-reloadable change with `vllm-sr config apply`. If the plan reports
`RESTART_REQUIRED`, use the deployment workflow. For an authorized local-stack
replacement, run `vllm-sr serve --config candidate.yaml --replace-active-config`.
Confirm readiness and the active revision before testing.

Set `ENTRYPOINT` to the published entrypoint you are evaluating, then preview a
case from your dataset:

```bash
vllm-sr route preview \
  --endpoint "$ROUTER_ORIGIN" --model "$ENTRYPOINT" \
  --prompt 'Give a brief definition of a readiness probe.' \
  --trace --json --timeout 300
```

Recombining saved signal values can help isolate a rule change, but it is not
a new Preview result. First reproduce the baseline decisions and heuristic
matches exactly, then test the candidate with real requests.

Preview runs configured classifiers and embeddings without backend generation.
Check the matched signals, projection results, decision, algorithm, selection
status, and errors. Preserve the full messages and tool fields for multi-turn
cases; use the discovered Preview HTTP schema when the CLI cannot express a
request shape.

Report policy coverage and deployment coverage separately. A correct decision
with no eligible backend shows a capacity or assignment problem. An immediate
response needs no selected model. A multi-model plan still needs execution.
None of these outcomes should be presented as a successful backend call.

## Verify delivery and the application result

Send the same request through the inference listener:

```bash
vllm-sr route probe \
  --config candidate.yaml \
  --base-url "$INFERENCE_BASE_URL" --model "$ENTRYPOINT" \
  --prompt 'Give a brief definition of a readiness probe.' \
  --expect-recipe "$RECIPE" --expect-decision "$DECISION" \
  --expect-selected-model "$SELECTED_MODEL" \
  --expect-response-model "$RESPONSE_MODEL" \
  --timeout 300
```

Set the expected values from your test case and the backend's verified response
identity. The selected-model header and upstream response model are separate
assertions. Omit the response-model assertion only when the backend does not
expose a stable identity.

Check completed output as well as routing. An HTTP 200 containing only reasoning,
an empty answer, or a truncated response is not successful delivery. Use a
completion budget that fits the actual input and leaves room for the final
answer. When the request omits a limit, a configured
`request_params.default_max_tokens` supplies the decision default; otherwise
the backend default applies.

Assign at least two eligible, reachable models when comparing selection policies.
With one candidate, the test verifies delivery but cannot measure a choice between
models.

Measure cold startup separately from warm median and p95 latency. Compare route
quality, answer quality, classifier work, backend calls, token use, and cost.
For multi-model algorithms, verify that the intended distinct workers and final
stage actually ran.

## Test retrieval, risk handling, and agent continuity

**Retrieval.** Attach [RAG and neural reranking](../tutorials/plugin/rag#neural-reranking)
to a path with a real knowledge base. Retrieve a wider candidate set and use
`rag.rerank` to retain the most relevant documents. Check document identities,
retrieval coverage, ranking, grounded answers, and added latency. Preview selects
the plugin; a routed request executes it. Start with a single-model path before
adding retrieval to every stage of a multi-model workflow.

**Risk handling.** [Guard](../tutorials/signal/learned/jailbreak) detects prompt
attacks; [Safety and Hazard](../tutorials/signal/learned/safety) identify content
risks and categories. Test harmful facilitation against help-seeking and benign
analysis before choosing refusal rules. PII can select a restricted model pool;
it does not redact content or establish a provider's retention policy.

**Agent continuity.** Use [Router Learning protection](../tutorials/learning/protection)
with stable session and conversation identities. Test a full tool cycle,
continuation, explicit correction, backend failure, decision change, and a new
conversation. Compare `apply`, `observe`, and `bypass`: an observed recommendation
to keep a model is different from an actual hold. Check the selected backend,
route headers, Replay API, and Dashboard together. Repeat a session ID across
recipes to verify their isolation. A hold must not retain an ineligible model.

Built-in recipe initialization enables conversation protection and leaves online
adaptation off, preserving any settings already present in the base configuration.
Clients need stable identities for protection to retain a model. Adopt adaptation
after evaluating your application's outcomes.

Preview can report `execution_required` when protection participates in selection.
It evaluates request signals without advancing conversation state; verify an
actual hold or switch through a routed request and its Replay record.

Inspect Replay and the Dashboard before activating the next candidate. The
default in-memory Replay store is cleared by configuration reloads and restarts;
save the traces needed for comparison first.

## Keep changes that improve the objective

Run baseline and candidate against the same dataset. Review regressions by
language, input length, use case, and session stage. Keep the candidate when it
improves the chosen outcome without violating the quality floor or hard
constraints; otherwise restore the baseline and preserve the evidence.

Use `vllm-sr benchmark catalog` for broader routing workloads. For a full model
or virtual-model comparison, discover the fixed suite with
`vllm-sr benchmark intelligence list` and `plan --help`. A virtual model must be
measured through its actual endpoint; its score cannot be assembled from member
scores. Partial runs help guide tuning but do not establish a full suite score.

The [agent tuning reference](https://vllm-sr.ai/install/agent/vllm-sr/references/recipe-tuning.md)
provides a reusable checklist. Keep raw evaluation outputs outside Git and
credentials in environment variables named by `--token-env` or `--api-key-env`.
