# Decision Balance Recipe Model Card

## Overview

Decision Balance turns three open models into one virtual model,
`vllm-sr/auto`, and spends reasoning only where a request needs it. The
Vela-2.0-4B decision model answers every routing question in one call per
request: what kind of work the request is, how much reasoning it needs,
whether exact facts matter, what a good answer needs, and whether the user is
correcting an earlier answer. A projection turns those answers into a
reasoning effort; decisions and algorithms turn the effort into a model and
its reasoning setting.

## Model details

| Model | Role | Relative cost per 1M output tokens |
| --- | --- | ---: |
| `zai/glm-5.3-flash` | Agentic work, exact facts, corrections, long inputs and research-level requests | 4.35 |
| `qwen/qwen3.8-flash-next` | Fast reasoning for code, mathematics and instruction following | 1.99 |
| `qwen/qwen3.8-27b` | Default model; images and everyday requests | 1.00 |

Costs are relative GPU-seconds per output token, measured on AMD Instinct
MI325X at 32 concurrent requests with each model on its own replica (GLM on
four GPUs, Flash-Next on two, the 27B on one), and close to that 4 : 2 : 1 GPU
ratio. Input tokens are priced at a tenth of output tokens. The currency is
`XXX`, the ISO code for no currency: these are units, not prices.

The Qwen3.8 models run with thinking off (`use_reasoning: false`) or at
`reasoning_effort` `medium` or `xhigh`. GLM-5.3-Flash always thinks; the recipe
sets its `reasoning_effort` to `low`, `high` or `max` per decision.

## Intended use

Use this recipe to serve one virtual model over a pool with a strong but
expensive model, a fast mid-size model and a small default, when most traffic
needs little reasoning and a minority needs a lot. It suits mixed assistant
traffic: chat, writing, code, mathematics, factual questions, tool use and
image questions.

It is not a privacy or compliance policy: every model in the pool may see any
request. Choose a privacy recipe when requests must stay on a restricted
model.

## Routing behavior

The decision model answers these questions in the call that also answers the
prompt guard, safety and PII signals:

| Question | Type | Asks |
| --- | --- | --- |
| `task` | choice | What kind of work the request asks for: chat, writing, code, stem, facts, analysis, agentic or document |
| `difficulty` | score | How much reasoning a strong expert needs, from 0 (none) to 4 (research level) |
| `precise_facts` | noul | Whether the answer depends on specific facts most people would have to look up |
| `needs` | set | Which of deliberation, tools, verification, long output and creativity a good answer needs |
| `correction` | noul | Whether the user says the assistant's previous answer was wrong |

No decision reads the `long_output` and `creativity` answers yet; they are
reported with the request's other signals.

Heuristic signals cover what needs no model: images, declared tools, an active
tool loop, an earlier assistant answer, input length and a request for a brief
answer.

The effort score is `0.3 × difficulty + 0.35 × P(deliberation) +
0.1 × P(verification) − 0.35 × brief request`, banded into off (below 0.4),
medium (below 0.675), high (below 1.15) and max. A STEM task that the decision
model rates at least multi-step (`difficulty` 2 or more) also runs at high
effort: asking for only the final answer lowers the deliberation answer, not
the reasoning the problem needs.

Decisions, highest priority first:

| Decision | When | Model | Algorithm |
| --- | --- | --- | --- |
| `guard` | Prompt attack or unsafe request | No model; a fixed reply | `fast_response` plugin |
| `long_context` | Input of 200K tokens or more | GLM-5.3-Flash, high | `static` |
| `vision` | The request carries an image | Qwen3.8-27B, medium | `static` |
| `recovery` | An earlier answer exists and the user says it was wrong | GLM-5.3-Flash, max | `static` |
| `agentic` | An active tool loop, or declared tools the request needs | GLM-5.3-Flash, high | `static` |
| `facts` | Facts most people would look up, for a facts, analysis or writing task below max effort | GLM-5.3-Flash, low | `static` |
| `frontier` | Max effort | GLM-5.3-Flash max or Flash-Next xhigh | `decision`: the decision model chooses from the candidates' descriptions |
| `code` | A code task at medium or high effort | Qwen3.8-Flash-Next, medium | `static` |
| `hard` | High effort, or a STEM task rated multi-step | Flash-Next or 27B, both xhigh | `multi_factor` |
| `standard` | Medium effort | Flash-Next or 27B, both medium | `multi_factor` |
| `fast` | Everything else | Qwen3.8-27B, thinking off | `static` |

`frontier` uses no second decision model: its `algorithm: decision` names no
deployment, so Vela-2.0-4B, which already answered the request's questions,
also chooses the model. `hard` and `standard` weigh quality (0.25), cost
(0.4), latency (0.1) and current load (0.25). With equal load the 27B wins on
cost; when it carries more in-flight requests than Flash-Next, Flash-Next takes
the request.

The effort rules follow per-effort measurements on public benchmark samples.
On GPQA Diamond, medium effort lost 8 to 15 points against extra-high on both
Qwen models, so hard STEM runs at extra-high. On LiveCodeBench, Flash-Next at
medium effort solved more problems than either Qwen model at extra-high, with
less than half the output tokens, so code goes to `code` instead of `standard`
or `hard`. With thinking off, both Qwen models lost 27 to 40 points on
multiple-choice knowledge and STEM questions, so `fast` keeps only requests
whose effort score stays below 0.4. The 0.675 boundary between medium and high,
the STEM difficulty of 2 and the `code` lane were chosen together by repeated
cross-validation on those samples.

## Requirements

- OpenAI-compatible endpoints for the three models, served under their model
  names (`--served-model-name zai/glm-5.3-flash` and so on) or with
  `provider_model_id` changed to match. Enable reasoning and tool-call parsing
  on each server.
- GLM-5.3-Flash with a 1M-token context for the `long_context` decision.
- A GPU for the Router: Vela-2.0-4B runs on a GPU only, so serve with
  `--platform amd` or `--platform nvidia`.
- Backends that accept the reasoning controls above: top-level
  `reasoning_effort` with `chat_template_kwargs.enable_thinking` for Qwen3.8
  (`false` for the thinking-off `fast` lane), `chat_template_kwargs.reasoning_effort`
  for GLM-5.3.
- On AMD GPUs with vLLM 0.31:
  - Qwen3.8-27B served with `--attention-backend TRITON_ATTN`. The default
    backend falls back to a decode kernel without KV splitting for the 27B's
    256-wide attention heads, so its cost per output token grows with input
    length and passes Flash-Next's from about 8K tokens.
  - GLM-5.3-Flash with the sparse-attention indexer fix of
    [vllm-project/vllm#59412](https://github.com/vllm-project/vllm/pull/59412)
    until a release includes it. Without it, answers that depend on more
    than about 2K tokens of prompt and reasoning silently degrade.

## Data handling and safety

The decision model locates personal data in every request with its PII span
head. Only types that identify a person count: names, contact details,
addresses, identity and account numbers. Places, organizations, dates,
titles, domain names and nationality, religious or political group names do
not count on their own, so "What is the capital of France?" is not personal
data. Personal data changes no route, but `routing.data_policy.replay_personal_data:
false` keeps such a request's content out of Router Replay: when Replay is
enabled, its record keeps the route, model, signals and detected PII types,
and no request or response body, prompt or tool trace. The recipe itself does
not enable Replay.

Locating personal data is the largest part of the decision model's cost on
long inputs. The span head reads the whole latest user turn in windows, while
the routing questions read only its beginning. On one AMD MI325X, the one
decision call took these median times without and with the PII question:

| Prompt | Without PII | With PII |
| --- | --- | --- |
| One line | 57 ms | 89 ms |
| 2K tokens | 0.11 s | 0.39 s |
| 16K tokens | 0.53 s | 1.75 s |

A deployment that keeps Router Replay off can remove the `personal_data` rule
to save that time. Replay then has no detector to keep personal data out, so
keep the rule whenever Replay is on.

Prompt attacks and unsafe requests receive a fixed reply without reaching any
model. The guard thresholds favor precision, because role-play and code
requests can score high on the prompt guard: a request the guard misses still
reaches a model, which applies its own safety training.

Every routed request's text, up to the decision model's input limit, is read
by the decision model inside the Router. Requests are sent to whichever
backend the route selects; all three models may see any request.

## Quick start

```bash
vllm-sr config validate --config config/recipes/decision-balance/config.yaml
vllm-sr serve --config config/recipes/decision-balance/config.yaml --platform amd
```

Point the three `backend_refs` endpoints at your servers first. Then send
requests to `vllm-sr/auto`; the `x-vsr-selected-decision` and
`x-vsr-selected-model` response headers show the lane and model.

## Evaluation

The probes cover every decision and the entrypoint, with negative variants for
well-known facts, declared but unused tools, follow-ups that are not
corrections and a one-line code fix, collision variants at the effort
boundaries, plus multilingual, multi-turn, tool, image and long-input
variants. See [`probes.yaml`](probes.yaml) and the
[conformance guide](../CONFORMANCE.md).

Quality evidence comes from the built-in catalog's third-party Artificial
Analysis records: `hard` ranks candidates on the `vllm-sr/reasoning@1.0.0`
index (GPQA Diamond and HLE) at xhigh effort. At medium effort the catalog has
no record for Flash-Next, so `standard` uses an operator index,
`decision-balance/operator-reasoning@1.0.0`, whose records are labelled
operator ratings: the 27B's from its medium-effort third-party records, and
Flash-Next's estimated from its xhigh records and the 27B's medium-to-xhigh
ratio.

## Limitations

- Routing quality depends on the decision model's answers. Its difficulty
  score separates everyday from multi-step requests well, but the boundary
  between high and max effort is coarse.
- `multi_factor` compares two candidates, so each factor favors one of them
  outright; small load differences can switch the model.
- Relative costs were measured on one hardware and serving stack with about 1K
  input tokens. Prices are per token, so they do not model a decode cost that
  grows with input length.
- Operator ratings are estimates, not measurements.
- The effort rules were checked on 60 to 150 questions per benchmark; other
  workloads can need other thresholds.
- A route to a stronger model does not guarantee a correct answer.
- The `long_context` threshold uses the Router's token estimate, about four
  bytes per token for prose until provider usage from requests with at least
  4 KiB of text calibrates it. For English prose it counted 1.2 to 1.5 times
  the backend's tokens, so requests from about 135K backend tokens can reach
  GLM.
- The recipe does not provision the inference backends it references.

## Changes

- 0.3.2: bug fixes only; lanes, thresholds and models are unchanged.
  `personal_data` no longer matches places, organizations, dates, titles,
  domain names or group names on their own. With Router fixes in the same
  release, `long_context` matches near 200K backend tokens again instead of
  about 55K, and `vllm-sr serve` keeps the stack up when the sr-bench worker
  cannot be replaced.
- 0.3.1: high effort from 0.675, wider probe margins, serving requirements.
- 0.3.0: `code` lane and hard STEM questions.

## References

- [Recipe metadata](metadata.yaml)
- [Runtime configuration](config.yaml)
- [Routing DSL](recipe.dsl)
- [Evaluation probes](probes.yaml)
- [Recipe authoring and conformance](../CONFORMANCE.md)
- [Decision signal](../../../website/docs/tutorials/signal/learned/decision.md)
- [Decision model selection](../../../website/docs/tutorials/algorithm/selection/decision.md)
- [Multi Factor](../../../website/docs/tutorials/algorithm/selection/multi-factor.md)
