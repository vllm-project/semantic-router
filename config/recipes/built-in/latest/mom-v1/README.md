# MoM V1 Model Card

## Overview

Five ways to route a mixed workload. Choose a goal, connect your models, and
publish one API entrypoint.

| Recipe | Goal | Decisions |
| --- | --- | --- |
| **Balance** | Reasoning effort only where a request needs it | `simple`, `medium`, `reasoning` |
| **Speed** | Fast interaction and efficient streaming | `fast`, `tools`, `reasoning` |
| **Cost** | Lower serving cost with targeted escalation | `economy`, `tools`, `reasoning` |
| **Accuracy** | Strong answers and deliberate use of multiple models | `simple`, `reasoning`, `review`, `agent` |
| **Vault** | Private processing within your assigned deployment boundary | `private`, `sensitive`, `guard` |

## Model details

MoM V1 is a family of routing policies. Policy version **4.0** keeps the
decision names above and the existing public model IDs. Its recipes ship with
vLLM Semantic Router; you supply the generation backends.

Every recipe routes on named questions to the router's decision model. The
questions a recipe asks about a request travel in one call, and each decision
reads the answers to named questions plus request facts such as declared tools,
an active tool loop or an earlier assistant answer. Balance runs the
[Decision Balance](../../../decision-balance/README.md) policy over the models
you assign. No recipe uses keyword, embedding, complexity or classifier signals.
The bundle declares Vela 2.0 0.8B as its decision model; see
[Requirements](#requirements).

## Intended use

Use Balance for mixed workloads, Speed for interactive applications, Cost for
cost-sensitive serving, Accuracy for demanding work, and Vault for an approved
private deployment. Each recipe can use a different set of connected models.

## Routing behavior

Each recipe's questions read the start of the latest user message. Answers are
probabilities; a question matches at the threshold shown. Request facts come
from the request itself: `has_tools` (tool definitions), `tool_required` and
`tool_disabled` (`tool_choice`), `tool_loop` (an active client tool loop) and
`flow` (a resumed Accuracy workflow).

**Balance** spends reasoning only where a request needs it. In one call, the
router's decision model answers five questions about the latest user message:
the kind of work (`task`), how much reasoning a strong expert needs
(`difficulty`, 0 to 4), whether a correct answer depends on facts most people
would look up (`precise_facts`), what a good answer needs (`needs`:
deliberation, tools, verification, long output, creativity) and whether the
user says the previous answer was wrong (`correction`). The effort score
`0.3 × difficulty + 0.35 × P(deliberation) + 0.1 × P(verification) − 0.35 ×
brief request` is banded into off (below 0.4), medium (below 0.675), high
(below 1.15) and max.

The three decisions group Decision Balance's eight lanes by the reasoning
effort each lane runs at, so one assignment per decision covers them:

| Decision | Lanes, highest priority first | Assign |
| --- | --- | --- |
| `reasoning` | **recovery**: an earlier answer exists, no tool loop is active, and both the correction question and the user feedback judgment say it was wrong. **agentic**: an active tool loop, or declared tools that the request needs, unless `tool_choice` is `none`. **frontier**: max effort. **hard**: high effort, except code and exact facts, or a STEM problem rated multi-step | Strong reasoning models that support tools, at high effort or above |
| `simple` | **fast**: effort off, except exact facts | Efficient models with thinking off |
| `medium` | **facts**: facts most people would look up, for a facts, analysis or writing task. **code**: a code task at medium or high effort. **standard**: medium effort. Also every request whose routing answers are unknown | Balanced models at medium effort |

Decision Balance runs each lane at its own effort on named models: recovery at
maximum effort, frontier at maximum or extra-high, hard at extra-high, agentic
at high, code and standard at medium, facts at low and fast with thinking off.
Here each decision runs at the effort you assign, so frontier and recovery use
the reasoning pool's effort and facts the medium pool's. To tune every lane
separately, use the standalone Decision Balance recipe, or split a decision in
a copy of this recipe.

Images and long inputs use the same decisions: candidate checks keep only the
pool's models that accept them, so a hard question about an image still goes
to `reasoning`. A brief-answer request lowers the effort score, but a STEM
problem rated multi-step stays on `reasoning`. Correction wording without an
earlier answer stays on effort routing.

**Speed** favors first-token latency for ordinary conversation and tools, and
per-token latency for reasoning. **Cost** keeps requests on one model and
prefers the lowest estimated request cost within each decision's quality band.
They ask the same questions, and Cost adds two:

| Question | Matches at | Decides |
| --- | --- | --- |
| `tool_use` | 0.4 | The user asks to call a tool or function now: `tools`, when the request declares tools and `tool_choice` is not `none`. Explaining or quoting a tool instruction does not count. |
| `depth` | 0.6 | A hard problem that needs investigation or careful multi-step reasoning: `reasoning`, which outranks `tools`. If the question goes unanswered, the request also goes to `reasoning`, so an outage never lowers quality. |
| `correction` (Cost) | 0.6 | The user reports a wrong answer or result and asks for a corrected one: `reasoning`. |
| `quoted` (Cost) | 0.5 | The message gives text to translate, quote, define or explain: a correction inside such text does not escalate. |

`tools` also takes an active tool loop and `tool_choice: required`; everything
else goes to `fast` or `economy`.

**Accuracy** normally uses one strong model:

| Question | Matches at | Decides |
| --- | --- | --- |
| `mode` (one, workers, experts) | 0.7 | `workers`: the user asks for several workers or agents that each run part of the work, which selects `agent`. `experts`: the user asks several models or experts to answer the same question independently, which selects `review`. One assistant comparing hypotheses, splitting work into workstreams, planning only or limited to one worker is `one`. |
| `depth` | 0.6 | A hard problem that needs investigation or careful multi-step reasoning: `reasoning`. |
| `verification` | 0.7 | The user asks to fact-check, verify or source an answer: `reasoning`. |
| `stakes` | 0.6 | A health, legal, business or economic question where a wrong answer could cause serious harm: `reasoning`. |
| `correction` | 0.6 | The user reports a wrong answer and asks for a corrected one: `reasoning`. |
| `quoted` | 0.7 | Text to translate, quote, define or explain: none of the questions above escalates it. |

`agent` and `review` never run during a client tool loop or a required tool
call; those requests go to `reasoning`. A resumed workflow (`flow`) continues on
`agent`.
`agent` runs at most three steps with two workers in parallel; `review`
compares two independent responses and synthesizes them. An unanswered `depth`
question sends the request to `reasoning`; an unanswered `mode` never fans out.

**Vault** uses separate pools for ordinary and sensitive requests:

| Signal | Matches at | Decides |
| --- | --- | --- |
| Prompt guard (built-in) | 0.75 | A prompt attack anywhere in the conversation: `guard` responds immediately without calling a backend. |
| `attack` | 0.9 | The latest message tells the assistant to ignore its rules or reveal its hidden instructions: `guard`. Quoted text that the user asks to analyze does not count. |
| PII (built-in) | any identifier | Personal identifiers anywhere in the conversation, including earlier turns: `sensitive`. A ZIP code or an age on its own does not count; an IP address does. |
| `confidential` | 0.5 | Confidential or internal material, such as private code, internal documents or a non-public deal, or a request to keep the data on site: `sensitive`. |
| `personal_data` | 0.5 | Personal data about a specific person, including the person's nationality, religion or political view: `sensitive`. Quoted contact details still count. |
| `risk` | 0.5 | A risk of harm to the user or someone else, such as suicidal thoughts or a request to hurt someone: `sensitive`, where the assigned model can help or refuse. |
| `personal_advice` | 0.5 | Advice on the user's own health, legal, financial or emotional situation: `sensitive`. |

Everything else goes to `private`. The prompt guard and PII judgments read the
whole conversation; the named questions read the start of the latest message.

The PII check favors recall. On a labelled set of 28 real identifiers and 30
look-alikes, Vela 2.0 0.8B's PII check found 26 of the identifiers, including
public IP addresses and names with street addresses. It missed a passport
number and a Chinese ID number, which the `personal_data` question caught
(0.62 and 0.94). It raised 14 false alarms, on networking terms such as
"private IP address", an email format example, times and city names. Those
requests take the sensitive pool. A ZIP code or an age does not count on its
own, because alone it does not identify a person, and the check read plain
numbers as both ("17" as a ZIP code at 0.80, "25" as an age at 0.66). With a
name or an ID beside them, the request still counts through the name or ID.
An IP address does count alone: allowing it too would remove 5 false alarms
but miss "My home IP address is 73.12.4.5".
If any of them goes unanswered, Vault fails the request closed rather than
guessing.

Built-in initialization enables conversation protection for single-model routes.
Send stable `x-session-id` and `x-conversation-id` headers to keep an eligible
model during an active tool cycle. At portable turn boundaries, protection weighs
switching costs; a new conversation can choose a new model. Defaults use a
300-second idle timeout and one turn before switching. Missing identity leaves
the base choice unchanged. Vault and multi-model execution bypass protection.

These defaults apply to the configuration's `global.router.learning` block;
initialization preserves every explicit setting, including `enabled: false`.
Online adaptation is disabled by default. Enable `adaptation.enabled` separately
to learn model choices from real outcomes within the matched decision's eligible
pool. Preview reports `execution_required` when protection or adaptation
determines the final choice during execution. Vault and multi-model execution
also bypass adaptation.

## Requirements

Assign at least one qualified model to each single-model decision and at least
two distinct worker models to Accuracy's `review` and `agent`. Vault's `guard`
responds immediately and needs no model assignment.

Models must declare their capabilities, context window, and maximum output
limit. Images and long inputs use the same decisions as other work; candidate
checks enforce their requirements after routing signals complete. Vault's
triage budget applies before these candidate checks. Missing metadata or an
insufficient eligible pool produces an explicit error.

Single-model decisions compare the catalog's versioned `general`, `reasoning`,
or `agentic` quality index at the assigned reasoning effort. Custom models need
relevant benchmark evidence. Cost comparisons require comparable configured
prices; latency comparisons use observations from real requests.

Absent client output limits default to 4,096 tokens, or 8,192 for reasoning and
Accuracy's single-model answers. Explicit limits are preserved. Multi-model
calls use declared stage budgets and are checked again before dispatch.
Accuracy's `agent` gives its planner, workers, and final synthesis 4,096 tokens
per call, matching its request default.

The bundle declares its decision model: a `primary` deployment of
`vllm-sr/Vela-2.0-0.8B` with no fixed device, named by
`global.model_catalog.system.decision_model`. With no device the router uses a
GPU where one exists and the CPU otherwise. `vllm-sr recipe builtin init`
copies only the learning and Replay defaults, so add this block to your
configuration to use the model the probes were calibrated on:

```yaml
global:
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-0.8B
    system:
      decision_model:
        deployment: primary
```

On a CPU the decision call is slow. Measured through the router with its
container limited to 4 CPUs and one request at a time, a short message takes
about 4 seconds at the median and 6 seconds at the 90th percentile; a long
input takes about 75 seconds, because the model reads 16,384 tokens of it.
Serve the decision model on a GPU for live traffic: set
`device: rocm` or `device: cuda` on the deployment to require one. A smaller
model, such as `vllm-sr/Vela-2.0-0.3B` in `artifact`, answers faster; re-run the
probes after any change. See
[Choose a model](https://vllm-sr.ai/docs/model-runtime/choose-a-model) for sizes
and hardware.

Every recipe's questions read the start of the latest user message, up to the
decision model's input limit (16,384 tokens for Vela 2.0 0.8B): put the request
before a long attachment. Vault's prompt guard and PII judgments read the whole
conversation, up to the decision model's scan budget; content past it was not
read, and Vault fails such a request closed with HTTP 503 and `signal_errors`
before any backend is called.

When a question goes unanswered, Preview lists it under `signal_errors`, and the
router counts it in `vsr_model_runtime_unknown_answers_total` by deployment and
reason, so a decision-model outage is visible.

## Data handling and safety

Vault routes personal identifiers to its sensitive pool. A nationality,
religion or political view counts when it belongs to a specific person; general
groups and ideas do not, and quoted personal records still qualify.

Every Vault path disables client tools, strips tool history, and disables Router
memory, response caching, and learning adaptation. It also
suppresses new Responses object writes. Replay capture is configured separately
and is enabled by default as described below. These restrictions apply regardless of
the prompt guard, PII and question verdicts; an unanswered one fails closed.

Assign every Vault backend to infrastructure that meets your privacy
requirements. The recipe does not establish physical placement, change provider
retention, delete earlier stored conversations, or disable operational usage
metadata.

## Replay and Insights

All five recipes, including Vault, enable PostgreSQL Replay by default so
Dashboard Insights can show their requests and routing decisions. The defaults
keep records for seven days, limit raw request and response excerpts to 4,096
bytes each, and limit structured tool traces to 100 steps. Expired PostgreSQL
records are removed during subsequent writes.

`vllm-sr recipe builtin init` preserves explicit operator settings, including
`enabled: false`. Local `vllm-sr serve` manages PostgreSQL and its persistent
volume. Other deployment methods need a PostgreSQL connection and credentials
in `global.services.router_replay.postgres`.

- To disable default capture, set `global.services.router_replay.enabled: false`.
  Individual decisions can still opt in.
- To disable capture for a decision, set its `router_replay` plugin to
  `configuration: {enabled: false}`. There is no recipe-level Replay override.
- Set `capture_personal_data: false` globally or in the decision plugin to omit
  content when PII is detected or its status is unknown.

Vault's other data-handling restrictions still apply. Replay settings do not
change provider retention or delete previously captured conversations. See the
[Replay guide](../../../../../website/docs/tutorials/plugin/router-replay.md)
for capture limits and access controls.

## Quick start

```bash
vllm-sr serve
```

Connect models in the Dashboard, choose a recipe, assign its decision pools,
and publish an entrypoint. Use **Preview** to inspect signals, decisions, and
candidate selection. Then send a real request through the entrypoint to verify
backend execution and latency.

When upgrading to policy 4.0, every recipe keeps its decisions, but the
decision model now selects them: named questions replace the keyword,
embedding, complexity and classifier signals, and Vault no longer loads a
separate content-risk classifier. In Balance, tool loops and requests that need
their declared tools move to `reasoning`, so give that pool models that support
tools. Since policy 3.0, content risk in Vault selects a responsible private
model instead of a fixed denial, and prompt attacks still receive an immediate
response. Existing published versions and their assignments remain unchanged.

## Evaluation

[`probes.yaml`](probes.yaml) covers multilingual requests, negative examples,
priority collisions, tools, images, retained conversation, and long inputs.
Validate routing behavior and actual backend eligibility against your deployment;
a correct decision alone does not prove that an assigned model can serve it.
See the [conformance guide](../../../CONFORMANCE.md) for commands.

The probes are calibrated on Vela 2.0 0.8B: through the router, with its
container limited to 4 CPUs, all 347 reach their expected decision. Re-run them with the
decision model you deploy. Because the decision model reads long inputs slowly
on a CPU, CI splits the bundle's probes into 16 jobs of similar cost
(`evaluation.shards` in `probes.yaml`) and merges their reports.

## Limitations

Quality bands compare the assigned pool; they do not guarantee answer accuracy.
Routing depends on the decision model's answers, and some sit close to their
thresholds on Vela 2.0 0.8B: a request to summarize a due diligence report
scores just above the `confidential` threshold, and a correction wrapped in a
translation request scores just above `quoted`. Validate representative
traffic, and prefer a larger decision model where routing errors are costly.
Context accounting includes conversation history and output reserves but remains
an estimate rather than a provider-tokenizer guarantee. Learned signals can
misclassify requests, so validate representative traffic before rollout.

Knowledge-base retrieval, reranking, caching, and memory can be added for
applications that need them. These reusable recipes require no knowledge base
and inject no system prompt. Use [recipe tuning](https://vllm-sr.ai/docs/benchmarking/agent-evaluation-loop)
to measure these additions and session continuity against real requests.

## References

[Documentation](https://vllm-sr.ai/) ·
[GitHub](https://github.com/vllm-project/semantic-router) ·
[Recipe conformance](../../../CONFORMANCE.md)
