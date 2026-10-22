# MoM V1 Model Card

## Overview

Five ways to route a mixed workload. Choose a goal, connect your models, and
publish one API entrypoint.

| Recipe | Goal | Decisions |
| --- | --- | --- |
| **Balance** | Reasoning effort only where a request needs it | `long_context`, `vision`, `recovery`, `agentic`, `facts`, `frontier`, `code`, `hard`, `standard`, `fast` |
| **Speed** | Fast interaction and efficient streaming | `fast`, `tools`, `reasoning` |
| **Cost** | Lower serving cost with targeted escalation | `economy`, `tools`, `reasoning` |
| **Accuracy** | Strong answers and deliberate use of multiple models | `simple`, `reasoning`, `review`, `agent` |
| **Vault** | Private processing within your assigned deployment boundary | `private`, `sensitive`, `guard` |

## Model details

MoM V1 is a family of routing policies. Policy version **4.0** keeps the
existing public model IDs and the decision names of Speed, Cost, Accuracy and
Vault; Balance's decisions are the lanes of the Decision Balance policy. Its
recipes ship with vLLM Semantic Router; you supply the generation backends.

Every recipe routes on named questions to the router's decision model. The
questions a recipe asks about a request travel in one call, and each decision
reads the answers to named questions plus request facts such as declared tools,
an active tool loop or an earlier assistant answer. No recipe uses keyword,
embedding, complexity or classifier signals. The bundle declares Vela 2.0 0.8B
as its decision model; see [Requirements](#requirements).

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

**Balance** is the Decision Balance policy: it spends reasoning only where a
request needs it. In one call, the router's decision model answers six
questions about the latest user message: the kind of work (`task`), how much
reasoning a strong expert needs (`difficulty`, 0 to 4), whether a correct
answer depends on facts most people would look up (`precise_facts`), what a
good answer needs (`needs`: deliberation, tools, verification, long output,
creativity), whether the user says the previous answer was wrong
(`correction`) and whether the user asks to call a tool now (`tool_use`). The
effort score `0.3 × difficulty + 0.35 × P(deliberation) + 0.1 ×
P(verification) − 0.35 × brief request` is banded into off (below 0.4), medium
(below 0.675), high (below 1.15) and max.

Every lane of the policy is its own decision, with its own pool and the
reasoning effort you assign to it. Decisions, highest priority first:

| Decision | When | Algorithm | Assign |
| --- | --- | --- | --- |
| `long_context` | An input of 200K tokens or more | `static` | A model with a 1M-token context window, at high effort |
| `vision` | The request carries an image | `static` | A model strong on documents and screenshots, at medium effort |
| `recovery` | An earlier answer exists, no tool loop is active, and both the correction question (0.6) and the user feedback judgment say it was wrong | `static` | A stronger model, of another family where possible, at maximum effort |
| `agentic` | An active tool loop, or declared tools that the request needs (an agentic task, tools needed at 0.5, `tool_use` at 0.4 or `tool_choice: required`), unless `tool_choice` is `none` | `static` | The strongest agent, at high effort |
| `facts` | Facts most people would look up (`precise_facts` at 0.8) for a facts, analysis or writing task, below max effort | `static` | The model that hallucinates least, at low effort |
| `frontier` | Max effort | `decision` | Two or more strong models, at maximum or extra-high effort |
| `code` | A code task at medium or high effort | `static` | The strongest coder, at medium effort |
| `hard` | High effort, or a STEM task rated multi-step (`difficulty` 2 or more) | `multi_factor` | Reasoning models at extra-high effort |
| `standard` | Medium effort, and every request whose routing answers are unknown | `multi_factor` | Reasoning models at medium effort |
| `fast` | Everything else | `static` | An efficient model with thinking off |

`static` takes the first eligible model of the pool. `hard` and `standard`
weigh quality on the `vllm-sr/reasoning@1.0.0` index (0.25), cost (0.4),
latency per output token (0.1) and current load (0.25), and take the cheapest
candidate when none has a quality record. `frontier` loads no second decision
model: the decision model that answered the request's questions also chooses
one of the frontier models from their descriptions, so describe what each one
is strongest at.

An image or a long input selects its lane before effort does, so a hard
question about an image goes to `vision`; candidate checks still keep only the
pool's models that accept the image and the context. A brief-answer request
lowers the effort score, but a STEM problem rated multi-step stays on `hard`.
Correction wording without an earlier answer stays on effort routing.

Balance has no guard decision; contain prompt attacks with Vault. The prompt
guard and safety checks read the whole input, and on a CPU Vela 2.0 0.8B reads
at most 65,536 tokens of it. A jailbreak rule matches content it did not read,
so a guard would refuse every long input there.

The policy was measured on three open models. To reproduce it, assign them as
follows:

| Decisions | Models and effort |
| --- | --- |
| `long_context`, `agentic` | `zai/glm-5.3-flash`, high |
| `recovery` | `zai/glm-5.3-flash`, max |
| `facts` | `zai/glm-5.3-flash`, low |
| `frontier` | `zai/glm-5.3-flash`, max, and `qwen/qwen3.8-flash-next`, xhigh |
| `code` | `qwen/qwen3.8-flash-next`, medium |
| `hard` | `qwen/qwen3.8-flash-next` and `qwen/qwen3.8-27b`, xhigh |
| `standard` | `qwen/qwen3.8-flash-next` and `qwen/qwen3.8-27b`, medium |
| `vision` | `qwen/qwen3.8-27b`, medium |
| `fast` | `qwen/qwen3.8-27b`, thinking off |

Their relative costs per output token are 4.35, 1.99 and 1.00, in GPU-seconds
measured on AMD Instinct MI325X at 32 concurrent requests with each model on
its own replica (GLM on four GPUs, Flash-Next on two, the 27B on one). The
effort rules follow per-effort measurements on public benchmark samples. On
GPQA Diamond, medium effort lost 8 to 15 points against extra-high on both
Qwen models, so hard STEM runs at extra-high. On LiveCodeBench, Flash-Next at
medium effort solved more problems than either Qwen model at extra-high, with
less than half the output tokens, so code has its own lane. With thinking off,
both Qwen models lost 27 to 40 points on multiple-choice knowledge and STEM
questions, so `fast` keeps only requests whose effort score stays below 0.4.
The 0.675 boundary, the STEM difficulty of 2 and the code lane were chosen
together by repeated cross-validation on those samples, with Vela 2.0 4B as
the decision model.

The Qwen3.8 models take top-level `reasoning_effort` with
`chat_template_kwargs.enable_thinking` (`false` for thinking off); GLM-5.3-Flash
always thinks and takes `chat_template_kwargs.reasoning_effort`. On AMD GPUs
with vLLM 0.31, serve Qwen3.8-27B with `--attention-backend TRITON_ATTN`: the
default backend decodes its 256-wide attention heads without KV splitting, so
its cost per output token grows with input length and passes Flash-Next's from
about 8K tokens. Serve GLM-5.3-Flash with the sparse-attention indexer fix of
[vllm-project/vllm#59412](https://github.com/vllm-project/vllm/pull/59412)
until a release includes it; without it, answers that depend on more than
about 2K tokens of prompt and reasoning silently degrade.

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

Assign at least one qualified model to each single-model decision, at least
two distinct worker models to Accuracy's `review` and `agent`, and at least
two distinct models to Balance's `frontier`. Vault's `guard` responds
immediately and needs no model assignment.

Models must declare their capabilities, context window, and maximum output
limit. Outside Balance's `vision` and `long_context`, images and long inputs
use the same decisions as other work; candidate checks enforce their
requirements after routing signals complete. Vault's triage budget applies
before these candidate checks. Missing metadata or an insufficient eligible
pool produces an explicit error.

Multi-factor decisions compare the catalog's versioned `general`, `reasoning`,
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
  content when PII is detected or its status is unknown. Balance declares no
  PII rule, so with that setting its records keep no request or response
  content.

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

When upgrading to policy 4.0, Speed, Cost, Accuracy and Vault keep their
decisions, but the decision model now selects them: named questions replace the
keyword, embedding, complexity and classifier signals, and Vault no longer
loads a separate content-risk classifier. Balance's three decisions become its
ten lanes: give the former `reasoning` pool to `recovery`, `agentic`,
`frontier` and `hard`, the `medium` pool to `facts`, `code` and `standard`, and
the `simple` pool to `fast`, then assign `long_context` and `vision`, a second
model to `frontier`, and the effort each lane runs at. Since policy 3.0,
content risk in Vault selects a responsible private model instead of a fixed
denial, and prompt attacks still receive an immediate response. Existing
published versions and their assignments remain unchanged.

## Evaluation

[`probes.yaml`](probes.yaml) covers multilingual requests, negative examples,
priority collisions, tools, images, retained conversation, and long inputs.
Validate routing behavior and actual backend eligibility against your deployment;
a correct decision alone does not prove that an assigned model can serve it.
See the [conformance guide](../../../CONFORMANCE.md) for commands.

The probes are calibrated on Vela 2.0 0.8B: through the router, with its
container limited to 4 CPUs, all 365 reach their expected decision. Re-run them with the
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

Balance's effort boundaries and its code lane were chosen with Vela 2.0 4B on
60 to 150 questions per benchmark; other workloads can need other thresholds.
Its difficulty score separates everyday from multi-step requests well, but the
boundary between high and max effort is coarse, and on Vela 2.0 0.8B some
requests fall on the other side of a boundary (the probes note where). With two
candidates, each `multi_factor` factor favors one of them outright, so small
load differences can switch the model. `long_context` reads the Router's token
estimate, about four bytes per token for prose until provider usage from
requests with at least 4 KiB of text calibrates it; for English prose it
counted 1.2 to 1.5 times a backend's tokens.

Knowledge-base retrieval, reranking, caching, and memory can be added for
applications that need them. These reusable recipes require no knowledge base
and inject no system prompt. Use [recipe tuning](https://vllm-sr.ai/docs/benchmarking/agent-evaluation-loop)
to measure these additions and session continuity against real requests.

## References

[Documentation](https://vllm-sr.ai/) ·
[GitHub](https://github.com/vllm-project/semantic-router) ·
[Recipe conformance](../../../CONFORMANCE.md)
