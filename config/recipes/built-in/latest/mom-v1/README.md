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

Balance runs the [Decision Balance](../../../decision-balance/README.md) policy
over the models you assign: the router's decision model answers its routing
questions in one call per request. Speed, Cost, Accuracy and Vault use
embedding, complexity, keyword and classifier signals. Their embedding and
complexity rules use rule-level settings to keep all declared candidate texts
for scoring instead of compressing them to a global bank limit. Task prototypes
cover both direct operations and work that requires design, diagnosis, proof,
or synthesis.

## Intended use

Use Balance for mixed workloads, Speed for interactive applications, Cost for
cost-sensitive serving, Accuracy for demanding work, and Vault for an approved
private deployment. Each recipe can use a different set of connected models.

## Routing behavior

A concise answer can still require hard reasoning. Speed, Cost and Accuracy
retain learned hard-task routing when the requested answer is brief. Hard tasks
with client tools use the reasoning pool, with tool capabilities still enforced
when selecting a model.

Cost and Accuracy recognize explicit requests to repair an incorrect answer,
recommendation, or explanation. Feedback classification requires an earlier
assistant reply. Without an earlier reply, explicit correction wording can
still route to reasoning. Quoted corrections, negated repair requests, and
style edits alone do not qualify.

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
per-token latency for reasoning. Lightweight semantic and request-shape signals
keep routing overhead small.

**Cost** keeps requests on one model. Its base selector prefers the lowest
estimated request cost within the decision's quality band, with stronger pools
for reasoning and active tool use. After an assistant answer, correctness
feedback or a direct request to correct the result can escalate answer recovery.
A self-contained error report with an instruction to repair it also qualifies.
Repetition or a formatting edit alone does not establish that an answer was wrong.

**Accuracy** normally uses one strong model. An explicit request to obtain fresh
independent assessments selects `review`, which compares two responses and
synthesizes them. Comparing already supplied opinions stays on one model.
An explicit request to delegate work to multiple workers selects `agent`, with
at most three steps and two workers in parallel. A request for only one worker
does not authorize fan-out; independence alone does not establish multiple
workers. Breaking a task into workstreams alone also stays on one model. The
router plans the internal stages and combines results; users need
not name those stages. Describing, translating, or prohibiting a workflow does
not authorize execution. Long context or a subject label does not trigger fan-out.
A client-owned tool loop stays on the single-model reasoning path.

**Vault** uses separate pools for ordinary and sensitive requests. `guard`
contains detected prompt attacks before calling a backend. Safety and Hazard
route content risks to `sensitive`, where the assigned model can provide help,
explain risky material, or refuse harmful assistance. Hazard uses its published
per-category operating point independently of the binary Safety verdict.
Personal action guidance also uses `sensitive` when a relevant domain accompanies
an explicit request or consequential intent. An explicit personal action request
can also qualify when both consequential and supportive intent exceed
informational intent and at least one of those intent signals is strong. This path
allows assistance when the domain label is incomplete. Definitions and quoted
requests alone do not establish that intent;
content-risk protection still applies.

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

Balance's questions name no deployment, so the router's default decision model
answers them, and the questions that one request stage asks a model travel in
one call. A new configuration uses Vela 2.0 0.3B on the CPU. To answer them
with a larger decision model, point `global.model_catalog.system.decision_model`
at a deployment of it, as
[Decision Balance](../../../decision-balance/config.yaml) does with Vela 2.0 4B
on ROCm. See [Choose a model](https://vllm-sr.ai/docs/model-runtime/choose-a-model)
for sizes and hardware.

Vault uses the default logical `hazard` deployment for its category classifier.
It loads only when used. Override that deployment for qualified accelerator
execution while retaining the model's `operating_point.json`. Verify the sensitive
pool on help-seeking, benign analysis, and harmful requests; a general quality
score does not certify responsible answers.

The default Vela Guard, PII, and Hazard tasks each accept at most 32,768 input
tokens for triage, including the conversation text evaluated by that task.
Their tokenizers can differ from the backend's tokenizer. Exceeding a triage
budget fails closed before backend context checks or dispatch; assigning a
backend with a larger context window does not raise this limit. Preview returns
HTTP 503 with `signal_errors` and no selected model. The maintained Vault
long-input probes stay within this budget and retain their routing assertions.

The default Hazard deployment runs on CPU. Long inputs can make triage slow
even within the supported budget; measure Preview latency for your traffic and
qualify an accelerated Hazard deployment when needed.

## Data handling and safety

Vault routes personal identifiers to its sensitive pool. Detected nationality,
religion and political affiliation require an explicit personal attribute or profile context
in the current request; quoted personal records also qualify.

Every Vault path disables client tools, strips tool history, and disables Router
memory, response caching, and learning adaptation. It also
suppresses new Responses object writes. Replay capture is configured separately
and is enabled by default as described below. These restrictions apply regardless of
the Guard, Safety, Hazard, and PII verdicts; unavailable triage fails closed.

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

When upgrading to policy 4.0, Balance keeps its three decisions, but the
decision model now selects them, and tool loops and requests that need their
declared tools move to `reasoning`, so give that pool models that support
tools. Speed, Cost, Accuracy and Vault keep their decisions. Since policy 3.0,
content risk in Vault selects a responsible private model instead of a fixed
denial, and prompt attacks still receive an immediate response. Existing
published versions and their assignments remain unchanged.

## Evaluation

[`probes.yaml`](probes.yaml) covers multilingual requests, negative examples,
priority collisions, tools, images, retained conversation, and long inputs.
Validate routing behavior and actual backend eligibility against your deployment;
a correct decision alone does not prove that an assigned model can serve it.
See the [conformance guide](../../../CONFORMANCE.md) for commands.

Balance's probes cover its three decisions and all eight lanes with the default
Vela 2.0 0.3B, so CPU conformance exercises the whole policy. Re-run them with
the decision model you deploy.

For local task difficulty in Speed, Cost and Accuracy, the score is the
hard-prototype score minus the
easy-prototype score. A margin above `hard_above: 0.025` selects hard; a margin
below `easy_below: -0.08` selects easy. Scores between the boundaries, including
either boundary itself, remain medium. Keeping the easy boundary farther from
zero requires stronger evidence before choosing the cheapest or simplest pool.

Calibrate candidate phrases and both boundaries together on representative
easy, ordinary, and demanding requests in your supported languages. Adjust the
hard and easy sides separately, checking unnecessary escalation as well as
premature downgrading. Freeze the chosen configuration before evaluating fresh
requests, and keep any requests used to choose a boundary in the calibration
results. Use Preview to verify actual signal scores, routing, and latency.

## Limitations

Quality bands compare the assigned pool; they do not guarantee answer accuracy.
Balance's routing depends on the decision model's answers. Its questions read
the beginning of the latest user message, so a very long input is judged by its
first tokens: put the request before a long attachment. The default Vela 2.0
0.3B rates some plain explanations as multi-step, reads some copy tasks on a
supplied record as fact lookups, and can miss the tool need of a request that
comes with a long record.
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
