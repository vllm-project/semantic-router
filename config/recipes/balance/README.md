# Balanced Routing Recipe Model Card

## Overview

Balanced Routing is a general-purpose policy for sharing traffic across local,
fast, specialist, and premium model lanes. The router's decision model answers
four named questions about each request in one call, and two conversation
facts come from the request itself. Each of the 15 routing decisions reads one
of these answers, so every route can be traced to the question that chose it.
The recipe declares the decision model it was calibrated with, vela3-lite
(Vela 3.0 0.8B).

The recipe exposes `vllm-sr/auto`. It is intended as a starting policy that
operators calibrate against their own models and traffic.

## Model details

| Role | Reference alias |
| --- | --- |
| Local default and creative work | `qwen/qwen3.5-rocm` |
| Fast, lower-cost specialist | `google/gemini-2.5-flash-lite` |
| Complex specialist and verification | `google/gemini-3.1-pro` |
| Deep reasoning and formal math | `openai/gpt5.4` |
| Premium legal analysis | `anthropic/claude-opus-4.6` |
| Secondary high-care health reviewer | `anthropic/claude-opus-4.6` |
| Visual understanding | `local/omni` |

These are routing aliases, not required vendor choices. Connect the Models you
operate, then review each decision assignment before publishing the Entrypoint.

## Intended use

Use this recipe for mixed assistant traffic where:

- most questions should use an efficient local or fast lane;
- complex reasoning and specialist domains justify stronger models;
- a follow-up saying that an answer was wrong should trigger a recovery path;
- explanations that ask for sources need a verified lane;
- image-bearing requests need an explicit visual-capability lane; and
- one request-facing model name is preferred over several client-side tiers.

Choose a more specialized recipe when privacy, minimum latency, or minimum cost
must dominate every decision.

## Decision questions

| Question | Type | Asks | Decides |
| --- | --- | --- | --- |
| `task` | Choice of 11 | What kind of work does this request ask for? | Which work lane a request takes. |
| `verification` | Yes or no | Does the user ask for sources, citations or evidence, or ask the assistant to fact-check a claim or verify an answer? | Whether an explanation takes the verified lane. |
| `correction` | Yes or no | In the latest message, does the user say the assistant's previous answer was wrong and ask for a corrected answer? | Whether a follow-up takes the recovery lane. |
| `clarify` | Yes or no | In the latest message, does the user ask the assistant to explain its previous answer again, more clearly, more simply or with an example? | Whether a follow-up takes the clarification lane. |

`task` counts when its most likely answer has probability at least 0.4. A yes
counts for `verification` at 0.5, for `clarify` at 0.6, and for `correction`
at 0.85. Each label of `task` has a description in
[`config.yaml`](config.yaml); the descriptions are part of the policy, and the
examples in them separate neighbouring kinds of work.

Two conversation facts need no model: `has_image` is true when the request
carries image content, and `has_prior_answer` is true when the conversation
already contains an assistant answer.

## Routing behavior

Decisions are tried by priority, and the first that matches wins.

| Priority | Decision | Reads | Candidates |
| --- | --- | --- | --- |
| 300 | `omni` | `has_image` | `local/omni` |
| 260 | `premium_legal` | `task`: legal | `anthropic/claude-opus-4.6`, `openai/gpt5.4` |
| 252 | `formal_math_proof` | `task`: proof | `openai/gpt5.4`, `anthropic/claude-opus-4.6` |
| 250 | `reasoning_deep` | `task`: argument | `google/gemini-3.1-pro`, `openai/gpt5.4` |
| 242 | `complex_specialist` | `task`: systems or science | `google/gemini-3.1-pro`, `openai/gpt5.4` |
| 232 | `feedback_wrong_answer_verified` | `has_prior_answer` and `correction` | `google/gemini-2.5-flash-lite`, `google/gemini-3.1-pro` |
| 220 | `medium_code_general` | `task`: code | `qwen/qwen3.5-rocm`, `google/gemini-2.5-flash-lite` |
| 218 | `verified_health` | `task`: health | `google/gemini-3.1-pro`, `anthropic/claude-opus-4.6` |
| 214 | `verified_explainer` | `task`: explain and `verification` | `google/gemini-2.5-flash-lite`, `google/gemini-3.1-pro` |
| 212 | `feedback_need_clarification` | `has_prior_answer` and `clarify`, not `correction` | `qwen/qwen3.5-rocm`, `google/gemini-2.5-flash-lite` |
| 208 | `medium_explainer` | `task`: explain, not `verification` | `qwen/qwen3.5-rocm`, `google/gemini-2.5-flash-lite` |
| 200 | `medium_creative` | `task`: creative | `qwen/qwen3.5-rocm`, `google/gemini-2.5-flash-lite` |
| 184 | `fast_qa` | `task`: facts | `qwen/qwen3.5-rocm`, `google/gemini-2.5-flash-lite` |
| 170 | `simple_general` | `task`: chat | `qwen/qwen3.5-rocm`, `google/gemini-2.5-flash-lite` |
| 10 | `casual_chat` | Nothing; the fallback | `qwen/qwen3.5-rocm`, `google/gemini-2.5-flash-lite` |

`task` gives each request one kind of work, so the work lanes never compete
with each other. Priority matters for images and follow-ups. An image goes to
`omni` whatever the text asks. A correction outranks the code, health,
explanation, creative and short-answer lanes, but not the legal, proof,
argument and specialist lanes, which already use stronger models. Systems and
science are two labels for the same lane.

When no answer counts, the request falls to `casual_chat`. The same happens
when the decision model cannot answer, for example for a blank message: every
rule that reads a question treats an unknown answer as no match.

## Calibration notes

The thresholds were calibrated on the maintained probes with vela3-lite:

- On a first message the model sometimes answers yes to `clarify`: of the 63
  first-message probes it answers, 4 reach the `clarify` threshold, and none
  scores more than 0.59 for `correction`. Both questions therefore count only
  after an earlier assistant answer.
- After an earlier answer, the 5 corrections score 0.97 or more for
  `correction`; the 4 clarification requests score 0.78 or more for `clarify`
  and 0.75 or less for `correction`; the 3 follow-ups that ask for something
  new score 0.06 or less for `correction` and 0.34 or less for `clarify`.
  `correction` counts at 0.85, inside that gap; at 0.7, two clarification
  requests took the recovery lane.
- Explanations that ask for sources score 0.61 or more for `verification`,
  and the other explanations 0.03 or less. A wording that began with
  verifying the answer scored a request to "compare recent peer-reviewed
  findings with citations" 0.39, so the question leads with sources,
  citations and evidence.
- The lane-deciding `task` answers have probability 0.78 or more.

The descriptions come from the calibration on Vela 2.0 0.8B, where
"Verify with a source whether light travels faster than sound" was read as
`science` (0.91) until `facts` named basic facts of science and nature, even
when the user asks for a source, and `science` named hard problems that need
multi-step reasoning, and "Who are you?" was read as `facts` until `chat` named
questions about the assistant itself.

Two descriptions come from an earlier calibration on Vela 2.0 0.3B: with
engineering plans and advanced science described as one label, that model
read a comparison of qubit decoherence sources as an argument, so `systems`
and `science` are two labels; and it read "What is 2 + 2? Answer briefly." as
a proof until the `proof` description named proving a theorem.

When you change a description or a threshold, or move to another decision
model, rerun the probes; a change in one label can move requests between
neighbouring labels.

## Requirements

- Reachable OpenAI-compatible endpoints for the configured aliases.
- The decision model for the four questions. The recipe declares it as the
  `primary` deployment of vela3-lite, which
  `global.model_catalog.system.decision_model` names. Keep that binding: an
  unset decision model follows the router default, vela3-fast, on which these
  thresholds were not calibrated.
  In the CPU conformance run, with the router limited to 4 CPUs and 4
  requests in flight, the median request took 19 s and the 95th
  percentile 41 s. The model reads the whole message once per question, so a
  long message costs more on a CPU: the repeated question in the probes is
  asked 100 times, because at 500 its four questions passed the decision
  deadline there.
  Serve it on a GPU for live traffic. With no device set, the runtime uses a
  GPU when the router has one and the CPU otherwise; set `device: rocm` or
  `device: cuda` on the `primary` deployment to require one. To use another
  decision model, change its `artifact` or point
  `global.model_catalog.system.decision_model.deployment` at another
  deployment, and rerun the probes, because the thresholds were calibrated on
  vela3-lite.
- A response-cache store when cache behavior is desired.
- Calibrated cost and capability metadata for the actual deployment.

## Data handling and safety

The decision model reads each request's messages to answer the four
questions. Review where it runs before configuring a different decision model.

The checked-in decisions opt into in-memory Router Replay. Replay captures up
to 4 KiB each from the request and response body for a routed request. The
records are process-local and non-durable, but they can still contain prompts,
answers, and other sensitive application data while the Router is running.

Review body capture, access, and retention before production use. Disable
replay or body capture when the deployment must not retain request content.

## Quick start

```bash
vllm-sr serve
```

In the Dashboard, connect the physical Models, choose **Balanced Routing** in
**Recipes**, assign a Model to each decision, and publish the resulting
Mixture-of-Model Entrypoint. The checked-in YAML and probes remain the
maintainer-facing source and conformance fixtures for this built-in Recipe.

## Evaluation

The 75 probes cover every decision, priority collisions between follow-ups and
work lanes, follow-ups that ask for something new, verification requests,
English and Chinese requests, a long repeated question, and the fallback for a
blank message. CPU conformance runs them with the declared vela3-lite,
which reaches the expected decision for all 75. With the router limited to 4
CPUs, as on a 4-vCPU CI runner, and 4 probes in flight, the median latency
was 19 seconds. See
[`probes.yaml`](probes.yaml) and the
[conformance guide](../CONFORMANCE.md).

## Limitations

- Quality, latency, and cost trade-offs depend on measured deployment data;
  the reference coefficients are not pricing or performance guarantees.
- The decision model can misread unfamiliar kinds of work or languages; a
  misread request takes another lane, and an unclear one takes `casual_chat`.
- A route to a stronger model does not guarantee correctness.
- The recipe does not provision the inference backends it references.

## References

- [Recipe metadata](metadata.yaml)
- [Runtime configuration](config.yaml)
- [Routing DSL](recipe.dsl)
- [Evaluation probes](probes.yaml)
- [Recipe authoring and conformance](../CONFORMANCE.md)
