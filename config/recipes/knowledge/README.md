# Knowledge Routing Recipe Model Card

## Overview

Knowledge Routing uses measured domain evidence to decide whether a question
should stay on a small local model or move to a larger frontier model. The
evidence is seeded from MMLU domain results: subjects where the larger model
gains the most escalate, and the others stay local. The router's decision
model names each question's subject in one call, and the routing decisions
read that answer. The recipe declares the decision model it was calibrated
with, Vela 2.0 0.8B. Replace the subject split with an organization's own
benchmarks and feedback.

## Model details

| Lane | Reference alias | Purpose |
| --- | --- | --- |
| Local | `local/small-7b` | Lower-cost default for domains with little measured uplift. |
| Frontier | `cloud/frontier-72b` | Escalation lane for domains with meaningful measured uplift. |
| Visual | `local/omni` | Dedicated image-understanding lane. |

Clients use `vllm-sr/auto`. Both aliases are examples and should be rebound to
the models represented by the deployment's own evidence.

## Intended use

Use this recipe when:

- model selection should be justified by benchmark or feedback data;
- a small model handles many domains well enough;
- frontier capacity should be reserved for domains with measurable benefit; or
- operators need an explainable subject label behind each escalation.
- image-bearing questions require a model with explicit visual capability.

It is not suitable when the subject split is stale, unrelated to production
traffic, or based on too little evidence to support the routing decision.

## Decision question

| Question | Type | Asks | Decides |
| --- | --- | --- | --- |
| `knowledge_subject` | Choice of 14, at 0.4 | Which subject is this question about? | Whether the question escalates to the frontier lane. |

The 14 subjects are biology, business, computer science, economics,
engineering, health, history, law, mathematics, philosophy, physics,
chemistry, psychology, and other (general knowledge, geography and culture);
each has a description in [`config.yaml`](config.yaml). One conversation fact
needs no model: `knowledge_has_images` is true when the request carries image
content.

## Routing behavior

Decisions are tried by priority, and the first that matches wins.

| Priority | Decision | Reads | Behavior |
| --- | --- | --- | --- |
| 300 | `omni` | `knowledge_has_images` | Send the request to the visual-language alias. |
| 200 | `escalate_72b` | `knowledge_subject`: biology, business, computer science, economics, history, mathematics, philosophy, psychology or other | Send the request to the frontier alias. |
| 100 | `keep_7b` | Nothing; the fallback | Keep the request on the local alias. |

Engineering, health, law, physics and chemistry stay local, and so does a
question whose subject has no answer with probability 0.4 or more. When the
decision model cannot answer, the request stays local too, so the recipe
needs no knowledge-base or keyword fallback.

## Calibration notes

On the maintained probes with Vela 2.0 0.8B, every domain question gets its
intended subject. The escalating answers have probability 0.85 or more; the
lowest local answer is engineering for an ideal-transformer question (0.64,
against physics at 0.33), and both of those subjects stay local. A
zero-determinant question about a data table is read as mathematics (0.88).

Earlier versions matched questions against a knowledge base of MMLU examples;
on the same probes it mislabelled 6 of 15 domain questions, for example a
matrix question as physics.

## Requirements

- Reachable OpenAI-compatible endpoints for the local, frontier, and visual aliases.
- The decision model for the subject question. The recipe declares it as the
  `primary` deployment of Vela 2.0 0.8B, which
  `global.model_catalog.system.decision_model` names. The deployment sets no
  device, so the model runtime uses a GPU when the router has one and the CPU
  otherwise.
  In the CPU conformance run, with the router limited to 4 CPUs and 4
  requests in flight, the median request took 9.8 s and the 95th
  percentile 10.0 s.
  Serve it on a GPU for live traffic. With no device set, the runtime uses a
  GPU when the router has one; set `device: rocm` or `device: cuda` on the
  `primary` deployment to require one. For a smaller model, change its
  `artifact` (for example to `vllm-sr/Vela-2.0-0.3B`) or point
  `global.model_catalog.system.decision_model.deployment` at another
  deployment, and rerun the probes, because the threshold was calibrated on
  the 0.8B.
- Production benchmark or feedback data before using the subject split for a
  materially different workload.

## Data handling and safety

The checked-in recipe does not enable Router Replay or response caching. The
decision model reads each question to name its subject, then the request goes
to one selected provider.

Provider-side logging and retention still apply. Run the decision model where
the questions may be read.

## Quick start

```bash
vllm-sr serve
```

In the Dashboard, connect the physical Models, choose **Knowledge Routing** in
**Recipes**, assign the local and frontier decisions, and publish the resulting
Mixture-of-Model Entrypoint.

## Evaluation

The 16 probes cover each escalation and local subject, a general-knowledge
question, a boundary question, and an image. They validate route selection
without generating model responses, and CPU conformance runs them with the
declared Vela 2.0 0.8B, which reaches the expected decision for all 16. See
[`probes.yaml`](probes.yaml) and the [conformance guide](../CONFORMANCE.md).

## Limitations

- MMLU is only seed evidence and may not represent a production workload.
- Benchmark uplift can change after model, prompt, or serving updates.
- A subject label does not guarantee factual correctness, and the decision
  model can misread a question that spans subjects.
- The reference prices and aliases are examples, not billing or performance
  claims.

## References

- [Recipe metadata](metadata.yaml)
- [Runtime configuration](config.yaml)
- [Routing DSL](recipe.dsl)
- [Evaluation probes](probes.yaml)
- [Recipe authoring and conformance](../CONFORMANCE.md)
