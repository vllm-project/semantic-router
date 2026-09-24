---
slug: decision-models
title: "Introducing Decision 1.0: Open Decision Foundation Models"
description: "Meet Decision 1.0: six Open Decision Foundation Models for routing, policy, agent actions, and batch decisions, with a path to native vLLM-SR integration."
authors:
  - name: "vLLM Semantic Router Team"
    url: "https://github.com/vllm-project/semantic-router"
tags: [release, decision, mixture-of-models, semantic-router]
image: /img/blog/decision-1-0/social-card.png
---

import { ArticleFigure, ArticleVideo } from '@site/src/components/ArticleMedia';

<ArticleVideo
  src="/videos/decision-1-0/decision-1-0-launch.mp4"
  poster="/img/blog/decision-1-0/launch-poster.png"
  title="Decision 1.0: Open Decision Foundation Models"
  landscape
>
  From the next move to a whole queue of decisions. Meet Decision 1.0. Sound on.
</ArticleVideo>

**AI is moving from generating answers to making decisions inside software.**

Those decisions are everywhere: route an intent, gate an action by confidence, combine signals into a score, or ask many questions at once. Small decisions can shape the behavior of an entire system.

That vision is deeply aligned with **vLLM Semantic Router**: understand each request, apply policy, and choose what should happen next. It is why we invested in **Decision 1.0: six open-weight Decision Foundation Models**—and why we will keep improving them in the open.

[**Read the paper →**](/decision-paper) · [**Explore the six models →**](https://huggingface.co/collections/llm-semantic-router/decision-10) · [**Try Decision Studio →**](https://huggingface.co/spaces/llm-semantic-router/decision-studio)

<!-- truncate -->

## Why the decision layer matters

AI can generate an answer. An application still has to decide what happens next.

A useful decision layer has to keep up with the application. Available models change. Policies evolve. A fast answer may be the right outcome for one request, while another deserves more computation. The application needs a way to express those differences and evaluate the available choices.

<ArticleVideo
  src="/videos/decision-1-0/decision-1-0-tetris.mp4"
  poster="/img/blog/decision-1-0/tetris-poster.png"
  title="Decision 1.0 speed comparison in Tetris"
  showcase
>
  <em>Feel the speed.</em>
</ArticleVideo>

Decision 1.0 makes the decision itself programmable:

**State + questions + criteria → typed probability distributions.**

The application defines what matters and what is allowed. The model evaluates the choices. The system stays in control.

Route a request. Select an action. Apply a policy. Score an outcome. One interface, many decisions.

This is the layer between intelligence and action: open, programmable, and built to keep software in control.

## Built to decide

Decision scores the candidates you provide directly. Each candidate is evaluated in the context of the evidence and the question, so an option can be a short label or a description of what an action means.

Two architectural branches make this possible. **Kai and Lex** use three bidirectional encoder paths built on Vela, with separate interaction layers and readouts for Choice, Noul, and Score. **Eos, Sol, Nox, and Lux** adapt Qwen3.5 text backbones, combining Gated DeltaNet with full attention and a shared candidate head.

<ArticleFigure
  src="/img/blog/decision-1-0/architecture.png"
  width={3000} height={3620}
  alt="Lux architecture: state, question and candidates enter a 32-layer causal text backbone with Gated DeltaNet and full attention; a shared candidate head produces typed probabilities."
>
  Inside Lux-9B. All candidates for one question are scored in a single forward pass. Independent questions are processed in batches.
</ArticleFigure>

The result is a model designed around the decision itself: compare the alternatives, return their probabilities, and let the application take the next step.

<details open>
<summary>Explore the encoder branch and candidate readout</summary>

<ArticleFigure
  src="/img/blog/decision-1-0/kai-architecture.png"
  width={3000} height={3700}
  alt="Kai architecture with three 22-layer bidirectional paths, shared input embeddings, and separate Choice, Noul and Score interaction layers and readouts."
>
  Kai's three-path encoder architecture is also used by Lex, whose weights specialize it for operational workflows.
</ArticleFigure>

<ArticleFigure
  src="/img/blog/decision-1-0/readout.png"
  width={3000} height={2420}
  alt="Lux candidate readout combines each contextual candidate endpoint with the global query vector, scores candidates with a shared head, and normalizes their probabilities."
>
  Lux's shared candidate readout. Candidate descriptions come from the request, rather than a fixed vocabulary of task labels.
</ArticleFigure>

[Kai architecture](https://huggingface.co/llm-semantic-router/Decision-1.0-Kai-0.6B/blob/3ec2d25838bf50b60d56cacb03fde220ab9d638a/ARCHITECTURE.md)

</details>

## Six models, one decision interface

Start with the model that fits your workload. Keep the questions, options, and rubrics as you explore the family.

| Model | Built for | Input budget |
| --- | --- | ---: |
| [**Decision-1.0-Kai-0.6B**](https://huggingface.co/llm-semantic-router/Decision-1.0-Kai-0.6B) | A compact, general-purpose starting point for routing, conditions, and action selection. | 1,024 tokens |
| [**Decision-1.0-Lex-0.6B**](https://huggingface.co/llm-semantic-router/Decision-1.0-Lex-0.6B) | Operational decisions: customer service, invoices, security incidents, and agent traces. | 1,024 tokens |
| [**Decision-1.0-Eos-0.8B**](https://huggingface.co/llm-semantic-router/Decision-1.0-Eos-0.8B) | The smallest hybrid decoder, bringing longer evidence into a compact model tier. | 16,384 tokens |
| [**Decision-1.0-Sol-2B**](https://huggingface.co/llm-semantic-router/Decision-1.0-Sol-2B) | The next step in capacity for decisions over extended context. | 16,384 tokens |
| [**Decision-1.0-Nox-4B**](https://huggingface.co/llm-semantic-router/Decision-1.0-Nox-4B) | More capacity for combining conditions, applying rules, and choosing actions. | 16,384 tokens |
| [**Decision-1.0-Lux-9B**](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B) | Our largest model and strongest overall result on the released decision suite. | 16,384 tokens |

Input budgets cover the complete state, question, and candidate descriptions. The model cards document each release's architecture and runtime requirements.

<ArticleVideo
  src="/videos/decision-1-0/decision-1-0-nox-4b-robot-arm.mp4"
  poster="/img/blog/decision-1-0/nox-4b-robot-arm-poster.png"
  title="Nox-4B action selection in a robot-arm simulation"
  landscape
>
  Nox-4B chooses actions step by step in a robot-arm simulation; the on-screen number is model latency for that step.
</ArticleVideo>

## Three answers your software can use

**Choice** picks an option. **Noul** judges a condition. **Score** applies an ordered rubric. Together, they cover the decisions inside a much larger workflow.

Consider a customer who reports a damaged parcel and requests a replacement today:

| Type | Ask the model | Use the answer |
| --- | --- | --- |
| **Choice** | Which team should handle this: delivery, billing, or technical support? | Route the request using the selected candidate ID and its distribution. |
| **Noul** | Does the customer request action today? | Use the probability of yes to decide whether to escalate. |
| **Score** | Rate urgency on a scale: can wait, this week, today. | Prioritize the queue using the level distribution and expected score. |

The options are yours. A Choice can select a tool, a backend, or an available game action. A Noul can check a refund condition or whether an answer is supported by evidence. A Score can apply your review rubric to documents or completed agent runs. Score levels are ordered from zero; Noul returns a probability, with the action threshold set by your application.

## One context. Many questions. Whole batches.

A single support record can need a destination, a refund check, an escalation decision, and a priority score. Apply those four questions to 128 records and you have **512 decisions in one local SDK batch submission**.

Kai and Lex's native Python API accepts up to 128 independent requests and 512 total decisions, preserving request and question order. The runtime groups work into physical batches; 512 describes submission capacity, not simultaneous forwards or measured throughput. [Batch API](https://huggingface.co/llm-semantic-router/Decision-1.0-Kai-0.6B/blob/52c81702356711b43b1a68e4aca8c98c84230155/SYSTEM_ONE.md).

This is where a decision model becomes useful across an entire operation: apply a policy to a stack of invoices, triage a stream of incidents, or assess thousands of agent traces in successive batches. Define the questions once, bring new contexts, and collect structured answers ready for the next stage of the workflow.

## Measured across 54 tasks

The published comparison spans **3,766 scored decisions across a selected 54-task suite** covering decisions, composition, reading, inference, and transfer. **Lux reaches 76.94 overall**, ahead of the displayed open-model references, including **Kev-9B at 71.89**. Nox reaches **73.09**, followed by Sol at **66.32**, Eos at **61.89**, and Kai at **53.52**.

<ArticleFigure
  src="/img/blog/decision-1-0/decision-ranking.png"
  width={2376} height={2233}
  alt="Published weighted decision benchmark ranking: hosted Jev reference 81.05, Lux-9B 76.94, Nox-4B 73.09, followed by the complete open reference and Decision roster."
>
  Lux leads the displayed open-model references on this suite. The hosted Jev reference remains higher overall at 81.05.
</ArticleFigure>

The capability matrix shows where each model stands. Lux reaches **84.10** on Decisions and **91.46** on Inference. Nox leads the displayed Decision models in Composition. Use the panels to find a starting point for the work you want to build, then evaluate on your own examples.

<ArticleFigure
  src="/img/blog/decision-1-0/decision-matrix.png"
  width={2376} height={1782}
  alt="Capability matrix comparing Decisions, Composition, Reading, Inference, Transfer, and weighted overall accuracy for the 15 released benchmark rows."
>
  One family, different strengths. Results use the release's selected tasks and weighted overall metric.
</ArticleFigure>

## Bring your System One workflow

Decision uses the upstream **System One request format**: `state`, `model`, and named `questions`. The latest model cards show how to use the **official TypeSafe Python SDK** or an equivalent HTTP request with your own SystemOne-compatible deployment.

Configure that deployment to serve `Decision-1.0-Lux-9B`, then replace the placeholder URL and key below. The model repositories provide weights and local inference code; an HTTP endpoint must be deployed separately.

```bash
pip install typesafe-sdk
```

Published request example from the [Lux model card](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B/blob/ec7001aa04b2fe2a682e02aed9572e047bd68993/USAGE.md):

```python
from typesafe_sdk import Choice, Noul, TypeSafeClient

with TypeSafeClient(
    api_key="YOUR_ENDPOINT_API_KEY",
    base_url="https://your-decision-endpoint.example",
    model="Decision-1.0-Lux-9B",
) as client:
    result = client.system_one(
        state="Customer reports a duplicate charge and asks for a refund.",
        questions={
            "route": Choice(
                instructions="Which team should handle this request?",
                criteria={
                    "billing": "Payments and refunds",
                    "technical": "Product faults",
                },
            ),
            "refund_requested": Noul(
                instructions="Did the customer request a refund?"
            ),
        },
    )
    print(result.choices["route"].choice)
    print(result.nouls["refund_requested"].noul)
```

The same questions with curl:

```bash
curl -X POST 'https://your-decision-endpoint.example/v1/systemone' \
  -H 'Authorization: Bearer YOUR_ENDPOINT_API_KEY' \
  -H 'Content-Type: application/json' \
  --data-raw '{
    "model": "Decision-1.0-Lux-9B",
    "state": "Customer reports a duplicate charge and asks for a refund.",
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
          "billing": "Payments and refunds",
          "technical": "Product faults"
        }
      },
      "refund_requested": {
        "type": "noul",
        "instructions": "Did the customer request a refund?"
      }
    }
  }'
```

To try Kai, configure its `Decision-1.0-Kai-0.6B` deployment alias and keep the same request structure. Its [SDK and curl guide](https://huggingface.co/llm-semantic-router/Decision-1.0-Kai-0.6B/blob/52c81702356711b43b1a68e4aca8c98c84230155/USAGE.md) walks through delivery routing and urgency. Add questions to inspect another dimension of the state, or reuse the questions with new contexts.

## Next: an open decision runtime

Our next step is **native Decision integration in vLLM Semantic Router**. We want vLLM-SR to become a runtime for decision models, with support for an **Open Decision API** that brings model execution and application decisions together.

For routing, that means using request context, candidate capabilities, and measured operating signals to make better model choices under quality, cost, and latency constraints. Alongside that native path, we plan to retain a **System One-compatible general decision API**, so the same family can serve agent actions, policy checks, and evaluation workflows beyond routing.

This is the roadmap ahead. Today's release provides the models and local inference interfaces; native vLLM-SR integration and Open Decision API support are planned next.

## Build your next move

Start in Decision Studio, download a model, or bring the System One format into your application. Kai and Lex also include [fine-tuning tools](https://huggingface.co/llm-semantic-router/Decision-1.0-Kai-0.6B/blob/52c81702356711b43b1a68e4aca8c98c84230155/FINETUNING.md) for adapting decisions to your own data. Decision's contributions use Apache 2.0, with retained upstream terms documented in each repository.

**Open models. Your questions. Your next move.**

[**Get Decision 1.0 →**](https://huggingface.co/collections/llm-semantic-router/decision-10) · [**Explore the playground →**](https://huggingface.co/spaces/llm-semantic-router/decision-studio) · [**Build with vLLM-SR →**](https://github.com/vllm-project/semantic-router)
