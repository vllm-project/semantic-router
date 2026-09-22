---
slug: decision-models
title: "Introducing Decision 1.0: Open Decision Foundation Models"
description: "Six open models for choosing actions, judging conditions, and applying rubrics. Meet Decision 1.0: runtime-defined questions, native probabilities, and batch decisions for the systems you build."
authors:
  - name: "vLLM Semantic Router Team"
    url: "https://github.com/vllm-project/semantic-router"
tags: [release, decision, mixture-of-models, semantic-router]
image: /img/blog/decision-1-0/launch-poster.png
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

**Introducing Decision 1.0: six Open Decision Foundation Models that turn context into choices, judgments, and scores.**

An agent needs its next action. A support queue needs the right destination. An invoice needs a policy check. A thousand records need the same rubric. These are small decisions that determine what an entire system does next.

Decision gives them a dedicated model family. Supply the evidence, ask your questions, and define the possible answers at runtime. Receive structured decisions and probability distributions, ready for your application to use. **Your questions define the task.**

[**Explore the six models →**](https://huggingface.co/collections/llm-semantic-router/decision-10) · [**Try Decision Studio →**](https://huggingface.co/spaces/llm-semantic-router/decision-studio)

<!-- truncate -->

## Intelligence at the point of action

We build [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) for systems that bring different models together. Those systems make decisions continuously: select a specialist, assess a condition, rank an option, or decide whether to escalate.

Decision turns that pattern into a reusable interface: **state + questions + criteria → answers**. Candidate descriptions travel with each request, so a new queue, action set, or scoring rubric does not require a new fixed classification head. The model scores the supplied candidates directly, without generating an explanation that an application must parse.

The launch film shows the same idea in motion: game state becomes an action, then the environment advances. Doom and chess use structured state and available actions supplied by their adapters. These are recordings from earlier checkpoints, illustrating the decision loop rather than measuring the latest releases; the shown Sol–Nox chess game ended in a draw.

## Built to decide

Decision has two architectural branches with the same three answer types.

**Kai and Lex** use three bidirectional encoder paths built on Vela. They share input embeddings, while Choice, Noul, and Score have separate interaction layers and candidate readouts. **Eos, Sol, Nox, and Lux** adapt Qwen3.5 text backbones, combining Gated DeltaNet with full attention. Their shared candidate head reads contextual candidate endpoints against a final query representation.

<ArticleFigure
  src="/img/blog/decision-1-0/architecture.png"
  width={3000} height={3620}
  alt="Lux architecture: complete state, question and candidates enter a 32-layer causal text backbone with Gated DeltaNet and full attention; a shared candidate head produces typed probabilities."
>
  Inside Lux-9B. All supplied candidates are scored in one forward pass per question; independent questions are grouped into physical batches. The released model contains the text backbone and decision head.
</ArticleFigure>

This makes the output space explicit. The application names the alternatives, the model evaluates them against the supplied evidence, and the application chooses how to act on the result.

<details>
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

[Lux model details](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B/blob/c22a05deaf2c4c492465f7e7048ed80f0d342e81/METHODS.md) · [Kai architecture](https://huggingface.co/llm-semantic-router/Decision-1.0-Kai-0.6B/blob/3ec2d25838bf50b60d56cacb03fde220ab9d638a/ARCHITECTURE.md)

</details>

## Six models, one decision interface

Choose a starting point for your workload, then evaluate it with your own states, options, and rubrics.

| Model | Starting point | Deployed parameters | Complete-input limit |
| --- | --- | ---: | ---: |
| [**Decision-1.0-Kai-0.6B**](https://huggingface.co/llm-semantic-router/Decision-1.0-Kai-0.6B) | Compact, general typed decisions with an encoder architecture. | 572M | 1,024 tokens |
| [**Decision-1.0-Lex-0.6B**](https://huggingface.co/llm-semantic-router/Decision-1.0-Lex-0.6B) | A Kai-derived specialist for customer service, invoices, security incidents, and agent traces. | 572M | 1,024 tokens |
| [**Decision-1.0-Eos-0.8B**](https://huggingface.co/llm-semantic-router/Decision-1.0-Eos-0.8B) | The smallest hybrid decoder in the family, with room for longer evidence. | 753M | 16,384 tokens |
| [**Decision-1.0-Sol-2B**](https://huggingface.co/llm-semantic-router/Decision-1.0-Sol-2B) | A larger hybrid decoder for decisions over extended context. | 1.884B | 16,384 tokens |
| [**Decision-1.0-Nox-4B**](https://huggingface.co/llm-semantic-router/Decision-1.0-Nox-4B) | A stronger decision model on the released suite, including action selection and rule application. | 4.208B | 16,384 tokens |
| [**Decision-1.0-Lux-9B**](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B) | Our largest decision model and highest-scoring family member on that suite. | 7.941B | 16,384 tokens |

Size suffixes identify model tiers; deployed counts above describe the released decision networks. For example, Lux adapts a 9B foundation while retaining a 7.941B text backbone and decision head. Input limits include the state, question, candidates, and formatting; oversized inputs are rejected.

## Three answers your software can use

A single state can support several independent questions. Consider this structured record:

```json
{"owner": "Lee", "status": "active", "severity": "medium"}
```

| Type | Question and criteria | Published Lux response |
| --- | --- | --- |
| **Choice** | “Choose the owner.” Candidates: `lee`, `sam`. | `lee`, with probability **99.57%**. |
| **Noul** | “Is the status active?” | **P(true) = 99.57%**. |
| **Score** | “Apply the severity scale.” Ordered levels: low, medium, high. | Expected index **1.006** on the 0–2 scale; **95.81%** probability on medium. |

These are rounded values from the release's [recorded example](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B/blob/c22a05deaf2c4c492465f7e7048ed80f0d342e81/model-card-example.json), not a new benchmark. Choice supports 2–255 candidates; the shared SystemOne interface supports 2–10 Score levels. Noul returns a probability, while Score returns a distribution and its expected ordinal index. Applications set their own thresholds and action policies; these numbers are not guarantees of correctness.

## One context. Many questions. Whole batches.

A support record might need a destination, a refund check, an escalation decision, and a priority score. Apply those four questions to 128 records and you have **512 decisions in one SDK batch submission**.

Kai and Lex's Python batch API supports up to 128 independent requests and 512 total decisions, preserving request and question order. The default physical batch is eight; an optional scheduler groups up to 32 compatible, same-type questions without increasing padding. The 512 figure is an API capacity, not 512 simultaneous model forwards or a measured throughput rate. Each state–question pair is encoded independently. [Batch API and limits](https://huggingface.co/llm-semantic-router/Decision-1.0-Kai-0.6B/blob/3ec2d25838bf50b60d56cacb03fde220ab9d638a/SYSTEM_ONE.md).

This is useful beyond queues: apply a policy across invoices, assess incident records, or score agent traces with a shared rubric. The work becomes a batch of explicit questions with structured answers.

We also publish question-count scaling for Lux's architecture and runtime:

<ArticleFigure
  src="/img/blog/decision-1-0/question-scaling.png"
  width={1980} height={1056}
  alt="Lux local request latency grows from 33.17 milliseconds for one question to 150.41 for eight and 600.74 for 32 distinct questions, with earlier weights on an idle AMD gfx942 GPU."
>
  Architecture/runtime measurements with earlier Lux weights: 499 input tokens per distinct question, 30 requests per point across six fresh processes. Warm local Python latency includes tokenization and inference; model loading and network transport are excluded.
</ArticleFigure>

These measurements describe that fixed workload and hardware, not concurrent HTTP throughput or the latest checkpoint's measured speed. [Full latency protocol, p95, and memory](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B/blob/c22a05deaf2c4c492465f7e7048ed80f0d342e81/QUESTION-SCALING.md).

## Measured across 54 tasks

The released comparison covers **3,766 scored decisions across 54 tasks**: general decisions, composition, reading, inference, and external transfer. On its weighted overall metric, **Lux reaches 76.94**, compared with **71.89 for Kev-9B**. Nox reaches **73.09**, Sol **66.32**, Eos **61.89**, and Kai **53.52**.

<ArticleFigure
  src="/img/blog/decision-1-0/decision-ranking.png"
  width={2376} height={2233}
  alt="Published weighted decision benchmark ranking: hosted Jev reference 81.05, Lux-9B 76.94, Nox-4B 73.09, followed by the complete open reference and Decision roster."
>
  Lux leads the displayed open-model references on this selected suite. The hosted Jev snapshot remains higher overall at 81.05.
</ArticleFigure>

The matrix makes the differences visible. Lux scores **84.10** on Decisions and **91.46** on Inference. Nox's Composition score is slightly higher than Lux's; external references lead some other panels. The right choice depends on the work you need done.

<ArticleFigure
  src="/img/blog/decision-1-0/decision-matrix.png"
  width={2376} height={1782}
  alt="Full capability matrix comparing Decisions, Composition, Reading, Inference, Transfer, and weighted overall accuracy for the 15 released benchmark rows."
>
  Every panel matters. An overall lead does not imply a win on every task, better probability calibration, or lower latency.
</ArticleFigure>

The overall weights are **30% Decisions, 25% Composition, and 15% each for Reading, Inference, and Transfer**. They reflect product priorities chosen after observing results. This is an observed regression suite, not a fresh blind test or a universal ranking. The public release includes [all task rows](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B/blob/c22a05deaf2c4c492465f7e7048ed80f0d342e81/TASKS.md), [evaluation methods and uncertainty](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B/blob/c22a05deaf2c4c492465f7e7048ed80f0d342e81/EVALUATION.md), and [weight sensitivity](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B/blob/c22a05deaf2c4c492465f7e7048ed80f0d342e81/WEIGHTING.md).

**Lex has a separate specialist evaluation:** 78.15% on a 2,000-decision operational test, versus 76.60% for the fine-tuned Laya Typed Decisions reference. Its observed gain has a paired 95% interval of −0.20 to +3.15 points. That test is not part of the 54-task ranking. [Lex evaluation](https://huggingface.co/llm-semantic-router/Decision-1.0-Lex-0.6B/blob/7983c480803fd003a3b79c4c8dafeb2131e1f94e/README.md).

## Make your first decision

The releases provide weights and inference code. Decision's own contributions use Apache 2.0; retained upstream component terms are documented in the model repositories, including [Kai's license scope](https://huggingface.co/llm-semantic-router/Decision-1.0-Kai-0.6B/blob/3ec2d25838bf50b60d56cacb03fde220ab9d638a/LICENSING_STATUS.md).

Download Lux, then follow its [qualified ROCm setup](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B/blob/c22a05deaf2c4c492465f7e7048ed80f0d342e81/RUNTIME.md). The release bundles its local Python API; runtime support is documented per model.

```bash
hf download llm-semantic-router/Decision-1.0-Lux-9B \
  --revision c22a05deaf2c4c492465f7e7048ed80f0d342e81 \
  --local-dir decision-model
```

With the downloaded repository mounted at `/model` in that environment:

```python
from decision import DecisionModel

model = DecisionModel.from_pretrained("/model", local_files_only=True)
result = model.decide(
    state={"owner": "Lee", "status": "active", "severity": "medium"},
    questions={
        "owner": {
            "type": "choice",
            "instructions": "Choose the owner.",
            "criteria": {"lee": "Lee", "sam": "Sam"},
        },
        "active": {"type": "noul", "instructions": "Is the status active?"},
        "severity": {
            "type": "score",
            "instructions": "Apply the severity scale.",
            "criteria": ["low", "medium", "high"],
        },
    },
)
print(result["answers"])
```

Start with your real states and candidate descriptions. Inspect errors and probability behavior, then choose thresholds against the outcomes your application needs. Kai and Lex also include [fine-tuning tools](https://huggingface.co/llm-semantic-router/Decision-1.0-Kai-0.6B/blob/3ec2d25838bf50b60d56cacb03fde220ab9d638a/FINETUNING.md) for adapting decisions to your own data.

**Open models. Your questions. Your next move.**

[**Get Decision 1.0 →**](https://huggingface.co/collections/llm-semantic-router/decision-10) · [**Explore the playground →**](https://huggingface.co/spaces/llm-semantic-router/decision-studio) · [**Build with vLLM-SR →**](https://github.com/vllm-project/semantic-router)
