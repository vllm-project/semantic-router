---
slug: system-one-auto
title: "Decision Models Need a Router, Too"
description: "Meet System One Auto: native routing for decision models. Start with Kai 0.6B, escalate to Vega 27B, and measure the accuracy–latency tradeoff behind one API."
authors: [Xunzhuo]
tags: [routing, evaluation, decision-models, semantic-router]
image: /img/blog/system-one-auto/hero.jpg
---

import { ArticleChartGallery, ArticleFigure } from '@site/src/components/ArticleMedia';

<ArticleFigure
  src="/img/blog/system-one-auto/hero.jpg"
  width={1672} height={941}
  alt="System One Auto: Decision Models Need a Router, Too. The white vLLM Semantic Router logo above a golden horizon and star-filled sky."
/>

**Which model should make this decision?**

[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) put a simple interface in focus: give a model the state, a question and the allowed answers; get a structured judgment back. Open models such as [Decision 2.0](https://huggingface.co/collections/vllm-sr/decision-20) bring that interface to different model sizes. An application can use a small model for speed or a larger one for accuracy. Choosing one for every request leaves something on the table.

vLLM Semantic Router already tackles model selection for LLMs. **System One Auto brings native decision models into the same routing architecture.** Call `vllm-sr/auto` with your questions. Start small, check the answer, and spend a larger model call when the configured gate asks for one.

Our first Kai 0.6B → Vega 27B experiment improves public-suite accuracy from **65.80% to 82.25%**, while **54.11% of requests finish with Kai**. The larger model still wins on accuracy; the cascade wins on average latency in our warm, serial test. Below, we show both sides of that tradeoff.

[**Try the cascade →**](/docs/next/tutorials/algorithm/native/cascade) · [**Download the evidence →**](/img/blog/system-one-auto/evidence.zip)

<!-- truncate -->

## A new model pool. A familiar routing problem.

Decision models classify intent, score difficulty and check conditions. The application supplies the choices and criteria; the model returns typed answers and probabilities. As those models become interchangeable behind an API, selection becomes part of the serving system.

<ArticleFigure
  src="/img/blog/system-one-auto/ecosystem.svg"
  width={1800} height={1160}
  alt="Two model-selection problems: Chat and Responses requests route to LLMs; native System One requests route to decision models and return Choice, Score and Noul answers."
  diagram
>
  The same motivation, with a different response contract. This experiment uses Decision 2.0 Kai and Vega; broader provider evaluation is future work.
</ArticleFigure>

System One Auto keeps the native contract end to end. Model bindings can point to a local runtime, a separate vLLM-SR Engine, or a compatible external System One API. Recipes can select different decisions and cascades. The application keeps its questions when the serving policy changes.

## More accurate than Kai, with fewer Vega calls

All three paths received the same 231 public JevBench requests. Auto answered 125 with Kai and escalated 106 to Vega. Those upgrades rescued **41 Kai errors** and introduced **three errors** where Kai had been right.

<ArticleChartGallery label="Explore quality and model calls" charts={[
  { label: 'Accuracy', src: '/img/blog/system-one-auto/quality.svg', width: 3672, height: 2244, alt: 'Exact-label accuracy: direct Kai 65.80 percent, Auto 82.25 percent, direct Vega 87.88 percent. Auto gains 16.45 percentage points over Kai and remains 5.63 points below Vega.', children: 'The same 231 public items, scored with the unchanged JevBench scorer. Intervals resample the 195 source groups together across serving paths.' },
  { label: 'Request paths and calls', src: '/img/blog/system-one-auto/paths.svg', width: 3672, height: 2244, alt: 'Of 231 requests, 125 return Kai and 106 upgrade to Vega. Auto makes 231 Kai calls and 106 Vega calls: 337 native calls total.', children: '54.11% of requests finish with Kai. Every request still calls Kai, so fast-path coverage is different from the fraction of physical model calls.' },
]} />

The net gain is **16.45 percentage points over Kai**. Direct Vega scores **87.88%**, leaving a **5.63-point gap**: this result improves on Kai but does not establish Vega-equivalent quality. This is the pinned **231-item public JevBench suite**, not the official v1.6.1 sealed leaderboard.

The gate targeted 70% Kai coverage on separate calibration data and achieved 54.11% here. Coverage changes with the workload.

## Count the tail, too

We measured the real Router frontend, including the separate Engine backend used for Vega. Across two warm passes, Auto averaged approximately **88 ms versus 111 ms for direct Vega**—about **20% lower**. Its p95 was approximately **381 ms versus Vega's 351 ms**, about **8.5% higher**: escalated requests pay for both stages.

<ArticleFigure
  src="/img/blog/system-one-auto/latency.svg"
  width={3672} height={2244}
  alt="Warm passes show Auto mean latency of 88.15 and 88.59 milliseconds versus Vega's 110.98 and 111.03. Auto p95 is 380.45 and 381.93 milliseconds versus Vega's 350.83 and 351.31."
>
  Mean and tail belong in the same comparison. Both warm passes contain the same 231 requests per path, at concurrency one. The substantially different first-pass timings are retained separately.
</ArticleFigure>

This serial test used **two resident GPUs**, one per model. It measures a frontend-latency tradeoff. Fewer Vega calls do not, by themselves, establish a smaller GPU bill or higher throughput.

<details>
<summary>Measurement method, uncertainty and limits</summary>

**Quality.** The authentic [JevBench public suite at commit `b6b8fff7e345b98c060ad26c13308860ddc67004`](https://github.com/fstandhartinger/jevbench/tree/b6b8fff7e345b98c060ad26c13308860ddc67004) contains 139 Choice, 74 Noul and 18 Score items, representing 195 source groups. The unchanged [upstream scorer](https://github.com/fstandhartinger/jevbench/blob/b6b8fff7e345b98c060ad26c13308860ddc67004/jevbench/scoring.py) uses the modal label for Choice and Score and maps Noul to yes/no. This is exact-label public-suite accuracy, **not the official v1.6.1 sealed leaderboard or its composite score**. Direct Kai, Auto and direct Vega scored 152, 190 and 203 correct out of 231. The paired intervals use 5,000 source-group bootstrap samples with seed 20261010.

**Uncertainty and calibration.** Auto's gain over Kai has paired 95% interval **[11.34, 21.72] percentage points**; its gap below Vega has interval **[2.59, 8.89]**. The gate threshold, **0.6059704079536342**, was frozen using a separate 192-source calibration set before this evaluation. For Noul it checks `max(p, 1-p)`, so a confident “no” can exit early. The [complete YAML](/docs/next/tutorials/algorithm/native/cascade#define-the-cascade) reproduces this operating point; it is not a universal default.

**Requests.** The controlled collector uses the pinned upstream TypeSafe request mapping, adds `require_full_input: true` to each question and requests `options.return_meta`. The stock upstream CLI does not add those options. Three seeded, sequential passes compare direct Kai, direct Vega and Auto: 2,079 timed frontend requests plus nine separate typed warmups. There are no retries or discarded outliers. All measured requests succeeded and retained complete-input and runtime-identity evidence. Quality uses only pass 0; later passes check timing and determinism, not additional independent quality samples.

**Timing and calls.** Native HTTP request counters, read outside the timed request, confirmed 231 Kai calls and 106 Vega calls for Auto in each pass. These count inference exchanges, not GPU forward passes. Model choices and answers were identical across the three passes. First-pass mean latency was 210.98 ms for Vega and 126.61 ms for Auto; it is not pooled with the two warm passes. Their paired mean-latency ratio intervals are [0.723, 0.856] and [0.726, 0.859]. The p95 figures are point estimates; this study does not establish a tail-latency confidence bound or a throughput advantage.

**Scope.** Both native models used the exact ROCm profile. The evidence package pins the checkpoint revisions, runtime identities, image content digests and the measured source snapshot. These receipts identify the historical measurement; a rerun on another software or numerical profile needs its own validation. Native compute time, reservation limits and fewer large-model calls are not dollar-cost measurements.

</details>

## Answer first. Escalate when needed.

The first cascade is deliberately simple. **Kai answers the original question bundle.** If every required answer passes the gate, return it. Otherwise, send the original bundle to Vega. There is no extra Kai classifier call, and Auto returns one model's complete response with its native probabilities.

<ArticleFigure
  src="/img/blog/system-one-auto/cascade.svg"
  width={1800} height={1010}
  alt="The application sends a question bundle to Kai. If every answer passes the gate, return Kai. Otherwise, ask Vega the original questions and return its complete native response."
  diagram
>
  The selected cascade makes at most two native model calls in this example. A stage must also satisfy the common answer-validity checks; if no acceptable result fits the budget, the request remains unresolved.
</ArticleFigure>

This path supports **Choice, Score and Noul**. Native probabilities provide evidence for escalation; they are not calibrated guarantees of correctness. Choose the gate on separate calibration data, freeze it, then measure what happens on held-out requests.

## One name in your application

Follow the [cascade guide](/docs/next/tutorials/algorithm/native/cascade) to connect Kai and Vega and publish `vllm-sr/auto`. Your application sends an ordinary native request:

```bash
curl -sS http://localhost:8801/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "state": "I cannot sign into my account.",
    "questions": {
      "task": {
        "type": "choice",
        "require_full_input": true,
        "instructions": "Which team should handle this request?",
        "criteria": {
          "account": "Account access or authentication",
          "billing": "Payments, invoices or refunds"
        }
      }
    }
  }'
```

For a protected listener, add `-H "Authorization: Bearer $SYSTEMONE_API_KEY"`. The response's `routing` object reports the model, stage and algorithm-call count. Try the same route in Dashboard's **System One → Decision Playground** and inspect execution in **Decision Monitoring**.

Stages, acceptance rules and `algorithm.budget` live in YAML. Signals can choose a decision before its cascade runs; their calls and timeouts are separate from that algorithm's budget. Our measured path has no model-backed routing signal.

## Run it, or verify the recorded result offline

The [maintained evaluation tooling](https://github.com/vllm-project/semantic-router/tree/main/tools/calibration/systemone_auto) compares direct Kai, direct Vega and Auto without fitting a gate on the test labels. The [evidence package](/img/blog/system-one-auto/evidence.zip) includes the original schedule, safe saved responses, per-request timings and a replay script. You can check the reported totals without running a model.

<details>
<summary>Exact source pins and reproduction commands</summary>

Pin these model checkpoints in the runtime deployments:

| Alias | Model | Revision |
| --- | --- | --- |
| `kai` | `vllm-sr/Decision-2.0-Kai-0.6B` | `cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764` |
| `vega` | `vllm-sr/Decision-2.0-Vega-27B` | `7aec49ae11a18741706da549ab626b9052795fe7` |

Check out the benchmark and unpack the evidence download:

```bash
git clone https://github.com/fstandhartinger/jevbench.git
git -C jevbench checkout --detach b6b8fff7e345b98c060ad26c13308860ddc67004
unzip evidence.zip
```

Verify the archived answers and measurements through the unchanged upstream scorer, without network or model calls:

```bash
python3 system-one-auto-evidence/replay_evidence.py \
  --jevbench ./jevbench \
  --evidence ./system-one-auto-evidence/evidence
```

For a new live run, follow the cascade guide, publish the direct `kai` and `vega` aliases alongside `vllm-sr/auto`, and verify that the service loaded the pinned configuration. From the Semantic Router repository:

```bash
make harness-bootstrap
PYTHONPATH=tools/calibration .venv-agent/bin/python -m systemone_auto jevbench \
  --jevbench-checkout ./jevbench \
  --endpoint http://127.0.0.1:8801 \
  --config config.yaml \
  --gate-threshold 0.6059704079536342 \
  --schedule ./system-one-auto-evidence/evidence/request-schedule.json \
  --warmups ./system-one-auto-evidence/evidence/warmup-requests.json \
  --output-dir .agent-harness/jevbench-controlled
```

Add `--key-env SYSTEMONE_API_KEY` for a protected listener. `--plan-only` writes the request plan without calling the service. The output directory must be empty. Every planned failure remains in the quality denominator; interrupted runs stay incomplete. This command records the local configuration hash, so separately verify what the service actually loaded.

To verify native request-call counts, also pass `--kai-metrics` and `--vega-metrics` with the two dedicated runtimes' metrics endpoints. These counter reads happen outside the HTTP timing measurement; missing counters are unknown, not zero. Keep other traffic off those runtimes during the comparison. Without metrics endpoints, the collector still measures frontend timing and quality but cannot claim measured call-count deltas.

The generator, frozen figure data and vector figures are available with the [figure sources](https://github.com/vllm-project/semantic-router/tree/main/website/static/img/blog/system-one-auto/source). For a new application, calibrate on its own representative data, freeze the gate, and then evaluate quality, unresolved requests, mean and tail together.

</details>

Confidence cascades already have public precedents, including [Jev-Style's native decision cascade](https://github.com/lawrence3699/jev-style/commit/5c8149c26256ff4dc1b1e8693a70935b40f90e75). System One Auto brings that serving pattern into vLLM-SR: native model bindings, recipes, bounded execution and a result you can reproduce.

**Keep the questions. Make the model choice configurable.** [Build your first System One Auto route →](/docs/next/tutorials/algorithm/native/cascade)
