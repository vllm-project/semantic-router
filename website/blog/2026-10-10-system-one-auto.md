---
slug: system-one-auto
title: "Decision Models Need a Router, Too"
description: "System One Auto gains 16.45 accuracy points over Kai on the public JevBench suite, with over half of requests staying on the small model."
authors: [Xunzhuo]
tags: [routing, evaluation, decision-models, semantic-router]
image: /img/blog/system-one-auto/hero.png
---

import { ArticleFigure } from '@site/src/components/ArticleMedia';
import { AutoResults, AutoCascade, ModelEcosystems } from '@site/src/components/SystemOneAuto';

<ArticleFigure
  src="/img/blog/system-one-auto/hero.png"
  width={1672} height={941}
  alt="System One Auto intelligently selects from a decision-model pool. An Auto routing hub highlights a selected path among Decision 2.0, Cloudflare Clef, Perplexity Decider, Laya, Fastino GLiDE and TypeSafe Jev. Route language. Route decisions."
>
  One API for a growing decision-model ecosystem. Provider logos illustrate the ecosystem; the experiment below measures Kai → Vega.
</ArticleFigure>

**Save your largest decision model for the requests that need it.**

vLLM Semantic Router already routes across open and closed LLMs. **Decision models are becoming a model pool of their own.** Open families such as [Decision 2.0](https://huggingface.co/collections/vllm-sr/decision-20), Clef, Decider and Laya sit alongside hosted APIs such as [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) and GLiDE. They classify intent, score candidates and check conditions—but which model should handle each request?

**Meet System One Auto.** Connect self-hosted decision models or compatible hosted APIs behind `vllm-sr/auto`. Configure how to select a model, when to escalate through a cascade, and whether an LLM judge should review the final candidates. Your application keeps one native API; you choose the model pool and routing policy.

<!-- truncate -->

## Better decisions. Fewer large-model calls.

For this experiment, we configured **Decision 2.0 Kai 0.6B → Vega 27B**: Kai answers first, and Vega handles requests that fail the configured gate. On the same **231 public JevBench requests**, this cascade improved accuracy over Kai while keeping most requests on the small model.

<AutoResults />

**54.11% estimated inference-cost savings.** At an equal price per Vega call, treating Kai’s cost as negligible, Auto makes **106 Vega calls instead of 231**. The model pool is configurable; this estimate uses the Kai → Vega experiment above.

The benefit is **more accurate answers than Kai, with lower average latency than Vega**. The tradeoff: Auto remains 5.63 accuracy points below Vega and its p95 is about 8.5% higher, because upgrades execute both stages.

## One router. Two model ecosystems.

<ModelEcosystems />

## A small model first. A bigger model when needed.

Kai answers the **original questions**. A configurable gate checks every required answer before deciding whether to return Kai's response or send the same input and questions to Vega.

<AutoCascade />

The first cascade supports **Choice, Score and Noul**. Its probabilities guide escalation; they do not guarantee correctness. Calibrate the gate on representative data, then evaluate it on separate requests.

## Keep the questions. Configure the model choice.

Your application sends a regular System One request:

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

Connect a **local runtime**, a separate **vLLM-SR Engine**, or a compatible **external System One API**. YAML defines the models, gates and execution budget. Existing signals and decisions can select different models or cascades; changing that policy does not change your application's questions.

For a protected listener, add `-H "Authorization: Bearer $SYSTEMONE_API_KEY"`. Try the route in Dashboard's **System One → Decision Playground**, then see which model answered in **Decision Monitoring**.

**[Build your first Auto route →](/docs/next/tutorials/algorithm/native/cascade)**

## Verify the result

The [evidence download](/img/blog/system-one-auto/evidence.zip) includes saved answers, timings and an offline replay through the unchanged upstream scorer. The [evaluation tooling](https://github.com/vllm-project/semantic-router/tree/main/tools/calibration/systemone_auto) lets you repeat the comparison on your own deployment.

<details>
<summary>Measurement method, uncertainty and limits</summary>

**Quality.** The authentic [JevBench public suite at commit `b6b8fff7e345b98c060ad26c13308860ddc67004`](https://github.com/fstandhartinger/jevbench/tree/b6b8fff7e345b98c060ad26c13308860ddc67004) contains 139 Choice, 74 Noul and 18 Score items, representing 195 source groups. The unchanged [upstream scorer](https://github.com/fstandhartinger/jevbench/blob/b6b8fff7e345b98c060ad26c13308860ddc67004/jevbench/scoring.py) uses the modal label for Choice and Score and maps Noul to yes/no. This is exact-label public-suite accuracy, **not the official v1.6.1 sealed leaderboard or its composite score**. Direct Kai, Auto and direct Vega scored 152, 190 and 203 correct out of 231. The paired intervals use 5,000 source-group bootstrap samples with seed 20261010.

**Uncertainty and calibration.** Of the 106 upgrades, 41 rescued Kai errors and three introduced errors where Kai had been right. The calibration target was 70% Kai coverage; the public suite achieved 54.11%, illustrating that coverage changes with the workload.

Auto's gain over Kai has paired 95% interval **[11.34, 21.72] percentage points**; its gap below Vega has interval **[2.59, 8.89]**. The gate threshold, **0.6059704079536342**, was frozen using a separate 192-source calibration set before this evaluation. For Noul it checks `max(p, 1-p)`, so a confident “no” can exit early. The [complete YAML](/docs/next/tutorials/algorithm/native/cascade#define-the-cascade) reproduces this operating point; it is not a universal default.

**Requests.** The controlled collector uses the pinned upstream TypeSafe request mapping, adds `require_full_input: true` to each question and requests `options.return_meta`. The stock upstream CLI does not add those options. Three seeded, sequential passes compare direct Kai, direct Vega and Auto: 2,079 timed frontend requests plus nine separate typed warmups. There are no retries or discarded outliers. All measured requests succeeded and retained complete-input and runtime-identity evidence. Quality uses only pass 0; later passes check timing and determinism, not additional independent quality samples.

**Timing and calls.** Native HTTP request counters, read outside the timed request, confirmed 231 Kai calls and 106 Vega calls for Auto in each pass. These count inference exchanges, not GPU forward passes. Model choices and answers were identical across the three passes. First-pass mean latency was 210.98 ms for Vega and 126.61 ms for Auto; it is not pooled with the two warm passes. Their paired mean-latency ratio intervals are [0.723, 0.856] and [0.726, 0.859]. The p95 figures are point estimates; this study does not establish a tail-latency confidence bound or a throughput advantage.

**Cost model.** The estimated saving assumes a constant price `C_vega` for each Vega call and negligible Kai cost. Vega-only costs `231 × C_vega`; Auto costs `231 × C_kai + 106 × C_vega`. Its relative saving is `125 / 231 − C_kai / C_vega`: **54.11% when `C_kai ≈ 0`**. Every request calls Kai, so any nonzero Kai cost applies to all 231 requests. This is a call-pricing estimate, separate from measured latency and accuracy. Length-dependent prices, serving overhead and reserved GPU capacity require their own cost model; the pilot retained two resident GPUs and did not measure a billing reduction.

**Scope.** The measured path used no model-backed routing signal. `algorithm.budget` applies after decision selection; signals use their own timeouts and request cancellation. Both native models used the exact ROCm profile. The evidence package pins the checkpoint revisions, runtime identities, image content digests and the measured source snapshot. These receipts identify the historical measurement; a rerun on another software or numerical profile needs its own validation. Native compute time, reservation limits and fewer large-model calls are not dollar-cost measurements.

</details>

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

The frozen data and original downloadable charts are available with the [figure sources](https://github.com/vllm-project/semantic-router/tree/main/website/static/img/blog/system-one-auto/source). For a new application, calibrate on its own representative data, freeze the gate, and then evaluate quality, unresolved requests, mean and tail together.

</details>

<details>
<summary>Model ecosystem and related work</summary>

The cover references [Decision 2.0](https://huggingface.co/collections/vllm-sr/decision-20), [Cloudflare Clef](https://huggingface.co/Cloudflare/clef), [Perplexity Decider](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) and [Laya](https://github.com/NandhaKishorM/laya) as open-model examples. Clef is also available as a hosted service. [Fastino GLiDE](https://fastino.ai/blog/introducing-glide-the-first-thinking-decision-model) and [TypeSafe Jev](https://docs.typesafe.ai/introduction/quickstart) represent hosted decision APIs. GLiDE’s no-thinking label refers to the [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) variant; at the time of writing, its open weights were announced but not released, and we have not verified a public API selector for that variant.

External bindings require a compatible native API and provider-specific validation. Direct native model requests bypass recipe routing; Responses requires its service to be enabled. The logos indicate the broader ecosystem, not tested adapters or partnerships. This article’s measured results use only Decision 2.0 Kai and Vega.

Confidence cascades have public precedents, including [Jev-Style's native decision cascade](https://github.com/lawrence3699/jev-style/commit/5c8149c26256ff4dc1b1e8693a70935b40f90e75). System One Auto integrates this pattern with vLLM-SR's native model bindings, signals, decisions and bounded execution. The experiment above evaluates Kai → Vega; it does not benchmark every supported provider or routing strategy.

</details>
