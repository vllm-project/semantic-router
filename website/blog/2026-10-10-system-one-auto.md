---
slug: system-one-auto
title: "System One Auto: Small First, Strong When It Counts"
description: "A small-first decision cascade behind one typed API: more accuracy than Kai, fewer Vega calls, and an explicit latency–quality tradeoff."
authors: [Xunzhuo]
tags: [routing, evaluation, decision-models, semantic-router]
image: /img/blog/system-one-auto/hero.jpg
---

import { ArticleChartGallery, ArticleFigure } from '@site/src/components/ArticleMedia';

<ArticleFigure
  src="/img/blog/system-one-auto/hero.jpg"
  width={1920} height={1080}
  alt="System One Auto. Small first. Strong when it counts. Decision 2.0 Kai to Vega, with the white vLLM Semantic Router logo."
/>

Applications make decisions long before they generate an answer: identify an intent, score a request, check whether a condition holds. A small model can handle many of those judgments quickly. A stronger model can resolve more of the difficult cases. The useful question is **when to spend the second model call**.

**System One Auto puts that choice behind one typed API.** Kai 0.6B answers the original questions. An explicit gate checks its answer probabilities. Requests that do not pass continue to Vega 27B.

On the pinned JevBench public suite, this cascade answers **54.11% of requests with Kai** and raises accuracy from **65.80% to 82.25%**. Warm mean frontend latency is **about 20% lower than direct Vega**. The tradeoff matters: Vega reaches **87.88% accuracy**, and Auto's p95 is **about 8.5% higher**. These are measured choices between quality and latency, rather than a claim that one path wins everywhere.

[**Try the cascade →**](/docs/tutorials/algorithm/native/cascade) · [**Download the evidence →**](/img/blog/system-one-auto/evidence.zip)

<!-- truncate -->

## Let the answer decide whether to continue

Kai answers the application's actual question bundle once. Its native probabilities provide the early-exit evidence; there is no additional classifier call. On escalation, Vega receives the same original state, instructions and answer criteria. Auto returns one model's complete response, preserving its answers and probabilities.

<ArticleFigure
  src="/img/blog/system-one-auto/cascade.svg"
  width={1800} height={920}
  alt="The application sends a question bundle to Kai. If every answer passes the gate, return Kai. Otherwise, ask Vega the original questions and return its complete native response."
  diagram
>
  The selected cascade makes at most two native model calls in this example. A stage must also satisfy the common answer-validity checks; if no acceptable result fits the budget, the request remains unresolved.
</ArticleFigure>

The first path supports **Choice, Score and Noul** bundles. Every required answer must pass its check. A high native probability is useful evidence, but it is not a calibrated probability of correctness. The threshold should be chosen on separate calibration data, then evaluated on held-out requests.

## One API, an explicit cascade

Publish an `api: systemone` entrypoint called `vllm-sr/auto` and bind the two models. A binding can target the local runtime, another vLLM-SR Engine frontend, or a compatible external System One service. Applications keep the same request format:

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

For a protected listener, add `-H "Authorization: Bearer $SYSTEMONE_API_KEY"`. The response's `routing` object reports the selected model, stage and algorithm-call count. Dashboard's **System One → Decision Playground** runs the same published route; **Decision Monitoring** shows its execution.

<details>
<summary>Reproduce the frozen gate with the current configuration</summary>

The [complete cascade guide](/docs/tutorials/algorithm/native/cascade) provides the provider and listener configuration. The selected decision uses this algorithm:

```yaml
algorithm:
  type: cascade
  budget: {deadline: 10s, max_calls: 2}
  quality:
    type: uncalibrated
    acceptance:
      rules:
        - {question_type: choice, field: top_probability, predicate: {gte: 0}}
        - {question_type: score, field: top_probability, predicate: {gte: 0}}
        - {question_type: noul, field: top_probability, predicate: {gte: 0}}
  stages:
    - name: fast
      kind: native
      model: kai
      accept:
        rules:
          - {question_type: choice, field: top_probability, predicate: {gte: 0.6059704079536342}}
          - {question_type: score, field: top_probability, predicate: {gte: 0.6059704079536342}}
          - {question_type: noul, field: top_probability, predicate: {gte: 0.6059704079536342}}
    - {name: strong, kind: native, model: vega}
```

This threshold was selected using a separate 192-source calibration set before the public evaluation. It reproduces this operating point; it is not a universal default. For Noul, top probability is `max(p, 1-p)`, so a confident “no” can exit early too. The common zero floor still requires valid probability evidence and complete answers; it does not promise a particular accuracy.

`algorithm.budget` limits the selected algorithm's physical model calls, transport retries and deadline. Routing signals run beforehand with their own timeouts and request cancellation. A recipe can select different decisions, each with its own native model or cascade and budget. The measured example has no model-backed routing signal.

</details>

## More accurate than Kai, with fewer Vega calls

All three paths received the same 231 public JevBench requests. Auto answered 125 with Kai and escalated 106 to Vega. Those upgrades rescued **41 Kai errors** and introduced **three errors** where Kai had been right.

<ArticleChartGallery label="Explore quality and model calls" charts={[
  { label: 'Accuracy', src: '/img/blog/system-one-auto/quality.svg', width: 3672, height: 2244, alt: 'Exact-label accuracy: direct Kai 65.80 percent, Auto 82.25 percent, direct Vega 87.88 percent. Auto gains 16.45 percentage points over Kai and remains 5.63 points below Vega.', children: 'The same 231 public items, scored with the unchanged JevBench scorer. Intervals resample the 195 source groups together across serving paths.' },
  { label: 'Request paths and calls', src: '/img/blog/system-one-auto/paths.svg', width: 3672, height: 2244, alt: 'Of 231 requests, 125 return Kai and 106 upgrade to Vega. Auto makes 231 Kai calls and 106 Vega calls: 337 native calls total.', children: '54.11% of requests finish with Kai. Every request still calls Kai, so fast-path coverage is different from the fraction of physical model calls.' },
]} />

The net gain over Kai is **16.45 percentage points**, with a paired 95% interval of **[11.34, 21.72]**. Auto remains **5.63 points below Vega**, with gap interval **[2.59, 8.89]**. This operating point improves on the small model; it does not establish Vega-equivalent quality.

The gate targeted 70% Kai coverage on calibration data and achieved 54.11% on the public suite. A threshold checks evidence, not a fixed traffic quota.

## Count the tail, too

We measured the real Router frontend, including the separate Engine backend used for Vega. Across two warm passes, Auto's mean latency fell from approximately **111 ms to 88 ms**. Its p95 rose from approximately **351 ms to 381 ms**: escalated requests pay for both stages.

<ArticleFigure
  src="/img/blog/system-one-auto/latency.svg"
  width={3672} height={2244}
  alt="Warm passes show Auto mean latency of 88.15 and 88.59 milliseconds versus Vega's 110.98 and 111.03. Auto p95 is 380.45 and 381.93 milliseconds versus Vega's 350.83 and 351.31."
>
  Mean and tail belong in the same comparison. Both warm passes contain the same 231 requests per path, at concurrency one. The first pass is retained separately because it shows a substantial warm-state effect.
</ArticleFigure>

This experiment used **two resident GPUs**, one for each model. Fewer Vega invocations are useful, but they do not alone establish a smaller GPU bill or better concurrent throughput. The result here is a serial frontend-latency tradeoff.

<details>
<summary>Measurement method, uncertainty and limits</summary>

**Quality.** The authentic [JevBench public suite at commit `b6b8fff7e345b98c060ad26c13308860ddc67004`](https://github.com/fstandhartinger/jevbench/tree/b6b8fff7e345b98c060ad26c13308860ddc67004) contains 139 Choice, 74 Noul and 18 Score items, representing 195 source groups. The unchanged [upstream scorer](https://github.com/fstandhartinger/jevbench/blob/b6b8fff7e345b98c060ad26c13308860ddc67004/jevbench/scoring.py) uses the modal label for Choice and Score and maps Noul to yes/no. This is exact-label public-suite accuracy, **not the official v1.6.1 sealed leaderboard or its composite score**. Direct Kai, Auto and direct Vega scored 152, 190 and 203 correct out of 231. The paired intervals use 5,000 source-group bootstrap samples with seed 20261010.

**Requests.** The controlled collector uses the pinned upstream TypeSafe request mapping, adds `require_full_input: true` to each question and requests `options.return_meta`. The stock upstream CLI does not add those options. Three seeded, sequential passes compare direct Kai, direct Vega and Auto: 2,079 timed frontend requests plus nine separate typed warmups. There are no retries or discarded outliers. All measured requests succeeded and retained complete-input and runtime-identity evidence. Quality uses only pass 0; later passes check timing and determinism, not additional independent quality samples.

**Timing and calls.** Physical runtime counters, read outside the timed request, confirmed 231 Kai calls and 106 Vega calls for Auto in each pass. Model choices and answers were identical across the three passes. First-pass mean latency was 210.98 ms for Vega and 126.61 ms for Auto; it is not pooled with the two warm passes. Their paired mean-latency ratio intervals are [0.723, 0.856] and [0.726, 0.859]. The p95 figures are point estimates; this study does not establish a tail-latency confidence bound or a throughput advantage.

**Scope.** Both native models used the exact ROCm profile. The evidence package pins the checkpoint revisions, runtime identities, image content digests and the measured source snapshot. These receipts identify the historical measurement; a rerun on another software or numerical profile needs its own validation. Native compute time, reservation limits and fewer large-model calls are not dollar-cost measurements.

</details>

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

To verify physical call counts, also pass `--kai-metrics` and `--vega-metrics` with the two dedicated runtimes' metrics endpoints. These counter reads happen outside the HTTP timing measurement; missing counters are unknown, not zero. Keep other traffic off those runtimes during the comparison. Without metrics endpoints, the collector still measures frontend timing and quality but cannot claim measured physical-call deltas.

The generator, frozen figure data and vector figures are available with the [figure sources](https://github.com/vllm-project/semantic-router/tree/main/website/static/img/blog/system-one-auto/source). For a new application, calibrate on its own representative data, freeze the gate, and then evaluate quality, unresolved requests, mean and tail together.

</details>

The first cascade is easy to inspect: **one small-model answer, an explicit acceptance gate, and a bounded upgrade**. Exploratory learned escalation policies did not show a stable useful gain, so they remain offline research. System One Auto starts with a path you can understand, configure and measure in your own application.
