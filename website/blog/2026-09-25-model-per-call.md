---
slug: model-per-call
title: "Per-Call Model Selection: Measuring Cost and Quality Across a Multi-Agent Run"
description: "Multi-agent frameworks bind a model to each agent before the work starts. vLLM Semantic Router picks the model for every call instead. We ran one four-agent crew four ways and measured calls, cost, time, quality and the cost of choosing."
authors: [abhinav]
tags: [routing, agents, multi-agent, mixture-of-models, evaluation, observability, community, vllm, semantic-router]
image: /img/blog/model-per-call/hero.png
---

Every multi-agent framework asks one question before the first task arrives: which model does each agent use?

```python
planner    = Agent(role="planner",    model="small-model")
summarizer = Agent(role="summarizer", model="small-model")
reviewer   = Agent(role="reviewer",   model="frontier-model")
writer     = Agent(role="writer",     model="small-model")
```

It looks like configuration. It is really a prediction, made once, about work nobody has seen yet.

A task never reaches a model as one request. It arrives as a stream of calls: list the changed files, summarize a module, search for a concurrency bug, write the final comment. When a model is bound to an agent, every call that agent makes gets the same answer, whether the call is hard or trivial.

vLLM Semantic Router moves that choice onto the request. Every agent sends the same model name, `MoM`, short for mixture of models. It is not a model, it is a virtual name the router resolves per call, based on what the call contains, using signals and decisions written in YAML.

Two words appear throughout this post. **Local** means a small model running on your own machine, free to call and slower. **Frontier** means a large hosted model, billed per token and fast.

To see what changes, we ran one four-agent crew four ways (all frontier, model per agent, model per call, all local) and recorded every call: which model served it, which decision chose it, what it cost, and how long the router took to decide.

Per-call routing sent **2 of 16** calls to the frontier model and cost **$1.37 per 1,000 reviews**. That is **one fifth** the cost of sending everything to the frontier model, and **less than half** the cost of a hand-tuned per-agent setup. It found **4 of 6** seeded bugs, matching the all-frontier arm and beating per-agent binding, which found 3. The router took **1.86 ms** at p50 to make each choice.

<!-- truncate -->

<p align="center">
<picture>
<img src="/img/blog/model-per-call/hero.png" alt="Four agents send every call through vLLM Semantic Router, which chooses a local or frontier model per call and records each choice" width="94%" />
</picture>
<br />
<em>Figure 1: One task is a stream of calls. vLLM Semantic Router chooses the model for each one and records why.</em>
</p>

## One Task, Many Calls

The example workload is a pull-request review, a common multi-agent pattern: one agent explores the change, one summarizes it, one searches for defects, one writes the comment. The pattern matters more than the domain. Each agent makes calls of very different difficulty.

| Agent | What its calls do | Calls per task |
|---|---|---:|
| Planner | Explores the change with tools, then plans the review | 5 |
| Summarizer | Writes one sentence per changed file | 4 |
| Reviewer | Searches each code file for defects | 6 |
| Writer | Turns the notes into the final comment | 1 |
| **Total** | | **16** |

The reviewer row is where per-agent binding overpays. Reviewing a pagination helper and reviewing unsynchronized shared state are the same agent doing very different work. A per-agent setup has to send both to the same model.

## The Model Becomes a Property of the Request

With vLLM Semantic Router in front of the models, the per-agent model strings collapse into one client:

```python
from openai import OpenAI

router = OpenAI(base_url="http://localhost:8899/v1", api_key="unused")

# every agent, every call
router.chat.completions.create(model="MoM", messages=messages, tools=tools)
```

Any OpenAI-compatible agent framework takes the same two settings: a base URL and a model name. The crew code no longer holds a model choice at all. The choice lives in the router's configuration, where an operator can change it without touching the agents.

Per-agent binding still works on the same endpoint. An agent that sends `local` or `frontier` by name is served by that model directly. That is how the comparison arms below all run through one router.

<p align="center">
<picture>
<img src="/img/blog/model-per-call/per-agent-vs-per-call.png" alt="Model per agent: fixed lanes chosen at setup. Model per call: every agent through one router that chooses per request" width="92%" />
</picture>
<br />
<em>Figure 2: Per-agent binding decides once, at setup time. Per-call routing decides on every request, from its content.</em>
</p>

## How vLLM Semantic Router Chooses

The choice is three configured parts. **Signals** observe the request, **decisions** combine them with `AND` or `OR` rules ranked by priority, and each decision names the model it gets. The whole policy for this run reads in one screen:

```yaml
routing:
  signals:
    keywords:
      - name: defect_search          # what the call asks for
        operator: OR
        keywords: ["find the defects"]
      - name: subtle_code            # code whose bugs are subtle
        operator: OR
        keywords: [threading, asyncio, multiprocessing, concurrent,
                   subprocess, pickle, password, secret_key]

  decisions:
    - name: deep-review
      priority: 200
      rules:
        operator: AND
        conditions:
          - { type: keyword, name: defect_search }
          - { type: keyword, name: subtle_code }
      modelRefs:
        - model: frontier

    - name: default-local
      priority: 10
      rules: { operator: AND, conditions: [] }
      modelRefs:
        - model: local
```

A call goes to the frontier model only when it asks for a defect search **and** the code involves concurrency, processes or secrets. Summarizing a threaded module stays local, and so does searching a pagination helper. The fall-through is local on purpose: a gap in the policy then shows up as a missed defect in the results, never as a surprise on the bill.

Keyword signals need no classifier model, which keeps the cost of choosing small enough to measure on every call. The same decisions accept the router's other signal families, such as embedding similarity, complexity and domain, without any change to the agents.

One property matters a lot for agent traffic: **signals read the latest user message, and tool results never change a decision.** Inside a tool loop every follow-up call carries the same instruction, so the loop stays on the model that started it. The choice changes where the work changes, between agents and between files.

<p align="center">
<picture>
<img src="/img/blog/model-per-call/policy-flow.png" alt="Request, signals, decision, model and accounting stages for one call" width="92%" />
</picture>
<br />
<em>Figure 3: Each call passes through signals and a decision before a model is selected, and leaves with its routing and cost recorded.</em>
</p>

## Every Call Is Accounted For

Choosing per call is only useful if every choice can be explained afterwards. vLLM Semantic Router returns the routing facts on each response and keeps running totals on its Prometheus endpoint:

| Surface | Field | What it tells you |
|---|---|---|
| Response header | `x-vsr-selected-model` | The model that served this call |
| Response header | `x-vsr-selected-decision` | The decision that chose it |
| Response header | `x-vsr-routing-latency-ms` | Time the router spent choosing |
| Response header | `x-vsr-cost`, `x-vsr-cost-currency` | This call's cost, priced from the model's `pricing` block |
| `/metrics` | `llm_model_cost_total{model,currency}` | Running cost per model |
| `/metrics` | `llm_model_routing_latency_seconds` | Routing time across all calls |

One request shows all of it:

<p align="center">
<picture>
<img src="/img/blog/model-per-call/headers.png" alt="Response headers from one call showing the selected model, the decision, routing latency and cost" width="88%" />
</picture>
<br />
<em>Figure 4: The router explains every call it serves, on the response itself.</em>
</p>

Cost comes from a `pricing` block on each model, giving the price per million prompt, cached and completion tokens. The router multiplies it by the real token counts, so the crew computes nothing itself. Every number below is read from these headers and counters, which means the same numbers are available to any deployment running the router.

Prices are xAI's published list prices as of 2026-09-11, so cost here is the router's own accounting at configured prices, not a provider invoice.

## Evaluation Setup

The crew reviews one pull request that touches four files and carries six seeded defects, from obvious to subtle:

| # | File | Defect |
|---|---|---|
| D1 | `counter.py` | Shared total is read and written back without the lock |
| D2 | `counter.py` | Error counter is not synchronized either |
| D3 | `billing.py` | `parse_amount` fails on `None` or an empty string |
| D4 | `billing.py` | Bare `except: pass` silently drops line items |
| D5 | `billing.py` | Money is computed with `float` |
| D6 | `pagination.py` | Slice ends one short, so every page drops its last row |

Four arms run the identical crew, prompts and file order. Only the model name each agent sends changes: `frontier` everywhere, `frontier` for the reviewer and `local` for the rest, `MoM` everywhere, or `local` everywhere.

The model-per-agent arm is what a careful team would write by hand, putting the expensive model only where the hard thinking happens. It is the baseline per-call routing has to beat. The all-local arm shows what routing buys over simply going cheap.

| Setting | Value |
|---|---|
| Router image | `sha256:81adc125`, built 2026-09-24 |
| Local model | `llama3.2:3b`, served by Ollama on CPU |
| Frontier model | `grok-4.20-0309-non-reasoning`, xAI |
| Runs per arm | 3, temperature 0 |
| Limits | 1,024 output tokens per call, 6 calls per conversation |
| Response cache | Off, so every call reaches a model and is priced |
| Quality | Defects named in the final review, graded by hand |

Grading was blind: the reviews were shuffled, the arm labels hidden, and each was scored against a fixed answer key written before the runs. Quality is graded on the final review comment, which is what a user of the crew receives. Every table below reports the median of 3 runs, with the range where it matters.

## Result 1: One Task Is 16 Calls, and 2 Need the Frontier Model

<p align="center">
<picture>
<img src="/img/blog/model-per-call/call-tape.png" alt="Every model call in one per-call run, coloured by the model that served it" width="92%" />
</picture>
<br />
<em>Figure 5: One per-call run, the one with the median number of calls. Each tick is one model call, coloured by the model the router selected.</em>
</p>

| Agent | Calls to `local` | Calls to `frontier` |
|---|---:|---:|
| Planner | 5 | 0 |
| Summarizer | 4 | 0 |
| Reviewer | 4 | 2 |
| Writer | 1 | 0 |
| **Total** | **14** | **2** |

The policy escalated exactly two calls, and both were the reviewer working on `counter.py`, the one file that uses threads. The reviewer's calls on `billing.py` and `pagination.py` asked for a defect search too, but their code has nothing subtle in it by this policy's definition, so they stayed local. Summarizing `counter.py` also stayed local, because a summary is not a defect search.

That is the difference a per-agent setting cannot express. The reviewer is one agent doing six calls, and only two of them were worth the expensive model.

The router also keeps the tool loop stable. The planner's five calls are one conversation with tool results in the middle, and all five stayed on the same model, because the instruction that started the loop is what the signals read.

## Result 2: Cost, Time and Quality Across the Four Arms

| | All frontier | Model per agent | Model per call | All local |
|---|---:|---:|---:|---:|
| Model calls per task | 14 | 16 | 16 | 16 |
| Calls served by frontier | 14 | 6 | 2 | 0 |
| Prompt tokens | 7,918 | 7,725 | 7,605 | 7,770 |
| Completion tokens | 986 | 899 | 1,651 | 2,115 |
| **Cost per 1,000 reviews** | **$6.88** | **$2.87** | **$1.37** | **$0** |
| **Defects found (of 6)** | **4** (4 to 4) | **3** (3 to 4) | **4** (0 to 5) | **0** (0 to 1) |
| Wall time (s) | 19.3 | 80.0 | 193.8 | 290.0 |

Median of 3 runs per arm.

<p align="center">
<picture>
<img src="/img/blog/model-per-call/scorecard.png" alt="Cost per 1,000 reviews and defects found for the four arms" width="92%" />
</picture>
<br />
<em>Figure 6: The same crew, four ways. Cost from the router's configured pricing, defects graded on the final review.</em>
</p>

Costs are shown per thousand reviews because one review costs a fraction of a cent, which is hard to compare at a glance.

**Against per-agent binding**, per-call routing cost $1.37 against $2.87 and found 4 bugs against 3. The hand-tuned baseline paid for six frontier calls because its reviewer was bound to the frontier model. Per-call routing paid for two and lost nothing by it. The saving comes from splitting one agent's work, which a per-agent setting cannot express.

**Against the ceiling**, it found the same median of 4 bugs for a fifth of the cost. **Against the floor**, the difference is the review itself: the all-local arm found a median of 0 bugs, and two of its three runs reported no defects at all and described the code as already fixed.

**Time is the honest trade.** Per-call routing was slower than per-agent binding, 193.8 seconds against 80.0, because it sent 14 of 16 calls to a 3B model on CPU while the per-agent arm sent 6 to a fast hosted model. That column describes the local deployment, not the router, whose own share of the time was 0.019 percent. Serving the same local model on faster hardware changes this column and nothing else in the table, because the model, the cost and the answers all stay the same.

## Result 3: Choosing Costs 1.86 ms

| Measure | Value |
|---|---:|
| Calls measured | 48 |
| p50 | 1.86 ms |
| p95 | 5.07 ms |
| Max | 10.52 ms |
| Median model call | 4.38 s |
| Routing share of wall time | 0.019% |

Routing time is measured by the router itself and returned on every response in `x-vsr-routing-latency-ms`, so this is not an estimate from outside. A median model call in this crew took 4.38 seconds, while the decision in front of it took under 2 milliseconds.

<p align="center">
<picture>
<img src="/img/blog/model-per-call/routing-time.png" alt="Routing time per call compared with the time of one model call" width="92%" />
</picture>
<br />
<em>Figure 7: Choosing is a very small slice of the time an answer takes.</em>
</p>

These figures are for keyword signals. Signal families that run a model, such as embedding similarity or complexity scoring, add their own inference time and should be measured separately.

## What the Policy Missed

| Defect | All frontier | Model per agent | Model per call | All local |
|---|---|---|---|---|
| D1 unlocked read-modify-write | 3 of 3 | 3 of 3 | 2 of 3 | 0 of 3 |
| D2 unsynchronized error counter | 3 of 3 | 1 of 3 | 1 of 3 | 0 of 3 |
| D3 `None` amount | 0 of 3 | 0 of 3 | 2 of 3 | 0 of 3 |
| D4 bare `except` | 3 of 3 | 3 of 3 | 2 of 3 | 1 of 3 |
| D5 float money | 0 of 3 | 0 of 3 | 0 of 3 | 0 of 3 |
| D6 pagination off-by-one | 3 of 3 | 3 of 3 | 2 of 3 | 0 of 3 |

How many of the 3 runs of each arm named the bug in the final review.

Three things here are worth stating plainly.

**D5 was never found, by any arm.** Money computed with `float` instead of `Decimal` went unreported even when every call went to the frontier model. No policy can route around a bug that no model in the fleet reports.

**D3 looks like a per-call win, and it is not.** The `None` amount was found in 2 of 3 per-call runs and none of the others, but it lives in `billing.py`, which per-call routing kept local. At three runs per arm that is variation, not evidence.

**The per-call arm has one zero, and the reason is the writer.** Its runs scored 5, 4 and 0. In the zero run the router sent the reviewer's `counter.py` calls to the frontier model as designed, the frontier model found the race, and then the local writer turned the notes into a comment saying the defects had already been addressed. Quality is graded on what the user receives, so that run scored nothing.

That is a policy gap, and it is worth being precise about it. The write step's prompt contains the findings but not the words the policy escalates on, so it falls through to the local model. Two honest fixes: escalate the write step as well, which raises cost by one call, or keep it local and accept that a small model sometimes loses what a large one found. Either way the router shows you the choice it made, so the gap is visible instead of silent.

We did not change the keyword list to improve any of these numbers.

## Reproduce It

The crew, the router configuration, the pull-request fixture and the grading key are in [`bench/agent_crew/`](https://github.com/vllm-project/semantic-router/tree/main/bench/agent_crew). With Ollama serving `llama3.2:3b` and an xAI key in `XAI_API_KEY`:

```bash
cd bench/agent_crew/configs
vllm-sr serve --minimal --config crew.yaml
```

Then, from `bench/agent_crew`:

```bash
for arm in all-local per-call per-agent all-frontier; do
  python crew.py --arm $arm --repeat 3 \
    --router-metrics-url http://127.0.0.1:9190/metrics
done

python grade.py --ungraded --blind
python report.py --figures
```

One arm at a time, because the `/metrics` counters cover the whole router. Any OpenAI-compatible backend can stand in for the frontier model: change its `backend_refs` and `pricing` block, and the arms and the reported cost follow.

Each run records every call, the model that served it and what it cost, and `report.py` turns those files into the table above. The grading key, `defects.json`, was written before the runs, so anyone who disagrees with a score can re-grade the same reviews by a different standard.

## What This Changes for vLLM Users

- **Model choice moves out of agent code** and into the router, versioned with the rest of the serving configuration.
- **The unit of choice becomes the call.** Hard and trivial calls from the same agent can go to different models, which a per-agent setting cannot express. In this run that was the whole saving: 2 frontier calls instead of 6, at the same measured quality.
- **Choosing is cheap enough to ignore.** 1.86 ms at p50, 0.019 percent of the task.
- **Per-agent binding still works on the same endpoint**, so teams can adopt per-call routing one agent at a time.
- **Every choice is attributable.** Selected model, decision, routing time and cost come back on each response and accumulate on `/metrics`, so a policy can be tuned from production traffic instead of guessed at.
- **Start with a local fall-through.** A policy that escalates only on clear signals fails toward lower cost, and its misses show up in quality results rather than hidden in the bill.

## Join Us

vLLM Semantic Router is building out routing for agent workloads, and there is open work on signal families for agent traffic, per-call evaluation suites, and observability for multi-agent runs.

- GitHub: [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)
- Documentation: [vllm-sr.ai](https://vllm-sr.ai)
- Community: join the **#semantic-router** channel on [vLLM Slack](https://vllm-dev.slack.com/archives/C09CTGF8KCN)
