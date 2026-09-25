# Agent Crew Routing

This directory contains a four-agent code-review crew that runs against the
router four ways, to compare choosing a model **per agent** with choosing one
**per call**.

A planner, a summarizer, a reviewer and a writer review one pull request that
carries six seeded defects. The same crew runs four times over, and the only
thing that changes is the model name each agent sends:

| Arm | Model name sent | What it measures |
|---|---|---|
| `all-frontier` | `frontier` everywhere | the quality ceiling, and its cost |
| `per-agent` | `frontier` for the reviewer, `local` for the rest | a hand-tuned baseline |
| `per-call` | `MoM` everywhere | the router chooses per call |
| `all-local` | `local` everywhere | the cost floor, and what it loses |

Every measurement comes from the router: `x-vsr-selected-model`,
`x-vsr-selected-decision`, `x-vsr-routing-latency-ms` and `x-vsr-cost` on each
response, plus the `/metrics` delta per run. The crew computes none of them.

The pull-request fixture and its seeded defects are development fixtures for
comparing arms on identical work. They are not a code-review benchmark and
results from them must not be reported as one.

## Run a comparison

Start the router on the bundled config, which defines a `local` model, a
`frontier` model and a keyword policy that escalates only defect searches on
concurrency or security code:

```bash
cd bench/agent_crew/configs
vllm-sr serve --minimal --config crew.yaml
```

`crew.yaml` expects Ollama serving `llama3.2:3b` and an xAI key in
`XAI_API_KEY`. Any OpenAI-compatible backend works in its place: change the
`frontier` model's `backend_refs` and its `pricing` block, and the arms and the
reported cost follow.

Then run each arm against the same crew, one at a time:

```bash
python bench/agent_crew/crew.py --arm all-local    --repeat 3 \
  --router-metrics-url http://127.0.0.1:9190/metrics
python bench/agent_crew/crew.py --arm per-call     --repeat 3 \
  --router-metrics-url http://127.0.0.1:9190/metrics
python bench/agent_crew/crew.py --arm per-agent    --repeat 3 \
  --router-metrics-url http://127.0.0.1:9190/metrics
python bench/agent_crew/crew.py --arm all-frontier --repeat 3 \
  --router-metrics-url http://127.0.0.1:9190/metrics
```

Run one arm at a time. The `/metrics` counters cover the whole router, so a
second client mixes into the totals.

Each run writes `calls.jsonl` (one line per call: agent, step, model requested
and selected, decision, tokens, routing ms, cost), `review.md` (the final
review) and `summary.json` (run totals and the `/metrics` delta) under
`runs/<arm>/<run>/`.

## Grade and report

```bash
python bench/agent_crew/grade.py --ungraded --blind
python bench/agent_crew/report.py --figures
```

`grade.py --blind` shuffles the reviews and hides which arm produced each one,
then asks about each seeded defect in turn and writes `grades.json`. A defect
counts only if the review names that problem in that file: "consider error
handling" does not count for the bare `except`, "the bare except hides failed
line items" does. The answer key is `defects.json`.

`report.py` writes `runs/results.md`, with every arm side by side as a median
and range, and optional charts. Re-running it against existing run directories
rebuilds the report without calling a model.

## Tests

```bash
python -m pytest bench/agent_crew/test_crew.py bench/agent_crew/test_check_provider.py
```

These use a stub router, so they need no models, no keys and no network.

## Checking a provider

`check_provider.py` calls a provider directly and reports any reply field the
router's strict decoder would reject. Useful before wiring a new backend into a
config.

```bash
python bench/agent_crew/check_provider.py \
  --base-url https://api.x.ai/v1 \
  --api-key-env XAI_API_KEY \
  --model grok-4.20-0309-non-reasoning
```
