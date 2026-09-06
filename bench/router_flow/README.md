# Router Flow Evaluation

This directory contains two evaluation paths for Router Flow:

- `flow_eval.py` is a lightweight development harness for comparing several
  model arms on the same local prompt set.
- [`real_eval/`](real_eval/README.md) runs maintained EvalScope adapters and is
  the path for benchmark scores intended for external comparison.

The lightweight harness is useful for prompt, output-contract, timeout, and
trace regression checks. Its bundled prompts are proxies; they must not be
reported as official SWE-bench, TerminalBench, LiveCodeBench, GPQA, or other
benchmark results.

See [`SIGNAL_DECISION_EVAL_PLAN.md`](SIGNAL_DECISION_EVAL_PLAN.md) for the
durable signal, decision, and evaluation design rules used by benchmark-specific
recipes under `configs/`.

## Run a development comparison

Start an OpenAI-compatible endpoint that exposes the model aliases named in
the command, then run each arm against the same prompt file:

```bash
python bench/router_flow/flow_eval.py \
  --dataset bench/router_flow/frontier_proxy_prompts.jsonl \
  --base-url http://127.0.0.1:8899/v1 \
  --arm auto=vllm-sr/auto \
  --arm single=worker-model \
  --limit 5 \
  --output-dir bench/router_flow/results/local-smoke
```

Without a judge, the harness records responses and request metrics but does not
claim answer quality. To score answers, provide a separate OpenAI-compatible
judge and name the environment variable that contains its key:

```bash
export ROUTER_FLOW_JUDGE_KEY='<set outside shell history when possible>'
python bench/router_flow/flow_eval.py \
  --dataset bench/router_flow/frontier_proxy_prompts.jsonl \
  --base-url http://127.0.0.1:8899/v1 \
  --arm auto=vllm-sr/auto \
  --arm single=worker-model \
  --judge-base-url https://judge.example.com/v1 \
  --judge-api-key-env ROUTER_FLOW_JUDGE_KEY \
  --judge-model judge-model \
  --limit 5 \
  --output-dir bench/router_flow/results/local-judged
```

For a local Qwen endpoint whose default chat template hides final answer text,
pass the endpoint-specific option explicitly:

```bash
--request-extra-json '{"chat_template_kwargs":{"enable_thinking":false}}'
```

Use `--judge-request-extra-json` for the same adjustment on the judge. These
options change the evaluated request and must be recorded with the result.

Each run writes:

- `samples.jsonl`, containing per-item responses, grades when present, latency,
  token usage, and looper headers;
- `summary.json`, containing aggregate counts and metrics by arm and category.

## Render a development report

```bash
python bench/router_flow/render_report.py \
  --results bench/router_flow/results/local-judged \
  --reference-json bench/router_flow/public_reference_scores.json \
  --output-dir bench/router_flow/results/local-report
```

The renderer creates Markdown and CSV tables, SVG charts, and metadata. Values
loaded from `public_reference_scores.json` are contextual references, not
locally reproduced results. Preserve that distinction in titles, legends, and
external write-ups.

`bench/router_flow/results/` is ignored by Git. Review prompts, responses, and
headers for credentials or private data before sharing an artifact.
