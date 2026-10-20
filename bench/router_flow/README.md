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

## Model-free standalone HTTP timing

`standalone_http.py` measures transport and rule routing, not Router Flow answer
quality. It requires a controlled provider returning `ok`, with code-keyword
requests routed to `code-model` and the default routed to `general-model`.
It rejects wrong-model and wrong-content responses instead of counting them
as fast successful requests.

Prepare the provider environment described in the
[native smoke guide](../../tools/docker/riscv64/README.md#minimal-native-smoke).
In one terminal, start the provider on loopback:

```bash
PYTHONPATH=tools/test/services/provider-mocker PROVIDER_MOCKER_SCENARIO=cli \
  "$VSR_SMOKE_WORK/provider/bin/python" -m provider_mocker \
  --host 127.0.0.1 --port 9009
```

In another terminal, generate a credential-free configuration from the same
fixture as the smoke test and run the locally built image:

```bash
VSR_HTTP_WORK=$(mktemp -d)
python3 - "$VSR_HTTP_WORK/config.json" <<'PY'
import runpy
import sys
from pathlib import Path

fixture = runpy.run_path("tools/test/smoke/riscv64-standalone.py")
fixture["write_configuration"](Path(sys.argv[1]), "127.0.0.1:9009", 8899)
PY
docker run --detach --name vsr-http-bench --network host \
  --cpus 2 --memory 1g --pids-limit 256 --env GOMAXPROCS=2 \
  --user 65532:65532 --read-only --cap-drop ALL --security-opt no-new-privileges \
  --tmpfs /tmp:rw,noexec,nosuid,mode=1777 \
  --tmpfs /app/config:rw,noexec,nosuid,uid=65532,gid=65532,mode=0700 \
  --tmpfs /app/models:rw,noexec,nosuid,uid=65532,gid=65532,mode=0700 \
  --volume "$VSR_HTTP_WORK/config.json:/app/config/config.yaml:ro" \
  semantic-router-router:public -gateway=standalone -listener-address=127.0.0.1
curl --fail http://127.0.0.1:8899/ready
python3 bench/router_flow/standalone_http.py \
  --endpoint http://127.0.0.1:8899 --seconds 30 --repeats 3 \
  --output "$VSR_HTTP_WORK/http.json"
docker rm --force --volumes vsr-http-bench
```

Use a free container name and free ports, and stop the provider after the run.
For x86, build the `vllm-sr-attached` target in
`tools/docker/Dockerfile.attached` with an amd64 tag
and substitute that tag above. Keep the source revision, CPU/memory limits,
configuration and provider inputs the same across hosts.

The client warms up with 20 requests, then runs concurrency 1, 4 and 8 for
each repeat. Each request uses a fresh HTTP connection. JSON output retains
raw latency samples, errors, request counts, throughput and nearest-rank
p50/p95. Timing includes client and provider work on the host, outside the
Router's CPU limit. Record host hardware and background load; these numbers
are not ISA-only performance measurements or saturated Router throughput.

Run its local response-validation tests with:

```bash
python3 -m unittest discover -s bench/router_flow -p test_standalone_http.py
```
