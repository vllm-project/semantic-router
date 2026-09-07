# Benchmark and Evaluation Tools

The `bench/` tree contains runnable evaluations for routing quality, reasoning
mode, session continuity, hallucination detection, Router Flow, and signal
runtime performance. It is a contributor workspace rather than a collection of
published benchmark results. Generated artifacts belong in an output directory
and are ignored by Git.

For an interpretation of maintained benchmark coverage, see the
[benchmarking overview](../website/docs/benchmarking/overview.md).

## Find the right runner

| Question | Runner |
| --- | --- |
| Does the router improve reasoning-task selection over a direct backend? | `vllm-semantic-router-bench` |
| Does a model benefit from its reasoning mode, and what does that cost? | `reasoning-mode-eval` |
| Does session routing preserve continuity and tool-loop invariants? | `agentic_routing_experiment.py` and `agentic_routing_live_benchmark.py` |
| Does production protection obey maintained per-turn contracts? | [`make bench-agent-routing-protection`](../website/docs/benchmarking/agent-routing-protection.md) |
| Does a routed model complete maintained multi-turn agent tasks? | `agent_task_live_benchmark.py` |
| Does a backend report prompt-cache usage through the router? | `cache_token_probe.py` |
| Do Router Flow arms improve answer quality? | [`router_flow/`](router_flow/README.md) |
| Does grounding-aware fusion help on DRACO? | [`grounded_fusion/`](grounded_fusion/README.md) |
| How do hallucination detectors compare? | [`hallucination/`](hallucination/README.md) |
| What is the signal latency across CPU, GPU, attention, and body modes? | [`cpu-vs-gpu/`](cpu-vs-gpu/README.md) |

Start with the smallest dataset or `--dry-run` mode supported by the selected
runner. Run `python <script> --help` or the installed command's `--help` before
launching a full evaluation; the CLI is the source of truth for flags and
defaults.

## Install

From the repository root:

```bash
python3 -m venv .venv-bench
. .venv-bench/bin/activate
python -m pip install -e 'bench[dev]'
```

Router Flow's EvalScope runner has additional dependencies:

```bash
python -m pip install -e 'bench[real_eval]'
```

Most live evaluations require one or more OpenAI-compatible endpoints. Keep API
keys in environment variables; do not put credentials in commands, result
files, or committed configuration.

## Reasoning evaluations

List the datasets registered by the installed package, then run a small
comparison:

```bash
vllm-semantic-router-bench list-datasets
vllm-semantic-router-bench test \
  --dataset mmlu \
  --samples 5 \
  --mode both
```

`test` is a quick sample. `compare` accepts explicit router and direct-vLLM
endpoints, and `comprehensive` runs multiple registered datasets. Use
`vllm-semantic-router-bench <command> --help` for their endpoint and output
options.

### Compare a model's reasoning modes

`reasoning-mode-eval` sends the same dataset samples with reasoning disabled
and enabled. It reports accuracy, token use, response time, and time per output
token.

```bash
reasoning-mode-eval \
  --datasets mmlu gpqa \
  --model qwen3-14b \
  --reasoning-family qwen3 \
  --endpoint http://127.0.0.1:8000/v1 \
  --samples-per-category 10 \
  --output-dir results/reasoning-mode
```

The run writes `vsr_canonical_patch.yaml` and
`vsr_canonical_patch_recommendation.json` alongside its detailed results. The
patch is a review aid, not an automatically safe deployment change. Confirm the
model name, reasoning family, measured trade-offs, and existing config entries
before merging it.

A known-family patch uses the current v0.3 provider and Model Card fields:

```yaml
providers:
  defaults:
    reasoning_families:
      qwen3:
        type: chat_template_kwargs
        parameter: enable_thinking
    default_reasoning_effort: medium
  models:
    - name: qwen3-14b
      reasoning_family: qwen3
routing:
  modelCards:
    - name: qwen3-14b
```

Reasoning is enabled per decision reference, after the evaluated model has been
merged into the existing configuration:

```yaml
routing:
  decisions:
    - name: math-reasoning
      rules:
        operator: AND
        conditions:
          - type: domain
            name: math
      modelRefs:
        - model: qwen3-14b
          use_reasoning: true
          reasoning_effort: high
```

Supported generated family mappings are `qwen3`, `deepseek`, and `gpt-oss`.
When `--reasoning-family` is omitted or unknown, the recommendation records a
manual follow-up instead of inventing a provider mapping.

## Session-routing evaluations

The session tools cover two distinct boundaries:

- `agentic_routing_experiment.py` is deterministic and does not call a live
  model. Use it for scenario matrices, policy ablations, replay input, and the
  Router Learning architecture gate.
- `agentic_routing_live_benchmark.py` calls a router and can compare it with a
  direct backend. It measures success, latency, selected diagnostic headers,
  continuity, context portability, and recovery behavior.

Run the deterministic gate with the maintained PR thresholds:

```bash
make bench-router-learning
```

Use `PROFILE=release` only when performing a release evaluation. Profile
semantics and output files are documented in
[`profiles/router_learning/`](profiles/router_learning/README.md).

Before sending traffic, inspect a live scenario without network calls:

```bash
python bench/agentic_routing_live_benchmark.py \
  --scenario balanced \
  --sessions 2 \
  --turns 3 \
  --dry-run \
  --output-dir results/session-routing-dry-run
```

For a live comparison, provide both endpoints and set explicit acceptance
thresholds. `--require-router-diagnostics` checks the maintained `x-vsr-*`
response-header contract.

The related tools are intentionally separate:

| Tool | Purpose |
| --- | --- |
| `agent_task_live_benchmark.py` | Score maintained smoke or long-horizon tasks and their tool transitions |
| `cache_token_probe.py` | Repeat a session prefix and classify cached-token reporting as missing, zero, or positive |
| `openai_fault_proxy.py` | Inject controlled upstream failures, plus optional fixed and jittered response latency, for recovery tests |
| `session_routing_branch_image_probe.py` | Record diagnostics from a reviewed branch image |
| `session_routing_branch_image_benchmark.py` | Assemble the diagnostic, live, failure, task, and cache summaries |
| `session_routing_ga_report.py` | Apply release thresholds to the assembled machine-readable evidence |
| `plot_session_routing_figures.py` | Render figures from existing summaries without rerunning traffic |

The branch-image and GA tools validate evidence; they do not build, deploy, or
approve an image. Use immutable image tags or digests and record the reviewed
source ref whenever those artifacts are used for a release decision.

## Results and reproducibility

Use a new output directory for each run. At minimum, record:

- source revision and immutable image identity;
- endpoint implementation and served model name;
- dataset revision, sample count, seed, and filters;
- generation settings and concurrency;
- hardware and relevant runtime versions;
- every threshold used to produce a pass/fail verdict.

Do not present a smoke limit, synthetic fixture, or dry run as a full benchmark.
Do not commit raw prompts, responses, headers, or environment captures until
they have been checked for credentials and user data.

## Contributor checks

Run the lightweight benchmark tests without starting model servers:

```bash
python -m pytest \
  bench/test_*.py \
  bench/grounded_fusion/test_*.py \
  bench/router_flow/real_eval/test_*.py
```

When changing a CLI, update its `--help`, tests, and this index only if the
reader's choice of runner or first command changes. Detailed experiment design
belongs with the runner that implements it.
