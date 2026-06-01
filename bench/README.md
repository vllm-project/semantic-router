# vLLM Semantic Router Benchmark Suite

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A comprehensive benchmark suite for evaluating **semantic router** performance against **direct vLLM** across multiple reasoning datasets. Perfect for researchers and developers working on LLM routing, evaluation, and performance optimization.

## 🎯 Key Features

- **6 Major Reasoning Datasets**: MMLU-Pro, ARC, GPQA, TruthfulQA, CommonsenseQA, HellaSwag
- **Router vs vLLM Comparison**: Side-by-side performance evaluation
- **Multiple Evaluation Modes**: NR (neutral), XC (explicit CoT), NR_REASONING (auto-reasoning)
- **Reasoning Mode Evaluation** (Issue #42): Dedicated standard vs reasoning mode comparison
- **Research-Ready Output**: CSV files and publication-quality plots
- **Dataset-Agnostic Architecture**: Easy to extend with new datasets
- **CLI Tools**: Simple command-line interface for common operations

## 🚀 Quick Start

### Installation

```bash
pip install vllm-semantic-router-bench
```

### Live Session-Aware Routing Benchmark

Use the live benchmark when a router or Envoy stack is already running and you
need repeatable system evidence for session-aware agentic routing:

```bash
python3 bench/agentic_routing_live_benchmark.py \
  --base-url http://127.0.0.1:8977/v1 \
  --metrics-url http://127.0.0.1:9279/metrics \
  --model auto \
  --scenario tool-heavy \
  --sessions 8 \
  --turns 12 \
  --concurrency 2
```

For GA evidence, run the same schedule against the router and a direct backend
in one invocation so the output includes both router metrics and an overhead
comparison:

```bash
python3 bench/agentic_routing_live_benchmark.py \
  --base-url http://127.0.0.1:8899/v1 \
  --baseline-base-url http://127.0.0.1:8090/v1 \
  --baseline-model qwen/qwen3.5-rocm \
  --metrics-url http://127.0.0.1:9279/metrics \
  --model auto \
  --scenario idle-heavy \
  --sessions 16 \
  --turns 24 \
  --concurrency 4 \
  --idle-pause-seconds 65 \
  --require-router-diagnostics \
  --min-success-rate 1.0 \
  --max-tool-loop-violations 0 \
  --max-context-portability-violations 0
```

The output is written under `.agent-harness/experiments/live-agentic-routing/`
by default and includes per-turn JSONL/CSV plus a summary with success rate,
latency percentiles, selected-model switches, tool-loop switch violations,
context-portability violations, token usage, cached-token ratio, and VSR
decision headers. It also reports how many sessions experienced transient
errors and then recovered on a later turn, which is the failure-recovery metric
to use in backend disruption runs. When `--baseline-base-url` is set, the
benchmark also writes `baseline/summary.json`, `comparison.json`, and
`comparison.md` with router latency overhead, throughput ratio, and status-count
deltas. Threshold flags fail the run when success, latency, overhead, or
session-continuity invariants fall outside the configured GA bounds.
Use repeated `--require-router-header` flags when the run should prove
observability readiness, not only routing continuity. This is useful for GA
evidence that must show every successful router request emitted the selected
model, selected decision, replay id, or other `x-vsr-*` diagnostics needed to
debug a session after the fact. Use `--require-router-diagnostics` for the
standard GA diagnostic gate: selected model, selected decision, replay id,
selected confidence, and context-token count. The benchmark treats missing
headers as failures and also validates that confidence is numeric in `[0, 1]`
and context-token count is a non-negative integer.

For backend disruption experiments, run the fault proxy between the router and
the upstream vLLM-compatible backend, then point the router endpoint at the
proxy. The proxy injects selected OpenAI-compatible HTTP errors and forwards all
other requests unchanged, which keeps the benchmark workload deterministic:

```bash
python3 bench/openai_fault_proxy.py \
  --listen-host 0.0.0.0 \
  --listen-port 18090 \
  --upstream-base-url http://127.0.0.1:8000 \
  --fail-turns 1 \
  --fail-session-mod 2 \
  --fail-session-remainder 0 \
  --log-jsonl .agent-harness/experiments/live-agentic-routing/faults.jsonl
```

Configure the router endpoint `base_url` for this run to
`http://127.0.0.1:18090/v1`, then gate recovery with explicit thresholds.
For `vllm-sr serve` stacks, apply the proxy endpoint before serving or
regenerate/restart the generated Envoy config as well; changing only the router
runtime config after startup can leave Envoy still forwarding to the old
backend cluster. When the stack uses a per-run Docker network, start the proxy
on the same network as both Envoy and the upstream backend; otherwise Envoy will
return `no healthy upstream` even though the proxy is healthy from the host.

```bash
python3 bench/agentic_routing_live_benchmark.py \
  --base-url http://127.0.0.1:8899/v1 \
  --model auto \
  --scenario tool-heavy \
  --sessions 16 \
  --turns 8 \
  --concurrency 4 \
  --min-success-rate 0.85 \
  --min-sessions-with-errors 8 \
  --min-session-recovery-rate 1.0 \
  --max-tool-loop-violations 0
```

Use `--fail-phases tool_loop,provider_state` when the run needs to target
agentic phases instead of fixed turn numbers. The default fault policy injects
at most one fault per selected session so the recovery metric measures whether
later turns can continue after a transient backend failure.
Use `--repeat-failures` for stress runs that should inject on every selected
matching phase instead of only once per selected session.

### Cache Token Reporting Probe

Use the cache-token probe when the question is whether an OpenAI-compatible
backend reports cached input tokens through the router or direct-backend path.
The probe sends a stable repeated prefix for several turns and classifies
reporting as `positive`, `reported_zero`, or `missing`:

```bash
python3 bench/cache_token_probe.py \
  --base-url http://127.0.0.1:8899/v1 \
  --model auto \
  --session-id cache-probe-router \
  --repeats 8 \
  --baseline-base-url http://127.0.0.1:8090/v1 \
  --baseline-model qwen/qwen3.5-rocm \
  --min-cached-token-reporting reported_zero
```

The output is written under `.agent-harness/experiments/cache-token-probe/` by
default. It includes per-path `summary.json` files plus an
`aggregate-summary.json` shaped for the GA report and branch-image assembler.
Each summary records `probe_kind` set to `repeated-prefix-cache-token-probe`,
the repeated-prefix hash, the prefix length, and the repeat count so GA
evidence is tied to this workload rather than an arbitrary JSON aggregate. The
probe recognizes Chat Completions style
`usage.prompt_tokens_details.cached_tokens`, Responses style
`usage.input_tokens_details.cached_tokens`, and common backend root counters
such as `usage.cached_tokens` or `usage.prompt_cache_hit_tokens`; summaries
record `cached_token_source_counts` so the evidence names the backend field that
was observed. Treat `missing` as an observability limitation: the backend
response does not include a cached-token field, so router-level cache accounting
cannot claim a positive cache-hit ratio from that run. For GA cache-accounting
evidence, run with `--min-cached-token-reporting positive`,
`--min-cached-token-field-rate 1.0`, and a deployment-specific
`--min-cached-prompt-ratio` threshold. The GA report requires the repeated-prefix
probe metadata and a direct-backend baseline before positive cached-token
evidence can pass, because the cached-token claim must prove backend behavior
rather than only router summary shaping. The router and direct-backend summaries
must share the same stable-prefix hash, prefix length, and unique-suffix pattern;
otherwise the GA report treats them as different probes. They must also record
their `base_url` and `model`, and the direct-backend `base_url` must differ from
the router `base_url`; otherwise a router-only probe cannot masquerade as
backend-positive evidence. The probe writes this as an aggregate validation
failure immediately, and the GA report and branch-image assembler enforce the
same rule. The same cache-reporting gates apply to both router and
direct-backend paths.

### Live Agent Task Benchmark

Use the agent-task benchmark when routing quality needs to be measured by
deterministic multi-turn task completion instead of only continuity counters.
The benchmark runs small coding-agent traces with simulated tool observations
and scores the final turn against exact required labels, so the result is
replayable without a judge model:

```bash
python3 bench/agent_task_live_benchmark.py \
  --base-url http://127.0.0.1:8899/v1 \
  --model auto \
  --baseline-base-url http://127.0.0.1:8090/v1 \
  --baseline-model qwen/qwen3.5-rocm \
  --include-previous-response-id \
  --min-success-rate 1.0 \
  --min-task-success-rate 0.75 \
  --max-tool-loop-violations 0 \
  --max-context-portability-violations 0
```

The output is written under `.agent-harness/experiments/live-agent-tasks/` by
default and includes per-turn CSV/JSONL, a machine-readable summary, a Markdown
summary, and a router-vs-direct-backend comparison when a baseline endpoint is
configured.

Use the long-horizon suite when the experiment should stress longer traces,
provider-state continuations, topic drift, cache-accounting decisions,
backend-failure/session-recovery triage, observability explanations, PR review
follow-up, codebase refactor planning, research artifact review, tool-error
recovery, research synthesis, maintainer issue/PR board reconciliation,
configuration contract review, repo bisect debugging, dependency-upgrade
regression repair, literature/data extraction, stale PR rebase triage,
benchmark-regression analysis, paper figure quality review, and maintainer
handoff:

```bash
python3 bench/agent_task_live_benchmark.py \
  --base-url http://127.0.0.1:8899/v1 \
  --model auto \
  --baseline-base-url http://127.0.0.1:8090/v1 \
  --baseline-model qwen/qwen3.5-rocm \
  --include-previous-response-id \
  --suite long-horizon \
  --task-repetitions 3 \
  --max-tokens 192 \
  --require-router-diagnostics \
  --min-success-rate 1.0 \
  --min-task-success-rate 0.75 \
  --max-tool-loop-violations 0 \
  --max-context-portability-violations 0
```

The GA readiness gate expects the current long-horizon suite rather than stale
diagnostics: 23 task types, all required phases (`user_turn`, `tool_loop`,
`provider_state`, `topic_drift`, `idle_boundary`, `final`), and 3 repetitions.
That produces at least 399 requests and 69 scored task instances for the
maintained suite. Smaller historical summaries remain useful diagnostics, but
they intentionally block GA.

### Session Routing GA Readiness Report

Use the branch-image diagnostic probe first when validating a freshly built
router stack. It sends one OpenAI-compatible chat completion and writes a
diagnostic summary that proves the standard router headers are present:

```bash
python3 bench/session_routing_branch_image_probe.py \
  --base-url http://127.0.0.1:8899/v1 \
  --model auto \
  --ref "$(git rev-parse --short HEAD)" \
  --image-tag "$TAG" \
  --output-dir .agent-harness/experiments/branch-image-diagnostic/current
```

The probe fails when the running stack does not emit selected model, selected
decision, replay id, selected confidence, or context-token count headers. A
passing diagnostic probe is useful evidence, but it does not satisfy the GA
branch-image benchmark requirement by itself. The readiness report treats
diagnostic-only or mounted-binary artifacts as blockers until a full
branch-image benchmark summary is available.

After the branch image has produced the diagnostic probe, live matrix, failure
recovery, and expanded agent-task summaries, assemble the full branch-image
artifact. The live, failure-recovery, cache-token, and agent-task benchmark
commands used for branch-image evidence must include the same identity fields:

```bash
--evidence-ref "$(git rev-parse --short HEAD)" --evidence-image-tag "$TAG"
```

The assembler rejects child summaries that are missing those fields or whose
identity does not match the requested commit ref and image tag. The diagnostic
probe is part of the same identity contract: it must be a
`branch-image-diagnostic-probe` summary with the same `--ref` and `--image-tag`
used by the full branch-image assembler:

```bash
python3 bench/session_routing_branch_image_benchmark.py \
  --diagnostic-summary .agent-harness/experiments/branch-image-diagnostic/current/summary.json \
  --live-aggregate .agent-harness/experiments/live-agentic-routing/branch-image-long-session/aggregate-summary.json \
  --failure-aggregate .agent-harness/experiments/live-agentic-routing/branch-image-repeat-failure/aggregate-summary.json \
  --agent-task-summary .agent-harness/experiments/live-agent-tasks/branch-image-long-horizon/summary.json \
  --cache-aggregate .agent-harness/experiments/cache-token-probe/branch-image-cache/aggregate-summary.json \
  --ref "$(git rev-parse --short HEAD)" \
  --image-tag "$TAG" \
  --output-dir .agent-harness/experiments/live-agentic-routing/branch-image-ga
```

This assembler is the only branch-image path that writes
`validation_kind: full-branch-image-benchmark` and
`branch_image_benchmark: true`. It still exits non-zero if the image tag or
label looks like a mounted-binary run, if the diagnostic probe failed, if live
or recovery summaries contain continuity failures, or if the agent-task summary
does not meet the expanded GA gate. It also requires a repeated-prefix
cache-token probe summary with a direct-backend baseline and the same router
evidence identity. The branch-image assembler does not require positive
cached-token reporting by default; that remains a separate GA readiness blocker
owned by the cache-token report gate. It does require the router and
direct-backend cache summaries to use the same repeated-prefix identity and
different recorded `base_url` values.

Use the GA readiness report after local, AMD, agent-task, cache-token, and
branch-image runs have produced machine-readable summaries. The report does not
replace the individual benchmark gates; it verifies that the required evidence
exists together and turns missing evidence into explicit blockers:

Generate the synthetic ablation input with `bench/agentic_routing_experiment.py
--ablation`. The maintained ablation matrix must include `single-turn` for the
non-session-aware routing baseline, `acr-initial` for the merged #1974
implementation baseline, and `acr-full` for the current full ACR policy under
evaluation. The
readiness report blocks GA when the initial-implementation baseline is absent,
because the release claim must compare the final policy against both the
non-session-aware path and the first merged agentic-routing implementation.

Generate publication figures from the maintained CSV outputs with the plotting
companion script. The experiment generator remains standard-library-only for
AMD hosts; the plotting script is intentionally separate and requires the bench
plotting dependencies:

```bash
python3 bench/plot_session_routing_figures.py \
  --scenario-summary .agent-harness/experiments/agentic-routing/current-initial-baseline-matrix/scenario_summary.csv \
  --scenario-seed-summary .agent-harness/experiments/agentic-routing/current-initial-baseline-matrix/scenario_seed_summary.csv \
  --ablation-summary .agent-harness/experiments/agentic-routing/current-initial-baseline-ablation/ablation_summary.csv \
  --output-dir .agent-harness/experiments/agentic-routing/current-initial-baseline-figures
```

The script writes `experiment-matrix.png`, `policy-ablation.png`, and
`seed-stability.png`. When agent-task summaries are provided, it also writes
`agent-task-readiness.png` from the same GA thresholds used by the report:
399 requests, 23 task types, 69 scored instances, and the required router
diagnostic headers. Use image-generated bitmap assets for explanatory blog or
paper schematics; do not hand-draw SVG architecture diagrams for this
workstream. Measured result charts should come from this CSV-to-plot path.

```bash
python3 bench/session_routing_ga_report.py \
  --synthetic-matrix-summary .agent-harness/experiments/agentic-routing/current-initial-baseline-matrix/summary.json \
  --synthetic-ablation-summary .agent-harness/experiments/agentic-routing/current-initial-baseline-ablation/summary.json \
  --live-aggregate .agent-harness/experiments/live-agentic-routing/amd-long-session-20260531/aggregate-summary.json \
  --failure-aggregate .agent-harness/experiments/live-agentic-routing/amd-repeat-failure-20260531/aggregate-summary.json \
  --agent-task-summary .agent-harness/experiments/live-agent-tasks/amd-long-agent-task-observability-20260531/summary.json \
  --cache-aggregate .agent-harness/experiments/cache-token-probe/amd-cache-token-20260531/aggregate-summary.json \
  --branch-image-summary .agent-harness/experiments/live-agentic-routing/branch-image-ga/summary.json
```

The output is written under `.agent-harness/reports/session-routing-ga/` by
default as `ga-readiness.json` and `ga-readiness.md`. A strict GA invocation
exits non-zero when any requirement is missing or blocked. Use
`--allow-blockers` only when generating an interim report that should document
known gaps, such as missing positive cache-token evidence or a pending
branch-image AMD run. The command also prints a small JSON summary to stdout
with `ga_ready`, `blocker_count`, and each blocker `id`, `title`, and `status`;
maintainer cron jobs can include that summary directly in a daily release brief
while linking the full Markdown and JSON reports for details.

The full branch-image summary must identify itself as real branch-image
benchmark evidence, for example with `validation_kind:
full-branch-image-benchmark` or `branch_image_benchmark: true`. A diagnostic
probe generated by `session_routing_branch_image_probe.py` intentionally writes
`validation_kind: branch-image-diagnostic-probe`, which keeps GA blocked even
when its required headers pass. The GA report also requires the full
branch-image summary's child checks to pass, including diagnostic, live matrix,
failure-recovery, expanded agent-task, cache-token probe, mounted-binary, and
overall branch-image checks; an old or hand-written summary that only sets the
top-level marker still blocks GA.
The report also requires the assembler identity and evidence shape: non-empty
`ref`, non-empty `image_tag`, and the diagnostic, live aggregate,
failure-recovery aggregate, agent-task summary, and cache-token aggregate
sections written by `session_routing_branch_image_benchmark.py`.

### Basic Usage

```bash
# Quick test on MMLU dataset
vllm-semantic-router-bench test --dataset mmlu --samples 5

# Full comparison between router and vLLM
vllm-semantic-router-bench compare --dataset arc --samples 10

# Reasoning mode evaluation (Issue #42)
vllm-semantic-router-bench reasoning-eval --datasets mmlu gpqa --samples 10

# List available datasets
vllm-semantic-router-bench list-datasets

# Run comprehensive multi-dataset benchmark
vllm-semantic-router-bench comprehensive
```

### Reasoning Mode Evaluation (Issue #42)

Dedicated benchmark comparing standard vs reasoning mode with key metrics:

```bash
# Run reasoning mode evaluation
reasoning-mode-eval --datasets mmlu gpqa truthfulqa --samples-per-category 10

# Or use the shell script
./reasoning_mode_eval.sh
```

**Key Metrics Evaluated:**

- **Response Correctness**: Accuracy on MMLU(-Pro) and non-MMLU test sets
- **Token Usage Ratio**: `completion_tokens / prompt_tokens`
- **Time per Output Token**: Response time efficiency metric (ms)

**Automated vSR Canonical Patch Generation:**

The benchmark automatically generates a canonical v0.3 patch that can be merged into `config/config.yaml`:

```bash
# Generate a ready-to-merge canonical patch with reasoning family specification
reasoning-mode-eval \
  --datasets mmlu gpqa \
  --model qwen3-14b \
  --reasoning-family qwen3 \
  --samples-per-category 20
```

**Output includes:**

- `vsr_canonical_patch.yaml` - Ready-to-merge canonical YAML patch
- `vsr_canonical_patch_recommendation.json` - Detailed performance analysis, merge guidance, and recommendations
- Automatic recommendation based on accuracy vs. cost/latency trade-offs

**Example generated patch:**

```yaml
providers:
  defaults:
    reasoning_families:
      qwen3:
        type: chat_template_kwargs
        parameter: enable_thinking
  models:
    - name: qwen3-14b
      reasoning_family: qwen3
routing:
  modelCards:
    - name: qwen3-14b
```

**Supported reasoning families:**

- `qwen3` - Emits `chat_template_kwargs.enable_thinking`
- `deepseek` - Emits `chat_template_kwargs.thinking`
- `gpt-oss` - Emits `reasoning_effort`

### Python API

```python
from reasoning import DatasetFactory, list_available_datasets

# Load a dataset
factory = DatasetFactory()
dataset = factory.create_dataset("mmlu")
questions, info = dataset.load_dataset(samples_per_category=10)

print(f"Loaded {len(questions)} questions from {info.name}")
print(f"Categories: {info.categories}")
```

## 📊 Supported Datasets

| Dataset | Domain | Categories | Difficulty | CoT Support |
|---------|--------|------------|------------|-------------|
| **MMLU-Pro** | Academic Knowledge | 57 subjects | Undergraduate | ✅ |
| **ARC** | Scientific Reasoning | Science | Grade School | ❌ |
| **GPQA** | Graduate Q&A | Graduate-level | Graduate | ❌ |
| **TruthfulQA** | Truthfulness | Truthfulness | Hard | ❌ |
| **CommonsenseQA** | Common Sense | Common Sense | Hard | ❌ |
| **HellaSwag** | Commonsense NLI | ~50 activities | Moderate | ❌ |

## 🔧 Advanced Usage

### Custom Evaluation Script

```python
import subprocess
import sys

# Run detailed benchmark with custom parameters
cmd = [
    "router-bench",  # Main benchmark script
    "--dataset", "mmlu",
    "--samples-per-category", "20", 
    "--run-router", "--router-models", "auto",
    "--run-vllm", "--vllm-models", "openai/gpt-oss-20b",
    "--vllm-exec-modes", "NR", "NR_REASONING",
    "--output-dir", "results/custom_test"
]

subprocess.run(cmd)
```

### Plotting Results

```bash
# Generate plots from benchmark results
bench-plot --router-dir results/router_mmlu \
           --vllm-dir results/vllm_mmlu \
           --output-dir results/plots \
           --dataset-name "MMLU-Pro"
```

## 📈 Research Output

The benchmark generates research-ready outputs:

- **CSV Files**: Detailed per-question results and aggregated metrics
- **Master CSV**: Combined results across all test runs
- **Plots**: Accuracy and token usage comparisons
- **Summary Reports**: Markdown reports with key findings

### Generated Output Structure

**Note**: The following directory structure is created locally when you run the benchmark. These files are not committed to the repository.

```
results/  # Created locally when running benchmarks
├── research_results_master.csv          # Main research data
├── comparison_20250115_143022/
│   ├── router_mmlu/
│   │   └── detailed_results.csv
│   ├── vllm_mmlu/  
│   │   └── detailed_results.csv
│   ├── plots/
│   │   ├── accuracy_comparison.png
│   │   └── token_usage_comparison.png
│   └── RESEARCH_SUMMARY.md
└── reasoning_mode_eval/                  # Issue #42 evaluation results
    ├── reasoning_mode_eval_summary.json  # Full evaluation summary with all metrics
    ├── vsr_canonical_patch.yaml          # Ready-to-merge canonical patch
    ├── vsr_canonical_patch_recommendation.json  # Detailed recommendation & analysis
    ├── REASONING_MODE_EVALUATION_REPORT.md   # Human-readable report
    ├── plots/
    │   ├── MMLU-Pro_overall_comparison.png
    │   ├── MMLU-Pro_category_accuracy.png
    │   ├── MMLU-Pro_token_usage_ratio.png
    │   └── MMLU-Pro_time_per_token.png
    └── MMLU-Pro/
        ├── detailed_results.csv
        ├── standard_mode_results.csv
        └── reasoning_mode_results.csv
```

## 🚀 Using Generated vSR Patch in Production

After running the reasoning mode evaluation, merge the generated canonical patch into your semantic-router deployment:

### 1. Review the Recommendation

```bash
# Check the detailed recommendation
cat results/reasoning_mode_eval/vsr_canonical_patch_recommendation.json

# View the generated patch
cat results/reasoning_mode_eval/vsr_canonical_patch.yaml
```

### 2. Integrate into config.yaml

Merge the generated patch into the existing `providers.defaults.reasoning_families` and `routing.modelCards` sections of `config/config.yaml`:

```yaml
# config/config.yaml

providers:
  defaults:
    reasoning_families:
      qwen3:
        type: chat_template_kwargs
        parameter: enable_thinking
  models:
    - name: qwen3-14b
      reasoning_family: qwen3

routing:
  modelCards:
    - name: qwen3-14b
```

### 3. Enable Reasoning in Routes (Optional)

To enable reasoning mode for specific routes, update `routing.decisions[].modelRefs[]` and optionally set a provider-wide default effort:

```yaml
# config/config.yaml

providers:
  defaults:
    default_reasoning_effort: medium

routing:
  decisions:
    - name: math_reasoning_route
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

### 4. End-to-End Pipeline Example

```bash
# 1. Run evaluation
reasoning-mode-eval \
  --datasets mmlu gpqa truthfulqa \
  --model qwen3-14b \
  --reasoning-family qwen3 \
  --endpoint http://your-vllm-server:8000/v1 \
  --samples-per-category 50

# 2. Review results
cat results/reasoning_mode_eval/REASONING_MODE_EVALUATION_REPORT.md

# 3. If recommendation is positive, merge generated config
cp results/reasoning_mode_eval/vsr_canonical_patch.yaml /tmp/vsr_canonical_patch.yaml

# 4. Merge the patch into config/config.yaml
#    - add providers.defaults.reasoning_families entries if missing
#    - update the matching routing.modelCards entry for the evaluated model
#    - enable use_reasoning in the relevant routing.decisions modelRefs

# 5. Restart semantic-router with updated config
kubectl rollout restart deployment semantic-router  # For K8s
# OR
docker-compose restart semantic-router  # For Docker Compose
```

## 🛠️ Development

### Local Installation

```bash
git clone https://github.com/vllm-project/semantic-router
cd semantic-router/bench
pip install -e ".[dev]"
```

### Adding New Datasets

1. Create a new dataset implementation in `dataset_implementations/`
2. Inherit from `DatasetInterface`
3. Register in `dataset_factory.py`
4. Add tests and documentation

```python
from reasoning import DatasetInterface, Question, DatasetInfo

class MyDataset(DatasetInterface):
    def load_dataset(self, **kwargs):
        # Implementation here
        pass
    
    def format_prompt(self, question, style="plain"):
        # Implementation here  
        pass
```

## 📋 Requirements

- Python 3.8+
- OpenAI API access (for model evaluation)
- Hugging Face account (for dataset access)
- 4GB+ RAM (for larger datasets)

### Dependencies

- `openai>=1.0.0` - OpenAI API client
- `datasets>=2.14.0` - Hugging Face datasets
- `pandas>=1.5.0` - Data manipulation
- `matplotlib>=3.5.0` - Plotting
- `seaborn>=0.11.0` - Advanced plotting
- `tqdm>=4.64.0` - Progress bars

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Common Contributions

- Adding new datasets
- Improving evaluation metrics
- Enhancing visualization
- Performance optimizations
- Documentation improvements

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Documentation**: https://vllm-semantic-router.com
- **GitHub**: https://github.com/vllm-project/semantic-router
- **Issues**: https://github.com/vllm-project/semantic-router/issues
- **PyPI**: https://pypi.org/project/vllm-semantic-router-bench/

## 📞 Support

- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Comprehensive guides and API reference
- **Community**: Join our discussions and get help from other users

---

**Made with ❤️ by the vLLM Semantic Router Team**
