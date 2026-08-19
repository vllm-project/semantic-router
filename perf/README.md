# Performance Framework

The performance framework measures Semantic Router overhead independently from
model-serving latency. It gives local development and CI one versioned
manifest, one result schema, one comparison policy, and the same JSON,
Markdown, and HTML reports.

Benchmark functions live beside the package hot path they measure. `perf/`
owns orchestration, environment metadata, baseline policy, and reports; it
does not duplicate Router behavior in synthetic helper code.

## Quick start

Run the fast CPU profile from the repository root:

```bash
make perf-bench-quick
```

Run the same fail-closed profile used by pull requests:

```bash
make perf-check
```

Generated artifacts are written to
`reports/perf/<environment>-<profile>/`:

```text
current.json       complete measurements and environment metadata
comparison.json    baseline comparison and coverage inventory
report.json        stable machine-readable report
report.md          terminal and GitHub job summary
report.html        standalone interactive review artifact
suites/*.log       raw output from every selected producer
```

Open `report.html` locally or read `report.md`. CI appends the Markdown report
to the job summary and uploads the complete directory.

## Performance layers

This first gate protects deterministic Router overhead:

| Suite | Real code path | Main dimensions |
| --- | --- | --- |
| `request-shape` | `pkg/extproc` request extraction and body handling | context tokens, JSON bytes, messages, tools, parallel requests |
| `decision-topology` | `pkg/decision` rule evaluation | decisions, match position, parallel requests |
| `selection` | `pkg/selection` cache affinity and context fit | candidate models, context utilization |
| `looper-core` | pure Looper/Fusion/ReMoM/Workflow helpers | algorithm, candidates, distribution |
| `looper-orchestration` | real algorithm orchestration with deterministic localhost model stubs | fanout, rounds, workers, quorum overhead |
| `classification` | model-backed CPU classification | classifier batch size and parallel calls |
| `semantic-cache` | semantic-cache operations | entries, hit/miss path, request concurrency |

The PR profile runs the first five suites. Model-backed classification and
cache suites remain opt-in in `cpu-full` until their model and store fixtures
are made hermetic enough for the PR gate.

Model time is a separate measurement layer. Future real-model suites must
record direct-backend and routed results together, then report the Router
delta:

```text
router_delta = routed_latency - direct_backend_latency
```

TTFT, TPOT/inter-token latency, end-to-end latency, tokens/s, upstream-call
count, and token amplification belong in those external result producers.
They must not be presented as component `ns/op`.

## Manifest contract

[`config/perf.yaml`](config/perf.yaml) is the source of truth:

- `environments` declare CPU/GPU kind, accelerator, capabilities, and runtime
  variables;
- `profiles` select a bounded suite inventory, sample count, benchmark time,
  and timeout;
- `suites` select the real package or external producer and document its
  dimensions and owned source paths.

The shipped environments are:

| Environment | Status | Purpose |
| --- | --- | --- |
| `cpu` | gated | Portable host-side Router regression coverage |
| `amd-gpu` | contract ready | Future ROCm/model-serving producers |
| `nvidia-gpu` | contract ready | Future CUDA/model-serving producers |

The runner contains no accelerator-specific branch. A GPU suite uses the same
manifest and result schema, and CI supplies an appropriate runner label.

Go suites use `runner: go_benchmark`. An accelerator or load generator can use
`runner: external`; the command receives these variables:

```text
VSR_PERF_ENVIRONMENT
VSR_PERF_PROFILE
VSR_PERF_SUITE
VSR_PERF_RESULT_FILE
```

The external producer must write the `current.json` schema to
`VSR_PERF_RESULT_FILE`. The framework then merges, compares, and reports it in
exactly the same way as a Go benchmark.

## Profiles and overrides

| Profile | Intended use | Sampling |
| --- | --- | --- |
| `quick` | local edit loop | one short sample |
| `ci` | CPU pull-request gate | repeated bounded samples |
| `nightly` | CPU timing trend | longer repeated samples |
| `cpu-full` | opt-in model/cache investigation | all current CPU suites |

Make variables expose the manifest without adding a second configuration
layer:

```bash
make perf-run \
  PERF_ENV=cpu \
  PERF_PROFILE=ci \
  PERF_OUTPUT_DIR="$PWD/reports/perf/my-run"
```

Validate or test only the framework code:

```bash
make perf-validate
make perf-unit
```

The full CPU profile needs the benchmark model artifacts:

```bash
make download-models-perf
make perf-bench
```

## Metrics and gating

Go benchmark repetitions are aggregated by median. Reports also record sample
count and the `ns/op` coefficient of variation so a noisy runner is visible.

- `allocs/op` and `B/op` are portable blocking metrics.
- `ns/op` is advisory on shared machines.
- A fixed same-class runner can enable `gate_ns_per_op` for a narrow threshold.
- External suites gate p95 latency growth, throughput loss, upstream-call
  growth, and token amplification against their environment baseline.
- An unbaselined current measurement fails the CI completeness gate.
- A baseline measurement missing from the selected suite also fails the gate.
- A suite producing zero measurements is an execution error.

Threshold patterns and bounds live in
[`config/thresholds.yaml`](config/thresholds.yaml). The first matching pattern
wins; unmatched benchmarks use the default.

Performance does not replace correctness. Every hot path must still pass its
normal unit and integration tests, and benchmark setup must validate the
result rather than accepting a faster wrong path.

## Baselines

The CPU PR inventory is reviewed in
[`testdata/baselines/cpu-ci.json`](testdata/baselines/cpu-ci.json). Scheduled
CI never updates it automatically.

After explaining and reviewing an intentional allocation change:

```bash
make perf-baseline-update
git diff -- perf/testdata/baselines/cpu-ci.json
```

The command captures the entire CPU CI profile and promotes that result. Commit
the baseline with the code change that requires it. Never refresh a baseline
to hide an unexplained regression.

Absolute timing from a different host is useful context but not a portable
claim. Reports retain commit, branch, Go version, OS/architecture, CPU model,
environment, profile, suites, and exact dimensions to make comparisons
auditable.

## Add or extend coverage

For a Router hot path:

1. Add `BenchmarkXxx` beside the package implementation. Keep fixture setup
   outside the timed region and call `b.ReportAllocs()`.
2. Encode actual factors in sub-benchmark names, for example
   `tokens=16384/messages=64/tools=8`.
3. Add or update a suite in `config/perf.yaml`; declare environments,
   capabilities, dimensions, and owned source paths.
4. Add the narrowest threshold pattern in `config/thresholds.yaml`.
5. Run `make perf-bench-quick`, then `make perf-check`.
6. Promote the complete baseline only after reviewing the result.

Avoid a full Cartesian product. Use single-factor scaling sweeps, bounded
pairwise combinations, production-like cases, and explicit worst corners.
Important Router axes include:

- input tokens, exact JSON bytes, message count, tools and schema bytes;
- request concurrency and classifier-internal batch size;
- decision count, rule depth, match position, and no-match scans;
- candidate models, Looper fanout, rounds, workers, and quorum;
- streaming chunks, cache hit/miss/write, cold/warm state, and fallback paths.

For a GPU or end-to-end producer, use `runner: external`, preserve direct and
routed measurements, and populate latency percentiles, throughput,
upstream-call count, and token amplification in the shared result schema.

## CI selection

Performance CI is selected for changes under the covered ExtProc, decision,
selection, and Looper hot paths, for `perf/**`, and for the performance Make
contract. The reusable workflow is
[`performance-test.yml`](../.github/workflows/performance-test.yml); the nightly
workflow calls it with the longer CPU profile.

The current PR boundary is deliberately CPU-only. Adding a GPU gate requires a
hermetic external producer, a stable runner label, a reviewed environment
baseline, and a workflow caller. It does not require changes to parsing,
comparison, or report generation.
