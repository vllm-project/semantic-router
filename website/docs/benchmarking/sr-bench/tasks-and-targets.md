---
title: Prepare reusable tasks and targets
---

# Prepare reusable tasks and targets

An evaluation runs the same frozen tasks against each selected target. This
page covers preparing those tasks from Dashboard or the CLI, reserving named
evaluation history before a holdout, registering targets, and supplying the
judges and simulators that some benchmarks require.

## Prepare tasks

Open **Dashboard → Evaluation → Create evaluation** and choose benchmarks, an
evaluation size, models or recipes, and limits. Benchmarks don't need to be
downloaded first:

1. **Review plan** reuses verified data, and prepares missing datasets and
   their supported data dependencies.
2. The creation page shows progress through installation, download and
   freezing, then presents the frozen plan.
3. **Start evaluation** begins model requests, only after that review.

For independent data management, **Datasets → Prepare dataset** uses the same
shared worker. Accepted preparation jobs continue if the page closes, and
completed datasets become available to both Dashboard and the CLI.

### From the CLI

The CLI uses the same service operation by default. It waits for completion
and writes the frozen manifest to standard output, so it can still be
redirected to a file. `--url` selects a remote worker; downloads and frozen
dataset files stay on that worker, not on the CLI host.

```bash
vllm-sr benchmark --store ./data/sr-bench dataset prepare \
  --benchmark mmlu-pro --benchmark gpqa-diamond \
  --profile quick > quick-dataset.json
```

Repeating `--benchmark` creates one service-owned collection job:

- **What it does.** It pins eligible existing sources, prepares only the
  missing benchmarks with the same seed, and validates the final composition.
- **Conflicts.** Source or seed conflicts need explicit resolution; they
  aren't treated as missing data.
- **Retries.** A failed job can be retried explicitly, and can reuse verified
  completed items.
- **Alternatives.** Single-benchmark preparation and explicit
  `dataset combine` are still available.

Use `--no-wait` to return the preparation job immediately. Closing the page or
interrupting the CLI wait doesn't cancel the worker's preparation, and both
clients can inspect the same jobs and prepared datasets:

```bash
vllm-sr benchmark --url http://127.0.0.1:8090 dataset options
vllm-sr benchmark --url http://127.0.0.1:8090 dataset prepare \
  --benchmark simpleqa-verified --profile smoke --no-wait
vllm-sr benchmark --url http://127.0.0.1:8090 dataset preparations
vllm-sr benchmark --url http://127.0.0.1:8090 dataset preparations PREPARATION_ID
vllm-sr benchmark --url http://127.0.0.1:8090 dataset show
```

### Permissions and dependencies

- **Permissions.** Preparation needs Evaluation write permission in Dashboard;
  reading options, progress and datasets needs read permission. Read-only
  Dashboard mode disables preparation, while benchmark and profile selection
  stay available for browsing. If access couldn't be checked, use
  **Refresh access** to retry the settings and account checks.
- **No model calls.** Preparing data doesn't need model generation
  permission, start an evaluation or make model requests.
- **What it installs.** Automatic dependency installation is limited to the
  allowlisted data preparation packages. Execution harnesses, sandbox images
  and model servers stay explicit worker setup operations.
- **Gated sources** need access approval and the appropriate Hugging Face
  credential in the **worker environment**; browser or local CLI credentials
  aren't uploaded.
- **Failures.** Installation and download failures stay visible on the
  preparation job, and can be retried explicitly once the cause is fixed.

### Import local files

Local file imports and advanced history options use the explicit `--local`
mode. For it, install `vllm-sr[bench]` on the preparation host for Parquet
sources. A local file is never implicitly uploaded to a remote worker, and
`--local` can't be combined with `--url` or `SR_BENCH_URL`:

```bash
vllm-sr benchmark --store ./data/sr-bench dataset prepare --local \
  --benchmark mmlu-pro --profile smoke \
  --source-path ./tasks.parquet --revision imported-v1
```

- **Local task imports** need `--source-path` and `--revision`; their actual
  bytes are hashed.
- **`--limit`** creates a labeled custom subset, in either mode.
- **Never edit a prepared file in place.** New questions, selection rules or
  source bytes create a new identity.

## Reserve named evaluation history

For repeated evaluations, prepare native sources with `--source-partition` to
record their upstream partition and canonical task identity. The partition is
the source's `test`, `dev` or other upstream task collection, independent of
sr-bench's evaluation split and seed. Use the same partition and exact source
provenance throughout a history comparison.

```bash
vllm-sr benchmark --store ./data/sr-bench dataset prepare --local \
  --benchmark mmlu-pro --profile quick --source-partition test > quick.json
# Use the dataset ID returned above; --dataset and --run may be repeated.
vllm-sr benchmark --store ./data/sr-bench dataset exclusions \
  --dataset DATASET_ID --run RUN_ID --output history.json
vllm-sr benchmark --store ./data/sr-bench dataset prepare --local \
  --benchmark mmlu-pro --profile standard --source-partition test \
  --exclusion-snapshot history.json --evaluation-role holdout > standard.json
```

### The exclusion snapshot

Snapshot compilation reads only explicitly named prepared datasets and frozen
run manifests from the selected local store:

- **All memberships are reserved,** including planned or failed cases.
  Membership doesn't establish that a model generated a response or that a
  person read it.
- **No question or answer bodies.** The immutable snapshot contains task
  identity hashes, reference digests and source provenance.
- **Bounded reads.** Named-reference reads are bounded; oversized inputs fail
  without publishing a selection.

### The Standard holdout

Standard keeps the existing deterministic ordering, and excludes the union of
its original Quick membership and the frozen history **once**:

- **Exact count or failure.** Preparation either produces the exact requested
  count, or fails before publishing a dataset. It never fills a shortfall with
  excluded tasks, or changes the profile count.
- **Provenance.** The snapshot and per-family counts become part of the new
  dataset identity, and combining datasets and freezing plans keep that
  provenance. Existing artifacts are unchanged.
- **Without the new options,** preparation keeps its original behavior, and
  makes no extra history qualification.

### Task identity

The identity policy requires exact source bytes, revision, normalizer and
upstream partition, with native task IDs (including the domain for τ³). GPQA
uses the full hash of its native, unformatted question within that source.

These fail explicitly, and need separate provenance reconciliation before they
can support an exclusion claim:

- older prepared artifacts without this identity;
- normalized imports;
- missing native IDs;
- cross-source or cross-revision mappings.

Aliases and message hashes don't establish equivalence.

### Retests

Freeze `--evaluation-role retest` explicitly for a family that's being
retested:

- **No disjointness claim.** It can be used without a history snapshot. A
  snapshot, if supplied, still excludes its memberships, and retest is never
  an automatic fallback after exhaustion.
- **GPQA.** An explicitly prepared GPQA retest can stay in the default
  protocol with a retest disclosure. Its aggregate must stay separate from any
  claimed unseen aggregate.
- **Separate labels.** The preparation role is separate from the existing
  evaluation split label.

:::caution
Named-history exclusion is a finite, local provenance claim. It doesn't
certify complete browsing or human exposure history, or the absence of
upstream contamination. Reports keep that limitation and identify explicit
retest families; these options don't introduce a new unseen-only scoring
aggregate.
:::

## Register targets

Register operator-owned targets for Dashboard:

```bash
vllm-sr benchmark --store ./data/sr-bench target register --file targets.json
vllm-sr benchmark target list
```

`targets.json` is an array of target objects:

| Field | Holds |
| --- | --- |
| `id` | The target's ID |
| `kind` | `single` or `mom` |
| `base_url`, `model` | Where and what to call |
| `api_key_env` | The credential's environment variable, when needed |
| `prices` | For priced runs, keyed by the actual returned model identity. Every entry has `input`, `cached_input`, `cache_write` and `output` rates, in USD per million tokens. |

Record the pricing basis, and use exactly the same rates in paired runs.

### MoM targets

- **Runtime binding.** A MoM target also binds the expected runtime
  `config_hash`; preview uses its `preview_url`.
- **Pricing.** Priced MoM runs declare `max_inference_calls`. The current
  direct MoM adapter requires one fully accounted inference call; unsupported
  compound usage can't be priced from the final selected model alone.
- **Read-only in Dashboard.** Dashboard can select registered targets, but
  can't edit their destinations or credentials.

### Sessions and generation settings

- **Sessions.** A target with `session_mode: session_aware` sends one session
  identity per task on its subject calls. Configure the mode on each
  registered target; targets are `stateless` by default. See
  [the results guide](./results.md) for request headers and report labels.
- **Frozen generation settings.** An operator can freeze native generation
  settings in a target's `request_params`. They override the run's `sampling`
  defaults, including temperature, reasoning options and output length when
  present, so inspect the effective profile before comparing targets.
- **Output caps.** The run's output cap must accommodate a target's fixed
  `max_tokens`; lowering the cap doesn't rewrite that profile. Choose another
  operator-registered profile when different native settings are needed.

## Judges, simulators and harnesses

- **Judged benchmarks** need a fixed single-model judge and
  `grader_version: sr-bench-reference-judge-v1`.
- **τ³** also needs a fixed simulator, and release `1.0.1`.
- **Where they're set.** Operators supply these in `benchmark-options.json`
  in the store.
- **External adapters** discover the pinned environments installed by
  `benchmark setup`. Set `SR_BENCH_{LCB,SCICODE,TERMINAL,TAU3}_PYTHON` and the
  corresponding `_ROOT` variables to override those locations.
- **Pinned sources.** Exact source revisions and, for code or terminal tasks,
  digest-pinned sandbox images are still required.

Preflight reports missing prerequisites before dispatch.

## Repeated-prefix cache diagnostic

The opt-in `cache-token-probe-v1` adapter runs the repeated-prefix workload from
`bench/cache_token_probe.py`. It observes provider-reported cache fields without
grading the answer. The bundled source snapshot preserves the system message,
13,839-character prefix and indexed suffix.

Prepare its single sequence on the worker host:

```bash
python -m cli.sr_bench.cache_probe \
  --store ./data/sr-bench --repeats 6 > cache-probe-dataset.json
```

Preparation freezes the source and prefix digests, repeat count and native request
settings. It requires no downloaded dataset or optional harness. The supported
profiles are smoke/quick, and a sequence has 1-32 repeats. Use the returned dataset
object in an ordinary live run manifest with `seed: 0` and
`cost_policy: capability_only`. Select this custom diagnostic explicitly; it stays
outside automatic source selection. These settings preserve the native defaults:

```json
{
  "sampling": {"temperature": 0, "max_tokens": 16},
  "limits": {
    "concurrency": 1,
    "max_calls_per_case": 6,
    "total_timeout_s": 120
  },
  "benchmark_options": {
    "cache-token-probe-v1": {
      "thresholds": {
        "min_success_rate": 1.0,
        "min_cached_token_reporting": "reported_zero",
        "min_cached_token_field_rate": 1.0,
        "min_cached_prompt_ratio": 0.0
      }
    }
  }
}
```

Set every subject target to `session_mode: session_aware`. One Context sends the
whole sequence with a stable session ID; different arms and runs get different
IDs. A run supports one arm or a Router target followed by one direct backend.
Use `kind: mom` and the real config hash for the Router, and `kind: single` for
the backend. Paired URLs must differ, but a different URL does not establish
model equivalence or matched cache state.

The shared worker uses `x-vsr-session-id` and `x-conversation-id`. When comparing
against the retained script on the current Router, pass
`--session-header x-vsr-session-id` explicitly; the script keeps its original
default. Router phase, config-hash and inference-count evidence comes from
`x-vsr-trace`, not the retired individual debug headers.

Native overrides use preparation's `--max-tokens`, `--temperature` and `--timeout`.
The manifest's effective generation settings and request timeout must match that
frozen source. Preflight rejects extra generation fields, stateless targets,
concurrent execution and insufficient call budgets. The existing output, idle,
case and run limits still apply.

Both endpoints must support streaming. Errors, cancellation and output limits stop
later dispatch; remaining repeats stay visible. The shared normalizer still rejects
an explicit null nested cached-token count, which the retained script treats as
missing. There is no buffered fallback or automatic retry.

The adapter works in installed and copied-CLI workers. The retained native script,
branch-image reports and GA consumers are unchanged. Read the
[cache-observation results](./results.md#cache-observation-diagnostics) before
interpreting thresholds; this slice does not establish full native parity or
authorize retirement.

For the retained release-report readers, freeze optional `evidence_ref` and
`evidence_image_tag` strings beside `thresholds` in this adapter's benchmark
options. They identify the operator's declared Router revision and image. They
are not inferred from the worker or from a config-hash acknowledgement, and the
direct-baseline arm does not inherit them. Target IDs supply the report labels.
Changing these declarations changes the frozen plan.

The ordinary report includes a versioned compatibility projection for a cache-only
live Router sequence and its optional direct baseline. Extract it without running
the probe again:

```bash
vllm-sr benchmark report RUN_ID > report.json
jq -e '.cache_probe_compatibility // error("No cache probe projection")' \
  report.json > cache-aggregate.json
```

The existing branch-image and GA commands accept `cache-aggregate.json` through
their `--cache-aggregate` option. Their other evidence requirements still apply.
Direct-only, mixed, preview and recovery runs have an unsupported projection with
explicit validation failures rather than an inferred Router/baseline pair.

## Scripted agent-task Chat diagnostic

The opt-in `agent-task-chat-v1` adapter runs the maintained tasks from
`bench/agent_task_live_benchmark.py`. It shares their task definitions, prompts and
substring grader with the retained script. Each call receives the scripted full
history, including tool-result fixtures. The diagnostic does not execute tools or
use earlier model replies to construct later requests.

| Suite | Tasks per repetition | Requests per repetition | Maximum calls per task |
| --- | --- | --- | --- |
| `smoke` | 4 | 12 | 3 |
| `long-horizon` | 28 | 163 | 6 |
| `all` | 32 | 175 | 6 |

Prepare the bundled tasks locally on the worker host:

```bash
python -m cli.sr_bench.agent_task \
  --suite smoke --task-repetitions 2 \
  --store ./data/sr-bench > agent-task-dataset.json
```

Use the returned dataset object in a run manifest with `profile: quick` and
`seed: 0`. Preparation preserves native task order and freezes all scripted turns,
repetition identities and the bundled kernel digest. Repetitions must be positive
and fit the dataset case limit; invalid values are rejected rather than clamped.
The supported profiles are `smoke` and `quick`, not a Standard holdout.

Set the subject target's `session_mode` to `session_aware`. The shared worker
sends one stable session ID per task instance. Keep `max_calls_per_case` large
enough for the selected suite, and freeze generation settings in the usual
`sampling` or target `request_params` fields. The retained defaults are
`temperature: 0` and `max_tokens: 96`. Set these benchmark options:

```json
{
  "agent-task-chat-v1": {
    "grader_version": "agent-task-native-substrings-v1"
  }
}
```

The shared streaming transport reads visible final content only and stops on
unexpected failures or truncated output. Failed and unattempted task grades stay
missing. It does not copy the retained runner's continuation after HTTP failures
or its grading of truncated or hidden content. Both targets must support streaming
for a live sr-bench comparison. Missing usage or model observations remain unknown;
use `capability_only` when complete pricing evidence is unavailable.

The CLI registers this adapter directly, including workers that copy the CLI
without package metadata. Its shared kernel uses only the standard library.
The source-only retained runner requires Python 3.10+ and an installed CLI;
the bench distribution, Looper dependencies and Python floor remain unchanged.
Responses, previous-response-ID continuation and existing plots remain in the
retained runner. Real-model qualification and owner review still gate retirement.

See [token-coverage results](./results.md#scripted-agent-task-diagnostics) before
interpreting the diagnostic.

## DRACO final-answer diagnostic

The opt-in `draco-routed-answer-v1` adapter preserves the existing grounded-fusion
rubric and runs subject and judge calls through the shared worker. It does not
change the published benchmark suite or provide a cached-panel policy comparison.

Prepare an existing DRACO JSON export locally on the worker host:

```bash
python -m cli.sr_bench.draco \
  --source-path "$DRACO_PATH" --revision "$DRACO_REVISION" \
  --source-partition "$DRACO_PARTITION" \
  --domains Medicine,Law --seed 42 --max-samples 8 \
  --store ./data/sr-bench > draco-dataset.json
```

The source revision and upstream partition are required. The output freezes
source bytes, native task identities and the legacy domain filter, shuffle and
cap. Use its dataset object in a run manifest with the same profile and seed.
Only smoke/quick diagnostics are supported; missing inputs require local
preparation, not an automatic download or an inferred holdout.

Register a fixed single-model auxiliary judge and set these benchmark options:

```json
{
  "draco-routed-answer-v1": {
    "judge": "rubric-judge",
    "grader_version": "draco-legacy-rubric-strict-v1",
    "batch_size": 8
  }
}
```

Both endpoints must support OpenAI-compatible streaming. The legacy
grounded-fusion Ollama proxy returns buffered JSON and cannot serve as this
adapter's judge endpoint. Freeze generation settings rather than bypassing the
transport's parameter policy. Each case needs one subject call plus one judge
call per criterion batch; preflight rejects an insufficient call budget.

For a Fusion subject, use `kind: mom`, `cost_policy: capability_only` and omit
`max_inference_calls`. The Router's priced-call gate supports a direct call limit
of one, not a Fusion bound. Do not relabel a compound route as a single model to
obtain a price; missing child usage or prices must remain unknown.

The adapter is registered in the trusted CLI initializer, including worker images
that copy the CLI without distribution metadata. Python 3.10+ and the updated CLI
are also required by the retained grounded-fusion commands that share its rubric
kernel. Reasoning/Looper distribution requirements are unchanged.

See [weighted diagnostic results](./results.md#weighted-rubric-diagnostics) before
interpreting a run.

## Next

- [Plan and run](./plan-and-run.md)
