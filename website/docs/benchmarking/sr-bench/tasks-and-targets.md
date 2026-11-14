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

## Legacy MMLU-Pro diagnostic

`legacy-mmlu-pro` runs the plain prompt and answer extraction from the
[bench reasoning suite](https://github.com/vllm-project/semantic-router/blob/a00f4b219357fbc77410748f3e4dbdc0a8e59e28/bench/reasoning/router_reason_bench_multi_dataset.py)
through sr-bench's normal request ledger and limits. It is a separate diagnostic,
not the published `mmlu-pro` adapter. Its report has `sr_bench_score: null`;
the published benchmarks and weights are unchanged.

Install `vllm-sr[bench]` on this host, then prepare the pinned MMLU-Pro source:

```bash
vllm-sr benchmark --store ./data/sr-bench dataset prepare --local \
  --benchmark legacy-mmlu-pro --profile smoke > legacy-mmlu-dataset.json
```

Extension preparation currently requires `--local`. It does not use the
shared-worker automatic download flow. The prepared dataset can be run by the
local benchmark service like any other frozen dataset.

For a backend already serving `served-model-name` at
`http://127.0.0.1:8000/v1`, write the run manifest from the prepared dataset:

```python
import json
from pathlib import Path

dataset = json.loads(Path("legacy-mmlu-dataset.json").read_text())
manifest = {
    "version": "sr-bench-1.0",
    "name": "Legacy MMLU plain comparison",
    "mode": "live",
    "profile": dataset["profile"],
    "seed": dataset["seed"],
    "dataset": dataset,
    "targets": [{
        "id": "direct",
        "kind": "single",
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "served-model-name",
    }],
    "cost_policy": "capability_only",
    "limits": {"concurrency": 1, "max_output_tokens": 4096},
    "sampling": {"temperature": 0, "max_tokens": 4096},
}
Path("legacy-mmlu-run.json").write_text(json.dumps(manifest, indent=2))
```

Set the actual endpoint and model identity before running. This example permits
unknown prices and cannot establish cost savings.

```bash
vllm-sr benchmark --store ./data/sr-bench plan \
  --manifest legacy-mmlu-run.json --output legacy-mmlu-frozen.json
vllm-sr benchmark --store ./data/sr-bench run \
  --manifest legacy-mmlu-frozen.json
vllm-sr benchmark --store ./data/sr-bench report RUN_ID \
  --output legacy-mmlu-report.json
```

Preparation uses sr-bench's existing MMLU profile sizes and deterministic
selection, not bench's per-category sampler. To compare runners, freeze the
same source question IDs, option order and reference answers. Use `--source-path`
and `--revision` for a saved native MMLU record file; `--limit` labels a smaller
comparison as a custom subset.

The diagnostic preserves the legacy extractor's precedence, including guesses
from prose that the published grader rejects. It still grades only the visible
final channel. Reasoning-only responses do not become answers, and truncated
responses remain output failures. The legacy non-streaming/reasoning-content
fallback, explicit-CoT and reasoning-mode comparisons are outside this
diagnostic's parity claim. The old package, aliases and runners remain available.

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

## Next

- [Plan and run](./plan-and-run.md)
