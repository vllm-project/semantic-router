---
title: Prepare reusable tasks and targets
---

# Prepare reusable tasks and targets

An evaluation runs the same frozen tasks against each selected target. This
page covers preparing those tasks from Dashboard or the CLI, reserving named
evaluation history before a holdout, registering targets, and supplying the
judges and simulators that some benchmarks require.

Open **Dashboard → Evaluation → Create evaluation** and choose benchmarks,
an evaluation size, models or recipes, and limits. Benchmarks do not need to be
downloaded first. **Review plan** automatically reuses verified data and prepares
missing datasets and their supported data dependencies. The creation page shows
progress through installation, download and freezing, then presents the frozen
plan. **Start evaluation** begins model requests only after that review.

For independent data management, **Datasets → Prepare dataset** uses the same
shared worker. Accepted preparation jobs continue if the page closes. Completed
datasets become available to both Dashboard and CLI.

The CLI uses the same service operation by default. It waits for completion and
writes the frozen manifest to standard output, so it can still be redirected to
a file. `--url` selects a remote worker; downloads and frozen dataset files remain
on that worker, not on the CLI host.

```bash
vllm-sr benchmark --store ./data/sr-bench dataset prepare \
  --benchmark mmlu-pro --benchmark gpqa-diamond \
  --profile quick > quick-dataset.json
```

Repeating `--benchmark` creates one service-owned collection job. It pins eligible
existing sources, prepares only missing benchmarks with the same seed, and
validates the final composition. Source or seed conflicts require explicit
resolution; they are not treated as missing data. A failed job can be retried
explicitly and can reuse verified completed items. Single-benchmark preparation
and explicit `dataset combine` remain available.

Use `--no-wait` to return the preparation job immediately. Closing the page or
interrupting the CLI wait does not cancel the worker's preparation. Both clients
can inspect the same jobs and prepared datasets:

```bash
vllm-sr benchmark --url http://127.0.0.1:8090 dataset options
vllm-sr benchmark --url http://127.0.0.1:8090 dataset prepare \
  --benchmark simpleqa-verified --profile smoke --no-wait
vllm-sr benchmark --url http://127.0.0.1:8090 dataset preparations
vllm-sr benchmark --url http://127.0.0.1:8090 dataset preparations PREPARATION_ID
vllm-sr benchmark --url http://127.0.0.1:8090 dataset show
```

Preparation requires Evaluation write permission in Dashboard; reading options,
progress and datasets requires read permission. Read-only Dashboard mode disables
preparation, while benchmark and profile selection remain available for browsing.
If access could not be checked, use **Refresh access** to retry the settings and
account checks. Preparing data does not require model generation permission,
start an evaluation, or make model requests. Automatic dependency installation is limited to the
allowlisted data preparation packages. It does not install execution harnesses,
build sandbox images, or provision model servers. Those remain explicit worker
setup operations. Gated sources require access approval and the appropriate
Hugging Face credential in the **worker environment**; browser or local CLI
credentials are not uploaded. Installation and download failures remain visible
on the preparation job and can be retried explicitly after the cause is fixed.

Local file imports and advanced history options use the explicit `--local` mode.
For this mode, install `vllm-sr[bench]` on the preparation host for Parquet sources.
A local file is never implicitly uploaded to a remote worker, and `--local` cannot
be combined with `--url` or `SR_BENCH_URL`:

```bash
vllm-sr benchmark --store ./data/sr-bench dataset prepare --local \
  --benchmark mmlu-pro --profile smoke \
  --source-path ./tasks.parquet --revision imported-v1
```

Local task imports require `--source-path` and `--revision`; their actual bytes
are hashed. `--limit` creates a labeled custom subset in either mode. Never edit
a prepared file in place. New questions, selection rules or source bytes create
a new identity.

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
record their upstream partition and canonical task identity. This partition is
the source's `test`, `dev`, or other upstream task collection; it is independent
of sr-bench's evaluation split and seed. Use the same partition and exact source
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

Snapshot compilation reads only explicitly named prepared datasets and frozen
run manifests from the selected local store. It reserves **all memberships**,
including planned or failed cases; membership does not establish that a model
generated a response or a person read it. The immutable snapshot contains task
identity hashes, reference digests, and source provenance, without question or
answer bodies. Named-reference reads are bounded; oversized inputs fail without
publishing a selection.

Standard keeps the existing deterministic ordering and excludes the union of
its original Quick membership and the frozen history **once**. Preparation
either produces the exact requested count or fails before publishing a dataset.
It never fills a shortfall with excluded tasks or changes the profile count.
The snapshot and per-family counts become part of the new dataset identity;
combining datasets and freezing plans preserve that provenance. Existing
artifacts are unchanged. Preparation without the new options retains its
original behavior and makes no additional history qualification.

This first identity policy requires exact source bytes, revision, normalizer and
upstream partition, with native task IDs (including the domain for τ³). GPQA
uses the full hash of its native, unformatted question within that source. Older
prepared artifacts without this identity, normalized imports, missing native
IDs, and cross-source or cross-revision mappings fail explicitly; aliases and
message hashes do not establish equivalence. They need separate provenance
reconciliation before they can support an exclusion claim.

Freeze `--evaluation-role retest` explicitly for a family that is being retested.
It can be used without a history snapshot and does not claim disjointness. A
snapshot, if supplied, still excludes its memberships; retest is never an
automatic fallback after exhaustion. An explicitly prepared GPQA retest can
remain in the default protocol with a retest disclosure. Its aggregate must
remain separate from any claimed unseen aggregate. The preparation role is
separate from the existing evaluation split label.

Named-history exclusion is a finite local provenance claim. It does not certify
complete browsing or human exposure history, or absence of upstream contamination.
Reports retain that limitation and identify explicit retest families; these
options do not introduce a new unseen-only scoring aggregate.

## Legacy plain reasoning diagnostics

The opt-in `legacy-*` adapters preserve the registered plain-prompt reasoning
configurations from `bench/reasoning`. They use the native formatters, answer
extraction and boolean correctness rules at a fixed source revision. They are
separate from the published benchmark protocols and do not change their weights.
Each configuration has its own benchmark ID and report entry.

| Native configuration | Source dataset and configuration | Native source splits |
| --- | --- | --- |
| `arc`, `arc-easy`, `arc-challenge` | `allenai/ai2_arc`, `ARC-Easy` and/or `ARC-Challenge` | `test` |
| `gpqa-main`, `gpqa-extended`, `gpqa-diamond` | `Idavidrein/gpqa`, `gpqa_main`, `gpqa_extended`, `gpqa_diamond` respectively | `train` |
| `gsm8k` | `gsm8k`, `main` | `test` |
| `aqua-rat` | `aqua_rat` | `test` |
| `drop` | `ucinlp/drop` | `validation` |
| `strategyqa` | `ChilleD/StrategyQA` | `test` |
| `sciq` | `sciq` | `test` |
| `openbookqa` | `openbookqa` | `test` |
| `truthfulqa` | `truthful_qa`, `multiple_choice` | `validation` |
| `commonsenseqa` | `commonsense_qa` | `train`, then `validation` |
| `hellaswag` | `hellaswag` | `train`, then `validation` |

The exporter accepts `gpqa` as an alias for `gpqa-main`; both produce
`legacy-gpqa-main` cases. Add the `legacy-` prefix to any other configuration
above to get its benchmark ID. MMLU-Pro has its separate legacy diagnostic.
MATH, OpenMathReasoning, explicit-CoT/XC and reasoning-mode evaluation are not part
of this catalog. Existing runners and plots remain available.

Install the optional environment on the preparation host and execution worker:

```bash
vllm-sr benchmark setup --benchmark legacy-reasoning --install
```

This installs a pinned native library and dependency lock in its own Python 3.12
environment. It does not download datasets or call models. Default `setup` does
not install this environment. The base CLI and Dashboard environment do not
import or install the native reasoning dependencies. Override its location with
`SR_BENCH_LEGACY_REASONING_ROOT` and `SR_BENCH_LEGACY_REASONING_PYTHON` when needed.

Supply authorized local JSONL records in the native source schema. A bundle
maps the native dataset, configuration and split to each file. Paths are relative
to the bundle. Each entry needs a declared revision; the exporter also hashes
the exact file bytes. It records local provenance, not a verified upstream
download. Gated datasets still require the operator's access approval and source
records. The exporter never attempts a network fallback.

For a synthetic GSM8K fixture, save this as `gsm8k.jsonl`:

```json
{"question":"There are two pairs of socks. How many socks are there?","answer":"Two times two is four. #### 4"}
```

Save its bundle as `sources.json`:

```json
{
  "version": 1,
  "kind": "fixture",
  "sources": [
    {
      "dataset": "gsm8k",
      "configuration": "main",
      "split": "test",
      "path": "gsm8k.jsonl",
      "revision": "synthetic-fixture-v1"
    }
  ]
}
```

Use `"kind": "local-import"` for an authorized dataset snapshot, retaining its
access and license records separately. Omit `configuration` for source calls
without one. ARC's combined configuration needs both source entries; CommonsenseQA
and HellaSwag need both native splits even when sampling.

Start with an ordinary JSON run template containing targets and limits, but no
`cases` or `dataset`. Export a complete run manifest and review its plan:

```bash
python -m cli.sr_bench.legacy_reasoning_export \
  --dataset gsm8k --source-bundle sources.json \
  --manifest run-template.json --output legacy-run.json \
  --seed 42 --samples-per-category 1
vllm-sr benchmark plan --manifest legacy-run.json
```

The exporter preserves native order and sampling, including GPQA's seeded choice
shuffle and SciQ's fixed choice order. Repeat `--category` to pass native category
filters. Omitting `--samples-per-category` keeps every native-selected question,
within the export bounds of 64 MiB of source/output data and 50,000 cases. The
export fails rather than silently truncating.

Native IDs, source digests, selection parameters, protocol and dependency lock
are frozen in the manifest. Some native IDs are assigned after sampling, so
case IDs also bind the source and selection. These are selection-bound
identities, not cross-seed task equivalence or exposure-history evidence.
Reference answers, solutions and native metadata stay inside the private
`answer` field. The output file is owner-readable and refuses overwrite.

Run the exported manifest through the ordinary service:

```bash
vllm-sr benchmark run --manifest legacy-run.json
```

Model requests use the same streaming transport, journal, limits and accounting
as other sr-bench tasks. Native grading sees only the completed visible final
string. It does not recover hidden reasoning, send judge requests or retry model
calls. An output-limit response remains a failed quality outcome, and reports
retain every planned case after transport or native-grader failures.

Use only `smoke` or `quick`. Neither the seed nor the native source split
establishes a holdout; `standard` is rejected. Native grading quirks remain
visible: for example, `Answer: AB` against an index reference raises the native
`TypeError` and leaves a failed case with its call and harness log.
`make test-sr-bench-legacy-reasoning` exercises all configurations on synthetic
records through a real local HTTP server and isolated workers. This fixture
coverage is not model-quality or full native-corpus qualification.

## Register targets

Register operator-owned targets for Dashboard:

```bash
vllm-sr benchmark --store ./data/sr-bench target register --file targets.json
vllm-sr benchmark target list
```

`targets.json` is an array of target objects. Each has `id`, `kind` (`single` or
`mom`), `base_url`, `model` and, when needed, `api_key_env`. Priced runs supply
`prices` keyed by actual returned model identity. Every price entry has `input`,
`cached_input`, `cache_write` and `output` rates in USD per million tokens.
Record the pricing basis and use the exact same rates in paired runs.

A MoM target also binds the expected runtime `config_hash`; preview uses its
`preview_url`. Priced MoM runs declare `max_inference_calls`. The current direct
MoM adapter requires one fully accounted inference call; unsupported compound
usage cannot be priced from the final selected model alone. Dashboard can
select registered targets but cannot edit their destinations or credentials.

A target with `session_mode: session_aware` sends one session identity per task
on its subject calls. Configure the mode on each registered target; targets are
`stateless` by default. See [the results guide](./results.md) for request headers
and report labels.

An operator can freeze native generation settings in a target's `request_params`.
These settings override the run's `sampling` defaults, including temperature,
reasoning options and output length when present. Inspect the effective profile
before comparing targets. The run's output cap must accommodate a target's fixed
`max_tokens`; lowering the cap does not rewrite that profile. Choose another
operator-registered profile when different native settings are required.

Judged benchmarks require a fixed single-model judge and
`grader_version: sr-bench-reference-judge-v1`. τ³ also needs a fixed simulator
and release `1.0.1`. Operators supply these in `benchmark-options.json` in the
store. External adapters discover the pinned environments installed by
`benchmark setup`. Set `SR_BENCH_{LCB,SCICODE,TERMINAL,TAU3}_PYTHON` and the
corresponding `_ROOT` variables to override those locations. Exact source
revisions and, for code or terminal tasks, digest-pinned sandbox images remain
required. Preflight reports missing prerequisites before dispatch.

## Next

- [Plan and run](./plan-and-run.md)
